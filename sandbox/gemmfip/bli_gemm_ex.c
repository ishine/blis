#include "blis.h"
#include <assert.h>


BLIS_INLINE void bli_handle_trans( obj_t* mat )
{
	if ( bli_obj_has_trans( mat ) )
	{
		dim_t  m = bli_obj_length    ( mat );
		dim_t  n = bli_obj_width     ( mat );
		inc_t rs = bli_obj_row_stride( mat );
		inc_t cs = bli_obj_col_stride( mat );
		bli_obj_set_length    (  n, mat );
		bli_obj_set_width     (  m, mat );
		bli_obj_set_row_stride( cs, mat );
		bli_obj_set_col_stride( rs, mat );
		bli_obj_toggle_trans  (     mat );
	}
}

void bli_gemm_ex
     (
       const obj_t*  alpha,
       const obj_t*  a,
       const obj_t*  b,
       const obj_t*  beta,
       const obj_t*  c,
       const cntx_t* cntx,
       const rntm_t* rntm
     )
{
	bli_init_once();

	// If C has a zero dimension, return early.
	if ( bli_obj_has_zero_dim( c ) ) return;

	// If alpha is zero, or if A or B has a zero dimension, scale C by beta
	// and return early.
	if ( bli_obj_equals( alpha, &BLIS_ZERO ) ||
	     bli_obj_has_zero_dim( a ) ||
	     bli_obj_has_zero_dim( b ) )
	{
		bli_scalm( beta, c );
		return;
	}

	// When the datatype is elidgible // & k is not too small,
	// invoke the sandbox method where dgemmsup and dpack interleaves
	// each other.
	if ( // ( k > m / 2 || k > n / 2 ) &&
		bli_obj_dt( a ) == BLIS_DOUBLE &&
		bli_obj_dt( b ) == BLIS_DOUBLE &&
		bli_obj_dt( c ) == BLIS_DOUBLE &&
		bli_obj_dt( alpha ) == BLIS_DOUBLE &&
		bli_obj_dt( beta  ) == BLIS_DOUBLE )
	{
		/* Detect small-m and transpose into small-n.
		 * TODO: For x86 only? It seems that arm64 does not need this. */
		if ( bli_obj_has_notrans( c ) ?
			( bli_obj_dim( BLIS_M, c ) < bli_min( bli_obj_dim( BLIS_N, c ), 50 ) ) :
			( bli_obj_dim( BLIS_N, c ) < bli_min( bli_obj_dim( BLIS_M, c ), 50 ) ) )
		{
			// Call C' += B'A' <=> C += A B.
			obj_t at, bt, ct;
			bli_obj_alias_to( a, &at );
			bli_obj_alias_to( b, &bt );
			bli_obj_alias_to( c, &ct );
			bli_obj_toggle_trans( &at );
			bli_obj_toggle_trans( &bt );
			bli_obj_toggle_trans( &ct );
			bli_gemm_ex ( alpha, &bt, &at, beta, &ct, cntx, rntm );
			return ;
		}

		obj_t a_loc, b_loc, c_loc;
		bli_obj_alias_to( a, &a_loc );
		bli_obj_alias_to( b, &b_loc );
		bli_obj_alias_to( c, &c_loc );
		bli_handle_trans( &a_loc );
		bli_handle_trans( &b_loc );
		bli_handle_trans( &c_loc );
		const inc_t rs_a = bli_obj_row_stride( &a_loc );
		const inc_t cs_b = bli_obj_col_stride( &b_loc );

		dim_t mr, nr;
		ukr_dgemm_sup_t milliker;
#if defined(__x86_64__) || defined(_M_X64) || defined(__i386) || defined(_M_IX86)
		if ( cs_b == 1 ) { mr = 6; nr = 8; milliker = bli_dgemmsup2_rv_haswell_asm_6x8m; }
		else { mr = 8; nr = 6; milliker = bli_dgemmsup2_cv_haswell_asm_8x6m; }
#elif defined(__aarch64__) || defined(__arm__) || defined(_M_ARM) || defined(_ARCH_PPC)
		milliker = rs_a == 1 ? bli_dgemmsup2_cv_armv8a_asm_8x6m :
			bli_dgemmsup2_rv_armv8a_asm_8x6m; mr = 8; nr = 6;
#else
		milliker = bli_dgemmsup2_ref; mr = mr_ref; nr = nr_ref;
#endif
		// Write mr, nr, and milliker to objects' customizable fields of a, b, and c.
		a_loc.ker_params = ( void * )mr;
		b_loc.ker_params = ( void * )nr;
		c_loc.ker_params = ( void * )milliker;

		rntm_t rntm_l;
		if ( rs_a == 1 || cs_b == 1 )
		{
			// Query the context for block size & packing kernels.
			if ( cntx == NULL ) cntx = bli_gks_query_cntx();
			
			// Check the operands.
			if ( bli_error_checking_is_enabled() )
				bli_gemm_check( alpha, a, b, beta, c, cntx );

			// Initialize runtime vars.
			bli_rntm_init_from_global( &rntm_l );

			// Parse and interpret the contents of the rntm_t object to properly
			// set the ways of parallelism for each loop, and then make any
			// additional modifications necessary for the current operation.
			bli_rntm_set_ways_for_op
			(
				BLIS_GEMM,
				BLIS_LEFT, // ignored for gemm/hemm/symm
				bli_obj_length( &c_loc ),
				bli_obj_width ( &c_loc ),
				bli_obj_width ( &a_loc ),
				&rntm_l
			);

			bli_l3_sup_thread_decorator
			(
				bls_dgemm, // shall become general-purpose later
				BLIS_GEMM, // operation family id
				alpha,
				&a_loc,
				&b_loc,
				beta,
				&c_loc,
				cntx,
				&rntm_l
			);
			return ;
		}

		// Otherwise, the program would pop back to the original path of exec.
	}

	if ( BLIS_SUCCESS == bli_gemmsup( alpha, a, b, beta, c, cntx, rntm ) ) return;

	// Initialize a local runtime with global settings if necessary. Note
	// that in the case that a runtime is passed in, we make a local copy.
	rntm_t rntm_l;
	if ( rntm == NULL ) { bli_rntm_init_from_global( &rntm_l ); }
	else                { rntm_l = *rntm;                       }

	// Default to using native execution.
	num_t dt = bli_obj_dt( c );
	ind_t im = BLIS_NAT;

	// If each matrix operand has a complex storage datatype, try to get an
	// induced method (if one is available and enabled). NOTE: Allowing
	// precisions to vary while using 1m, which is what we do here, is unique
	// to gemm; other level-3 operations use 1m only if all storage datatypes
	// are equal (and they ignore the computation precision).
	if ( bli_obj_is_complex( c ) &&
	     bli_obj_is_complex( a ) &&
	     bli_obj_is_complex( b ) )
	{
		// Find the highest priority induced method that is both enabled and
		// available for the current operation. (If an induced method is
		// available but not enabled, or simply unavailable, BLIS_NAT will
		// be returned here.)
		im = bli_gemmind_find_avail( dt );
	}

	// If necessary, obtain a valid context from the gks using the induced
	// method id determined above.
	if ( cntx == NULL ) cntx = bli_gks_query_ind_cntx( im );

	// Check the operands.
	if ( bli_error_checking_is_enabled() )
		bli_gemm_check( alpha, a, b, beta, c, cntx );

	// Invoke the operation's front-end and request the default control tree.
	bli_gemm_front( alpha, a, b, beta, c, cntx, &rntm_l );
}

