/*

   BLIS
   An object-based framework for developing high-performance BLAS-like
   libraries.

   Copyright (C) 2014, The University of Texas at Austin
   Copyright (C) 2018 - 2023, Advanced Micro Devices, Inc. All rights reserved.

   Redistribution and use in source and binary forms, with or without
   modification, are permitted provided that the following conditions are
   met:
    - Redistributions of source code must retain the above copyright
      notice, this list of conditions and the following disclaimer.
    - Redistributions in binary form must reproduce the above copyright
      notice, this list of conditions and the following disclaimer in the
      documentation and/or other materials provided with the distribution.
    - Neither the name(s) of the copyright holder(s) nor the names of its
      contributors may be used to endorse or promote products derived
      from this software without specific prior written permission.

   THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
   "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
   LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
   A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
   HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
   SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
   LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
   DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
   THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
   (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
   OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

*/

#include "blis.h"
//#define PRINT_SMALL_TRSM_INFO


void bli_trsm_front
     (
       side_t  side,
       obj_t*  alpha,
       obj_t*  a,
       obj_t*  b,
       cntx_t* cntx,
       rntm_t* rntm,
       cntl_t* cntl
     )
{
	AOCL_DTL_TRACE_ENTRY(AOCL_DTL_LEVEL_TRACE_3);
//	AOCL_DTL_LOG_TRSM_INPUTS(AOCL_DTL_LEVEL_TRACE_3, side, alpha, a, b);
	bli_init_once();

	obj_t   a_local;
	obj_t   b_local;
	obj_t   c_local;

	// Check parameters.
	if ( bli_error_checking_is_enabled() )
		bli_trsm_check( side, alpha, a, b, &BLIS_ZERO, b, cntx );

	// If alpha is zero, scale by beta and return.
	if ( bli_obj_equals( alpha, &BLIS_ZERO ) )
	{
		bli_scalm( alpha, b );
		AOCL_DTL_TRACE_EXIT(AOCL_DTL_LEVEL_TRACE_3);
		return;
	}

	// Alias A and B so we can tweak the objects if necessary.
	bli_obj_alias_to( a, &a_local );
	bli_obj_alias_to( b, &b_local );
	bli_obj_alias_to( b, &c_local );

	// We do not explicitly implement the cases where A is transposed.
	// However, we can still handle them. Specifically, if A is marked as
	// needing a transposition, we simply induce a transposition. This
	// allows us to only explicitly implement the no-transpose cases. Once
	// the transposition is induced, the correct algorithm will be called,
	// since, for example, an algorithm over a transposed lower triangular
	// matrix A moves in the same direction (forwards) as a non-transposed
	// upper triangular matrix. And with the transposition induced, the
	// matrix now appears to be upper triangular, so the upper triangular
	// algorithm will grab the correct partitions, as if it were upper
	// triangular (with no transpose) all along.
	if ( bli_obj_has_trans( &a_local ) )
	{
		bli_obj_induce_trans( &a_local );
		bli_obj_set_onlytrans( BLIS_NO_TRANSPOSE, &a_local );
	}

#if 1

	// If A is being solved against from the right, transpose all operands
	// so that we can perform the computation as if A were being solved
	// from the left.
	if ( bli_is_right( side ) )
	{
		bli_toggle_side( &side );
		bli_obj_induce_trans( &a_local );
		bli_obj_induce_trans( &b_local );
		bli_obj_induce_trans( &c_local );
	}

#else

	// NOTE: Enabling this code requires that BLIS NOT be configured with
	// BLIS_RELAX_MCNR_NCMR_CONSTRAINTS defined.
#ifdef BLIS_RELAX_MCNR_NCMR_CONSTRAINTS
	#error "BLIS_RELAX_MCNR_NCMR_CONSTRAINTS must not be defined for current trsm_r implementation."
#endif

	// If A is being solved against from the right, swap A and B so that
	// the triangular matrix will actually be on the right.
	if ( bli_is_right( side ) )
	{
		bli_obj_swap( &a_local, &b_local );
	}

#endif

	// Set each alias as the root object.
	// NOTE: We MUST wait until we are done potentially swapping the objects
	// before setting the root fields!
	bli_obj_set_as_root( &a_local );
	bli_obj_set_as_root( &b_local );
	bli_obj_set_as_root( &c_local );

#ifdef AOCL_DYNAMIC
	// If dynamic-threading is enabled, calculate optimum number
	//  of threads and update in rntm
	bli_nthreads_optimum(a, b, b, BLIS_TRSM, rntm );
#endif
	// Parse and interpret the contents of the rntm_t object to properly
	// set the ways of parallelism for each loop, and then make any
	// additional modifications necessary for the current operation.
	bli_rntm_set_ways_for_op
	(
	  BLIS_TRSM,
	  side,
	  bli_obj_length( &c_local ),
	  bli_obj_width( &c_local ),
	  bli_obj_width( &a_local ),
	  rntm
	);

	// If TRSM and GEMM have different blocksizes and blocksizes
	// are changed in global cntx object, when GEMM and TRSM are
	// called in parallel, blocksizes in global cntx object will
	// not be correct for GEMM
	// to fix this
	// create a local copy of cntx so that overriding the blocksizes does
	// not impact the global cntx object.
	cntx_t cntx_trsm = *cntx;

	// A sort of hack for communicating the desired pack schemas for A and B
	// to bli_trsm_cntl_create() (via bli_l3_thread_decorator() and
	// bli_l3_cntl_create_if()). This allows us to access the schemas from
	// the control tree, which hopefully reduces some confusion, particularly
	// in bli_packm_init().
	if ( bli_cntx_method( &cntx_trsm ) == BLIS_NAT )
	{
#if defined(BLIS_FAMILY_AMDZEN) ||  defined(BLIS_FAMILY_ZEN4) 
		/* Zen4 TRSM Fixme:
		 *
		 * On Zen4 we want to use AVX-512 kernels for GEMM and AVX2 kernels 
		 * for TRSM (Till we implement TRSM AVX-512 kernels)
		 * 
		 * The AVX2 kernels use different block sizes then AVX512 kernels
		 * Here we override the default block sizes in the context with AVX2 
		 * specific block size used in GEMMTRSM kernerls.
		 * 
		 * We need to revisit this when TRSM AVX-512 kernels are implemented.
		 */
		if ( (bli_arch_query_id() == BLIS_ARCH_ZEN4)  &&
			 ((bli_obj_dt(a) == BLIS_FLOAT) || (bli_obj_dt(a) == BLIS_DOUBLE)) )
		{
			bli_zen4_override_trsm_blkszs(&cntx_trsm);
		}
#endif
		bli_obj_set_pack_schema( BLIS_PACKED_ROW_PANELS, &a_local );
		bli_obj_set_pack_schema( BLIS_PACKED_COL_PANELS, &b_local );
	}
	else // if ( bli_cntx_method( cntx_trsm ) != BLIS_NAT )
	{
		pack_t schema_a = bli_cntx_schema_a_block( &cntx_trsm );
		pack_t schema_b = bli_cntx_schema_b_panel( &cntx_trsm );

		bli_obj_set_pack_schema( schema_a, &a_local );
		bli_obj_set_pack_schema( schema_b, &b_local );
	}

	// Invoke the internal back-end.
	bli_l3_thread_decorator
	(
	  bli_trsm_int,
	  BLIS_TRSM, // operation family id
	  alpha,
	  &a_local,
	  &b_local,
	  alpha,
	  &c_local,
	  &cntx_trsm,
	  rntm,
	  cntl
	);

	AOCL_DTL_TRACE_EXIT(AOCL_DTL_LEVEL_TRACE_3);
}

