/*

   BLIS
   An object-based framework for developing high-performance BLAS-like
   libraries.

   Copyright (C) 2014, The University of Texas at Austin.
   Copyright (C) 2019 - 2023, Advanced Micro Devices, Inc. All rights reserved.

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

//
// Define BLAS-to-BLIS interfaces.
//

#if defined(BLIS_KERNELS_ZEN4)

    #define TRMM_BLIS_IMPL(ch, blasname) \
        PASTEF77S(ch,blasname) ( side, uploa, transa, diaga, m, n, alpha, a, lda, b, ldb ); \
        arch_t id = bli_arch_query_id(); \
        if (id == BLIS_ARCH_ZEN4) \
        { \
            bli_zero_zmm(); \
        } \

#else

    #define TRMM_BLIS_IMPL(ch, blasname) \
        PASTEF77S(ch,blasname) ( side, uploa, transa, diaga, m, n, alpha, a, lda, b, ldb ); \

#endif

#ifdef BLIS_BLAS3_CALLS_TAPI

#undef  GENTFUNC
#define GENTFUNC( ftype, ch, blasname, blisname ) \
\
void PASTEF77S(ch,blasname) \
     ( \
       const f77_char* side, \
       const f77_char* uploa, \
       const f77_char* transa, \
       const f77_char* diaga, \
       const f77_int*  m, \
       const f77_int*  n, \
       const ftype*    alpha, \
       const ftype*    a, const f77_int* lda, \
             ftype*    b, const f77_int* ldb  \
     ) \
{ \
	AOCL_DTL_TRACE_ENTRY(AOCL_DTL_LEVEL_TRACE_1) \
	AOCL_DTL_LOG_TRMM_INPUTS(AOCL_DTL_LEVEL_TRACE_1, *MKSTR(ch), *side, \
			*uploa, *transa, *diaga, *m, *n, *alpha, *lda, *ldb) \
	side_t  blis_side; \
	uplo_t  blis_uploa; \
	trans_t blis_transa; \
	diag_t  blis_diaga; \
	dim_t   m0, n0; \
	inc_t   rs_a, cs_a; \
	inc_t   rs_b, cs_b; \
\
	/* Initialize BLIS. */ \
	bli_init_auto(); \
\
	/* Perform BLAS parameter checking. */ \
	PASTEBLACHK(blasname) \
	( \
	  MKSTR(ch), \
	  MKSTR(blasname), \
	  side, \
	  uploa, \
	  transa, \
	  diaga, \
	  m, \
	  n, \
	  lda, \
	  ldb  \
	); \
\
	/* Quick return if possible. */ \
	if ( *m == 0 || *n == 0 ) \
	{ \
	  AOCL_DTL_TRACE_EXIT(AOCL_DTL_LEVEL_TRACE_1); \
	  /* Finalize BLIS. */ \
	  bli_finalize_auto(); \
	  return; \
	} \
\
	/* Map BLAS chars to their corresponding BLIS enumerated type value. */ \
	bli_param_map_netlib_to_blis_side( *side,  &blis_side ); \
	bli_param_map_netlib_to_blis_uplo( *uploa, &blis_uploa ); \
	bli_param_map_netlib_to_blis_trans( *transa, &blis_transa ); \
	bli_param_map_netlib_to_blis_diag( *diaga, &blis_diaga ); \
\
	/* Typecast BLAS integers to BLIS integers. */ \
	bli_convert_blas_dim1( *m, m0 ); \
	bli_convert_blas_dim1( *n, n0 ); \
\
	/* Set the row and column strides of the matrix operands. */ \
	rs_a = 1; \
	cs_a = *lda; \
	rs_b = 1; \
	cs_b = *ldb; \
\
	/* If alphs is zero, set B to zero and return early */ \
	if( PASTEMAC(ch,eq0)( *alpha ) ) \
	{ \
		PASTEMAC2(ch,setm,_ex)( BLIS_NO_CONJUGATE, \
								0, \
								BLIS_NONUNIT_DIAG, \
								BLIS_DENSE, \
								m0, n0, \
								(ftype*) alpha, \
								(ftype*) b, rs_b, cs_b, \
								NULL, NULL \
							 ); \
		/* Finalize BLIS. */ \
		AOCL_DTL_TRACE_EXIT(AOCL_DTL_LEVEL_TRACE_1) \
		bli_finalize_auto(); \
		return; \
	} \
\
	/* Call BLIS interface. */ \
	PASTEMAC2(ch,blisname,BLIS_TAPI_EX_SUF) \
	( \
	  blis_side, \
	  blis_uploa, \
	  blis_transa, \
	  blis_diaga, \
	  m0, \
	  n0, \
	  (ftype*)alpha, \
	  (ftype*)a, rs_a, cs_a, \
	  (ftype*)b, rs_b, cs_b, \
	  NULL, \
	  NULL  \
	); \
\
	/* Finalize BLIS. */ \
	AOCL_DTL_TRACE_EXIT(AOCL_DTL_LEVEL_TRACE_1) \
	bli_finalize_auto(); \
} \
IF_BLIS_ENABLE_BLAS(\
void PASTEF77(ch,blasname) \
     ( \
       const f77_char* side, \
       const f77_char* uploa, \
       const f77_char* transa, \
       const f77_char* diaga, \
       const f77_int*  m, \
       const f77_int*  n, \
       const ftype*    alpha, \
       const ftype*    a, const f77_int* lda, \
             ftype*    b, const f77_int* ldb  \
     ) \
{ \
	TRMM_BLIS_IMPL(ch,blasname) \
} \
)

#else

#undef  GENTFUNC
#define GENTFUNC( ftype, ch, blasname, blisname ) \
\
void PASTEF77S(ch,blasname) \
     ( \
       const f77_char* side, \
       const f77_char* uploa, \
       const f77_char* transa, \
       const f77_char* diaga, \
       const f77_int*  m, \
       const f77_int*  n, \
       const ftype*    alpha, \
       const ftype*    a, const f77_int* lda, \
             ftype*    b, const f77_int* ldb  \
     ) \
{ \
	AOCL_DTL_TRACE_ENTRY(AOCL_DTL_LEVEL_TRACE_1) \
	AOCL_DTL_LOG_TRMM_INPUTS(AOCL_DTL_LEVEL_TRACE_1, *MKSTR(ch), *side, *uploa,\
		       	*transa, *diaga, *m, *n, (void*)alpha, *lda, *ldb) \
	side_t  blis_side; \
	uplo_t  blis_uploa; \
	trans_t blis_transa; \
	diag_t  blis_diaga; \
	dim_t   m0, n0; \
\
	/* Initialize BLIS. */ \
	bli_init_auto(); \
\
	/* Perform BLAS parameter checking. */ \
	PASTEBLACHK(blasname) \
	( \
	  MKSTR(ch), \
	  MKSTR(blasname), \
	  side, \
	  uploa, \
	  transa, \
	  diaga, \
	  m, \
	  n, \
	  lda, \
	  ldb  \
	); \
\
	/* Quick return if possible. */ \
	if ( *m == 0 || *n == 0 ) \
	{ \
	  AOCL_DTL_TRACE_EXIT(AOCL_DTL_LEVEL_TRACE_1); \
	  /* Finalize BLIS. */ \
	  bli_finalize_auto(); \
	  return; \
	} \
\
	/* Map BLAS chars to their corresponding BLIS enumerated type value. */ \
	bli_param_map_netlib_to_blis_side( *side,  &blis_side ); \
	bli_param_map_netlib_to_blis_uplo( *uploa, &blis_uploa ); \
	bli_param_map_netlib_to_blis_trans( *transa, &blis_transa ); \
	bli_param_map_netlib_to_blis_diag( *diaga, &blis_diaga ); \
\
	/* Typecast BLAS integers to BLIS integers. */ \
	bli_convert_blas_dim1( *m, m0 ); \
	bli_convert_blas_dim1( *n, n0 ); \
\
	/* Set the row and column strides of the matrix operands. */ \
	const inc_t rs_a = 1; \
	const inc_t cs_a = *lda; \
	const inc_t rs_b = 1; \
	const inc_t cs_b = *ldb; \
\
	/* If alphs is zero, set B to zero and return early */ \
	if( PASTEMAC(ch,eq0)( *alpha ) ) \
	{ \
		PASTEMAC2(ch,setm,_ex)( BLIS_NO_CONJUGATE, \
								0, \
								BLIS_NONUNIT_DIAG, \
								BLIS_DENSE, \
								m0, n0, \
								(ftype*) alpha, \
								(ftype*) b, rs_b, cs_b, \
								NULL, NULL \
							 ); \
		/* Finalize BLIS. */ \
		AOCL_DTL_TRACE_EXIT(AOCL_DTL_LEVEL_TRACE_1) \
		bli_finalize_auto(); \
		return; \
	} \
\
	const num_t   dt     = PASTEMAC(ch,type); \
\
	const struc_t struca = BLIS_TRIANGULAR; \
\
	obj_t       alphao = BLIS_OBJECT_INITIALIZER_1X1; \
	obj_t       ao     = BLIS_OBJECT_INITIALIZER; \
	obj_t       bo     = BLIS_OBJECT_INITIALIZER; \
\
	dim_t       mn0_a; \
\
	bli_set_dim_with_side( blis_side, m0, n0, &mn0_a ); \
\
	bli_obj_init_finish_1x1( dt, (ftype*)alpha, &alphao ); \
\
	bli_obj_init_finish( dt, mn0_a, mn0_a, (ftype*)a, rs_a, cs_a, &ao ); \
	bli_obj_init_finish( dt, m0,    n0,    (ftype*)b, rs_b, cs_b, &bo ); \
\
	bli_obj_set_uplo( blis_uploa, &ao ); \
	bli_obj_set_diag( blis_diaga, &ao ); \
	bli_obj_set_conjtrans( blis_transa, &ao ); \
\
	bli_obj_set_struc( struca, &ao ); \
\
	PASTEMAC(blisname,BLIS_OAPI_EX_SUF) \
	( \
	  blis_side, \
	  &alphao, \
	  &ao, \
	  &bo, \
	  NULL, \
	  NULL  \
	); \
\
	/* Finalize BLIS. */ \
	AOCL_DTL_TRACE_EXIT(AOCL_DTL_LEVEL_TRACE_1) \
	bli_finalize_auto(); \
} \
IF_BLIS_ENABLE_BLAS(\
void PASTEF77(ch,blasname) \
     ( \
       const f77_char* side, \
       const f77_char* uploa, \
       const f77_char* transa, \
       const f77_char* diaga, \
       const f77_int*  m, \
       const f77_int*  n, \
       const ftype*    alpha, \
       const ftype*    a, const f77_int* lda, \
             ftype*    b, const f77_int* ldb  \
     ) \
{ \
	TRMM_BLIS_IMPL(ch,blasname) \
} \
)

#endif

INSERT_GENTFUNC_BLAS( trmm, trmm )

