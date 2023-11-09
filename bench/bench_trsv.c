/*

   BLIS
   An object-based framework for developing high-performance BLAS-like
   libraries.

   Copyright (C) 2021 - 2023, Advanced Micro Devices, Inc. All rights reserved.

   Redistribution and use in source and binary forms, with or without
   modification, are permitted provided that the following conditions are
   met:
    - Redistributions of source code must retain the above copyright
      notice, this list of conditions and the following disclaimer.
    - Redistributions in binary form must reproduce the above copyright
      notice, this list of conditions and the following disclaimer in the
      documentation and/or other materials provided with the distribution.
    - Neither the name of The University of Texas nor the names of its
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

#ifdef WIN32
#include <io.h>
#else
#include <unistd.h>
#endif
#include "blis.h"



// Benchmark application to process aocl logs generated by BLIS library.
#ifndef DT
#define DT BLIS_DOUBLE
#endif


#define AOCL_MATRIX_INITIALISATION

//#define BLIS_ENABLE_CBLAS

/* For BLIS since logs are collected at BLAS interfaces
 * we disable cblas interfaces for this benchmark application
 */

/* #ifdef BLIS_ENABLE_CBLAS */
/* #define CBLAS */
/* #endif */

int main( int argc, char** argv )
{
    obj_t a, x;
    obj_t x_save;
    dim_t m;
    dim_t  p_inc = 0; // to keep track of number of inputs
    char  dt_ch;

    char uploa_c;
    uplo_t  uploa;
    f77_char f77_uploa;

    char transA;
    trans_t  transa;
    f77_char f77_transa;

    char  diaga_c;
    diag_t diaga;
    f77_char f77_diaga;


    inc_t lda;
    int   r, n_repeats;

    num_t dt;
    double dtime;
    double dtime_save;
    double gflops;

    inc_t incx;
    char tmp[256]; // to store function name, line no present in logs.

    FILE* fin  = NULL;
    FILE* fout = NULL;

    //bli_init();

    n_repeats = N_REPEAT;

    dt = DT;

    if (argc < 3)
    {
        printf("Usage: ./bench_trsv_XX.x input.csv output.csv\n");
        exit(1);
    }
    fin = fopen(argv[1], "r");
    if (fin == NULL)
    {
        printf("Error opening the file %s\n", argv[1]);
        exit(1);
    }
    fout = fopen(argv[2], "w");
    if (fout == NULL)
    {
        printf("Error opening output file %s\n", argv[2]);
        exit(1);
    }

    fprintf(fout, "Dt uploa\t transa\t diaga\t m\t lda\t incx\t gflops\n");

    // {S,D,C,Z} {uploa transa diaga m lda, incx}
    while (fscanf(fin, "%s %c %c %c %c " INT_FS INT_FS INT_FS "\n",
      tmp, &dt_ch, &uploa_c, &transA, &diaga_c, &m, &lda, &incx) == 8)
    {

        if      (dt_ch == 'D' || dt_ch == 'd') dt = BLIS_DOUBLE;
        else if (dt_ch == 'Z' || dt_ch == 'z') dt = BLIS_DCOMPLEX;
        else if (dt_ch == 'S' || dt_ch == 's') dt = BLIS_FLOAT;
        else if (dt_ch == 'C' || dt_ch == 'c') dt = BLIS_SCOMPLEX;
        else
        {
            printf("Invalid data type %c\n", dt_ch);
            continue;
        }


        if('l' == uploa_c || 'L' == uploa_c)
            uploa = BLIS_LOWER;
        else if('u' == uploa_c || 'U' == uploa_c)
            uploa = BLIS_UPPER;
        else
        {
            printf("Invalid entry for the argument 'uplo':%c\n",uploa_c);
            continue;
        }


        if      ( transA == 'n' || transA == 'N') transa = BLIS_NO_TRANSPOSE;
        else if ( transA == 't' || transA == 'T') transa = BLIS_TRANSPOSE;
        else if ( transA == 'c' || transA == 'C') transa = BLIS_CONJ_TRANSPOSE;
        else
        {
            printf("Invalid option for transA \n");
            continue;
        }


        if('u' == diaga_c || 'U' == diaga_c)
            diaga = BLIS_UNIT_DIAG;
        else if('n' == diaga_c || 'N' == diaga_c)
            diaga = BLIS_NONUNIT_DIAG;
        else
        {
            printf("Invalid entry for the argument 'diaga':%c\n", diaga_c);
            continue;
        }

        // Solving the linear system
        // transa(A) * x = y

        // where A is an m x m triangular matrix stored in the lower or upper triangle
        // as specified by uploa with unit/non-unit nature specified by diaga, and x and y
        // are vectors of length m.
        // The right-hand side vector operand y is overwritten with the solution vector x.

        bli_obj_create( dt, m, m, 1, lda, &a );
        bli_obj_create( dt, m, 1, incx, 1, &x );
        bli_obj_create( dt, m, 1, incx, 1, &x_save );

#ifdef AOCL_MATRIX_INITIALISATION
        bli_randm( &a );
        bli_randm( &x );
#endif

        bli_param_map_blis_to_netlib_uplo( uploa, &f77_uploa );
        bli_param_map_blis_to_netlib_trans( transa, &f77_transa );
        bli_param_map_blis_to_netlib_diag( diaga, &f77_diaga );

        bli_obj_set_struc( BLIS_TRIANGULAR, &a );
        bli_obj_set_uplo( uploa, &a );
        bli_obj_set_conjtrans( transa, &a );
        bli_obj_set_diag( diaga, &a );

        // Randomize A and zero the unstored triangle to ensure the
        // implementation reads only from the stored region.
        bli_randm( &a );
        bli_mktrim( &a );

        // Load the diagonal of A to make it more likely to be invertible.
        bli_shiftd( &BLIS_TWO, &a );

        bli_copym( &x, &x_save );

        dtime_save = DBL_MAX;

#ifdef PRINT
            bli_printm( "a", &a, "%4.1f", "" );
            bli_printm( "x", &x, "%4.1f", "" );
#endif
        for ( r = 0; r < n_repeats; ++r )
        {

            bli_copym( &x_save, &x );

            dtime = bli_clock();

#ifdef BLIS

            bli_trsv( &BLIS_ONE,
                      &a,
                      &x );

#else // BLIS Interface

#ifdef CBLAS

            enum CBLAS_TRANSPOSE cblas_transa;
            enum CBLAS_UPLO cblas_uplo;
            enum CBLAS_DIAG cblas_diag;

            if( bli_is_trans( transa ) )
              cblas_transa = CblasTrans;
            else if( bli_is_conjtrans( transa ) )
              cblas_transa = CblasConjTrans;
            else
              cblas_transa = CblasNoTrans;

            if(bli_is_upper(uploa))
                cblas_uplo = CblasUpper;
            else
                cblas_uplo = CblasLower;

            if(bli_is_unit_diag(diaga))
                cblas_diag = CblasUnit;
            else
                cblas_diag = CblasNonUnit;

#else

            f77_int  mm     = bli_obj_length( &a );
            f77_int  lda    = bli_obj_col_stride( &a );
            f77_int  incx   = bli_obj_vector_inc( &x );

            if ( bli_is_float( dt ) )
            {
                float*  ap     = bli_obj_buffer( &a );
                float*  xp     = bli_obj_buffer( &x );

#ifdef CBLAS

                cblas_strsv(
                    cblas_uplo,
                    cblas_transa,
                    cblas_diag,
                    mm,
                    *ap, lda,
                    *xp, incx
                );

#else

                strsv_( &f77_uploa,
                    &f77_transa,
                    &f77_diaga,
                    &mm,
                    ap, &lda,
                    xp, &incx );

#endif
            }
            else if ( bli_is_double( dt ) )
            {
                double*  ap     = bli_obj_buffer( &a );
                double*  xp     = bli_obj_buffer( &x );

#ifdef CBLAS

                cblas_dtrsv(
                    cblas_uplo,
                    cblas_transa,
                    cblas_diag,
                    mm,
                    *ap, lda,
                    *xp, incx
                );

#else

                dtrsv_( &f77_uploa,
                    &f77_transa,
                    &f77_diaga,
                    &mm,
                    ap, &lda,
                    xp, &incx );
#endif

            }
            else if ( bli_is_scomplex( dt ) )
            {
                scomplex* ap     = bli_obj_buffer( &a );
                scomplex* xp     = bli_obj_buffer( &x );

#ifdef CBLAS

                cblas_ctrsv(
                    cblas_uplo,
                    cblas_transa,
                    cblas_diag,
                    mm,
                    *ap, lda,
                    *xp, incx
                );

#else
                ctrsv_( &f77_uploa,
                    &f77_transa,
                    &f77_diaga,
                    &mm,
                    ap, &lda,
                    xp, &incx );

#endif
            }
            else if ( bli_is_dcomplex( dt ) )
            {
                dcomplex* ap     = bli_obj_buffer( &a );
                dcomplex* xp     = bli_obj_buffer( &x );

#ifdef CBLAS

                cblas_ztrsv(
                    cblas_uplo,
                    cblas_transa,
                    cblas_diag,
                    mm,
                    *ap, lda,
                    *xp, incx
                );

#else
                ztrsv_( &f77_uploa,
                    &f77_transa,
                    &f77_diaga,
                    &mm,
                    ap, &lda,
                    xp, &incx );

#endif

#endif
            }
#endif

            dtime_save = bli_clock_min_diff( dtime_save, dtime );
        }

#ifdef PRINT
        bli_printm( "x after", &x, "%4.1f", "" );
        exit(1);
#endif

        gflops = ( 1.0 * m * m ) / ( dtime_save * 1.0e9 );

#ifdef BLIS
        printf( "data_trsv_blis" );
#else
        printf( "data_trsv_%s", BLAS );
#endif
        p_inc++;
        printf( "( %2lu, 1:2 ) = [ %4lu %7.2f ];\n",
                ( unsigned long )p_inc,
                ( unsigned long )m, gflops );

        fprintf (fout, "%s\t %c\t %c\t %c\t %c\t %ld\t %ld\t %ld\t %6.3f\n",
                        tmp, dt_ch, uploa_c, transA, diaga_c, m, lda, incx, gflops);

        fflush(fout);

        bli_obj_free( &a );
        bli_obj_free( &x );
        bli_obj_free( &x_save );
    }

    // bli_finalize();
    return 0;
}
