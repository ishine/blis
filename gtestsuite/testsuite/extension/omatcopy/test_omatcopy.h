/*

   BLIS
   An object-based framework for developing high-performance BLAS-like
   libraries.

   Copyright (C) 2024, Advanced Micro Devices, Inc. All rights reserved.

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

#pragma once

#include "omatcopy.h"
#include "extension/ref_omatcopy.h"
#include "inc/check_error.h"
#include<cstdlib>

/**
 * @brief Generic test body for omatcopy operation.
 */

template<typename T>
static void test_omatcopy( char storage, char trans, gtint_t m, gtint_t n, T alpha, gtint_t lda_inc, gtint_t ldb_inc,
                          double thresh, bool is_memory_test = false, bool is_nan_inf_test = false, T exval = T{0.0} )
{
    // Set an alternative trans value that corresponds to only
    // whether the B matrix should be mxn or nxm(only transposing)
    char B_trans;
    B_trans = ( ( trans == 'n' ) || ( trans == 'r' ) )? 'n' : 't';

    // Compute the leading dimensions of A and B.
    gtint_t lda = testinghelpers::get_leading_dimension( storage, 'n', m, n, lda_inc );
    gtint_t ldb = testinghelpers::get_leading_dimension( storage, B_trans, m, n, ldb_inc );

    // Compute sizes of A and B, in bytes
    gtint_t size_a = testinghelpers::matsize( storage, 'n', m, n, lda ) * sizeof( T );
    gtint_t size_b = testinghelpers::matsize( storage, B_trans, m, n, ldb ) * sizeof( T );

    // Create the objects for the input and output operands
    // The API does not expect the memory to be aligned
    testinghelpers::ProtectedBuffer A_buf( size_a, false, is_memory_test );
    testinghelpers::ProtectedBuffer B_buf( size_b, false, is_memory_test );
    testinghelpers::ProtectedBuffer B_ref_buf( size_b, false, false );

    // Pointers to access the memory chunks
    T *A, *B, *B_ref;

    // Acquire the first set of greenzones for A and B
    A = ( T* )A_buf.greenzone_1;
    B = ( T* )B_buf.greenzone_1;
    B_ref = ( T* )B_ref_buf.greenzone_1; // For B_ref, there is no greenzone_2

    // Initiaize the memory with random data
    testinghelpers::datagenerators::randomgenerators( -10, 10, storage, m, n, A, 'n', lda );
    testinghelpers::datagenerators::randomgenerators( -10, 10, storage, m, n, B, B_trans, ldb );

    if( is_nan_inf_test )
    {
      gtint_t rand_m = rand() % m;
      gtint_t rand_n = rand() % n;
      gtint_t idx = ( storage == 'c' || storage == 'C' )? ( rand_m + rand_n * lda ) : ( rand_n + rand_m * lda );

      A[idx] = exval;
    }
    // Copying the contents of B to B_ref
    memcpy( B_ref, B, size_b );

    // Add signal handler for segmentation fault
    testinghelpers::ProtectedBuffer::start_signal_handler();
    try
    {
        // Call the API.
        // This call is made irrespective of is_memory_test.
        // This will check for out of bounds access with first redzone(if memory test is true)
        // Else, it will just call the ukr function.
        omatcopy<T>( trans, m, n, alpha, A, lda, B, ldb);

        if ( is_memory_test )
        {
            // Acquire the pointers near the second redzone
            A = ( T* )A_buf.greenzone_2;
            B = ( T* )B_buf.greenzone_2;

            // Copy the data for A and B accordingly
            // NOTE : The objects for A and B will have acquired enough memory
            //        such that the greenzones in each do not overlap.
            memcpy( A, A_buf.greenzone_1, size_a );
            memcpy( B, B_buf.greenzone_1, size_b );

            // Call the API, to check with the second redzone.
            omatcopy<T>( trans, m, n, alpha, A, lda, B, ldb);
        }
    }
    catch(const std::exception& e)
    {
        // Reset to default signal handler
        testinghelpers::ProtectedBuffer::stop_signal_handler();

        // Show failure in case seg fault was detected
        FAIL() << "Memory Test Failed";
    }
    // Reset to default signal handler
    testinghelpers::ProtectedBuffer::stop_signal_handler();

    //----------------------------------------------------------
    //    Call reference implementation to get ref results.
    //----------------------------------------------------------
    testinghelpers::ref_omatcopy<T>( storage, trans, m, n, alpha, A, lda, B_ref, ldb );

    //----------------------------------------------------------
    //              Compute component-wise error.
    //----------------------------------------------------------

    if( B_trans == 'n' )
      computediff<T>( "B", storage, m, n, B, B_ref, ldb, thresh, is_nan_inf_test );
    else
      computediff<T>( "B", storage, n, m, B, B_ref, ldb, thresh, is_nan_inf_test );

}

// Test-case logger : Used to print the test-case details based on parameters
template <typename T>
class omatcopyGenericPrint {
public:
    std::string operator()(
        testing::TestParamInfo<std::tuple<char,char,gtint_t,gtint_t,T,gtint_t,gtint_t,bool>> str) const {
        char storage   = std::get<0>(str.param);
        char trans     = std::get<1>(str.param);
        gtint_t m      = std::get<2>(str.param);
        gtint_t n      = std::get<3>(str.param);
        T alpha    = std::get<4>(str.param);
        gtint_t lda_inc = std::get<5>(str.param);
        gtint_t ldb_inc = std::get<6>(str.param);
        bool is_memory_test = std::get<7>(str.param);

        std::string str_name = API_PRINT;
        str_name += "_stor_" + std::string(&storage, 1);
        str_name += "_trans_" + std::string(&trans, 1);
        str_name += "_m_" + std::to_string(m);
        str_name += "_n_" + std::to_string(n);
        str_name += "_alpha_" + testinghelpers::get_value_string(alpha);
        gtint_t lda = testinghelpers::get_leading_dimension( storage, 'n', m, n, lda_inc );
        gtint_t ldb = testinghelpers::get_leading_dimension( storage, trans, m, n, ldb_inc );
        str_name += "_lda" + std::to_string(lda);
        str_name += "_ldb" + std::to_string(ldb);
        str_name += ( is_memory_test )? "_mem_test_enabled" : "_mem_test_disabled";

        return str_name;
    }
};

template <typename T>
class omatcopyEVTPrint {
public:
    std::string operator()(
        testing::TestParamInfo<std::tuple<char,char,gtint_t,gtint_t,T,gtint_t,gtint_t,T,bool>> str) const {
        char storage    = std::get<0>(str.param);
        char trans      = std::get<1>(str.param);
        gtint_t m       = std::get<2>(str.param);
        gtint_t n       = std::get<3>(str.param);
        T alpha  = std::get<4>(str.param);
        gtint_t lda_inc = std::get<5>(str.param);
        gtint_t ldb_inc = std::get<6>(str.param);
        T exval  = std::get<7>(str.param);

        std::string str_name = API_PRINT;
        str_name += "_stor_" + std::string(&storage, 1);
        str_name += "_trans_" + std::string(&trans, 1);
        str_name += "_m_" + std::to_string(m);
        str_name += "_n_" + std::to_string(n);
        str_name += "_alpha_" + testinghelpers::get_value_string(alpha);
        str_name = str_name + "_A_exval_" + testinghelpers::get_value_string(exval);
        gtint_t lda = testinghelpers::get_leading_dimension( storage, 'n', m, n, lda_inc );
        gtint_t ldb = testinghelpers::get_leading_dimension( storage, trans, m, n, ldb_inc );
        str_name += "_lda" + std::to_string(lda);
        str_name += "_ldb" + std::to_string(ldb);

        return str_name;
    }
};
