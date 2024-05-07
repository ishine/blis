/*

   BLIS
   An object-based framework for developing high-performance BLAS-like
   libraries.

   Copyright (C) 2023 - 2024, Advanced Micro Devices, Inc. All rights reserved.

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

#include "symv.h"
#include "level2/ref_symv.h"
#include "inc/check_error.h"
#include <stdexcept>
#include <algorithm>

template<typename T>
void test_symv( char storage, char uploa, char conja, char conjx, gtint_t n,
    T alpha, gtint_t lda_inc, gtint_t incx, T beta, gtint_t incy, double thresh )
{
    // Compute the leading dimensions of a.
    gtint_t lda = testinghelpers::get_leading_dimension( storage, 'n', n, n, lda_inc );

    //----------------------------------------------------------
    //        Initialize matrics with random integer numbers.
    //----------------------------------------------------------
    std::vector<T> a = testinghelpers::get_random_matrix<T>( -2, 5, storage, 'n', n, n, lda );
    testinghelpers::make_symm<T>( storage, uploa, n, a.data(), lda );
    testinghelpers::make_triangular<T>( storage, uploa, n, a.data(), lda );

    std::vector<T> x = testinghelpers::get_random_vector<T>( -3, 3, n, incx );
    std::vector<T> y;
    if (beta != testinghelpers::ZERO<T>())
        y = testinghelpers::get_random_vector<T>( -2, 5, n, incy );
    else
    {
        // Vector Y should not be read, only set.
        testinghelpers::set_vector( n, incy, y.data(), testinghelpers::aocl_extreme<T>() );
    }

    // Create a copy of y so that we can check reference results.
    std::vector<T> y_ref(y);
    //----------------------------------------------------------
    //                  Call BLIS function
    //----------------------------------------------------------
    symv<T>( storage, uploa, conja, conjx, n, &alpha, a.data(), lda,
                                  x.data(), incx, &beta, y.data(), incy );

    //----------------------------------------------------------
    //                  Call reference implementation.
    //----------------------------------------------------------
    testinghelpers::ref_symv<T>( storage, uploa, conja, conjx, n, &alpha,
                 a.data(), lda, x.data(), incx, &beta, y_ref.data(), incy );

    //----------------------------------------------------------
    //              check component-wise error.
    //----------------------------------------------------------
    computediff<T>( "y", n, y.data(), y_ref.data(), incy, thresh );

#ifdef CAN_TEST_INFO_VALUE
    gtint_t info = bli_info_get_info_value();
    computediff<gtint_t>( "info", info, 0 );
#endif
}

// Test-case logger : Used to print the test-case details based on parameters
template <typename T>
class symvGenericPrint {
public:
    std::string operator()(
        testing::TestParamInfo<std::tuple<char,char,char,char,gtint_t,T,T,gtint_t,gtint_t,gtint_t>> str) const {
        char storage   = std::get<0>(str.param);
        char uploa     = std::get<1>(str.param);
        char conja     = std::get<2>(str.param);
        char conjx     = std::get<3>(str.param);
        gtint_t n      = std::get<4>(str.param);
        T alpha   = std::get<5>(str.param);
        T beta    = std::get<6>(str.param);
        gtint_t incx   = std::get<7>(str.param);
        gtint_t incy   = std::get<8>(str.param);
        gtint_t ld_inc = std::get<9>(str.param);

        std::string str_name = API_PRINT;
        str_name += "_stor_" + storage;
        str_name += "_uploa_" + uploa;
        str_name += "_conja_" + conja;
        str_name += "_conjx_" + conjx;
        str_name += "_n_" + std::to_string(n);
        str_name += "_alpha_" + testinghelpers::get_value_string(alpha);
        str_name += "_beta_" + testinghelpers::get_value_string(beta);
        str_name += "_incx_" + testinghelpers::get_value_string(incx);
        str_name += "_incy_" + testinghelpers::get_value_string(incy);
        str_name    = str_name + "_" + std::to_string(ld_inc);
        return str_name;
    }
};
