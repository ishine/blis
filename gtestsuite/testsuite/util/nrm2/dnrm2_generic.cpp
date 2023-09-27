/*

   BLIS
   An object-based framework for developing high-performance BLAS-like
   libraries.

   Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.

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

#include <gtest/gtest.h>
#include "test_nrm2.h"

class dnrm2Test :
        public ::testing::TestWithParam<std::tuple<gtint_t, gtint_t>> {};

TEST_P( dnrm2Test, RandomData )
{
    using T = double;
    //----------------------------------------------------------
    // Initialize values from the parameters passed through
    // test suite instantiation (INSTANTIATE_TEST_SUITE_P).
    //----------------------------------------------------------
    // vector length:
    gtint_t n = std::get<0>(GetParam());
    // stride size for x:
    gtint_t incx = std::get<1>(GetParam());

    // Set the threshold for the errors:
    double thresh = std::sqrt(n)*testinghelpers::getEpsilon<T>();

    //----------------------------------------------------------
    //     Call test body using these parameters
    //----------------------------------------------------------
    test_nrm2<T>( n, incx, thresh );
}

// Prints the test case combination
class dnrm2TestPrint {
public:
    std::string operator()(
        testing::TestParamInfo<std::tuple<gtint_t, gtint_t>> str) const {
        gtint_t n     = std::get<0>(str.param);
        gtint_t incx  = std::get<1>(str.param);
#ifdef TEST_BLAS
        std::string str_name = "dnrm2_";
#elif TEST_CBLAS
        std::string str_name = "cblas_dnrm2";
#else  //#elif TEST_BLIS_TYPED
        std::string str_name = "bli_dnormfv";
#endif
        str_name    = str_name + "_" + std::to_string(n);
        std::string incx_str = ( incx > 0) ? std::to_string(incx) : "m" + std::to_string(std::abs(incx));
        str_name    = str_name + "_" + incx_str;
        return str_name;
    }
};

/**
 * dnrm2 implementation is composed by two parts:
 * - vectorized path for n>4
 *      - for-loop for multiples of 8 (F8)
 *      - for-loop for multiples of 4 (F4)
 * - scalar path for n<=4 (S)
*/

INSTANTIATE_TEST_SUITE_P(
        AT_1T,
        dnrm2Test,
        ::testing::Combine(
            // m size of vector
            ::testing::Values(gtint_t(1),  // trivial case n=1
                              gtint_t(3),  // will go through SSE and scalar
                              gtint_t(8),  // 1*8 - will only go through F8
                              gtint_t(24), // 3*8 - will go through F8
                              gtint_t(34), // 4*8 + 2 - will go through F8 & S
                              gtint_t(52), // 6*8 + 4 - will go through F8 & F4
                              gtint_t(71), // 8*8 + 4 + 3 - will go through F8 & F4 & S
                              gtint_t(89), // a few bigger numbers
                              gtint_t(122),
                              gtint_t(185),
                              gtint_t(217)
            ),
            // stride size for x
            ::testing::Values(gtint_t(1), gtint_t(3)
#ifndef TEST_BLIS_TYPED
            , gtint_t(-1), gtint_t(-7)
#endif
        )
        ),
        ::dnrm2TestPrint()
    );

// Multithreading unit tester
/*
    NOTE : The following instantiator is the most useful if BLIS
           configured with aocl-dynamic disabled,  since then it
           would be sufficient to verify functionality upto 64
           threads.

    The following instantiator has data points that would suffice
    the extreme value testing with <= 64 threads.

    Sizes from 256 to 259 ensure that each thread gets a minimum
    size of 4, with some sizes inducing fringe cases.

    Sizes from 512 to 515 ensure that each thread gets a minimum
    size of 8, with some sizes inducing fringe cases.

    Sizes from 768 to 771 ensure that each thread gets a minimum
    size of 12( i.e 8-block loop + 4-block loop), with some sizes
    inducing fringe cases.

    Non-unit strides are also tested, since they might get packed.
*/
INSTANTIATE_TEST_SUITE_P(
        AT_MT_Unit_Tester,
        dnrm2Test,
        ::testing::Combine(
            // m size of vector
            ::testing::Values(gtint_t(256),
                              gtint_t(257),
                              gtint_t(258),
                              gtint_t(259),
                              gtint_t(512),
                              gtint_t(513),
                              gtint_t(514),
                              gtint_t(515),
                              gtint_t(768),
                              gtint_t(769),
                              gtint_t(770),
                              gtint_t(771)
            ),
            // stride size for x
            ::testing::Values(gtint_t(1), gtint_t(3)
#ifndef TEST_BLIS_TYPED
            , gtint_t(-1), gtint_t(-7)
#endif
        )
        ),
        ::dnrm2TestPrint()
    );

// Instantiator if AOCL_DYNAMIC is enabled
/*
  The instantiator here checks for correctness of
  the compute with sizes large enough to bypass
  the thread setting logic with AOCL_DYNAMIC enabled
*/
INSTANTIATE_TEST_SUITE_P(
        AT_MT_AOCL_DYNAMIC,
        dnrm2Test,
        ::testing::Combine(
            // m size of vector
            ::testing::Values(gtint_t(2950000),
                              gtint_t(2950001),
                              gtint_t(2950002),
                              gtint_t(2950003)
            ),
            // stride size for x
            ::testing::Values(gtint_t(1), gtint_t(3)
#ifndef TEST_BLIS_TYPED
            , gtint_t(-1), gtint_t(-7)
#endif
        )
        ),
        ::dnrm2TestPrint()
    );