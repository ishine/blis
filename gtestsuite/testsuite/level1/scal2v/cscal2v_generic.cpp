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
#include "test_scal2v.h"

class cscal2vGenericTest :
        public ::testing::TestWithParam<std::tuple<char,
                                                   gtint_t,
                                                   gtint_t,
                                                   gtint_t,
                                                   scomplex,
                                                   char>> {};

GTEST_ALLOW_UNINSTANTIATED_PARAMETERIZED_TEST(cscal2vGenericTest);

// Tests using random integers as vector elements.
TEST_P( cscal2vGenericTest, RandomData )
{
  using T = scomplex;
  //----------------------------------------------------------
  // Initialize values from the parameters passed through
  // test suite instantiation (INSTANTIATE_TEST_SUITE_P).
  //----------------------------------------------------------
  // denotes whether alpha or conj(alpha) will be used:
  char conj_alpha = std::get<0>(GetParam());
  // vector length:
  gtint_t n = std::get<1>(GetParam());
  // stride size for x:
  gtint_t incx = std::get<2>(GetParam());
  // stride size for y:
  gtint_t incy = std::get<3>(GetParam());
  // alpha
  T alpha = std::get<4>(GetParam());
  // specifies the datatype for randomgenerators
  char datatype = std::get<5>(GetParam());

  // Set the threshold for the errors:
  double thresh = testinghelpers::getEpsilon<T>();
  //----------------------------------------------------------
  //     Call generic test body using those parameters
  //----------------------------------------------------------
  test_scal2v<T>(conj_alpha, n, incx, incy, alpha, thresh, datatype);
}

// Used to generate a test case with a sensible name.
// Beware that we cannot use fp numbers (e.g., 2.3) in the names,
// so we are only printing int(2.3). This should be enough for debugging purposes.
// If this poses an issue, please reach out.
class cscal2vGenericTestPrint {
public:
    std::string operator()(
        testing::TestParamInfo<std::tuple<char, gtint_t, gtint_t, gtint_t, scomplex, char>> str) const {
        char conj = std::get<0>(str.param);
        gtint_t n = std::get<1>(str.param);
        gtint_t incx = std::get<2>(str.param);
        gtint_t incy = std::get<3>(str.param);
        scomplex alpha = std::get<4>(str.param);
        char datatype = std::get<5>(str.param);
        std::string str_name = "bli_cscal2v";
        str_name += "_" + std::to_string(n);
        str_name += "_" + std::string(&conj, 1);
        std::string incx_str = ( incx > 0) ? std::to_string(incx) : "m" + std::to_string(std::abs(incx));
        str_name += "_" + incx_str;
        std::string incy_str = ( incy > 0) ? std::to_string(incy) : "m" + std::to_string(std::abs(incy));
        str_name += "_" + incy_str;
        std::string alpha_str = ( alpha.real > 0) ? std::to_string(int(alpha.real)) : ("m" + std::to_string(int(std::abs(alpha.real))));
                    alpha_str = alpha_str + "pi" + (( alpha.imag > 0) ? std::to_string(int(alpha.imag)) : ("m" + std::to_string(int(std::abs(alpha.imag)))));
        str_name = str_name + "_a" + alpha_str;
        str_name = str_name + "_" + datatype;
        return str_name;
    }
};
#ifdef TEST_BLIS_TYPED
// Black box testing for generic and main use of cscal2.
INSTANTIATE_TEST_SUITE_P(
        Blackbox,
        cscal2vGenericTest,
        ::testing::Combine(
            ::testing::Values('n','c'),                                      // n: use x, c: use conj(x)
            ::testing::Range(gtint_t(10), gtint_t(101), 10),                 // m size of vector takes values from 10 to 100 with step size of 10.
            ::testing::Values(gtint_t(1)),                                   // stride size for x
            ::testing::Values(gtint_t(1)),                                   // stride size for y
            ::testing::Values(scomplex{2.0, -1.0}, scomplex{-2.0, 3.0}),     // alpha
            ::testing::Values(ELEMENT_TYPE)                                  // i : integer, f : float  datatype type tested
        ),
        ::cscal2vGenericTestPrint()
    );


// Test for non-unit increments.
// Only test very few cases as sanity check.
// We can modify the values using implementantion details.
INSTANTIATE_TEST_SUITE_P(
        NonUnitIncrements,
        cscal2vGenericTest,
        ::testing::Combine(
            ::testing::Values('n','c'),                                      // n: use x, c: use conj(x)
            ::testing::Range(gtint_t(10), gtint_t(101), 10),                 // m size of vector takes values from 10 to 100 with step size of 10.
            ::testing::Values(gtint_t(2), gtint_t(11)),                      // stride size for x
            ::testing::Values(gtint_t(4)),                                   // stride size for y
            ::testing::Values(scomplex{4.0, 3.1}),                           // alpha
            ::testing::Values(ELEMENT_TYPE)                                  // i : integer, f : float  datatype type tested
        ),
        ::cscal2vGenericTestPrint()
    );
#endif