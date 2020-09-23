/*

   BLIS
   An object-based framework for developing high-performance BLAS-like
   libraries.

   Copyright (C) 2014, The University of Texas at Austin
   Copyright (C) 2020, Advanced Micro Devices, Inc. All rights reserved.

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


//
// Define template prototypes for level-1v kernels.
//

// Note: Instead of defining function prototype macro templates and then
// instantiating those macros to define the individual function prototypes,
// we simply alias the official operations' prototypes as defined in
// bli_l1v_ker_prot.h.

#undef  GENTPROT
#define GENTPROT ADDV_KER_PROT

INSERT_GENTPROT_BASIC0( addv_ker_name )


#undef  GENTPROT
#define GENTPROT AMAXV_KER_PROT

INSERT_GENTPROT_BASIC0( amaxv_ker_name )


#undef  GENTPROT
#define GENTPROT AMINV_KER_PROT

INSERT_GENTPROT_BASIC0( aminv_ker_name )


#undef  GENTPROT
#define GENTPROT AXPBYV_KER_PROT

INSERT_GENTPROT_BASIC0( axpbyv_ker_name )


#undef  GENTPROT
#define GENTPROT AXPYV_KER_PROT

INSERT_GENTPROT_BASIC0( axpyv_ker_name )


#undef  GENTPROT
#define GENTPROT COPYV_KER_PROT

INSERT_GENTPROT_BASIC0( copyv_ker_name )


#undef  GENTPROT
#define GENTPROT DOTV_KER_PROT

INSERT_GENTPROT_BASIC0( dotv_ker_name )


#undef  GENTPROT
#define GENTPROT DOTXV_KER_PROT

INSERT_GENTPROT_BASIC0( dotxv_ker_name )


#undef  GENTPROT
#define GENTPROT INVERTV_KER_PROT

INSERT_GENTPROT_BASIC0( invertv_ker_name )


#undef  GENTPROT
#define GENTPROT SCALV_KER_PROT

INSERT_GENTPROT_BASIC0( scalv_ker_name )


#undef  GENTPROT
#define GENTPROT SCAL2V_KER_PROT

INSERT_GENTPROT_BASIC0( scal2v_ker_name )


#undef  GENTPROT
#define GENTPROT SETV_KER_PROT

INSERT_GENTPROT_BASIC0( setv_ker_name )


#undef  GENTPROT
#define GENTPROT SUBV_KER_PROT

INSERT_GENTPROT_BASIC0( subv_ker_name )


#undef  GENTPROT
#define GENTPROT SWAPV_KER_PROT

INSERT_GENTPROT_BASIC0( swapv_ker_name )


#undef  GENTPROT
#define GENTPROT XPBYV_KER_PROT

INSERT_GENTPROT_BASIC0( xpbyv_ker_name )

