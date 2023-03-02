*This sandbox was originally named* `packsup`, *then* `gemmhot`.

*Currently only* `dgemm` *is supported*.

# GEMMFIP: Unifying large- and small-size GEMM with fused-in packings.

This sandbox is a `GEMM` solution to cover up packing overheads of regular the `GEMM` codepath. By integrating packing instructions to the `GEMMSUP` millikernel on the unpacked memory, `bls_?gemm` here tries to make a unified solution to kick calculation off without the need to wait for `memcpy`.
