#include "blis.h"
#include <assert.h>


BLIS_INLINE
void bli_dgemmsup2_ref_microkernel
    (
     dim_t            m,
     dim_t            n,
     dim_t            k,
     double *restrict alpha,
     double *restrict a, inc_t rs_a0, inc_t cs_a0,
     double *restrict b, inc_t rs_b0, inc_t cs_b0,
     double *restrict beta,
     double *restrict c, inc_t rs_c0, inc_t cs_c0,
     auxinfo_t       *data,
     cntx_t          *cntx,
     double *restrict a_p, int pack_a,
     double *restrict b_p, int pack_b
    )
{
  const void* a_next = bli_auxinfo_next_a( data );
  const void* b_next = bli_auxinfo_next_b( data );
  uint64_t cs_a_next = bls_aux_ls_ext_next( data );

  for (int i = 0; i < m; ++i)
  {
    for (int j = 0; j < n; ++j)
    {
      double c_ij = 0.0;

      for (int l = 0; l < k; ++l)
      {
        // Prefetch the next milli-kernel.
        // if (!l) should be removed after loop reordering.
        if (!l) { __builtin_prefetch(a_next); a_next += cs_a_next; }

        double a_il = a[i * rs_a0 + l * cs_a0];
        double b_lj = b[l * rs_b0 + j * cs_b0];

        // Optionally store loaded a & b elements to the packing space.
        if (pack_a) a_p[i + l * mr_ref] = a_il;
        if (pack_b) b_p[l * mr_ref + j] = b_lj;

        c_ij += a_il * b_lj;
      }
      if (*alpha != 1.0)
        c_ij *= *alpha;
      if (*beta != 0.0)
        c_ij += c[i*rs_c0 + j*cs_c0] * *beta;

      c[i*rs_c0 + j*cs_c0] = c_ij;
    }
  }
  __builtin_prefetch(b_next);
}

void bli_dgemmsup2_ref
    (
     dim_t            m_,
     dim_t            n,
     dim_t            k,
     double *restrict alpha,
     double *restrict a_, inc_t rs_a0, inc_t cs_a0,
     double *restrict b,  inc_t rs_b0, inc_t cs_b0,
     double *restrict beta,
     double *restrict c_, inc_t rs_c0, inc_t cs_c0,
     auxinfo_t       *data,
     cntx_t          *cntx,
     double *restrict a_p_, int pack_a,
     double *restrict b_p,  int pack_b,
     inc_t            ares_offset, // only has affect when pack_a == true
     bool            *semaphore_
    )
{
  inc_t ps_a_p    = bls_aux_ps_ext_p   ( data );
  inc_t ps_a      = bls_aux_ps_ext     ( data );
  inc_t cs_a_next = bls_aux_ls_ext_next( data );
  const void *next_a = bli_auxinfo_next_a( data );
  const void *next_b = bli_auxinfo_next_b( data );

  bli_auxinfo_set_next_b( b, data );
  bls_aux_set_ls_ext_next( cs_a0, data );

  // Initialize
  dim_t            m   = m_ -         mr_ref * ares_offset;
  double *restrict a   = a_ +         ps_a   * ares_offset;
  double *restrict a_p = a_p_ +       ps_a_p * ares_offset;
  double *restrict c   = c_ + mr_ref * rs_c0 * ares_offset;
  bool            *semaphore = semaphore_ +    ares_offset;

  do
  {
  for ( ; m != 0; )
  {
    dim_t m_loc = bli_min( m, mr_ref );
    if ( m <= mr_ref && !ares_offset )
    {
      bli_auxinfo_set_next_a( next_a, data );
      bli_auxinfo_set_next_b( next_b, data );
      bls_aux_set_ls_ext_next( cs_a_next, data );
    }
    else
      bli_auxinfo_set_next_a( a + bli_min(ps_a, 128 /* arch-dependent. don't prefetch too far away */), data );

    // Inline dispatch from millikernel to microkernel.
    if ( *semaphore )
      bli_dgemmsup2_ref_microkernel
        ( m_loc, n, k,
          alpha,
          a_p, 1, mr_ref,
          b, rs_b0, cs_b0,
          beta,
          c, rs_c0, cs_c0,
          data,
          cntx,
          a_p, 0,
          b_p, pack_b );
    else
      bli_dgemmsup2_ref_microkernel
        ( m_loc, n, k,
          alpha,
          a, rs_a0, cs_a0,
          b, rs_b0, cs_b0,
          beta,
          c, rs_c0, cs_c0,
          data,
          cntx,
          a_p, pack_a,
          b_p, pack_b );
    *semaphore = TRUE;

    m -= m_loc;
    a += ps_a;
    a_p += ps_a_p;
    c += m_loc * rs_c0;
    semaphore++;
    if ( pack_b ) {
      pack_b = 0;
      // Start reusing the packed B.
      // To avoid RAW hazards on some chips, one MAY want to delay a few steps before reuse.
      b = b_p;
      rs_b0 = nr_ref;
      cs_b0 = 1;
      bli_auxinfo_set_next_b( b_p, data );
    }
  }
  m   = mr_ref * ares_offset;
  a   = a_;
  a_p = a_p_;
  c   = c_;
  semaphore = semaphore_;
  ares_offset = 0;
  } while ( m );
}
