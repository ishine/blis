#include "blis.h"
#include <stdlib.h>
#include <assert.h>


BLIS_INLINE void abort_(const char *msg)
{ fprintf(stderr, "%s\n", msg); abort(); }

BLIS_INLINE void assert_(const bool cond, const char *msg)
{ if (!cond) abort_(msg); }

#define min_(a, b) ( (a) < (b) ? (a) : (b) )

err_t bls_dgemm
    (
     const obj_t*     _alpha,
     const obj_t*     _a,
     const obj_t*     _b,
     const obj_t*     _beta,
     const obj_t*     _c,
     const cntx_t*    cntx,
     const rntm_t*    rntm,
           thrinfo_t* thread
    )
{
#if 0
    assert( bli_obj_has_notrans( _a ) ); // Transpose should be handled at upper level.
    assert( bli_obj_has_notrans( _b ) );
    assert( bli_obj_has_notrans( _c ) );
#endif
    dim_t m0 = bli_obj_length( _c );
    dim_t n0 = bli_obj_width ( _c );
    dim_t k0 = bli_obj_width ( _a );
    double *restrict alpha0 = bli_obj_buffer( _alpha );
    double *restrict a      = bli_obj_buffer( _a     );
    double *restrict b      = bli_obj_buffer( _b     );
    double *restrict beta0  = bli_obj_buffer( _beta  );
    double *restrict c      = bli_obj_buffer( _c     );
    inc_t rs_a = bli_obj_row_stride( _a ), cs_a = bli_obj_col_stride( _a );
    inc_t rs_b = bli_obj_row_stride( _b ), cs_b = bli_obj_col_stride( _b );
    inc_t rs_c = bli_obj_row_stride( _c ), cs_c = bli_obj_col_stride( _c );
    // Borrowed GEMMFIP kernel metadata.
    dim_t mr = ( dim_t )_a->ker_params;
    dim_t nr = ( dim_t )_b->ker_params;
    ukr_dgemm_sup_t ukr_sup = ( ukr_dgemm_sup_t )_c->ker_params;

    const dim_t mc_= bli_cntx_get_blksz_def_dt( BLIS_DOUBLE, BLIS_MC, cntx );
    const dim_t nc_= bli_cntx_get_blksz_def_dt( BLIS_DOUBLE, BLIS_NC, cntx );
    const dim_t kc = bli_cntx_get_blksz_def_dt( BLIS_DOUBLE, BLIS_KC, cntx );
    const dim_t kc_max = kc + 47;
#if 0
    const dim_t mc = mc_;
    const dim_t nc = nc_;
    assert_( !(mc % mr), "MC not multiple of MR." );
    assert_( !(nc % nr), "NC not multiple of NR." );
#else
    const dim_t mc = ( mc_ + mr - 1 ) / mr * mr;
    const dim_t nc = ( nc_ + nr - 1 ) / nr * nr;
#endif
    const dim_t num_ir = mc / mr;
    const dim_t num_jr = nc / nr;
    const bool has_pack_a =
        n0 >= bli_cntx_get_blksz_def_dt( BLIS_DOUBLE, BLIS_NT, cntx ) &&
        k0 >= bli_cntx_get_blksz_def_dt( BLIS_DOUBLE, BLIS_KT, cntx );
    const bool has_pack_b =
        m0 >= bli_cntx_get_blksz_def_dt( BLIS_DOUBLE, BLIS_MT, cntx ) &&
        k0 >= bli_cntx_get_blksz_def_dt( BLIS_DOUBLE, BLIS_KT, cntx );

    auxinfo_t data;
    bli_auxinfo_set_next_a( 0, &data );
    bli_auxinfo_set_next_b( 0, &data );

    thrinfo_t* restrict thread_jc = bli_thrinfo_sub_node( thread );
    thrinfo_t* restrict thread_pc = bli_thrinfo_sub_node( thread_jc );
    thrinfo_t* restrict thread_pb = bli_thrinfo_sub_node( thread_pc );
    thrinfo_t* restrict thread_ic = bli_thrinfo_sub_node( thread_pb );
    thrinfo_t* restrict thread_pa = bli_thrinfo_sub_node( thread_ic );
    thrinfo_t* restrict thread_jr = bli_thrinfo_sub_node( thread_pa );

    // Limited threading capability. Discard non-master JC through PA.
    if ( thread_jc->work_id || thread_pc->work_id || thread_pb->work_id || 
         thread_ic->work_id || thread_pa->work_id )
        // TODO: There should be a one-time warning like "Specified ... but used ..."
        return BLIS_SUCCESS;

    bli_packm_sup_init_mem( has_pack_b, BLIS_BUFFER_FOR_B_PANEL, BLIS_DOUBLE, nc, kc_max, nr, thread_pb );
    bli_packm_sup_init_mem( has_pack_a, BLIS_BUFFER_FOR_A_BLOCK, BLIS_DOUBLE, mc, kc_max, mr, thread_pa );
    bli_packm_sup_init_mem( 1,          BLIS_BUFFER_FOR_GEN_USE, BLIS_FLOAT,  num_ir,  1,  1, thread_ic );

    mem_t* mem_b = bli_thrinfo_mem( thread_pb );
    mem_t* mem_a = bli_thrinfo_mem( thread_pa );

    double *b_panels = bli_mem_buffer( mem_b );
    double *a_panels = bli_mem_buffer( mem_a );

    // Constants.
    double one = 1.0;

    for ( dim_t jc_offset = 0; jc_offset < n0; jc_offset += nc ) {
        double *b_l4 = b + jc_offset * cs_b;
        double *c_l4 = c + jc_offset * cs_c;
        double *alpha = alpha0;
        double *beta  = beta0;

        for ( dim_t lc_offset = 0; lc_offset < k0; /* lc_offset += k_uker. */ ) {
            double *a_l3 = a    + lc_offset * cs_a;
            double *b_l3 = b_l4 + lc_offset * rs_b;
            dim_t k_uker = k0 - lc_offset < kc_max ? k0 - lc_offset : kc;
            // Determine whether to use k_uker * ?r or kc * ?r as the packing stride.
            // On CPU basically k_uker * ?r is better since it ensures equential HW prefetching.
            dim_t k_ps = k_uker;

            // At the end of the packing space is a semaphore for A-restreaming.
            bool *semaphore = bli_mem_buffer( bli_thrinfo_mem( thread_ic ) );
            if ( !thread_ic->thread_id )
                memset( semaphore, 0, sizeof( bool ) * num_ir );
            bli_thrinfo_barrier( thread_ic );

            for ( dim_t ic_offset = 0; ic_offset < m0; ic_offset += mc ) {
                double *a_l2 = a_l3 + ic_offset * rs_a;
                double *c_l2 = c_l4 + ic_offset * rs_c;

                double *a_uker, *b_uker;
                inc_t rs_a_uker, cs_a_uker;
                inc_t rs_b_uker, cs_b_uker;

                #ifdef HAS_BULK_KER
                if ( bli_rntm_pack_b( rntm ) && ic_offset == 0 )
                {
                    // Ahead-of-time packing case.
                    //
                    l1mukr_t dpackm = bli_cntx_get_blksz_def_dt( BLIS_DOUBLE, BLIS_MR, cntx ) == mr ?
                        bli_cntx_get_ukr_dt( BLIS_DOUBLE, BLIS_PACKM_NRXK_KER, cntx ) :
                        bli_cntx_get_ukr_dt( BLIS_DOUBLE, BLIS_PACKM_MRXK_KER, cntx );
                    for ( dim_t jr = 0; jr < num_jr && n0 - jc_offset - jr * nr > 0; ++jr )
                    {
                        double *b_l1 = b_l3 + jr * nr * cs_b;
                        double *b_p = b_panels + nr * k_ps * jr;
                        dim_t n_uker = min_(n0 - jc_offset - jr * nr, nr);
                        dpackm
                            (
                             BLIS_NO_CONJUGATE, BLIS_PACKED_ROWS,
                             n_uker, k_uker, k_uker,
                             &one,
                             b_l1, cs_b, rs_b,
                             b_p, nr,
                             cntx
                            );
                    }
                }
                #endif

                // Partitions JR & A-restreaming.
                // A-restream allocates packing workload w.r.t. IR to each JR worker.
                dim_t jr_start,    jr_end;
                dim_t ares_offset, ares_dummy;
                const dim_t num_jr_loc = bli_min( num_jr, ( n0 - jc_offset + nr - 1 ) / nr );
                const dim_t num_ir_loc = bli_min( num_ir, ( m0 - ic_offset + mr - 1 ) / mr );
                bli_thread_range_sub( thread_jr, num_jr_loc, 1, FALSE, &jr_start,    &jr_end     );
                bli_thread_range_sub( thread_jr, num_ir_loc, 1, FALSE, &ares_offset, &ares_dummy );

                for ( dim_t jr = jr_start; jr < jr_end; ++jr ) {
                    double *c_l1 = c_l2 + jr * nr * cs_c;
                    double *b_l1 = b_l3 + jr * nr * cs_b;;
                    double *b_p = b_panels + nr * k_ps * jr;
                    dim_t jr_offset = jc_offset + jr * nr;
                    dim_t n_uker = min_(n0 - jr_offset, nr);

                    // The B-repack strategy.
                    if ( bli_rntm_pack_b( rntm ) || ( ic_offset > 0 && has_pack_b ) ) {
                        // Reuse packed b.
                        b_uker = b_p;
                        rs_b_uker = nr;
                        cs_b_uker = 1;
                    } else {
                        b_uker = b_l1;
                        rs_b_uker = rs_b;
                        cs_b_uker = cs_b;
                    }

                    #ifdef HAS_BULK_KER
                    if ( bli_rntm_pack_a( rntm ) && jr == 0 )
                    {
                        // Ahead-of-time packing case.
                        //
                        double one = 1.0;
                        l1mukr_t dpackm = bli_cntx_get_blksz_def_dt( BLIS_DOUBLE, BLIS_MR, cntx ) == mr ?
                            bli_cntx_get_ukr_dt( BLIS_DOUBLE, BLIS_PACKM_MRXK_KER, cntx ) :
                            bli_cntx_get_ukr_dt( BLIS_DOUBLE, BLIS_PACKM_NRXK_KER, cntx );
                        for ( dim_t ir = 0; ir < num_ir && m0 - ic_offset - ir * mr > 0; ++ir )
                        {
                            double *a_l1 = a_l2 + ir * mr * rs_a;
                            double *a_p = a_panels + mr * k_ps * ir;
                            dim_t m_uker = min_(m0 - ic_offset - ir * mr, mr);
                            dpackm
                                (
                                 BLIS_NO_CONJUGATE, BLIS_PACKED_COLUMNS,
                                 m_uker, k_uker, k_uker,
                                 &one,
                                 a_l1, rs_a, cs_a,
                                 a_p, mr,
                                 cntx
                                );
                        }
                    }
                    #endif

                    // Set next_a.
                    // TODO: Consider threaded situation?
                    if ( jr + 1 < num_jr && jr_offset + n_uker < n0 ) {
                        if ( has_pack_a ) {
                            // Still using the already-packed a panels.
                            bli_auxinfo_set_next_a( a_panels, &data );
                            bls_aux_set_ls_ext_next( mr, &data ); // cs_a_next.
                        } else {
                            bli_auxinfo_set_next_a( a_l2, &data );
                            bls_aux_set_ls_ext_next( cs_a, &data ); // cs_a_next.
                        }
                    } else {
                        bls_aux_set_ls_ext_next( cs_a, &data ); // cs_a_next.
                        if ( ic_offset + mc < m0 )
                            bli_auxinfo_set_next_a( a_l2 + mc * rs_a, &data );
                        else
                            if ( lc_offset + kc < k0 )
                                bli_auxinfo_set_next_a( a_l3 + kc * cs_a, &data );
                            else
                                bli_auxinfo_set_next_a( a, &data );
                    }

                    // Set next_b
                    if ( jr + 1 < num_jr && jr_offset + n_uker < n0 ) {
                        if ( ic_offset > 0 && has_pack_b ) // Previous ic has packed b for the next jr.
                            bli_auxinfo_set_next_b( b_p + nr * k_ps, &data );
                        else
                            bli_auxinfo_set_next_b( b_l1 + nr * cs_b, &data );
                    } else if ( ic_offset + mc < m0 )
                        // Return jr.
                        bli_auxinfo_set_next_b( b_panels, &data );
                    else
                        if ( lc_offset + kc < k0 )
                            bli_auxinfo_set_next_b( b_l3 + kc * rs_b, &data );
                        else
                            bli_auxinfo_set_next_b( b_l3 + nc * cs_b, &data );

                    dim_t m_mker = min_( m0 - ic_offset, mc );
                    bls_aux_set_ps_ext_p( mr * k_ps, &data ); // ps_a_p.

                    // Regardless of whether A-repack or A-restream is deployed,
                    // one only needs to consider packing in the first millikernel.
                    if ( bli_rntm_pack_a( rntm ) || ( jr > jr_start && has_pack_a ) ) {
                        a_uker = a_panels;
                        rs_a_uker = 1;
                        cs_a_uker = mr;
                        bls_aux_set_ps_ext( bls_aux_ps_ext_p( &data ), &data ); // ps_a.
                    } else {
                        a_uker = a_l2;
                        rs_a_uker = rs_a;
                        cs_a_uker = cs_a;
                        bls_aux_set_ps_ext( rs_a * mr, &data ); // ps_a.
                    }

                    ukr_sup
                        (
                         m_mker, n_uker, k_uker,
                         alpha,
                         a_uker, rs_a_uker, cs_a_uker,
                         b_uker, rs_b_uker, cs_b_uker,
                         beta,
                         c_l1, rs_c, cs_c,
                         &data, cntx,
                         a_panels, a_uker != a_panels && jr_offset + n_uker < n0 && has_pack_a,
                         b_p,      b_uker != b_p                                 && has_pack_b,
                         ares_offset, semaphore
                        );

                }
            }
            beta = &one;
            lc_offset += k_uker;
            bli_thrinfo_barrier( thread_ic );
        }
    }
    return BLIS_SUCCESS;
}

