#include "blis_test.h"
#include "blis_utils.h"
#include "test_xpbyv.h"

// Local prototypes.
void libblis_test_xpbyv_deps (
  thread_data_t* tdata,
  test_params_t* params,
  test_op_t*     op
);

void libblis_test_xpbyv_impl (
  iface_t   iface,
  obj_t*    x,
  obj_t*    beta,
  obj_t*    y
);

double libblis_test_xpbyv_check (
  test_params_t* params,
  obj_t*         x,
  obj_t*         beta,
  obj_t*         y,
  obj_t*         y_orig
);

double libblis_ref_xpbyv(
  test_params_t* params,
  obj_t*  x,
  obj_t*  beta,
  obj_t*  y,
  obj_t*  y_save
) {
  double resid = 0.0;

  if((params->bitextf == 0) && (params->oruflw == BLIS_DEFAULT)) {
 	  // Perform checks.
    resid = libblis_test_xpbyv_check( params, x, beta, y, y_save );
  }
  else {
    if(params->oruflw == BLIS_DEFAULT) {
      resid = libblis_test_ixpbyv_check( params, x, beta, y, y_save );
    }
    else {
      resid = libblis_test_vector_check(params, x);
    }
  }
  return resid;
}

double libblis_test_bitrp_xpbyv(
  test_params_t* params,
  iface_t        iface,
  obj_t*         x,
  obj_t*         beta,
  obj_t*         y,
  obj_t*         y_orig,
  obj_t*         r,
  num_t          dt
) {
  double resid = 0.0;
	 unsigned int n_repeats = params->n_repeats;
	 unsigned int i;

  for(i = 0; i < n_repeats; i++) {
    bli_copyv( y_orig, r );
    libblis_test_xpbyv_impl( iface, x, beta, r );
    resid = libblis_test_bitrp_vector(y, r, dt);
  }
  return resid;
}

double libblis_test_op_xpbyv (
  test_params_t* params,
  iface_t        iface,
  char*          dc_str,
  char*          pc_str,
  char*          sc_str,
  tensor_t*      dim,
  atom_t         alpha
){
  num_t        datatype;
  dim_t        m;
  conj_t       conjx;
  obj_t        beta, x, y;
  obj_t        y_save;
  double       resid = 0.0;

  // Use the datatype of the first char in the datatype combination string.
  bli_param_map_char_to_blis_dt( dc_str[0], &datatype );

  // Map the dimension specifier to actual dimensions.
  m = dim->m;

  // Map parameter characters to BLIS constants.
  bli_param_map_char_to_blis_conj( pc_str[0], &conjx );

  // Create test scalars.
  bli_obj_scalar_init_detached( datatype, &beta );

  // Create test operands (vectors and/or matrices).
  libblis_test_vobj_create( params, datatype, sc_str[0], m, &x );
  libblis_test_vobj_create( params, datatype, sc_str[1], m, &y );
  libblis_test_vobj_create( params, datatype, sc_str[1], m, &y_save );

  // Set beta.
  if ( bli_obj_is_real( &y ) )
    bli_setsc( -2.0,  0.0, &beta );
  else
    bli_setsc(  0.0, -2.0, &beta );

  // Randomize x and y, and save y.
  if((params->bitextf == 0) && (params->oruflw == BLIS_DEFAULT)) {
    libblis_test_vobj_randomize( params, FALSE, &x );
    libblis_test_vobj_randomize( params, FALSE, &y );
  } else {
    libblis_test_vobj_irandomize( params, &x );
    libblis_test_vobj_irandomize( params, &y );
  }

  bli_copyv( &y, &y_save );

  // Apply the parameters.
  bli_obj_set_conj( conjx, &x );

  libblis_test_xpbyv_impl( iface, &x, &beta, &y );

#ifndef __GTEST_VALGRIND_TEST__
  if(params->bitrp) {
    obj_t r;

    libblis_test_vobj_create( params, datatype, sc_str[1], m, &r );

    resid = libblis_test_bitrp_xpbyv(params, iface, &x, &beta, &y,
                                                    &y_save, &r, datatype);

    bli_obj_free( &r );
  }
  else {
    resid = libblis_ref_xpbyv( params, &x, &beta, &y, &y_save );
  }
#endif

  // Zero out performance and residual if output vector is empty.
  libblis_test_check_empty_problem( &y, &resid );

  // Free the test objects.
  libblis_test_obj_free( &x );
  libblis_test_obj_free( &y );
  libblis_test_obj_free( &y_save );

  return abs(resid);
}

void libblis_test_xpbyv_impl (
  iface_t   iface,
  obj_t*    x,
  obj_t*    beta,
  obj_t*    y
) {
  switch ( iface )
  {
    case BLIS_TEST_SEQ_FRONT_END:
      bli_xpbyv( x, beta, y );
      break;

    default:
      libblis_test_printf_error( "Invalid interface type.\n" );
  }
}

double libblis_test_xpbyv_check (
  test_params_t* params,
  obj_t*         x,
  obj_t*         beta,
  obj_t*         y,
  obj_t*         y_orig
) {
  num_t  dt      = bli_obj_dt( y );
  num_t  dt_real = bli_obj_dt_proj_to_real( y );

  dim_t  m       = bli_obj_vector_dim( y );

  obj_t  x_temp, y_temp;
  obj_t  norm;

  double junk;
  double resid = 0.0;

  //
  // Pre-conditions:
  // - x is randomized.
  // - y_orig is randomized.
  // Note:
  // - beta should have a non-zero imaginary component in the complex
  //   cases in order to more fully exercise the implementation.
  //
  // Under these conditions, we assume that the implementation for
  //
  //   y := beta * y_orig + conjx(x)
  //
  // is functioning correctly if
  //
  //   normfv( y - ( beta * y_orig + conjx(x) ) )
  //
  // is negligible.
  //

  bli_obj_scalar_init_detached( dt_real, &norm );

  bli_obj_create( dt, m, 1, 0, 0, &x_temp );
  bli_obj_create( dt, m, 1, 0, 0, &y_temp );

  bli_copyv( x,      &x_temp );
  bli_copyv( y_orig, &y_temp );

  bli_scalv( beta, &y_temp );
  bli_addv( &x_temp, &y_temp );

  bli_subv( &y_temp, y );
  bli_normfv( y, &norm );
  bli_getsc( &norm, &resid, &junk );

  bli_obj_free( &x_temp );
  bli_obj_free( &y_temp );

  return resid;
}