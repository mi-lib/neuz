/* neuZ - Neural Network Library
 * (C)Copyright, Zhidao, since 2020, all rights are reserved.
 *
 * activator functions.
 */

#include <neuz/neuz_activator.h>

/* step function */
static double _nzActivatorStep(double val){ return val >= 0 ? 1 : 0; }
static double _nzActivatorStepDif(double val){ return 0; /* case val=0 is ignored. */ }

nzActivator nz_activator_step = {
  "step",
  _nzActivatorStep,
  _nzActivatorStepDif
};

/* sigmoid function */
static double _nzActivatorSigmoid(double val){ return zSigmoid( 4*val ); }
static double _nzActivatorSigmoidDif(double val){
  double u;
  u = exp( -4*val );
  return 4 * u / zSqr( 1 + u );
}

nzActivator nz_activator_sigmoid = {
  "sigmoid",
  _nzActivatorSigmoid,
  _nzActivatorSigmoidDif
};

/* rectified linear unit function */
static double _nzActivatorReLU(double val){ return zMax( val, 0 ); }
static double _nzActivatorReLUDif(double val){ return val >= 0 ? 1 : 0; /* case val=0 is ignored. */ }

nzActivator nz_activator_relu = {
  "relu",
  _nzActivatorReLU,
  _nzActivatorReLUDif
};

/* blunt ReLU */
static double _nzActivatorBluntReLU(double val){ return 0.5 * ( val + sqrt( val*val + 1 ) ); }
static double _nzActivatorBluntReLUDif(double val){ return 0.5 + 0.5*val / sqrt( val*val + 1 ); }

nzActivator nz_activator_blunt_relu = {
  "bluntrelu",
  _nzActivatorBluntReLU,
  _nzActivatorBluntReLUDif
};

/* add the handle to the following list when you create a new activator function. */
#define NZ_ACTIVATOR_ARRAY \
  nzActivator *_nz_activator[] = {\
    &nz_activator_step,\
    &nz_activator_sigmoid,\
    &nz_activator_relu,\
    &nz_activator_blunt_relu,\
    NULL,\
  }

/* query an activator function by a string. */
nzActivator *nzActivatorQuery(char *str)
{
  NZ_ACTIVATOR_ARRAY;
  nzActivator **activator;

  for( activator=_nz_activator; *activator; activator++ )
    if( strcmp( (*activator)->typestr, str ) == 0 ) return *activator;
  return NULL;
}
