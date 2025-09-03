#include <neuz/neuz.h>

zVec train_ref(const zVec input, zVec outref)
{
  zVecSetElem( outref, 0, 1.0*zVecElem(input,0) + 2.0*zVecElem(input,1) );
  zVecSetElem( outref, 1, 2.0*zVecElem(input,0) - 3.0*zVecElem(input,1) );
  return outref;
}

double train(nzNet *net, const zVec input, zVec output, zVec outref)
{
  train_ref( input, outref );
  nzNetPropagate( net, input );
  nzNetBackPropagate( net, input, outref, nzLossGradSquareSum );
  nzNetGetOutput( net, output );
  return nzLossSquareSum( output, outref );
}

void test(nzNet *net, const zVec input, zVec output, zVec outref)
{
  train_ref( input, outref );
  nzNetPropagate( net, input );
  nzNetGetOutput( net, output );
  printf( "%g %g %g %g\n", zVecElemNC(outref,0), zVecElemNC(outref,1), zVecElemNC(output,0), zVecElemNC(output,1) );
}

#define N0 2
#define N1 4
#define N2 2

#define N_TRAIN 100000
#define N_BATCH     10
#define RATE         0.001

int main(int argc, char *argv[])
{
  nzNet nn;
  zVec input, output, outref;
  double l;
  double rate;
  int i, j, n_train, n_batch;

  zRandInit();

  n_train = argc > 1 ? atoi( argv[1] ) : N_TRAIN;
  n_batch = argc > 2 ? atoi( argv[2] ) : N_BATCH;

  /* read or create network */
  nzNetInit( &nn );
  nzNetAddGroupSetActivator( &nn, N0, NULL );
  nzNetAddGroupSetActivator( &nn, N1, &nz_activator_softplus );
  nzNetAddGroupSetActivator( &nn, N2, &nz_activator_ident );
  nzNetConnectGroup( &nn, 0, 1 );
  nzNetConnectGroup( &nn, 1, 2 );
  input  = zVecAlloc( nzNetInputSize(&nn) );
  output = zVecAlloc( nzNetOutputSize(&nn) );
  outref = zVecAlloc( nzNetOutputSize(&nn) );

  /* train */
  for( rate=RATE, i=0; i<n_train; i++, rate*=0.9 ){
    nzNetInitGrad( &nn );
    for( l=0, j=0; j<n_batch; j++ ){
      zVecRandUniform( input, -10, 10 );
      l += train( &nn, input, output, outref );
    }
    eprintf( "%03d %.10g\n", i, l );
    if( zIsTiny( l ) ) break;
    nzNetTrainSDM( &nn, rate );
  }
  /* check */
  for( l=0, j=0; j<100; j++ ){
    zVecRandUniform( input, -10, 10 );
    test( &nn, input, output, outref );
  }

  nzNetDestroy( &nn );
  zVecFreeAtOnce( 3, input, output, outref );
  return 0;
}
