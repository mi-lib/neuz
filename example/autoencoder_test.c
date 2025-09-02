#include <neuz/neuz.h>

zVec train_ref(zVec input, double theta)
{
  double s, c;

  zSinCos( theta, &s, &c );
  return zVecSetElemList( input, 0.25*(s+1), 0.25*(c+1) );
}

double train(nzNet *net, zVec input, zVec output, double theta)
{
  train_ref( input, theta );
  nzNetPropagate( net, input );
  nzNetBackPropagate( net, input, input, nzLossGradSquredSum );
  nzNetGetOutput( net, output );
  return nzLossSquredSum( output, input );
}

void test(nzNet *net, zVec input, zVec output, double theta)
{
  train_ref( input, theta );
  nzNetPropagate( net, input );
  nzNetGetOutput( net, output );
  printf( "%g %g %g %g %g\n", theta, zVecElemNC(input,0), zVecElemNC(input,1), zVecElemNC(output,0), zVecElemNC(output,1) );
}

#define SIN_AE_ZTK "sin_ae.ztk"

#define N0 2
#define N1 5
#define N2 3

#define N_TRAIN 10000
#define N_BATCH    10
#define RATE        0.1

int main(int argc, char *argv[])
{
  nzNet nn;
  zVec input, output;
  double l;
  double rate;
  int i, j, n_train, n_batch;

  zRandInit();

  n_train = argc > 1 ? atoi( argv[1] ) : N_TRAIN;
  n_batch = argc > 2 ? atoi( argv[2] ) : N_BATCH;
  input  = zVecAlloc( N0 );
  output = zVecAlloc( N0 );

  /* read or create network */
  if( !nzNetReadZTK( &nn, SIN_AE_ZTK ) ){
    nzNetInit( &nn );
    nzNetAddGroupSetActivator( &nn, N0, NULL );
    nzNetAddGroupSetActivator( &nn, N1, &nz_activator_sigmoid );
    nzNetAddGroupSetActivator( &nn, N2, &nz_activator_sigmoid );
    nzNetAddGroupSetActivator( &nn, N1, &nz_activator_sigmoid );
    nzNetAddGroupSetActivator( &nn, N0, &nz_activator_sigmoid );
    nzNetConnectGroup( &nn, 0, 1 );
    nzNetConnectGroup( &nn, 1, 2 );
    nzNetConnectGroup( &nn, 2, 3 );
    nzNetConnectGroup( &nn, 3, 4 );
  }
  /* train */
  for( rate=RATE, i=0; i<n_train; i++, rate*=0.9 ){
    nzNetInitGrad( &nn );
    for( l=0, j=0; j<n_batch; j++ )
      l += train( &nn, input, output, zRandF(-zPI,zPI) );
    eprintf( "%03d %.10g\n", i, l );
    if( zIsTiny( l ) ) break;
    nzNetTrainSDM( &nn, RATE );
  }
  /* check */
  for( l=0, j=0; j<100; j++ )
    test( &nn, input, output, zRandF(-zPI,zPI) );

  nzNetWriteZTK( &nn, SIN_AE_ZTK );
  nzNetDestroy( &nn );
  zVecFreeAtOnce( 2, input, output );
  return 0;
}
