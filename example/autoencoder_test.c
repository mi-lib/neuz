#include <neuz/neuz.h>

#define N0 2
#define N1 5
#define N2 3

#define NT  1000000
#define NB        1
#define RATE      0.1

double loss(zVec output, zVec des)
{
  return 0.5 * zVecSqrDist( output, des );
}

double lossgrad(zVec output, zVec des, int i)
{
  return zVecElem(output,i) - zVecElem(des,i);
}

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
  nzNetBackPropagate( net, input, input, lossgrad );
  nzNetGetOutput( net, output );
  return loss( output, input );
}

void test(nzNet *net, zVec input, zVec output, double theta)
{
  train_ref( input, theta );
  nzNetPropagate( net, input );
  nzNetGetOutput( net, output );
  printf( "%g %g %g %g %g\n", theta, zVecElemNC(input,0), zVecElemNC(input,1), zVecElemNC(output,0), zVecElemNC(output,1) );
}

int main(int argc, char *argv[])
{
  nzNet nn;
  zVec input, output;
  double l;
  int i, j;

  zRandInit();

  input  = zVecAlloc( N0 );
  output = zVecAlloc( N0 );

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

  /* training */
  for( i=0; i<NT; i++ ){
    nzNetInitGrad( &nn );
    for( l=0, j=0; j<NB; j++ )
      l += train( &nn, input, output, zRandF(-zPI,zPI) );
    eprintf( "%03d %.10g\n", i, l );
    if( zIsTiny( l ) ) break;
    nzNetTrainSDM( &nn, RATE );
  }
  /* check */
  for( l=0, j=0; j<100; j++ )
    test( &nn, input, output, zRandF(-zPI,zPI) );

  nzNetWriteZTK( &nn, "sin.ztk" );
  nzNetDestroy( &nn );
  zVecFreeAO( 2, input, output );
  return 0;
}
