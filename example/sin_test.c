#include <neuz/neuz.h>

#define N0 2
#define N1 5
#define N2 3

#define NT  1000000
#define NB        1
#define RATE      0.05

double loss(zVec output, zVec des)
{
  return 0.5 * zVecSqrDist( output, des );
}

double lossgrad(zVec output, zVec des, int i)
{
  return zVecElem(output,i) - zVecElem(des,i);
}

double train(nzNet *net, zVec input, zVec output, double theta)
{
  double s, c;

  zSinCos( theta, &s, &c );
  zVecSetElemList( input, 0.5*(s+1), 0.5*(c+1) );
  nzNetPropagate( net, input );
  nzNetBackPropagate( net, input, input, lossgrad );
  nzNetGetOutput( net, output );
  return loss( output, input );
}

void test(nzNet *net, zVec input, zVec output, double theta)
{
  double s, c;

  zSinCos( theta, &s, &c );
  zVecSetElemList( input, 0.5*(s+1), 0.5*(c+1) );
  nzNetPropagate( net, input );
  nzNetGetOutput( net, output );
  printf( "%g %g %g %g %g\n", theta, s, c, 2*zVecElemNC(output,0)-1, 2*zVecElemNC(output,1)-1 );
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

  for( l=0, j=0; j<100; j++ )
    test( &nn, input, output, zRandF(-zPI,zPI) );

  nzNetWriteZTK( &nn, "sin.ztk" );

  nzNetDestroy( &nn );
  zVecFreeAO( 2, input, output );
  return 0;
}
