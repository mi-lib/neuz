#include <neuz/neuz.h>

#define N0 1
#define N1 5
#define N2 2

#define NT  1000000
#define NB       10
#define RATE      0.05

double loss(zVec output, zVec outref)
{
  return 0.5 * zVecSqrDist( output, outref );
}

double lossgrad(zVec output, zVec outref, int i)
{
  return zVecElem(output,i) - zVecElem(outref,i);
}

zVec train_ref(zVec input, zVec outref, double theta)
{
  double s, c;

  zSinCos( theta, &s, &c );
  zVecSetElem( input, 0, theta );
  return zVecSetElemList( outref, 0.25*(s+1), 0.25*(c+1) );
}

double train(nzNet *net, zVec input, zVec output, zVec outref, double theta)
{
  train_ref( input, outref, theta );
  nzNetPropagate( net, input );
  nzNetBackPropagate( net, input, outref, lossgrad );
  nzNetGetOutput( net, output );
  return loss( output, outref );
}

void test(nzNet *net, zVec input, zVec output, zVec outref, double theta)
{
  train_ref( input, outref, theta );
  nzNetPropagate( net, input );
  nzNetGetOutput( net, output );
  printf( "%g %g %g %g %g\n", theta, zVecElemNC(outref,0), zVecElemNC(outref,1), zVecElemNC(output,0), zVecElemNC(output,1) );
}

int main(int argc, char *argv[])
{
  nzNet nn;
  zVec input, output, outref;
  double l;
  int i, j;

  zRandInit();

  input  = zVecAlloc( N0 );
  output = zVecAlloc( N2 );
  outref = zVecAlloc( N2 );

  nzNetInit( &nn );
  nzNetAddGroupSetActivator( &nn, N0, NULL );
#if 1
  nzNetAddGroupSetActivator( &nn, N1, &nz_activator_sigmoid );
  nzNetAddGroupSetActivator( &nn, N2, &nz_activator_sigmoid );
#elif 0
  nzNetAddGroupSetActivator( &nn, N1, &nz_activator_relu );
  nzNetAddGroupSetActivator( &nn, N2, &nz_activator_relu );
#else
  nzNetAddGroupSetActivator( &nn, N1, &nz_activator_blunt_relu );
  nzNetAddGroupSetActivator( &nn, N2, &nz_activator_blunt_relu );
#endif
  nzNetConnectGroup( &nn, 0, 1 );
  nzNetConnectGroup( &nn, 1, 2 );

  /* training */
  for( i=0; i<NT; i++ ){
    nzNetInitGrad( &nn );
    for( l=0, j=0; j<NB; j++ )
      l += train( &nn, input, output, outref, zRandF(-zPI,zPI) );
    eprintf( "%03d %.10g\n", i, l );
    if( zIsTiny( l ) ) break;
    nzNetTrainSDM( &nn, RATE );
  }
  /* check */
  for( l=0, j=0; j<100; j++ )
    test( &nn, input, output, outref, zRandF(-zPI,zPI) );

  nzNetWriteZTK( &nn, "sin.ztk" );
  nzNetDestroy( &nn );
  zVecFreeAO( 3, input, output, outref );
  return 0;
}
