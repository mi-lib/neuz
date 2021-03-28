#include <neuz/neuz.h>

void test(nzNet *net, zVec input, zVec output, int i1, int i2)
{
  zVecSetElemList( input, (double)i1, (double)i2 );
  nzNetPropagate( net, input );
  nzNetGetOutput( net, output );
  printf( "I1=%g, I2=%g -> OR: %g, AND: %g, NAND: %g, XOR: %g\n", zVecElemNC(input,0), zVecElemNC(input,1), zVecElemNC(output,0), zVecElemNC(output,1), zVecElemNC(output,2), zVecElemNC(output,3) );
}

int main(int argc, char *argv[])
{
  nzNet nn;
  zVec input, output;

  nzNetReadZTK( &nn, argv[1] );

  input  = zVecAlloc( 2 );
  output = zVecAlloc( 4 );

  test( &nn, input, output, 0, 0 );
  test( &nn, input, output, 1, 0 );
  test( &nn, input, output, 0, 1 );
  test( &nn, input, output, 1, 1 );

  nzNetDestroy( &nn );
  zVecFreeAO( 2, input, output );
  return 0;
}
