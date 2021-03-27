#include <neuz/neuz.h>

#define N0 3
#define N1 5
#define N2 4

int main(int argc, char *argv[])
{
  nzNet nn;
  zVec input, output;

  input  = zVecAlloc( N0 );
  output = zVecAlloc( N2 );
  zVecRandUniform( input, 0, 10 );

  nzNetInit( &nn );
  nzNetAddGroup( &nn, N0 );
  nzNetAddGroup( &nn, N1 );
  nzNetAddGroup( &nn, N2 );
  nzNetConnectGroup( &nn, 0, 1 );
  nzNetConnectGroup( &nn, 1, 2 );
  nzNetFPrint( stdout, &nn );

  nzNetPropagate( &nn, input );
  nzNetGetOutput( &nn, output );
  zVecPrint( output );

  nzNetDestroy( &nn );
  zVecFree( input );
  zVecFree( output );
  return 0;
}
