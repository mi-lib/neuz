#include <iostream>
#include <neuz/neuz.h>

double train(nzNet &net, zVec input, zVec output, zVec des, int i1, int i2, int oo, int oa, int on, int ox)
{
  zVecSetElemList( input, (double)i1, (double)i2 );
  zVecSetElemList( des, (double)oo, (double)oa, (double)on, (double)ox );
  net.propagate( input );
  net.getOutput( output );
  net.backpropagate( input, des, nzLossGradSquredSum );
  return nzLossSquredSum( output, des );
}

void test(nzNet &net, zVec input, zVec output, int i1, int i2)
{
  zVecSetElemList( input, (double)i1, (double)i2 );
  net.propagate( input );
  net.getOutput( output );
  printf( "I1=%g, I2=%g -> OR: %g, AND: %g, NAND: %g, XOR: %g\n", zVecElemNC(input,0), zVecElemNC(input,1), zVecElemNC(output,0), zVecElemNC(output,1), zVecElemNC(output,2), zVecElemNC(output,3) );
}

#define XOR_ZTK "xor.ztk"

#define N0 2
#define N1 5
#define N2 4

#define N_TRAIN 10000
#define RATE    0.1

int main(int argc, char *argv[])
{
  nzNet nn;
  zVec input, output, des;
  double l;
  double rate;
  int i, n_train;

  zRandInit();

  n_train = argc > 1 ? atoi( argv[1] ) : N_TRAIN;
  input  = zVecAlloc( N0 );
  output = zVecAlloc( N2 );
  des = zVecAlloc( N2 );

  /* read or create network */
  if( !nn.readZTK( XOR_ZTK ) ){
    nn.init();
    nn.addGroup( N0, NULL );
    nn.addGroup( N1, &nz_activator_sigmoid );
    nn.addGroup( N2, &nz_activator_sigmoid );
    nn.connectGroup( 0, 1 );
    nn.connectGroup( 1, 2 );
  }
  /* train */
  for( rate=RATE, i=0; i<n_train; i++, rate*=0.9 ){
    nn.initGrad();
    l  = train( nn, input, output, des, 0, 0, 0, 0, 1, 0 );
    l += train( nn, input, output, des, 1, 0, 1, 0, 1, 1 );
    l += train( nn, input, output, des, 0, 1, 1, 0, 1, 1 );
    l += train( nn, input, output, des, 1, 1, 1, 1, 0, 0 );
    std::cout << i << ' ' << l << std::endl;
    if( zIsTiny( l ) ) break;
    nn.trainSDM( RATE );
  }

  test( nn, input, output, 0, 0 );
  test( nn, input, output, 1, 0 );
  test( nn, input, output, 0, 1 );
  test( nn, input, output, 1, 1 );

  nn.writeZTK( XOR_ZTK );

  nn.destroy();
  zVecFreeAtOnce( 3, input, output, des );
  return 0;
}
