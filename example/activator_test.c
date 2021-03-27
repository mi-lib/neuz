#include <neuz/neuz.h>

#define N 100

int main(int argc, char *argv[])
{
  int i;
  double val;

  for( i=-N; i<=N; i++ ){
    val = 2*(double)i/N;
    printf( "%.10g %.10g %.10g %.10g %.10g %.10g %.10g\n", val,
      nz_activator_step.f( val ), nz_activator_step.df( val ),
      nz_activator_sigmoid.f( val ), nz_activator_sigmoid.df( val ),
      nz_activator_relu.f( val ), nz_activator_relu.df( val ) );
  }
  return 0;
}
