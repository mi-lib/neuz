/* neuZ - Neural Network Library
 * (C)Copyright, Zhidao, since 2020, all rights are reserved.
 *
 * loss functions.
 */

#include <neuz/neuz_loss.h>

/* squared-sum function */
double nzLossSquredSum(zVec v, zVec v_ref)
{
  return 0.5 * zVecSqrDist( v, v_ref );
}

/* gradient of squared-sum function */
double nzLossGradSquredSum(zVec v, zVec v_ref, int i)
{
  return zVecElem(v,i) - zVecElem(v_ref,i);
}
