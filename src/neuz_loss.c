/* neuZ - Neural Network Library
 * (C)Copyright, Zhidao, since 2020, all rights are reserved.
 *
 * loss functions.
 */

#include <neuz/neuz_loss.h>

/* sum-of-squares loss function */
double nzLossSquareSum(zVec v, zVec v_ref)
{
  return 0.5 * zVecSqrDist( v, v_ref );
}

/* gradient of sum-of-squares loss function */
double nzLossGradSquareSum(zVec v, zVec v_ref, int i)
{
  return zVecElem(v,i) - zVecElem(v_ref,i);
}
