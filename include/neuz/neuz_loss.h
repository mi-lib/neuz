/* neuZ - Neural Network Library
 * (C)Copyright, Zhidao, since 2020, all rights are reserved.
 */
/*! \file neuz_loss.h
 * \brief loss functions.
 * \author Zhidao
 */

#ifndef __NEUZ_LOSS_H__
#define __NEUZ_LOSS_H__

#include <neuz/neuz_misc.h>

__BEGIN_DECLS

/*! \brief squared-sum loss function */
__NEUZ_EXPORT double nzLossSquredSum(zVec v, zVec v_ref);
__NEUZ_EXPORT double nzLossGradSquredSum(zVec v, zVec v_ref, int i);

__END_DECLS

#endif /* __NEUZ_LOSS_H__ */
