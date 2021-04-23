/* neuZ - Neural Network Library
 * (C)Copyright, Zhidao, since 2020, all rights are reserved.
 */
/*! \file neuz_errmsg.h
 * \brief error and warning messages.
 * \author Zhidao
 */

#ifndef __NEUZ_ERRMSG_H__
#define __NEUZ_ERRMSG_H__

/* NOTE: never include this header file in user programs. */

/* error messages */

#define NEUZ_ERR_UNKNOWN_ACTIVATOR "unknown activator type: %s"

#define NEUZ_ERR_GROUP_NOT_FOUND "neuron group %d not found"

#define NEUZ_ERR_NEURON_NOT_FOUND "neuron %d:%d not found"

/* warning messages */

#define NEUZ_WARN_GROUP_MISMATCH_SIZ "size mismatch between a neuron group (%d) and a vector (%d)"

#define NEUZ_WARN_NET_TOOFEWLAYER "cannot apply backpropagation to a two-or-less-layered network."

__END_DECLS

#endif /* __NEUZ_ERRMSG_H__ */
