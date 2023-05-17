/* neuZ - Neural Network Library
 * (C)Copyright, Zhidao, since 2020, all rights are reserved.
 */
/*! \file neuz_activator.h
 * \brief activator functions.
 * \author Zhidao
 */

#ifndef __NEUZ_ACTIVATOR_H__
#define __NEUZ_ACTIVATOR_H__

#include <neuz/neuz_misc.h>

__BEGIN_DECLS

/*! \brief activator function class */
typedef struct{
  const char *typestr;   /* a string to represent type */
  double (* f)(double);  /* function */
  double (* df)(double); /* derivative function */
} nzActivator;

/*! \brief step function */
__NEUZ_EXPORT nzActivator nz_activator_step;

/*! \brief sigmoid function */
__NEUZ_EXPORT nzActivator nz_activator_sigmoid;

/*! \brief rectified linear unit function */
__NEUZ_EXPORT nzActivator nz_activator_relu;

/*! \brief blunt ReLU */
__NEUZ_EXPORT nzActivator nz_activator_blunt_relu;

/*! \brief softplus */
__NEUZ_EXPORT nzActivator nz_activator_softplus;

/*! \brief assign an activator function queried by a string. */
__NEUZ_EXPORT nzActivator *nzActivatorQuery(char *str);

__END_DECLS

#endif /* __NEUZ_ACTIVATOR_H__ */
