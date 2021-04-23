/* neuZ - Neural Network Library
 * (C)Copyright, Zhidao, since 2020, all rights are reserved.
 */
/*! \file neuz_activator.h
 * \brief activator functions.
 * \author Zhidao
 */

#ifndef __NEUZ_ACTIVATOR_H__
#define __NEUZ_ACTIVATOR_H__

#include <zm/zm.h>
#include <neuz/neuz_errmsg.h>

__BEGIN_DECLS

/*! \brief activator function class */
typedef struct{
  char *typestr;         /* a string to represent type */
  double (* f)(double);  /* function */
  double (* df)(double); /* derivative function */
} nzActivator;

/*! \brief step function */
extern nzActivator nz_activator_step;

/*! \brief sigmoid function */
extern nzActivator nz_activator_sigmoid;

/*! \brief rectified linear unit function */
extern nzActivator nz_activator_relu;

/*! \brief blunt ReLU */
extern nzActivator nz_activator_blunt_relu;

/*! \brief assign an activator function queried by a string. */
__EXPORT nzActivator *nzActivatorQuery(char *str);

__END_DECLS

#endif /* __NEUZ_ACTIVATOR_H__ */
