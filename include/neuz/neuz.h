/* neuZ - Neural Network Library
 * (C)Copyright, Zhidao, since 2020, all rights are reserved.
 */

/*!
 * \mainpage

 neuZ is a neural network library written in C.
 */

#ifndef __NEUZ_H__
#define __NEUZ_H__

#include <zm/zm.h>

__BEGIN_DECLS

/*! \brief activator function class */
typedef struct{
  char *typestr;         /* a string to represent type */
  double (* f)(double);  /* function */
  double (* df)(double); /* derivative function */
} nzActivator;

/*! \brief step function */
__EXPORT double nzActivatorStep(double val);
__EXPORT double nzActivatorStepDif(double val);
extern nzActivator nz_activator_step;

/*! \brief sigmoid function */
__EXPORT double nzActivatorSigmoid(double val);
__EXPORT double nzActivatorSigmoidDif(double val);
extern nzActivator nz_activator_sigmoid;

/*! \brief rectified linear unit function */
__EXPORT double nzActivatorReLU(double val);
__EXPORT double nzActivatorReLUDif(double val);
extern nzActivator nz_activator_relu;

/*! \brief assign an activator function queried by a string. */
__EXPORT nzActivator *nzActivatorQueryAssign(char *str);

/*! \brief axon class */
typedef struct _nzAxon{
  double weight;
  void *upstream;
  double _dw;
  struct _nzAxon *next;
} nzAxon;

/*! \brief unit neuron class */
typedef struct _nzNeuronData{
  int gid; /* group identifier */
  int nid; /* neuron identifier */
  double input;
  double output;
  double bias;
  double _db;
  double _p;
  double _v;
  nzActivator *activator;
  nzAxon *axon;
} nzNeuronData;

/*! \brief neuron list class */
zListClass( nzNeuronList, nzNeuron, nzNeuronData );

/*! \brief initialize a new neuron unit. */
__EXPORT nzNeuron *nzNeuronInit(nzNeuron *neuron, int gid, int nid);

/*! \brief destroy a neuron unit. */
__EXPORT void nzNeuronDestroy(nzNeuron *neuron);

/*! \brief connect two neuron units. */
__EXPORT bool nzNeuronConnect(nzNeuron *nu, nzNeuron *nd, double weight);

/*! \brief propagate output values of upstream units to downstream. */
__EXPORT double nzNeuronPropagate(nzNeuron *neuron);

/*! \brief print a neuron unit. */
__EXPORT void nzNeuronFPrint(FILE *fp, nzNeuron *neuron);

/*! \brief neuron group class */
typedef struct{
  int id; /* identifier */
  nzNeuronList list;
} nzNeuronGroup;

/*! \brief initialize a neuron group. */
__EXPORT nzNeuronGroup *nzNeuronGroupInit(nzNeuronGroup *ng, int id);

/*! \brief add a neuron into a group. */
__EXPORT bool nzNeuronGroupAddOne(nzNeuronGroup *ng);

/*! \brief add multiple neurons into a group. */
__EXPORT bool nzNeuronGroupAdd(nzNeuronGroup *ng, int num);

/*! \brief destroy a neuron group. */
__EXPORT void nzNeuronGroupDestroy(nzNeuronGroup *ng);

/*! \brief find a neuron in a neuron group. */
__EXPORT nzNeuron *nzNeuronGroupFindNeuron(nzNeuronGroup *ng, int gid, int nid);

/*! \brief set activator functions of units in a neuron group. */
__EXPORT void nzNeuronGroupSetActivator(nzNeuronGroup *ng, nzActivator *activator);

/*! \brief set input values to a neuron group. */
__EXPORT bool nzNeuronGroupSetInput(nzNeuronGroup *ng, zVec input);

/*! \brief get output values from a neuron group. */
__EXPORT bool nzNeuronGroupGetOutput(nzNeuronGroup *ng, zVec output);

/*! \brief propagate output values of upstream group to downstream. */
__EXPORT void nzNeuronGroupPropagate(nzNeuronGroup *ng);

/*! \brief print a neuron group. */
__EXPORT void nzNeuronGroupFPrint(FILE *fp, nzNeuronGroup *ng);

/*! \brief neural network class */
zListClass( nzNet, nzNetCell, nzNeuronGroup );

#define nzNetInputLayer(net)  ( &zListTail(net)->data )
#define nzNetOutputLayer(net) ( &zListHead(net)->data )

/*! \brief initialize a neural network. */
#define nzNetInit(net) zListInit( net )

/*! \brief add a neuron group to a neural network. */
__EXPORT bool nzNetAddGroup(nzNet *net, int num);

/*! \brief add a neuron group with a specified activator to a neural network. */
__EXPORT bool nzNetAddGroupSetActivator(nzNet *net, int num, nzActivator *activator);

/*! \brief destroy a neural network. */
__EXPORT void nzNetDestroy(nzNet *net);

/*! \brief find a neuron group in a neural network. */
__EXPORT nzNeuronGroup *nzNetFindGroup(nzNet *net, int id);

/*! \brief find a neuron in a neural network. */
__EXPORT nzNeuron *nzNetFindNeuron(nzNet *net, int gid, int nid);

/*! \brief add a neuron to a neural network. */
__EXPORT bool nzNetAddNeuron(nzNet *net, int gid, int nid, nzActivator *activator, double bias);

/*! \brief connect two neuron groups in a neural network. */
__EXPORT bool nzNetConnectGroup(nzNet *net, int iu, int id);

/*! \brief connects two neurons in a neural network. */
__EXPORT bool nzNetConnect(nzNet *net, int ugid, int unid, int dgid, int dnid, double weight);

/*! \brief set input values to the input layer of a neural network. */
__EXPORT bool nzNetSetInput(nzNet *net, zVec input);

/*! \brief get output values from a neural network. */
__EXPORT bool nzNetGetOutput(nzNet *net, zVec output);

/*! \brief propagate input values to a neural network to the output. */
__EXPORT double nzNetPropagate(nzNet *net, zVec input);

/*! \brief initialize gradients of weights and bias of a neural network. */
__EXPORT void nzNetInitGrad(nzNet *net);

/*! \brief back-propagate loss and train a neural network. */
__EXPORT bool nzNetBackPropagate(nzNet *net, zVec input, zVec des, double (* lossgrad)(zVec,zVec,int));

/*! \brief train a neural network based on the steepest descent method. */
__EXPORT bool nzNetTrainSDM(nzNet *net, double rate);

/*! \brief print a neural network. */
__EXPORT void nzNetFPrint(FILE *fp, nzNet *net);

/* parse ZTK format */

#define NZ_NET_TAG "neuralnetwork"

/*! \brief register a definition of tag-and-keys for a neural network to a ZTK format processor. */
__EXPORT bool nzNetRegZTK(ZTK *ztk, char *tag);

/*! \brief read a neural network from a ZTK format processor. */
__EXPORT nzNet *nzNetFromZTK(nzNet *net, ZTK *ztk);

/*! \brief read a neural network from a ZTK format file. */
__EXPORT nzNet *nzNetReadZTK(nzNet *net, char filename[]);

/*! \brief print out a neural network to a file. */
__EXPORT void nzNetFPrintZTK(FILE *fp, nzNet *net);

/*! \brief write a neural network to a ZTK format file. */
__EXPORT bool nzNetWriteZTK(nzNet *net, char filename[]);

__END_DECLS

#endif /* __NEUZ_H__ */
