/* neuZ - Neural Network Library
 * (C)Copyright, Zhidao, since 2020, all rights are reserved.
 */
/*! \file neuz_neuron.h
 * \brief neuron unit and group.
 * \author Zhidao
 */

#ifndef __NEUZ_NEURON_H__
#define __NEUZ_NEURON_H__

#include <neuz/neuz_activator.h>

__BEGIN_DECLS

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
__NEUZ_EXPORT nzNeuron *nzNeuronInit(nzNeuron *neuron, int gid, int nid);

/*! \brief destroy a neuron unit. */
__NEUZ_EXPORT void nzNeuronDestroy(nzNeuron *neuron);

/*! \brief connect two neuron units. */
__NEUZ_EXPORT bool nzNeuronConnect(nzNeuron *nu, nzNeuron *nd, double weight);

/*! \brief propagate output values of upstream units to downstream. */
__NEUZ_EXPORT double nzNeuronPropagate(nzNeuron *neuron);

/*! \brief print a neuron unit. */
__NEUZ_EXPORT void nzNeuronFPrint(FILE *fp, nzNeuron *neuron);

/*! \brief neuron group class */
typedef struct{
  int id; /* identifier */
  nzNeuronList list;
} nzNeuronGroup;

/*! \brief initialize a neuron group. */
__NEUZ_EXPORT nzNeuronGroup *nzNeuronGroupInit(nzNeuronGroup *ng, int id);

/*! \brief add a neuron into a group. */
__NEUZ_EXPORT bool nzNeuronGroupAddOne(nzNeuronGroup *ng);

/*! \brief add multiple neurons into a group. */
__NEUZ_EXPORT bool nzNeuronGroupAdd(nzNeuronGroup *ng, int num);

/*! \brief destroy a neuron group. */
__NEUZ_EXPORT void nzNeuronGroupDestroy(nzNeuronGroup *ng);

/*! \brief find a neuron in a neuron group. */
__NEUZ_EXPORT nzNeuron *nzNeuronGroupFindNeuron(nzNeuronGroup *ng, int gid, int nid);

/*! \brief set activator functions of units in a neuron group. */
__NEUZ_EXPORT void nzNeuronGroupSetActivator(nzNeuronGroup *ng, nzActivator *activator);

/*! \brief set input values to a neuron group. */
__NEUZ_EXPORT bool nzNeuronGroupSetInput(nzNeuronGroup *ng, zVec input);

/*! \brief get output values from a neuron group. */
__NEUZ_EXPORT bool nzNeuronGroupGetOutput(nzNeuronGroup *ng, zVec output);

/*! \brief propagate output values of upstream group to downstream. */
__NEUZ_EXPORT void nzNeuronGroupPropagate(nzNeuronGroup *ng);

/*! \brief print a neuron group. */
__NEUZ_EXPORT void nzNeuronGroupFPrint(FILE *fp, nzNeuronGroup *ng);

/*! \brief neural network class */
zListClass( nzNet, nzNetCell, nzNeuronGroup );

#define nzNetInputLayer(net)  ( &zListTail(net)->data )
#define nzNetOutputLayer(net) ( &zListHead(net)->data )

/*! \brief initialize a neural network. */
#define nzNetInit(net) zListInit( net )

/*! \brief add a neuron group to a neural network. */
__NEUZ_EXPORT bool nzNetAddGroup(nzNet *net, int num);

/*! \brief add a neuron group with a specified activator to a neural network. */
__NEUZ_EXPORT bool nzNetAddGroupSetActivator(nzNet *net, int num, nzActivator *activator);

/*! \brief destroy a neural network. */
__NEUZ_EXPORT void nzNetDestroy(nzNet *net);

/*! \brief find a neuron group in a neural network. */
__NEUZ_EXPORT nzNeuronGroup *nzNetFindGroup(nzNet *net, int id);

/*! \brief find a neuron in a neural network. */
__NEUZ_EXPORT nzNeuron *nzNetFindNeuron(nzNet *net, int gid, int nid);

/*! \brief add a neuron to a neural network. */
__NEUZ_EXPORT bool nzNetAddNeuron(nzNet *net, int gid, int nid, nzActivator *activator, double bias);

/*! \brief connect two neuron groups in a neural network. */
__NEUZ_EXPORT bool nzNetConnectGroup(nzNet *net, int iu, int id);

/*! \brief connects two neurons in a neural network. */
__NEUZ_EXPORT bool nzNetConnect(nzNet *net, int ugid, int unid, int dgid, int dnid, double weight);

/*! \brief set input values to the input layer of a neural network. */
__NEUZ_EXPORT bool nzNetSetInput(nzNet *net, zVec input);

/*! \brief get output values from a neural network. */
__NEUZ_EXPORT bool nzNetGetOutput(nzNet *net, zVec output);

/*! \brief propagate input values to a neural network to the output. */
__NEUZ_EXPORT double nzNetPropagate(nzNet *net, zVec input);

/*! \brief initialize gradients of weights and bias of a neural network. */
__NEUZ_EXPORT void nzNetInitGrad(nzNet *net);

/*! \brief back-propagate loss and train a neural network. */
__NEUZ_EXPORT bool nzNetBackPropagate(nzNet *net, zVec input, zVec des, double (* lossgrad)(zVec,zVec,int));

/*! \brief train a neural network based on the steepest descent method. */
__NEUZ_EXPORT bool nzNetTrainSDM(nzNet *net, double rate);

/*! \brief print a neural network. */
__NEUZ_EXPORT void nzNetFPrint(FILE *fp, nzNet *net);

/* parse ZTK format */

#define NZ_NET_TAG "neuralnetwork"

/*! \brief read a neural network from a ZTK format processor. */
__NEUZ_EXPORT nzNet *nzNetFromZTK(nzNet *net, ZTK *ztk);

/*! \brief read a neural network from a ZTK format file. */
__NEUZ_EXPORT nzNet *nzNetReadZTK(nzNet *net, char filename[]);

/*! \brief print out a neural network to a file. */
__NEUZ_EXPORT void nzNetFPrintZTK(FILE *fp, nzNet *net);

/*! \brief write a neural network to a ZTK format file. */
__NEUZ_EXPORT bool nzNetWriteZTK(nzNet *net, char filename[]);

__END_DECLS

#endif /* __NEUZ_NEURON_H__ */
