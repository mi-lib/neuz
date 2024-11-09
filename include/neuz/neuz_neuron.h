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
ZDECL_STRUCT( nzAxon );
ZDEF_STRUCT( __NEUZ_CLASS_EXPORT, nzAxon ){
  double weight;
  void *upstream;
  double _dw;
  nzAxon *next;
};

/*! \brief unit neuron class */
ZDEF_STRUCT( __NEUZ_CLASS_EXPORT, nzNeuronData ){
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
};

/*! \brief neuron list class */
ZDECL_STRUCT( nzNeuron );
ZDEF_STRUCT( __NEUZ_CLASS_EXPORT, nzNeuron ){
  nzNeuron *prev, *next;
  nzNeuronData data;
#ifdef __cplusplus
  nzNeuron() : prev{this}, next{this} {}
  nzNeuron *init(int gid, int nid);
  void destroy();
  bool connectTo(nzNeuron *nd, double weight);
  double propagate();
  void fprint(FILE *fp);
#endif /* __cplusplus */
};

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

#ifdef __cplusplus
inline nzNeuron *nzNeuron::init(int gid, int nid){ return nzNeuronInit( this, gid, nid ); }
inline void nzNeuron::destroy(){ nzNeuronDestroy( this ); }
inline bool nzNeuron::connectTo(nzNeuron *nd, double weight){ return nzNeuronConnect( this, nd, weight ); }
inline double nzNeuron::propagate(){ return nzNeuronPropagate( this ); }
inline void nzNeuron::fprint(FILE *fp){ nzNeuronFPrint( fp, this ); }
#endif /* __cplusplus */

/*! \brief neuron list class */
ZDEF_STRUCT( __NEUZ_CLASS_EXPORT, nzNeuronList ){
  int size;
  nzNeuron root;
#ifdef __cplusplus
  nzNeuronList() : size{0} {}
#endif /* __cplusplus */
};

/*! \brief neuron group class */
ZDEF_STRUCT( __NEUZ_CLASS_EXPORT, nzNeuronGroup ){
  int id; /* identifier */
  nzNeuronList list;
#ifdef __cplusplus
  nzNeuronGroup *init(int id);
  bool add();
  bool add(int num);
  void destroy();
  nzNeuron *find(int gid, int nid);
  void setActivator(nzActivator *activator);
  bool setInput(zVec input);
  bool getOutput(zVec output);
  void propagate();
  void fprint(FILE *fp);
#endif /* __cplusplus */
};

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

#ifdef __cplusplus
inline nzNeuronGroup *nzNeuronGroup::init(int id){ return nzNeuronGroupInit( this, id ); }
inline bool nzNeuronGroup::add(){ return nzNeuronGroupAddOne( this ); }
inline bool nzNeuronGroup::add(int num){ return nzNeuronGroupAdd( this, num ); }
inline void nzNeuronGroup::destroy(){ nzNeuronGroupDestroy( this ); }
inline nzNeuron *nzNeuronGroup::find(int gid, int nid){ return nzNeuronGroupFindNeuron( this, gid, nid ); }
inline void nzNeuronGroup::setActivator(nzActivator *activator){ nzNeuronGroupSetActivator( this, activator ); }
inline bool nzNeuronGroup::setInput(zVec input){ return nzNeuronGroupSetInput( this, input ); }
inline bool nzNeuronGroup::getOutput(zVec output){ return nzNeuronGroupGetOutput( this, output ); }
inline void nzNeuronGroup::propagate(){ nzNeuronGroupPropagate( this ); }
inline void nzNeuronGroup::fprint(FILE *fp){ nzNeuronGroupFPrint( fp, this ); }
#endif /* __cplusplus */

/*! \brief neural network class */
ZDECL_STRUCT( nzNetCell );
ZDEF_STRUCT( __NEUZ_CLASS_EXPORT, nzNetCell ){
  nzNetCell *prev, *next;
  nzNeuronGroup data;
#ifdef __cplusplus
  nzNetCell() : prev{this}, next{this} {}
#endif /* __cplusplus */
};

ZDEF_STRUCT( __NEUZ_CLASS_EXPORT, nzNet ){
  int size;
  nzNetCell root;
#ifdef __cplusplus
  nzNet() : size{0} {}
  void init();
  void destroy();
  nzNeuronGroup *inputLayer();
  nzNeuronGroup *outputLayer();
  bool addGroup(int num);
  bool addGroup(int num, nzActivator *activator);
  nzNeuronGroup *findGroup(int id);
  bool addNeuron(int gid, int nid, nzActivator *activator, double bias);
  nzNeuron *findNeuron(int gid, int nid);
  bool connectGroup(int iu, int id);
  bool connectNeuron(int ugid, int unid, int dgid, int dnid, double weight);
  bool setInput(zVec input);
  bool getOutput(zVec output);
  double propagate(zVec input);
  void initGrad();
  bool backpropagate(zVec input, zVec des, double (* lossgrad)(zVec,zVec,int));
  bool trainSDM(double rate);
  void fprint(FILE *fp);

  nzNet *fromZTK(ZTK *ztk);
  nzNet *readZTK(const char filename[]);
  void fprintZTK(FILE *fp);
  bool writeZTK(const char filename[]);
#endif /* __cplusplus */
};

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

/*! \brief add a neuron to a neural network. */
__NEUZ_EXPORT bool nzNetAddNeuron(nzNet *net, int gid, int nid, nzActivator *activator, double bias);

/*! \brief find a neuron in a neural network. */
__NEUZ_EXPORT nzNeuron *nzNetFindNeuron(nzNet *net, int gid, int nid);

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

#define ZTK_TAG_NEUZ_NETWORK "neuz::neuralnetwork"

#define ZTK_KEY_NEUZ_NEURON  "neuron"
#define ZTK_KEY_NEUZ_CONNECT "connect"

/*! \brief read a neural network from a ZTK format processor. */
__NEUZ_EXPORT nzNet *nzNetFromZTK(nzNet *net, ZTK *ztk);

/*! \brief read a neural network from a ZTK format file. */
__NEUZ_EXPORT nzNet *nzNetReadZTK(nzNet *net, const char filename[]);

/*! \brief print out a neural network to a file. */
__NEUZ_EXPORT void nzNetFPrintZTK(FILE *fp, nzNet *net);

/*! \brief write a neural network to a ZTK format file. */
__NEUZ_EXPORT bool nzNetWriteZTK(nzNet *net, const char filename[]);

#ifdef __cplusplus
inline void nzNet::init(){ nzNetInit( this ); }
inline void nzNet::destroy(){ nzNetDestroy( this ); }
inline nzNeuronGroup *nzNet::inputLayer(){ return nzNetInputLayer( this ); }
inline nzNeuronGroup *nzNet::outputLayer(){ return nzNetOutputLayer( this ); }
inline bool nzNet::addGroup(int num){ return nzNetAddGroup( this, num ); }
inline bool nzNet::addGroup(int num, nzActivator *activator){ return nzNetAddGroupSetActivator( this, num, activator ); }
inline nzNeuronGroup *nzNet::findGroup(int id){ return nzNetFindGroup( this, id ); }
inline bool nzNet::addNeuron(int gid, int nid, nzActivator *activator, double bias){ return nzNetAddNeuron( this, gid, nid, activator, bias ); }
inline nzNeuron *nzNet::findNeuron(int gid, int nid){ return nzNetFindNeuron( this, gid, nid ); }
inline bool nzNet::connectGroup(int iu, int id){ return nzNetConnectGroup( this, iu, id ); }
inline bool nzNet::connectNeuron(int ugid, int unid, int dgid, int dnid, double weight){ return nzNetConnect( this, ugid, unid, dgid, dnid, weight ); }
inline bool nzNet::setInput(zVec input){ return nzNetSetInput( this, input ); }
inline bool nzNet::getOutput(zVec output){ return nzNetGetOutput( this, output ); }
inline double nzNet::propagate(zVec input){ return nzNetPropagate( this, input ); }
inline void nzNet::initGrad(){ nzNetInitGrad( this ); }
inline bool nzNet::backpropagate(zVec input, zVec des, double (* lossgrad)(zVec,zVec,int)){ return nzNetBackPropagate( this, input, des, lossgrad ); }
inline bool nzNet::trainSDM(double rate){ return nzNetTrainSDM( this, rate ); }
inline void nzNet::fprint(FILE *fp){ nzNetFPrint( fp, this ); }
inline nzNet *nzNet::fromZTK(ZTK *ztk){ return nzNetFromZTK( this, ztk ); }
inline nzNet *nzNet::readZTK(const char filename[]){ return nzNetReadZTK( this, filename ); }
inline void nzNet::fprintZTK(FILE *fp){ nzNetFPrintZTK( fp, this ); }
inline bool nzNet::writeZTK(const char filename[]){ return nzNetWriteZTK( this, filename ); }
#endif /* __cplusplus */

__END_DECLS

#endif /* __NEUZ_NEURON_H__ */
