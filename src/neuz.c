/* neuZ - Neural Network Library
 * (C)Copyright, Zhidao, since 2020, all rights are reserved.
 */

#include <neuz/neuz.h>

#define NEUZ_ERR_GROUP_NOT_FOUND "neuron group %d not found"
#define NEUZ_WARN_GROUP_MISMATCH_SIZ "size mismatch between a neuron group (%d) and a vector (%d)"
#define NEUZ_WARN_NET_TOOFEWLAYER "cannot apply backpropagation to a two-or-less-layered network."

/* activator functions */

/* step function */
nzActivator nz_activator_step = {
  nzActivatorStep,
  nzActivatorStepDif
};

double nzActivatorStep(double val){ return val >= 0 ? 1 : 0; }
double nzActivatorStepDif(double val){ return 0; /* case val=0 is ignored. */ }

/* sigmoid function */
nzActivator nz_activator_sigmoid = {
  nzActivatorSigmoid,
  nzActivatorSigmoidDif
};

double nzActivatorSigmoid(double val){ return zSigmoid( 4*val ); }
double nzActivatorSigmoidDif(double val){
  double u;
  u = exp( -4*val );
  return 4 * u / zSqr( 1 + u );
}

/* rectified linear unit function */
nzActivator nz_activator_relu = {
  nzActivatorReLU,
  nzActivatorReLUDif
};

double nzActivatorReLU(double val){ return zMax( val, 0 ); }
double nzActivatorReLUDif(double val){ return val >= 0 ? 1 : 0; /* case val=0 is ignored. */ }

/* unit neuron class */

/* initialize a new neuron unit. */
nzNeuron *nzNeuronInit(nzNeuron *neuron)
{
  zListCellInit( neuron );
  neuron->data.input = 0;
  neuron->data.output = 0;
  neuron->data.bias = zRandF( -1, 1 );
  neuron->data._db = 0;
  neuron->data._p = 0;
  neuron->data._v = 0;
  neuron->data.activator = &nz_activator_sigmoid;
  neuron->data.axon = NULL;
  return neuron;
}

/* destroy a neuron unit. */
void nzNeuronDestroy(nzNeuron *neuron)
{
  nzAxon *ap;

  while( ( ap = neuron->data.axon ) ){
    neuron->data.axon = ap->next;
    free( ap );
  }
}

/* connect two neuron units. */
bool nzNeuronConnect(nzNeuron *nu, nzNeuron *nd)
{
  nzAxon *axon;

  if( !( axon = zAlloc( nzAxon, 1 ) ) ){
    ZALLOCERROR();
    return false;
  }
  axon->upstream = nu;
  axon->weight = zRandF( -1, 1 );
  axon->_dw = 0;
  axon->next = nd->data.axon;
  nd->data.axon = axon;
  return true;
}

/* propagate output values of upstream units to downstream. */
double nzNeuronPropagate(nzNeuron *neuron)
{
  nzAxon *ap;

  if( !neuron->data.activator ){ /* input layer */
    return neuron->data.output = neuron->data.input;
  }
  neuron->data.input = neuron->data.bias;
  for( ap=neuron->data.axon; ap; ap=ap->next ){
    neuron->data.input += ap->weight * ((nzNeuron*)ap->upstream)->data.output;
  }
  return neuron->data.output = neuron->data.activator->f( neuron->data.input );
}

/* initialize gradients of weights and bias of a neuron. */
static void _nzNeuronInitGrad(nzNeuron *neuron)
{
  nzAxon *ap;

  for( ap=neuron->data.axon; ap; ap=ap->next ) ap->_dw = 0;
  neuron->data._db = 0;
}

/* initialize internal parameters for learning. */
static void _nzNeuronInitParam(nzNeuron *neuron)
{
  neuron->data._p = 0;
  neuron->data._v = neuron->data.activator ? neuron->data.activator->df( neuron->data.input ) : 0;
}

/* back-propagate loss and train a neuron unit of a neural network. */
static bool _nzNeuronBackPropagate(nzNeuron *neuron)
{
  nzAxon *ap;

  for( ap=neuron->data.axon; ap; ap=ap->next ){
    ((nzNeuron *)ap->upstream)->data._p += neuron->data._p * ap->weight * ((nzNeuron *)ap->upstream)->data._v;
    ap->_dw += neuron->data._p * ((nzNeuron *)ap->upstream)->data.output;
  }
  neuron->data._db += neuron->data._p;
  return true;
}

/* train a neuron based on the steepest descent method. */
static bool _nzNeuronTrainSDM(nzNeuron *neuron, double rate)
{
  nzAxon *ap;

  for( ap=neuron->data.axon; ap; ap=ap->next )
    ap->weight -= rate * ap->_dw;
  neuron->data.bias -= rate * neuron->data._db;
  return true;
}

/* print a neuron unit. */
void nzNeuronFPrint(FILE *fp, nzNeuron *neuron)
{
  nzAxon *ap;

  fprintf( fp, "<%p> (%.10g)\n", neuron, neuron->data.bias );
  for( ap=neuron->data.axon; ap; ap=ap->next )
    fprintf( fp, "  |-(%.10g)<- <%p>\n", ap->weight, ap->upstream );
}

/* neuron group class */

/* add a neuron into a group. */
bool nzNeuronGroupAddOne(nzNeuronGroup *ng)
{
  nzNeuron *neuron;

  if( !( neuron = zAlloc( nzNeuron, 1 ) ) ){
    ZALLOCERROR();
    return false;
  }
  nzNeuronInit( neuron );
  zListInsertHead( ng, neuron );
  return true;
}

/* add multiple neurons into a group. */
bool nzNeuronGroupAdd(nzNeuronGroup *ng, int num)
{
  while( --num >= 0 )
    if( !nzNeuronGroupAddOne( ng ) ) return false;
  return true;
}

/* destroy a neuron group. */
void nzNeuronGroupDestroy(nzNeuronGroup *ng)
{
  nzNeuron *np;

  while( zListIsEmpty( ng ) ){
    zListDeleteHead( ng, &np );
    nzNeuronDestroy( np );
    free( np );
  }
}

/* set activator functions of units in a neuron group. */
void nzNeuronGroupSetActivator(nzNeuronGroup *ng, nzActivator *activator)
{
  nzNeuron *np;

  zListForEach( ng, np )
    np->data.activator = activator;
}

/* set input values to a neuron group. */
bool nzNeuronGroupSetInput(nzNeuronGroup *ng, zVec input)
{
  nzNeuron *np;
  register int i = 0;

  zListForEach( ng, np ){
    if( i >= zVecSize(input) ){
      ZRUNWARN( NEUZ_WARN_GROUP_MISMATCH_SIZ, zListSize(ng), zVecSize(input) );
      break;
    }
    np->data.input = zVecElemNC(input,i++);
  }
  return true;
}

/* get output values from a neuron group. */
bool nzNeuronGroupGetOutput(nzNeuronGroup *ng, zVec output)
{
  nzNeuron *np;
  register int i = 0;

  zListForEach( ng, np ){
    if( i >= zVecSize(output) ){
      ZRUNWARN( NEUZ_WARN_GROUP_MISMATCH_SIZ, zListSize(ng), zVecSize(output) );
      break;
    }
    zVecElemNC(output,i++) = np->data.output;
  }
  return true;
}

/* propagate output values of upstream group to downstream. */
void nzNeuronGroupPropagate(nzNeuronGroup *ng)
{
  nzNeuron *np;

  zListForEach( ng, np )
    nzNeuronPropagate( np );
}

/* initialize gradients of weights and bias of a neuron group. */
static void _nzNeuronGroupInitGrad(nzNeuronGroup *ng)
{
  nzNeuron *np;

  zListForEach( ng, np )
    _nzNeuronInitGrad( np );
}

/* initialize internal parameters of a neuron group for learning. */
static void _nzNeuronGroupInitParam(nzNeuronGroup *ng)
{
  nzNeuron *np;

  zListForEach( ng, np )
    _nzNeuronInitParam( np );
}

/* back-propagate loss and train in a neuron group of a neural network. */
static bool _nzNeuronGroupBackPropagate(nzNeuronGroup *ng)
{
  nzNeuron *np;

  zListForEach( ng, np )
    _nzNeuronBackPropagate( np );
  return true;
}

/* train a neuron group based on the steepest descent method. */
static bool _nzNeuronGroupTrainSDM(nzNeuronGroup *ng, double rate)
{
  nzNeuron *np;
  bool ret = true;

  zListForEach( ng, np )
    if( !_nzNeuronTrainSDM( np, rate ) ) ret = false;
  return ret;
}

/* print a neuron group. */
void nzNeuronGroupFPrint(FILE *fp, nzNeuronGroup *ng)
{
  nzNeuron *np;

  fprintf( fp, "(%d neurons)\n", zListSize(ng) );
  zListForEach( ng, np )
    nzNeuronFPrint( fp, np );
}

/* neural network class */

/* add a neuron group to a neural network. */
bool nzNetAddGroup(nzNet *net, int num)
{
  nzNetCell *nc;

  if( !( nc = zAlloc( nzNetCell, 1 ) ) ){
    ZALLOCERROR();
    return false;
  }
  zListInit( &nc->data );
  if( !nzNeuronGroupAdd( &nc->data, num ) ){
    nzNeuronGroupDestroy( &nc->data );
    free( nc );
    return false;
  }
  zListInsertHead( net, nc );
  return true;
}

/* add a neuron group with a specified activator to a neural network. */
bool nzNetAddGroupSetActivator(nzNet *net, int num, nzActivator *activator)
{
  if( !nzNetAddGroup( net, num ) ) return false;
  nzNeuronGroupSetActivator( &zListHead(net)->data, activator );
  return true;
}

/* destroy a neural network. */
void nzNetDestroy(nzNet *net)
{
  nzNetCell *nc;

  while( !zListIsEmpty( net ) ){
    zListDeleteHead( net, &nc );
    nzNeuronGroupDestroy( &nc->data );
    free( nc );
  }
}

/* find a neuron group in a neural network. */
nzNeuronGroup *nzNetFindGroup(nzNet *net, int i)
{
  nzNetCell *nc;

  zListItem( net, i, &nc );
  return nc ? &nc->data : NULL;
}

/* connect two neuron groups in a neural network. */
bool nzNetConnectGroup(nzNet *net, int iu, int id)
{
  nzNeuronGroup *ngu, *ngd;
  nzNeuron *nu, *nd;

  if( !( ngu = nzNetFindGroup( net, iu ) ) ){
    ZRUNERROR( NEUZ_ERR_GROUP_NOT_FOUND, iu );
    return false;
  }
  if( !( ngd = nzNetFindGroup( net, id ) ) ){
    ZRUNERROR( NEUZ_ERR_GROUP_NOT_FOUND, id );
    return false;
  }
  zListForEach( ngd, nd ){
    zListForEach( ngu, nu ){
      if( !nzNeuronConnect( nu, nd ) ) return false;
    }
  }
  return true;
}

/* set input values to the input layer of a neural network. */
bool nzNetSetInput(nzNet *net, zVec input)
{
  return nzNeuronGroupSetInput( nzNetInputLayer(net), input );
}

/* get output values from a neural network. */
bool nzNetGetOutput(nzNet *net, zVec output)
{
  return nzNeuronGroupGetOutput( nzNetOutputLayer(net), output );
}

/* propagate input values to a neural network to the output. */
double nzNetPropagate(nzNet *net, zVec input)
{
  nzNetCell *nc;

  if( input )
    if( !nzNetSetInput( net, input ) ) return false;
  zListForEach( net, nc )
    nzNeuronGroupPropagate( &nc->data );
  return true;
}

/* initialize gradients of weights and bias of a neural network. */
void nzNetInitGrad(nzNet *net)
{
  nzNetCell *nc;

  zListForEach( net, nc )
    _nzNeuronGroupInitGrad( &nc->data );
}

/* initialize internal parameters of a neural network for learning. */
static void _nzNetInitParam(nzNet *net)
{
  nzNetCell *nc;

  zListForEach( net, nc )
    _nzNeuronGroupInitParam( &nc->data );
}

/* initialize parameters of each unit of a neural network for back-propagation. */
static bool _nzNetInitP(nzNet *net, zVec input, zVec des, double (* lossgrad)(zVec,zVec,int))
{
  zVec output;
  nzNeuron *np;
  register int i = 0;

  if( zVecSize(des) != zListSize( nzNetOutputLayer(net) ) ) return false;
  if( !( output = zVecAlloc( zVecSizeNC(des) ) ) ) return false;
  nzNetPropagate( net, input );
  nzNetGetOutput( net, output );
  _nzNetInitParam( net );
  zListForEach( nzNetOutputLayer(net), np )
    np->data._p = lossgrad( output, des, i++ ) * np->data._v;
  zVecFree( output );
  return true;
}

/* back-propagate loss and train a neural network. */
bool nzNetBackPropagate(nzNet *net, zVec input, zVec des, double (* lossgrad)(zVec,zVec,int))
{
  nzNetCell *nc;

  if( zListSize(net) < 3 ){
    ZRUNWARN( NEUZ_WARN_NET_TOOFEWLAYER );
    return false;
  }
  if( !_nzNetInitP( net, input, des, lossgrad ) ) return false;
  for( nc=zListHead(net); nc!=zListTail(net); nc=zListCellPrev(nc) )
    _nzNeuronGroupBackPropagate( &nc->data );
  return true;
}

/* train a neural network based on the steepest descent method. */
bool nzNetTrainSDM(nzNet *net, double rate)
{
  nzNetCell *nc;
  bool ret = true;

  for( nc=zListHead(net); nc!=zListTail(net); nc=zListCellPrev(nc) )
    if( !_nzNeuronGroupTrainSDM( &nc->data, rate ) ) ret = false;
  return ret;
}

/* print a neural network. */
void nzNetFPrint(FILE *fp, nzNet *net)
{
  nzNetCell *nc;
  int i = 0;

  fprintf( fp, "%d groups\n", zListSize(net) );
  zListForEach( net, nc ){
    fprintf( fp, "group #%d : ", i++ );
    nzNeuronGroupFPrint( fp, &nc->data );
  }
}
