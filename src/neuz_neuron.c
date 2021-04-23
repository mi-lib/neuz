/* neuZ - Neural Network Library
 * (C)Copyright, Zhidao, since 2020, all rights are reserved.
 *
 * neuron unit and group.
 */

#include <neuz/neuz_neuron.h>

/* unit neuron class */

/* initialize a new neuron unit. */
nzNeuron *nzNeuronInit(nzNeuron *neuron, int gid, int nid)
{
  zListCellInit( neuron );
  neuron->data.gid = gid;
  neuron->data.nid = nid;
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
bool nzNeuronConnect(nzNeuron *nu, nzNeuron *nd, double weight)
{
  nzAxon *axon;

  if( !( axon = zAlloc( nzAxon, 1 ) ) ){
    ZALLOCERROR();
    return false;
  }
  axon->upstream = nu;
  axon->weight = weight;
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

  neuron->data._p *= neuron->data._v;
  for( ap=neuron->data.axon; ap; ap=ap->next ){
    ((nzNeuron *)ap->upstream)->data._p += neuron->data._p * ap->weight;
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

  fprintf( fp, "<%p> [#%d:%d] (%.10g)\n", neuron, neuron->data.gid, neuron->data.nid, neuron->data.bias );
  for( ap=neuron->data.axon; ap; ap=ap->next )
    fprintf( fp, "  |-(%.10g)<- <%p>\n", ap->weight, ap->upstream );
}

/* neuron group class */

/* initialize a neuron group. */
nzNeuronGroup *nzNeuronGroupInit(nzNeuronGroup *ng, int id)
{
  ng->id = id;
  zListInit( &ng->list );
  return ng;
}

/* add a neuron into a group. */
bool nzNeuronGroupAddOne(nzNeuronGroup *ng)
{
  nzNeuron *neuron;

  if( !( neuron = zAlloc( nzNeuron, 1 ) ) ){
    ZALLOCERROR();
    return false;
  }
  nzNeuronInit( neuron, ng->id, zListSize(&ng->list) );
  zListInsertHead( &ng->list, neuron );
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

  while( !zListIsEmpty( &ng->list ) ){
    zListDeleteHead( &ng->list, &np );
    nzNeuronDestroy( np );
    free( np );
  }
}

/* find a neuron in a neuron group. */
nzNeuron *nzNeuronGroupFindNeuron(nzNeuronGroup *ng, int gid, int nid)
{
  nzNeuron *np;

  zListForEachRew( &ng->list, np )
    if( np->data.gid == gid && np->data.nid == nid ) return np;
  return NULL;
}

/* set activator functions of units in a neuron group. */
void nzNeuronGroupSetActivator(nzNeuronGroup *ng, nzActivator *activator)
{
  nzNeuron *np;

  zListForEach( &ng->list, np )
    np->data.activator = activator;
}

/* set input values to a neuron group. */
bool nzNeuronGroupSetInput(nzNeuronGroup *ng, zVec input)
{
  nzNeuron *np;
  register int i = 0;

  if( zListSize(&ng->list) != zVecSize(input) ){
    ZRUNWARN( NEUZ_WARN_GROUP_MISMATCH_SIZ, zListSize(&ng->list), zVecSize(input) );
    return false;
  }
  zListForEach( &ng->list, np )
    np->data.input = zVecElemNC(input,i++);
  return true;
}

/* get output values from a neuron group. */
bool nzNeuronGroupGetOutput(nzNeuronGroup *ng, zVec output)
{
  nzNeuron *np;
  register int i = 0;

  if( zListSize(&ng->list) != zVecSize(output) ){
    ZRUNWARN( NEUZ_WARN_GROUP_MISMATCH_SIZ, zListSize(&ng->list), zVecSize(output) );
    return false;
  }
  zListForEach( &ng->list, np )
    zVecElemNC(output,i++) = np->data.output;
  return true;
}

/* propagate output values of upstream group to downstream. */
void nzNeuronGroupPropagate(nzNeuronGroup *ng)
{
  nzNeuron *np;

  zListForEach( &ng->list, np )
    nzNeuronPropagate( np );
}

/* initialize gradients of weights and bias of a neuron group. */
static void _nzNeuronGroupInitGrad(nzNeuronGroup *ng)
{
  nzNeuron *np;

  zListForEach( &ng->list, np )
    _nzNeuronInitGrad( np );
}

/* initialize internal parameters of a neuron group for learning. */
static void _nzNeuronGroupInitParam(nzNeuronGroup *ng)
{
  nzNeuron *np;

  zListForEach( &ng->list, np )
    _nzNeuronInitParam( np );
}

/* back-propagate loss and train in a neuron group of a neural network. */
static bool _nzNeuronGroupBackPropagate(nzNeuronGroup *ng)
{
  nzNeuron *np;

  zListForEach( &ng->list, np )
    _nzNeuronBackPropagate( np );
  return true;
}

/* train a neuron group based on the steepest descent method. */
static bool _nzNeuronGroupTrainSDM(nzNeuronGroup *ng, double rate)
{
  nzNeuron *np;
  bool ret = true;

  zListForEach( &ng->list, np )
    if( !_nzNeuronTrainSDM( np, rate ) ) ret = false;
  return ret;
}

/* print a neuron group. */
void nzNeuronGroupFPrint(FILE *fp, nzNeuronGroup *ng)
{
  nzNeuron *np;

  fprintf( fp, "(%d neurons)\n", zListSize(&ng->list) );
  zListForEach( &ng->list, np )
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
  nzNeuronGroupInit( &nc->data, zListSize(net) );
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
nzNeuronGroup *nzNetFindGroup(nzNet *net, int id)
{
  nzNetCell *nc;

  zListForEachRew( net, nc )
    if( nc->data.id == id ) return &nc->data;
  return NULL;
}

/* find a neuron in a neural network. */
nzNeuron *nzNetFindNeuron(nzNet *net, int gid, int nid)
{
  nzNeuronGroup *ng;

  if( !( ng = nzNetFindGroup( net, gid ) ) ) return NULL;
  return nzNeuronGroupFindNeuron( ng, gid, nid );
}

/* add a neuron to a neural network. */
bool nzNetAddNeuron(nzNet *net, int gid, int nid, nzActivator *activator, double bias)
{
  nzNeuronGroup *ng;
  nzNeuron *np;

  while( !( ng = nzNetFindGroup( net, gid ) ) )
    if( !nzNetAddGroup( net, 0 ) ) return false;
  while( !( np = nzNeuronGroupFindNeuron( ng, gid, nid ) ) )
    if( !nzNeuronGroupAddOne( ng ) ) return false;
  np->data.bias = bias;
  np->data.activator = activator;
  return true;
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
  zListForEach( &ngd->list, nd ){
    zListForEach( &ngu->list, nu ){
      if( !nzNeuronConnect( nu, nd, zRandF( -1, 1 ) ) ) return false;
    }
  }
  return true;
}

/* connects two neurons in a neural network. */
bool nzNetConnect(nzNet *net, int ugid, int unid, int dgid, int dnid, double weight)
{
  nzNeuron *nu, *nd;

  if( !( nu = nzNetFindNeuron( net, ugid, unid ) ) ){
    ZRUNERROR( NEUZ_ERR_NEURON_NOT_FOUND, ugid, unid );
    return false;
  }
  if( !( nd = nzNetFindNeuron( net, dgid, dnid ) ) ){
    ZRUNERROR( NEUZ_ERR_NEURON_NOT_FOUND, dgid, dnid );
    return false;
  }
  return nzNeuronConnect( nu, nd, weight );
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

  if( zVecSize(des) != zListSize( &nzNetOutputLayer(net)->list ) ) return false;
  if( !( output = zVecAlloc( zVecSizeNC(des) ) ) ) return false;
  nzNetPropagate( net, input );
  nzNetGetOutput( net, output );
  _nzNetInitParam( net );
  zListForEach( &nzNetOutputLayer(net)->list, np )
    np->data._p = lossgrad( output, des, i++ );
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

/* parse ZTK format */

static void *_nzNetNeuronFromZTK(void *obj, int i, void *arg, ZTK *ztk)
{
  int gid, nid;
  nzActivator *activator;

  gid = ZTKInt(ztk);
  nid = ZTKInt(ztk);
  activator = nzActivatorQuery( ZTKVal(ztk) );
  ZTKValNext( ztk );
  return nzNetAddNeuron( (nzNet*)obj, gid, nid, activator, ZTKDouble(ztk) ) ? obj : NULL;
}

static void *_nzNetConnectFromZTK(void *obj, int i, void *arg, ZTK *ztk)
{
  int ugid, unid, dgid, dnid;

  ugid = ZTKInt(ztk);
  unid = ZTKInt(ztk);
  dgid = ZTKInt(ztk);
  dnid = ZTKInt(ztk);
  return nzNetConnect( (nzNet*)obj, ugid, unid, dgid, dnid, ZTKDouble(ztk) ) ? obj : NULL;
}

static ZTKPrp __ztk_prp_key_neuralnetwork[] = {
  { "neuron", -1, _nzNetNeuronFromZTK, NULL },
  { "connect", -1, _nzNetConnectFromZTK, NULL },
};

/* read a neural network from a ZTK format processor. */
nzNet *nzNetFromZTK(nzNet *net, ZTK *ztk)
{
  nzNetInit( net );
  if( !ZTKKeyRewind( ztk ) ) return NULL;
  return ZTKEvalKey( net, NULL, ztk, __ztk_prp_key_neuralnetwork );
}

static void *_nzNetFromZTK(void *net, int i, void *arg, ZTK *ztk){
  return nzNetFromZTK( net, ztk );
}

static ZTKPrp __ztk_prp_tag_neuralnetwork[] = {
  { "neuralnetwork", -1, _nzNetFromZTK, NULL },
};

/* read a neural network from a ZTK format file. */
nzNet *nzNetReadZTK(nzNet *net, char filename[])
{
  ZTK ztk;

  ZTKInit( &ztk );
  ZTKParse( &ztk, filename );
  net = ZTKEvalTag( net, NULL, &ztk, __ztk_prp_tag_neuralnetwork );
  ZTKDestroy( &ztk );
  return net;
}

/* print out a neural network to a file. */
void nzNetFPrintZTK(FILE *fp, nzNet *net)
{
  nzNetCell *nc;
  nzNeuron *np;
  nzAxon *ap;

  if( !net ) return;
  fprintf( fp, "[%s]\n", NZ_NET_TAG );
  zListForEach( net, nc ){
    zListForEach( &nc->data.list, np ){
      fprintf( fp, "neuron: %d %d %s %.10g\n", np->data.gid, np->data.nid, np->data.activator ? np->data.activator->typestr : "nil", np->data.bias );
      for( ap=np->data.axon; ap; ap=ap->next ){
        fprintf( fp, "connect: %d %d %d %d %.10g\n",
          ((nzNeuron*)ap->upstream)->data.gid, ((nzNeuron*)ap->upstream)->data.nid,
          np->data.gid, np->data.nid,
          ap->weight );
      }
    }
  }
  fprintf( fp, "\n" );
}

/* write a neural network to a ZTK format file. */
bool nzNetWriteZTK(nzNet *net, char filename[])
{
  FILE *fp;

  if( !( fp = zOpenZTKFile( filename, "w" ) ) ){
    ZOPENERROR( filename );
    return false;
  }
  nzNetFPrintZTK( fp, net );
  fclose( fp );
  return true;
}
