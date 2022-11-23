# Abstract implementation of RealNVP Flow

__all__ = ['RealNVPNetwork',
           'RealNVPBijector',
           'RealNVPFlow']

import numpy as np
import tensorflow as tf # type: ignore
import tensorflow.compat.v1 as tf1 # type: ignore
from tensorflow.keras import Input # type: ignore
from tensorflow.keras import layers, initializers, regularizers, constraints, callbacks, optimizers, metrics, losses # type: ignore
from tensorflow.keras.models import Model # type: ignore
from tensorflow.keras.layers import Layer #type: ignore
import tensorflow_probability as tfp # type: ignore
tfd = tfp.distributions
tfb = tfp.bijectors

from typing import Union, List, Dict, Callable, Tuple, Optional, NewType, Type, Generic, Any, TypeVar, TYPE_CHECKING
from typing_extensions import TypeAlias
from NF4HEP.utils.custom_types import Array, ArrayInt, ArrayStr, DataType, StrPath, IntBool, StrBool, StrList, FigDict, LogPredDict, Number, DTypeStr, DTypeStrList, DictStr
from NF4HEP.utils.verbosity import print, Verbosity
from NF4HEP.bijectors.bijectors_base import AbstractNeuralNetwork
from NF4HEP.bijectors.bijectors_base import AbstractBijector
from NF4HEP.bijectors.bijectors_base import Flow_base
from NF4HEP.utils import utils

header_string_1 = "=============================="
header_string_2 = "------------------------------"

# Singleton object representing "no value", in cases where "None" is meaningful.
UNSPECIFIED = object()

class RealNVPNetwork(Layer, Verbosity):
    # """
    # """
    # def define_network(self, verbose = None):
    #    verbose, verbose_sub = self.get_verbosity(verbose)
    #    self.NN = tfb.AutoregressiveNetwork(**self.model_define_inputs
    """
    Neural Network Architecture for calcualting s and t for Real-NVP
    model_define_inputs can be of the following form:
    .. code-block:: python

        model_define_inputs = {"ndims": 10, 
                               "hidden_layers": ["Dense(1000,activation='relu',kernel_initializer='glorot_uniform',kernel_regularizer=regularizers.L1L2(l1=0.001, l2=0.001))",
                                                 {"name": "Dense",
                                                  "args": [1000],#{"units": 1000},
                                                  "kwargs": {"activation": "relu", 
                                                             "use_bias": True,
                                                             "kernel_initializer": "GlorotUniform(seed=None)",
                                                             "bias_initializer": "zeros", 
                                                             "kernel_regularizer": "L1L2(l1=0.001, l2=0.001)",
                                                             "bias_regularizer": "L1(l1=0.0001)", 
                                                             "activity_regularizer": "L1(l1=0.0001)", 
                                                             "kernel_constraint": "MaxNorm(max_value=2, axis=0)",
                                                             "bias_constraint": None}},
                                                 "Dense(500)",
                                                 {"name": "Activation",
                                                  "args": ["relu"]},
                                                 [100,"selu"],
                                                 [100,"relu","glorot_uniform"],
                                                 {"name": "Dense",
                                                  "args": [1000],#{"units": 1000},
                                                  "kwargs": {"use_bias": True,
                                                             "kernel_initializer": {"name": "GlorotUniform",
                                                                                    "args": [],
                                                                                    "kwargs": {"seed": None}},
                                                             "bias_initializer": "zeros", 
                                                             "kernel_regularizer": {"name": "L1L2",
                                                                                    "args": [],
                                                                                    "kwargs": {"l1": 0.001,
                                                                                               "l2": 0.001}},
                                                             "bias_regularizer": "L1(l1=0.0001)", 
                                                             "activity_regularizer": "L1(l1=0.0001)", 
                                                             "kernel_constraint": {"name": "MaxNorm",
                                                                                    "args": [],
                                                                                    "kwargs": {"max_value": 2,
                                                                                               "axis": 0}},
                                                             "bias_constraint": "MaxNorm(max_value=2, axis=0)"}},
                                                 "Activation('selu')",
                                                 {"name": "BatchNormalization",
                                                  "args": [],
                                                  "kwargs": {"axis": -1, 
                                                             "momentum": 0.99, 
                                                             "epsilon": 0.001, 
                                                             "center": True, 
                                                             "scale": True,
                                                             "beta_initializer": "zeros", 
                                                             "gamma_initializer": "ones",
                                                             "moving_mean_initializer": "zeros",
                                                             "moving_variance_initializer": "ones", 
                                                             "beta_regularizer": None,
                                                             "gamma_regularizer": None, 
                                                             "beta_constraint": None, 
                                                             "gamma_constraint": None}},
                                                 {"name": "Dropout",
                                                  "args": [0.],
                                                  "kwargs": {"noise_shape": None, 
                                                             "seed": None}},
                                                 "BatchNormalization",
                                                 "AlphaDropout(0.1)",
                                                 {"name": "AlphaDropout",
                                                  "args": [0.], 
                                                  "kwargs": {"noise_shape": None, 
                                                              "seed": None}},
                                                 {"name": "Dense",
                                                  "args": [1000],
                                                  "kwargs": {"activation": "relu", 
                                                              "use_bias": True,
                                                              "kernel_initializer": "glorot_uniform",
                                                              "bias_initializer": "zeros", 
                                                              "kernel_regularizer": None,
                                                              "bias_regularizer": None, 
                                                              "activity_regularizer": None, 
                                                              "kernel_constraint": None,
                                                              "bias_constraint": None}},
                                                 "Dense(1,activation='linear',kernel_regularizer=regularizers.L1L2(l1=0.001, l2=0.001))"], 
                               "dropout_rate": 0,
                               "batch_norm": False}
    """
    def __init__(self,
                 model_define_inputs: Dict[str, Any],
                 verbose: Optional[IntBool] = None
                ) -> None:
        # Attributes type declarations
        self._batch_norm: StrBool
        self._dropout_rate: Union[np.float_,str]
        self._hidden_layers: List[Any]
        self._layers: List[Layer]
        self._layers_string: List[str]
        self._model_define_inputs: Dict[str, Any]
        self._ndims: int
        # Initialise parent Layer and Verbosity classes
        Layer.__init__(self)
        Verbosity.__init__(self, verbose)
        # Set verbosity
        verbose, _ = self.get_verbosity(verbose)
        # Initialize object
        print(header_string_1, "\nInitializing RealNVPNetwork object.\n", show = verbose)
        self.__set_model_define_inputs(model_define_inputs = model_define_inputs, verbose = verbose)
        self.__set_layers()
        super().__init__()

    @property
    def batch_norm(self) -> StrBool:
        return self._batch_norm

    @property
    def dropout_rate(self) -> Union[np.float_,str]:
        return self._dropout_rate

    @property
    def hidden_layers(self) -> List[Any]:
        return self._hidden_layers

    @property
    def layers(self) -> List[Any]:
        return self._layers

    @property
    def layers_string(self) -> List[str]:
        return self._layers_string

    @property
    def model_define_inputs(self) -> Dict[str, Any]:
        return self._model_define_inputs

    @property
    def ndims(self) -> int:
        return self._ndims

    def __set_model_define_inputs(self,
                                  model_define_inputs: Dict[str, Any],
                                  verbose: Optional[IntBool] = None
                                 ) -> None:
        verbose, verbose_sub = self.get_verbosity(verbose)
        self._model_define_inputs = model_define_inputs
        try:
            self._ndims = model_define_inputs["ndims"]
        except:
            raise KeyError("The 'model_define_inputs' argument misses the mandatory 'ndims' item.")
        try:
            self._hidden_layers = model_define_inputs["hidden_layers"]
        except:
            raise KeyError("The 'model_define_inputs' argument misses the mandatory 'hidden_layers' item.")
        utils.check_set_dict_keys(dic = model_define_inputs, 
                                  keys = ["ndims","hidden_layers","dropout_rate","batch_norm"],
                                  vals = [self._ndims, self._hidden_layers, 0, False],
                                  verbose = verbose_sub)
        self._dropout_rate = model_define_inputs["dropout_rate"]
        self._batch_norm = model_define_inputs["batch_norm"]
        self._model_define_inputs = model_define_inputs
                        
    def __set_layers(self,
                     verbose: Optional[IntBool] = None
                    ) -> None:
        """
        Method that defines strings representing the |tf_keras_layers_link| that are stored in the 
        :attr:`NF.layers_string <NF4HEP.NF.layers_string>` attribute.
        These are defined from the attributes

            - :attr:`NF.hidden_layers <NF4HEP.NF.hidden_layers>`
            - :attr:`NF.batch_norm <NF4HEP.NF.batch_norm>`
            - :attr:`NF.dropout_rate <NF4HEP.NF.dropout_rate>`

        If |tf_keras_batch_normalization_link| layers are specified in the 
        :attr:`NF.hidden_layers <NF4HEP.NF.hidden_layers>` attribute, then the 
        :attr:`NF.batch_norm <NF4HEP.NF.batch_norm>` attribute is ignored. Otherwise,
        if :attr:`NF.batch_norm <NF4HEP.NF.batch_norm>` is ``True``, then a 
        |tf_keras_batch_normalization_link| layer is added after the input layer and before
        each |tf_keras_dense_link| layer.

        If |tf_keras_dropout_link| layers are specified in the 
        :attr:`NF.hidden_layers <NF4HEP.NF.hidden_layers>` attribute, then the 
        :attr:`NF.dropout_rate <NF4HEP.NF.dropout_rate>` attribute is ignored. Otherwise,
        if :attr:`NF.dropout_rate <NF4HEP.NF.dropout_rate>` is larger than ``0``, then
        a |tf_keras_dropout_link| layer is added after each |tf_keras_dense_link| layer 
        (but the output layer).

        The method also sets the three attributes:

            - :attr:`NF.layers <NF4HEP.NF.layers>` (set to an empty list ``[]``, filled by the 
                :meth:`NF.model_define <NF4HEP.NF.model_define>` method)
            - :attr:`NF.model_params <NF4HEP.NF.model_params>`
            - :attr:`NF.model_trainable_params <NF4HEP.NF.model_trainable_params>`
            - :attr:`NF.model_non_trainable_params <NF4HEP.NF.model_non_trainable_params>`

        - **Arguments**

            - **verbose**

                See :argument:`verbose <common_methods_arguments.verbose>`.

        - **Produces file**

            - :attr:`NF.output_log_file <NF4HEP.NF.output_log_file>`
        """
        layer_string: str
        layer: Layer
        verbose, verbose_sub = self.get_verbosity(verbose)
        print(header_string_2,"\nSetting hidden layers\n", show = verbose)
        self._layers_string = []
        self._layers = []
        i = 0
        if "dropout" in str(self._hidden_layers).lower():
            insert_dropout = False
            self._dropout_rate = "custom"
        elif "dropout" not in str(self._hidden_layers).lower() and self._dropout_rate != 0:
            insert_dropout = True
        else:
            insert_dropout = False
        if "batchnormalization" in str(self._hidden_layers).lower():
            self._batch_norm = "custom"
        layer_string = ""
        for layer in self._hidden_layers:
            if type(layer) == str:
                if "(" in layer:
                    layer_string = "layers."+layer
                else:
                    layer_string = "layers."+layer+"()"
            elif type(layer) == dict:
                try:
                    name = layer["name"]
                except:
                    raise Exception("The layer ", str(layer), " has unspecified name.")
                try:
                    args = layer["args"]
                except:
                    args = []
                try:
                    kwargs = layer["kwargs"]
                except:
                    kwargs = {}
                layer_string = utils.build_method_string_from_dict("layers", name, args, kwargs)
            elif type(layer) == list:
                units = layer[0]
                activation = layer[1]
                try:
                    initializer = layer[2]
                except:
                    initializer = None
                if activation == "selu":
                    layer_string = "layers.Dense(" + str(units) + ", activation='" + activation + "', kernel_initializer='lecun_normal')"
                elif activation != "selu" and initializer != None:
                    layer_string = "layers.Dense(" + str(units) + ", activation='" + activation + "')"
                else:
                    layer_string = "layers.Dense(" + str(units)+", activation='" + activation + "', kernel_initializer='" + initializer + "')"
            else:
                layer_string = ""
                print("Invalid input for layer: ", layer, ". The layer will not be added to the model.", show = verbose)
            if layer_string != "":
                if self._batch_norm == True and "dense" in layer_string.lower():
                    self._layers_string.append("layers.BatchNormalization()")
                    print("Added hidden layer: layers.BatchNormalization()", show = verbose)
                    i = i + 1
                try:
                    eval(layer_string)
                    self._layers_string.append(layer_string)
                    print("Added hidden layer: ", layer_string, show = verbose)
                    i = i + 1
                except Exception as e:
                    print(e)
                    print("Could not add layer", layer_string, "\n", show = verbose)
                if insert_dropout:
                    try:
                        act = eval(layer_string+".activation")
                        if "selu" in str(act).lower():
                            layer_string = "layers.AlphaDropout(" + str(self._dropout_rate)+")"
                            self._layers_string.append(layer_string)
                            print("Added hidden layer: ",layer_string, show = verbose)
                            i = i + 1
                        elif "linear" not in str(act):
                            layer_string = "layers.Dropout(" + str(self._dropout_rate)+")"
                            self._layers_string.append(layer_string)
                            print("Added hidden layer: ", layer_string, show = verbose)
                            i = i + 1
                        else:
                            layer_string = ""
                    except:
                        layer_string = "layers.AlphaDropout(" + str(self._dropout_rate)+")"
                        self._layers_string.append(layer_string)
                        print("Added hidden layer: ", layer_string, show = verbose)
                        i = i + 1
        if layer_string != "":
            if self._batch_norm == True and "dense" in layer_string.lower():
                self._layers_string.append("layers.BatchNormalization()")
                print("Added hidden layer: layers.BatchNormalization()", show = verbose)
        t_layer_string = "layers.Dense(self._ndims, name='t')"
        log_s_layer_string = "layers.Dense(self._ndims, activation='tanh', name='log_s')"
        self._layers_string.append(t_layer_string)
        self._layers_string.append(log_s_layer_string)
        for layer_string in self._layers_string:
            try:
                print("Building layer:", layer_string)
                self._layers.append(eval(layer_string))
            except:
                print("Failed to evaluate:", layer_string)

    def call(self, x):
        """
        """
        # Define and return Model
        y = x
        for layer in self._layers[:-2]:
            y = layer(y)
        t = self._layers[-2](y)
        log_s = self._layers[-1](y)
        return t, log_s

class RealNVPBijector(tfb.Bijector, Verbosity): # type: ignore
    """
    Implementation of a Real-NVP for Denisty Estimation. L. Dinh “Density estimation using Real NVP,” 2016.
    """
    def __init__(self,
                 model_define_inputs: Dict[str, Any],
                 model_bijector_inputs: Dict[str, Any],
                 verbose: Optional[IntBool] = None
                ) -> None:
        # Attributes type declarations
        self._Model: Model
        self._model_bijector_inputs: Dict[str, Any]
        self._ndims: int
        self._NN: RealNVPNetwork
        self._rem_dims: int
        self._tran_ndims: int
        # Initialise parent Verbosity classe
        Verbosity.__init__(self, verbose)
        # Set verbosity
        verbose, _ = self.get_verbosity(verbose)
        # Initialize object
        print(header_string_1, "\nInitializing RealNVPBijector object.\n", show = verbose)
        self.__set_model_bijector_inputs(model_bijector_inputs = model_bijector_inputs, verbose = verbose)
        #self._ndims = self._NN._ndims
        # Initialize parent Bijector class
        bij_kargs = utils.dic_minus_keys(self.model_bijector_inputs,["rem_dims"])
        tfb.Bijector.__init__(self, **bij_kargs)
        self.NN = RealNVPNetwork(model_define_inputs)

    @property
    def NN(self) -> RealNVPNetwork:
        return self._NN
    
    @NN.setter
    def NN(self,
           nn: RealNVPNetwork
          ) -> None:
        self._NN = nn
        self._ndims = self._NN._ndims
        x = Input((self._rem_dims,))
        t, log_s = self._NN(x)
        self._Model = Model(x, [t, log_s])

    @property
    def Model(self) -> Model:
        return self._Model

    @property
    def model_bijector_inputs(self) -> Dict[str, Any]:
        return self._model_bijector_inputs

    @property
    def ndims(self) -> int:
        return self._ndims

    @property
    def rem_dims(self) -> int:
        return self._rem_dims

    @property
    def tran_ndims(self) -> int:
        return self._tran_ndims

    def __set_model_bijector_inputs(self,
                                    model_bijector_inputs: Dict[str, Any],
                                    verbose: Optional[IntBool] = None
                                   ) -> None:
        verbose, verbose_sub = self.get_verbosity(verbose)
        try:
            self._rem_dims = model_bijector_inputs["rem_dims"]
        except:
            raise KeyError("The 'model_bijector_inputs' argument misses the mandatory 'rem_dims' item.")
        utils.check_set_dict_keys(dic = model_bijector_inputs, 
                                  keys = ["graph_parents",
                                          "is_constant_jacobian",
                                          "validate_args",
                                          "dtype",
                                          "forward_min_event_ndims",
                                          #"inverse_min_event_ndims",
                                          "experimental_use_kahan_sum",
                                          "parameters",
                                          "name"],
                                  vals = [None,
                                          False,
                                          False,
                                          None,
                                          1,
                                          #UNSPECIFIED,
                                          False,
                                          None,
                                          None],
                                  verbose = verbose_sub)
        self._model_bijector_inputs = model_bijector_inputs
        self._tran_ndims = self._ndims-self._rem_dims
        if self._rem_dims < 1 or self._rem_dims > self._ndims - 1:
            raise Exception('ERROR: rem_dims must be 1<rem_dims<ndims-1')

    def _bijector_fn(self, x):
        t, log_s = self._Model(x)
        #print('this is t')
        # print(t)
        return tfb.Shift(shift=t)(tfb.Scale(log_scale=log_s))
        # return tfb.affine_scalar.AffineScalar(shift=t, log_scale=log_s)

    def _forward(self, x):
        #x_a, x_b = tf.split(x, 2, axis=-1)
        x_a = x[:, :self._tran_ndims]
        x_b = x[:, self._tran_ndims:]
        # print('x_a')
        # print(x_a)
        y_b = x_b
        y_a = self._bijector_fn(x_b).forward(x_a)
        y = tf.concat([y_a, y_b], axis=-1)
        return y

    def _inverse(self, y):
        #y_a, y_b = tf.split(y, 2, axis=-1)
        y_a = y[:, :self._tran_ndims]
        y_b = y[:, self._tran_ndims:]
        x_b = y_b
        x_a = self._bijector_fn(y_b).inverse(y_a)
        x = tf.concat([x_a, x_b], axis=-1)
        return x

    def _forward_log_det_jacobian(self, x):
        #x_a, x_b = tf.split(x, 2, axis=-1)
        x_a = x[:, :self._tran_ndims]
        x_b = x[:, self._tran_ndims:]
        return self._bijector_fn(x_b).forward_log_det_jacobian(x_a, event_ndims=1)

    def _inverse_log_det_jacobian(self, y):
        #y_a, y_b = tf.split(y, 2, axis=-1)
        y_a = y[:, :self._tran_ndims]
        y_b = y[:, self._tran_ndims:]
        return self._bijector_fn(y_b).inverse_log_det_jacobian(y_a, event_ndims=1)

class RealNVPChain(tfb.Chain, Verbosity): # type: ignore
    name = "RealNVP"
    """
    """
    def __init__(self,
                 model_define_inputs: Dict[str, Any],
                 model_bijector_inputs: Dict[str, Any],
                 model_flow_inputs: Dict[str, Any],
                 verbose: Optional[IntBool] = None
                ) -> None:
        # Attributes type declarations
        self._batch_normalization: bool
        self._Bijector: RealNVPBijector
        self._model_flow_inputs: Dict[str, Any]
        self._nbijectors: int
        # Initialise parent Verbosity classe
        Verbosity.__init__(self, verbose)
        # Set verbosity
        verbose, _ = self.get_verbosity(verbose)
        # Initialize object
        print(header_string_1, "\nInitializing RealNVPFlow object.\n", show = verbose)
        self.__set_model_flow_inputs(model_flow_inputs = model_flow_inputs, verbose = verbose)
        self._bijectors = []
        self._Bijector = RealNVPBijector(model_define_inputs, model_bijector_inputs)
        #self._NN = self._bijector._NN
        ndims = self._bijector._ndims
        permutation = tf.cast(np.concatenate((np.arange(int(ndims/2), ndims), np.arange(0, int(ndims/2)))), tf.int32)
        Permute = tfb.Permute(permutation=permutation)
        print("\n",tfb.BatchNormalization())
        print("\n",self._bijector)
        print("\n",Permute)
        for _ in range(self._num_bijectors):
            if self._batch_normalization:
                self._bijectors.append(tfb.BatchNormalization())
            self._bijectors.append(self._Bijector)
            self._bijectors.append(Permute)
        #self._chain = tfb.Chain(bijectors=list(reversed(self._bijectors[:-1])), name='realnvp_chain')
        tfb.Chain.__init__(self, bijectors=list(reversed(self._bijectors[:-1])), name='realnvp_chain')

    @property
    def batch_normalization(self) -> bool:
        return self._batch_normalization

    @property
    def Bijector(self) -> RealNVPBijector:
        return self._Bijector

    @property
    def NN(self) -> RealNVPNetwork:
        return self._Bijector._NN

    @property
    def model_flow_inputs(self) -> Dict[str, Any]:
        return self._model_flow_inputs

    @property
    def nbijectors(self) -> int:
        return self._nbijectors

    def __set_model_flow_inputs(self,
                                model_flow_inputs: Dict[str, Any],
                                verbose: Optional[IntBool] = None
                               ) -> None:
        verbose, verbose_sub = self.get_verbosity(verbose)
        try:
            self._num_bijectors = model_flow_inputs["nbijectors"]
        except:
            print("The 'model_flow_inputs' argument misses the mandatory 'nbijectors' item. The corresponding attribute will be set to a default of 2.")
        utils.check_set_dict_keys(dic = model_flow_inputs, 
                                  keys = ["nbijectors","batch_normalization"],
                                  vals = [2,False],
                                  verbose = verbose_sub)
        self._batch_normalization = model_flow_inputs["batch_normalization"]
        self._model_flow_inputs = model_flow_inputs