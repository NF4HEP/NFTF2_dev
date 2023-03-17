# Base implementation of Network, Bijector, Chain objects

__all__ = ['BaseNetwork',
           'BaseBijector'
           ]

from abc import ABC, abstractmethod
from numpy import typing as npt
from pathlib import Path

import numpy as np
import tensorflow as tf
#import tensorflow.compat.v1 as tf1
from tensorflow.python.keras import Input
from tensorflow.python.keras import layers, initializers, regularizers, constraints, callbacks, optimizers, metrics, losses
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.layers import Layer
import tensorflow_probability as tfp
from tensorflow_probability.python.bijectors import Bijector, Chain, Shift, Scale, Permute, BatchNormalization

from typing import Union, List, Dict, Callable, Tuple, Optional, NewType, Type, Generic, Any, TypeVar, TYPE_CHECKING
from typing_extensions import TypeAlias
from NF4HEP.utils.custom_types import Array, ArrayInt, ArrayStr, DataType, StrPath, IntBool, StrBool, StrList, StrArray, FigDict, LogPredDict, Number, DTypeStr, DTypeStrList, DictStr
from NF4HEP.utils.verbosity import print, Verbosity
from NF4HEP.utils import utils

header_string_1 = "=============================="
header_string_2 = "------------------------------"

 
class BaseNetwork(Layer, Verbosity):
    """
    Base Neural Network Architecture to be used as _bijector_fn function in the bijectors.
    The hidden_layers input is specified as in the following instance:
    
    .. code-block:: python

        hidden_layers = ["Dense(1000,activation='relu',kernel_initializer='glorot_uniform',kernel_regularizer=regularizers.L1L2(l1=0.001, l2=0.001))",
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
                 params: Optional[int] = None,
                 event_shape: Optional[Union[int,List[int]]] = None, # this parameter is determined automatically by the build method if it is not specified
                 conditional = False,
                 conditional_event_shape = None,
                 conditional_input_layers = 'all_layers',
                 input_order: StrArray = 'left-to-right',
                 hidden_degrees: str = 'equal',
                 hidden_layers: Optional[List[Any]] = None,
                 batch_norm: bool = False,
                 dropout_rate: Union[float,str] = 0.,
                 seed: Optional[int] = None,
                 validate_args: bool = True,
                 verbose: Optional[IntBool] = None,
                 **layer_kwargs
                ) -> None:
        # Attributes type declarations (attributes from input parameters)
        self._params: int
        self._event_shape: Optional[List[int]]
        self._conditional: bool
        self._conditional_event_shape: Optional[list]
        self._conditional_input_layers: str 
        self._input_order_param: StrArray
        self._hidden_degrees: str
        self._hidden_layers_input: List[Any]
        self._batch_norm: StrBool
        self._dropout_rate: Union[float,str]
        self._seed: Optional[int]
        self._validate_args: bool
        self._layer_kwargs: dict
        # Attributes type declarations (other attributes)
        self._event_size: Optional[int]
        self._event_ndims: Optional[int]
        self._conditional_size: Optional[int]
        self._conditional_ndims: Optional[int]
        self._input_order: Array # to be specified        
        self._hidden_layers: List[Layer]
        self._hidden_layers_string: List[str]
        self._output_layers: List[Layer]
        self._output_layers_string: List[str]
        self._masks: list
        self._network: Optional[Model]
        # Initialise parent Verbosity class
        Verbosity.__init__(self, verbose)
        # Set verbosity
        verbose, _ = self.get_verbosity(verbose)
        # Initialise parent Layer class
        Layer.__init__(self, **layer_kwargs)
        #Layer.__init__(self)
        # Initialize object
        # Parse inputs
        self._params = params if params is not None else 1
        self._event_shape = _list(event_shape) if event_shape is not None else None
        self._conditional = conditional
        self._conditional_event_shape = _list(conditional_event_shape) if conditional_event_shape is not None else None
        self._conditional_input_layers = conditional_input_layers
        self._input_order_param = input_order
        self._hidden_degrees = hidden_degrees
        self._hidden_layers_input = hidden_layers if hidden_layers is not None else []        
        self._batch_norm = batch_norm
        self._dropout_rate = _float(dropout_rate)
        self._seed = seed
        self._validate_args = validate_args
        self._layer_kwargs = layer_kwargs
        # Define additional attributes
        if self._event_shape is not None:
            self._event_size = self._event_shape[-1]
            self._event_ndims = len(self._event_shape)
            if self._event_ndims != 1:
                raise ValueError('Parameter `event_shape` must describe a rank-1 shape. `event_shape: {!r}`'.format(event_shape))
        else:
            self._event_size = None
            self._event_ndims = None
        if self._conditional:
            if self._event_shape is None:
                raise ValueError('`event_shape` must be provided when `conditional` is True')
            if self._conditional_event_shape is None:
                raise ValueError('`conditional_event_shape` must be provided when `conditional` is True')
            self._conditional_size = self._conditional_event_shape[-1]
            self._conditional_ndims = len(self._conditional_event_shape)
            if self._conditional_ndims != 1:
                raise ValueError('Parameter `conditional_event_shape` must describe a rank-1 shape')
            if not ((self.conditional_input_layers == 'first_layer') or (self.conditional_input_layers == 'all_layers')):
                raise ValueError('`conditional_input_layers` must be "first_layers" or "all_layers"')
        else:
            if self._conditional_event_shape is not None:
                raise ValueError('`conditional_event_shape` passed but `conditional` is set to False.')
        
        # To be built in `build`.
        self._hidden_layers = []
        self._hidden_layers_string = []
        self._output_layers = []
        self._output_layers_string = []
        self._input_order = []
        self._masks = []
        self._network = None
        
    @property
    def params(self) -> int:
        return self._params

    @property
    def event_shape(self) -> Optional[List[int]]:
        return self._event_shape
    
    @property
    def conditional(self) -> bool:
        return self._conditional
    
    @property
    def conditional_event_shape(self) -> Optional[list]:
        return self._conditional_event_shape
    
    @property
    def conditional_input_layers(self) -> str:
        return self._conditional_input_layers
    
    @property
    def input_order_param(self) -> StrArray:
        return self._input_order_param
    
    @property
    def hidden_degrees(self) -> str:
        return self._hidden_degrees
    
    @property
    def hidden_layers_input(self) -> List[Any]:
        return self._hidden_layers_input
    
    @property
    def batch_norm(self) -> StrBool:
        return self._batch_norm

    @property
    def dropout_rate(self) -> Union[float,str]:
        return self._dropout_rate
    
    @property
    def seed(self) -> Optional[int]:
        return self._seed
    
    @property
    def validate_args(self) -> bool:
        return self._validate_args
    
    @property
    def layer_kwargs(self) -> dict:
        return self._layer_kwargs
    
    @property
    def event_size(self) -> Optional[int]:
        return self._event_size
    
    @property
    def event_ndims(self) -> Optional[int]:
        return self._event_ndims

    @property
    def conditional_size(self) -> Optional[int]:
        return self._conditional_size
    
    @property
    def conditional_ndims(self) -> Optional[int]:
        return self._conditional_ndims
    
    @property
    def input_order(self) -> Array:
        return self._input_order

    @property
    def hidden_layers(self) -> List[Layer]:
        return self._hidden_layers

    @property
    def hidden_layers_string(self) -> List[str]:
        return self._hidden_layers_string
    
    @property
    def output_layers(self) -> List[Layer]:
        return self._output_layers

    @property
    def output_layers_string(self) -> List[str]:
        return self._output_layers_string
    
    @property
    def masks(self) -> list:
        return self._masks

    @property
    def network(self) -> Optional[Model]:
        return self._network

    def __set_hidden_layers(self,
                            verbose: Optional[IntBool] = None
                           ) -> None:
        """
        """
        layer_string: str
        layer: Layer
        verbose, _ = self.get_verbosity(verbose)
        print(header_string_2,"\nSetting hidden layers\n", show = verbose)
        self._hidden_layers_string = []
        self._hidden_layers = []
        i = 0
        if "dropout" in str(self.hidden_layers_input).lower():
            insert_dropout = False
            self._dropout_rate = "custom"
        elif "dropout" not in str(self.hidden_layers_input).lower() and self.dropout_rate != 0.:
            insert_dropout = True
        else:
            insert_dropout = False
        if "batchnormalization" in str(self.hidden_layers_input).lower():
            self._batch_norm = "custom"
        layer_string = ""
        for layer in self.hidden_layers_input:
            if isinstance(layer,str):
                if "(" in layer:
                    layer_string = "layers."+layer
                else:
                    layer_string = "layers."+layer+"()"
            elif isinstance(layer,dict):
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
            elif isinstance(layer,list):
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
                print("WARNING: Invalid input for layer: ", layer, ". The layer will not be added to the model.", show = True)
            if layer_string != "":
                if self._batch_norm == True and "dense" in layer_string.lower():
                    self._hidden_layers_string.append("layers.BatchNormalization()")
                    print("Added hidden layer: layers.BatchNormalization()", show = verbose)
                    i = i + 1
                try:
                    eval(layer_string)
                    self._hidden_layers_string.append(layer_string)
                    print("Added hidden layer: ", layer_string, show = verbose)
                    i = i + 1
                except Exception as e:
                    print(e)
                    print("WARNING: Could not add layer", layer_string, "\n", show = True)
                if insert_dropout:
                    try:
                        act = eval(layer_string+".activation")
                        if "selu" in str(act).lower():
                            layer_string = "layers.AlphaDropout(" + str(self._dropout_rate)+")"
                            self._hidden_layers_string.append(layer_string)
                            print("Added hidden layer: ",layer_string, show = verbose)
                            i = i + 1
                        elif "linear" not in str(act):
                            layer_string = "layers.Dropout(" + str(self._dropout_rate)+")"
                            self._hidden_layers_string.append(layer_string)
                            print("Added hidden layer: ", layer_string, show = verbose)
                            i = i + 1
                        else:
                            layer_string = ""
                    except:
                        layer_string = "layers.AlphaDropout(" + str(self._dropout_rate)+")"
                        self._hidden_layers_string.append(layer_string)
                        print("Added hidden layer: ", layer_string, show = verbose)
                        i = i + 1
        if layer_string != "":
            if self._batch_norm == True and "dense" in layer_string.lower():
                self._hidden_layers_string.append("layers.BatchNormalization()")
                print("Added hidden layer: layers.BatchNormalization()", show = verbose)
        for layer_string in self._hidden_layers_string:
            try:
                print("Building layer:", layer_string, show = verbose)
                self._hidden_layers.append(eval(layer_string))
            except:
                print("WARNING: Failed to evaluate:", layer_string, show = True)

    def get_config(self):
        config = Layer.get_config(self)
        config.update({"params": self.params,
                       "event_shape": self.event_shape,
                       "conditional": self.conditional,
                       "conditional_event_shape": self.conditional_event_shape,
                       "conditional_input_layers": self.conditional_input_layers,
                       "input_order": self.input_order_param,
                       "hidden_degrees": self.hidden_degrees,
                       "hidden_layers": self.hidden_layers,
                       "batch_norm": self.batch_norm,
                       "dropout_rate": self.dropout_rate,
                       "seed": self.seed,
                       "validate_args": self.validate_args,
                       "verbose": self._verbose,
                       "layer_kwargs": self.layer_kwargs})
        return config

    def from_config(self, config):
        return self(**config)


class BaseBijector(Bijector,Verbosity):
    """
    Base class for the Normalizing Flow Bijector.
    Class inspired by the `MaskedAutoregressiveFlow` bijector TensorFlow Probability implementation (v0.19.0).
    """
    def __init__(self,
                 shift_and_log_scale_fn: Optional[Callable] = None,
                 bijector_fn: Optional[Callable] = None,
                 is_constant_jacobian: bool = False,
                 validate_args: bool = False,
                 event_ndims: int = 1,
                 verbose: Optional[IntBool] = None
                ) -> None:
        # Attributes type declarations (attributes from input parameters)
        self._shift_and_log_scale_fn: Optional[Callable]
        self._bijector_fn: Optional[Callable]
        self._is_constant_jacobian: bool
        self._validate_args: bool
        self._event_ndims: int
        # Initialise parent Verbosity class
        Verbosity.__init__(self, verbose)
        # Set verbosity
        verbose, _ = self.get_verbosity(verbose)
        # Initialize object
        parameters = dict(locals())
        with tf.name_scope(self.__name__) as name:
            if bool(shift_and_log_scale_fn) == bool(bijector_fn):
                raise ValueError('Exactly one of `shift_and_log_scale_fn` and `bijector_fn` should be specified.')
            self._event_ndims = event_ndims
            if shift_and_log_scale_fn:
                #def _bijector_fn(x, **condition_kwargs):
                #    """
                #    This is from MaskedAutoregressiveFlow
                #    Args:
                #        x (_type_): _description_
                #    Returns:
                #        _type_: _description_
                #    """
                #    params = shift_and_log_scale_fn(x, **condition_kwargs)
                #    if tf.is_tensor(params):
                #        shift, log_scale = tf.unstack(params, num=2, axis=-1)
                #    else:
                #        shift, log_scale = params
                #      
                #    bijectors = []
                #    if shift is not None:
                #        bijectors.append(shift_lib.Shift(shift))
                #    if log_scale is not None:
                #        bijectors.append(scale_lib.Scale(log_scale=log_scale))
                #    return chain.Chain(bijectors, validate_event_size=False)
                def _bijector_fn(x0, input_depth, **condition_kwargs):
                    """
                    This is from RealNVP
                    Args:
                        x0 (_type_): _description_
                        input_depth (_type_): _description_
#
                    Returns:
                        _type_: _description_
                    """
                    shift, log_scale = shift_and_log_scale_fn(x0, input_depth, **condition_kwargs)
                      
                    bijectors = []
                    if shift is not None:
                        bijectors.append(Shift(shift))
                    if log_scale is not None:
                        bijectors.append(Scale(log_scale=log_scale))
                    return Chain(bijectors, validate_event_size=False)
                bijector_fn = _bijector_fn
            if validate_args:
                bijector_fn = _validate_bijector_fn(bijector_fn)
            # Still do this assignment for variable tracking.
            self._shift_and_log_scale_fn = shift_and_log_scale_fn
            self._bijector_fn = bijector_fn
            # Initialise parent Bijector class
            Bijector.__init__(self,
                              forward_min_event_ndims = self._event_ndims,
                              is_constant_jacobian = is_constant_jacobian,
                              validate_args = validate_args,
                              parameters = parameters,
                              name = name)
            
    @classmethod
    def _parameter_properties(cls, dtype):
        return dict()
            
    @property
    def shift_and_log_scale_fn(self) -> Callable:
        if self._shift_and_log_scale_fn is not None:
            return self._shift_and_log_scale_fn
        else:
            raise ValueError("Shift and log-scale bijector function `_shift_and_log_scale_fn` is not defined (None).")
    
    @property
    def bijector_fn(self) -> Callable:
        if self._bijector_fn is not None:
            return self._bijector_fn
        else:
            raise ValueError("Bijector function `_bijector_fn` is not defined (None).")
    
    @property
    def is_constant_jacobian(self) -> bool:
        return self._is_constant_jacobian
    
    @property
    def validate_args(self) -> bool:
        return self._validate_args
    
    @property
    def event_ndims(self) -> int:
        return self._event_ndims
    
    
class BaseChain(Chain,Verbosity): # type: ignore
    """
    model_chain_inputs and model_chain_inputs can be of the following form:
    .. code-block:: python

        model_chain_inputs = {num_masked: Optional[int] = None,
                              fraction_masked: Optional[float] = None,
                              shift_and_log_scale_fn: Optional[Callable] = None,
                              bijector_fn: Optional[Callable] = None,
                              is_constant_jacobian: bool = False,
                              validate_args: bool = False,
                              event_ndims: int = 1,
                              verbose: Optional[IntBool] = None}

        model_chain_inputs = {"nbijectors": 2,
                              "batch_normalization": False}
    """
    def __init__(self,
                 ndims: Optional[int] = None,
                 permutation: Union[str,ArrayInt,tf.Tensor] = "bi-partition",
                 nbijectors: Optional[int] = None,
                 batch_normalization: bool = False,
                 network_kwargs: Optional[Dict[str, Any]] = None,
                 bijector_kwargs: Optional[Dict[str, Any]] = None,
                 verbose: Optional[IntBool] = None
                ) -> None:
        # Attributes type declarations
        self._ndims: Optional[int]
        self._nbijectors: int
        self._batch_normalization: bool
        self._permutation: Union[str,ArrayInt,tf.Tensor]
        self._network_kwargs: Dict[str, Any]
        self._bijector_kwargs: Dict[str, Any]
        self._network_name: str
        self._bijector_name: str
        self._bijectors: List[Bijector]
        # Initialise parent Verbosity class
        Verbosity.__init__(self, verbose)
        # Set verbosity
        verbose, _ = self.get_verbosity(verbose)
        # Initialize object
        self._network_name = self.__name__.replace("Chain","Network")
        self._bijector_name = self.__name__.replace("Chain","Bijector")
        # Parsing inputs
        self._ndims = ndims
        self._permutation = permutation
        if nbijectors is not None:
            self._nbijectors = nbijectors
        else:
            self._nbijectors = 2
            print("WARNING: The number of bijectors `nbijectors` has not been specified and has been automatically set to 2.")
        self._batch_normalization = batch_normalization
        if network_kwargs is not None:
            self._network_kwargs = network_kwargs
        else:
            raise ValueError("The `network_kwargs` argument should be specified as required by the",self._network_name,"object.")
        if bijector_kwargs is not None:
            self._bijector_kwargs = bijector_kwargs
        else:
            raise ValueError("The `network_kwargs` argument should be specified as required by the",self._bijector_name,"object.")
        self._bijectors = []
        for _ in range(self.nbijectors):
            nn = eval(self.network_name)(**self.network_kwargs)
            bijector = eval(self.bijector_name)(shift_and_log_scale_fn = nn,**self.bijector_kwargs)
            permute = Permute(permutation = self.__get_permutation(ndims = self.ndims, permutation = self.permutation))
            if self._batch_normalization:
                self._bijectors.append(BatchNormalization())
            self._bijectors.append(bijector)
            self._bijectors.append(permute)
        if self._batch_normalization:
            self._bijectors.append(BatchNormalization())
        Chain.__init__(self, bijectors=list(reversed(self._bijectors[:-1])), name = self.__name__)        
                
    @property
    def ndims(self) -> Optional[int]:
        return self._ndims

    @property
    def nbijectors(self) -> int:
        return self._nbijectors
    
    @property
    def batch_normalization(self) -> bool:
        return self._batch_normalization
    
    @property
    def permutation(self) -> Union[str,ArrayInt,tf.Tensor]:
        return self._permutation
    
    @property
    def shuffle_style(self) -> str:
        return self._shuffle_style
    
    @property
    def network_kwargs(self) -> Dict[str,Any]:
        return self._network_kwargs
    
    @property
    def bijector_kwargs(self) -> Dict[str,Any]:
        return self._bijector_kwargs
    
    @property
    def network_name(self) -> str:
        return self._network_name
    
    @property
    def bijector_name(self) -> str:
        return self._bijector_name
    
    @property
    def bijectors(self) -> list[Bijector]:
        return self._bijectors

    def __get_permutation(self,
                          ndims: Optional[int] = None,
                          permutation: Union[str,ArrayInt,tf.Tensor] = "bi-partition"
                         ) -> tf.Tensor:
        if ndims is None:
            if not isinstance(permutation, (list, np.ndarray, tf.Tensor)):# (permutation is None or isinstance(permutation,str)):
                raise ValueError("If `permutation` is not an explicit permutation vector (list, array or tf.Tensor of integers), the `ndims` input argument must be different than `None`.")
            else:
                perm = tf.cast(permutation, tf.int32)
        else:
            if permutation is None:
                print("WARNING: no permutation has been specified. Proceeding with a default bi-partition permutation.", show = True)
                perm = _permute_bi_partition(ndims)
            elif isinstance(permutation,  str) and ndims is not None:
                if self._perm_style == "bi-partition":
                    perm = _permute_bi_partition(ndims)
                elif self._perm_style == "reverse":
                    perm = _permute_reverse(ndims)
                elif self._perm_style == "shuffle":
                    perm = _permute_shuffle(ndims)
                else:
                    print("WARNING: Invalid permutation string. Proceeding with `perm_style = 'bi-partition'`.", show = True)
                    perm = _permute_bi_partition(ndims)
            elif isinstance(permutation, (list, np.ndarray, tf.Tensor)):
                perm = tf.cast(permutation, tf.int32)
            else:
                print("WARNING: Invalid permutation string. Proceeding with `perm_style = 'bi-partition'`.", show = True)
                perm = _permute_bi_partition(ndims)
        if  isinstance(perm, tf.Tensor):
            return perm
        else:
            raise ValueError("Could not set the `_permutation` tensor.")
        
    
def _make_dense_autoregressive_masks(params: int,
                                     event_size: int,
                                     hidden_units: Optional[list],
                                     input_order_input: StrArray = 'left-to-right',
                                     hidden_degrees: str = 'equal',
                                     seed: Optional[int] = None,
                                    ) -> List:
    """Creates masks for use in dense MADE [Germain et al. (2015)][1] networks.
    See the documentation for `AutoregressiveNetwork` for the theory and
    application of MADE networks. This function lets you construct your own dense
    MADE networks by applying the returned masks to each dense layer. E.g. a
    consider an autoregressive network that takes `event_size`-dimensional vectors
    and produces `params`-parameters per input, with `num_hidden` hidden layers,
    with `hidden_size` hidden units each.
    ```python
    def random_made(x):
        masks = tfb._make_dense_autoregressive_masks(params=params,
                                                     event_size=event_size,
                                                     hidden_units=[hidden_size] * num_hidden)
        output_sizes = [hidden_size] * num_hidden
        input_size = event_size
        for (mask, output_size) in zip(masks, output_sizes):
            mask = tf.cast(mask, tf.float32)
            x = tf.matmul(x, tf.random.normal([input_size, output_size]) * mask)
            x = tf.nn.relu(x)
            input_size = output_size
        x = tf.matmul(x,
                      tf.random.normal([input_size, params * event_size]) * masks[-1])
        x = tf.reshape(x, [-1, event_size, params])
        return x
    y = random_made(tf.zeros([1, event_size]))
    assert [1, event_size, params] == y.shape
    ```
    Each mask is a Numpy boolean array. All masks have the shape `[input_size,
    output_size]`. For example, if we `hidden_units` is a list of two integers,
    the mask shapes will be: `[event_size, hidden_units[0]], [hidden_units[0],
    hidden_units[1]], [hidden_units[1], params * event_size]`.
    You can extend this example with trainable parameters and constraints if
    necessary.
    Args:
        params: Python integer specifying the number of parameters to output
          per input.
        event_size: Python integer specifying the shape of the input to this layer.
        hidden_units: Python `list`-like of non-negative integers, specifying
          the number of units in each hidden layer.
        input_order: Order of degrees to the input units: 'random', 'left-to-right',
          'right-to-left', or an array of an explicit order. For example,
          'left-to-right' builds an autoregressive model
          p(x) = p(x1) p(x2 | x1) ... p(xD | x<D).
        hidden_degrees: Method for assigning degrees to the hidden units:
          'equal', 'random'. If 'equal', hidden units in each layer are allocated
          equally (up to a remainder term) to each degree. Default: 'equal'.
        seed: If not `None`, seed to use for 'random' `input_order` and
          `hidden_degrees`.
    Returns:
        masks: A list of masks that should be applied the dense matrices of
          individual densely connected layers in the MADE network. Each mask is a
          Numpy boolean array.
    # References
    [1]: Mathieu Germain, Karol Gregor, Iain Murray, and Hugo Larochelle. MADE:
         Masked Autoencoder for Distribution Estimation. In _International
         Conference on Machine Learning_, 2015. https://arxiv.org/abs/1502.03509
    """
    if seed is None:
        input_order_seed = None
        degrees_seed = None
    else:
        input_order_seed, degrees_seed = np.random.RandomState(seed).randint(2**31, size=2)
    input_order = _create_input_order(input_size = event_size, 
                                      input_order_input = input_order_input, 
                                      seed = input_order_seed)
    masks = _create_masks(_create_degrees(input_size = event_size,
                                          hidden_units = hidden_units,
                                          input_order_input = input_order_input,
                                          hidden_degrees = hidden_degrees,
                                          seed = degrees_seed))
    # In the final layer, we will produce `params` outputs for each of the
    # `event_size` inputs.  But `masks[-1]` has shape `[hidden_units[-1],
    # event_size]`.  Thus, we need to expand the mask to `[hidden_units[-1],
    # event_size * params]` such that all units for the same input are masked
    # identically.  In particular, we tile the mask so the j-th element of
    # `tf.unstack(output, axis=-1)` is a tensor of the j-th parameter/unit for
    # each input.
    #
    # NOTE: Other orderings of the output could be faster -- should benchmark.
    masks[-1] = np.reshape(
        np.tile(masks[-1][..., tf.newaxis], [1, 1, params]),
        [masks[-1].shape[0], event_size * params])
    return masks


def _list(xs: Any) -> List:
    """Convert the given argument to a list."""
    try:
        return list(xs)
    except TypeError:
        return [xs]


def _float(xs: Any) -> float:
    """Convert the given argument to a float."""
    try:
        return float(xs)
    except TypeError:
        return xs
    

def _create_input_order(input_size: int,
                        input_order_input: StrArray = 'left-to-right', 
                        seed: Optional[int] = None
                       ) -> Array:
    """Returns a degree vectors for the input."""
    if isinstance(input_order_input, str):
        if input_order_input == 'left-to-right':
            return np.arange(start=1, stop=input_size + 1)
        elif input_order_input == 'right-to-left':
            return np.arange(start=input_size, stop=0, step=-1)
        elif input_order_input == 'random':
            ret = np.arange(start=1, stop=input_size + 1)
            if seed is None:
                rng = np.random
            else:
                rng = np.random
                rng.RandomState(seed)
            rng.shuffle(ret)
            return ret
    elif np.all(np.sort(np.array(input_order_input)) == np.arange(1, input_size + 1)):
        return np.array(input_order_input)
    raise ValueError('Invalid input order: "{}".'.format(input_order_input))


def _create_degrees(input_size: int,
                    hidden_units: Optional[list] = None,
                    input_order_input: StrArray = 'left-to-right', 
                    hidden_degrees: str = 'equal',
                    seed: Optional[int] = None
                   ) -> list:
    """
    Returns a list of degree vectors, one for each input and hidden layer.
    A unit with degree d can only receive input from units with degree < d. Output
    units always have the same degree as their associated input unit.
    Args:
        input_size: Number of inputs.
        hidden_units: list with the number of hidden units per layer. It does not
          include the output layer. Each hidden unit size must be at least the size
          of length (otherwise autoregressivity is not possible).
        input_order: Order of degrees to the input units: 'random', 'left-to-right',
          'right-to-left', or an array of an explicit order. For example,
          'left-to-right' builds an autoregressive model
          p(x) = p(x1) p(x2 | x1) ... p(xD | x<D).
        hidden_degrees: Method for assigning degrees to the hidden units:
          'equal', 'random'.  If 'equal', hidden units in each layer are allocated
          equally (up to a remainder term) to each degree.  Default: 'equal'.
        seed: If not `None`, use as a seed for the 'random' hidden_degrees.
    Raises:
        ValueError: invalid input order.
        ValueError: invalid hidden degrees.
    """
    input_order = _create_input_order(input_size = input_size, 
                                      input_order_input = input_order_input,
                                      seed = None)
    degrees = [input_order]
    if hidden_units is None:
        hidden_units = []
    for units in hidden_units:
        if isinstance(hidden_degrees, str):
            if hidden_degrees == 'random':
                if seed is None:
                    rng = np.random
                else:
                    rng = np.random
                    rng.RandomState(seed)
                # samples from: [low, high)
                degrees.append(rng.randint(low=min(np.min(degrees[-1]), input_size - 1),
                                           high=input_size,
                                           size=units))
            elif hidden_degrees == 'equal':
                min_degree = min(np.min(degrees[-1]), input_size - 1)
                degrees.append(np.maximum(
                    min_degree,
                    # Evenly divide the range `[1, input_size - 1]` in to `units + 1`
                    # segments, and pick the boundaries between the segments as degrees.
                    np.ceil(np.arange(1, units + 1)* (input_size - 1) / float(units + 1)).astype(np.int32)))
        else:
            raise ValueError('Invalid hidden order: "{}".'.format(hidden_degrees))
    return degrees


def _create_masks(degrees: list) -> list:
    """Returns a list of binary mask matrices enforcing autoregressivity."""
    return [
        # Create input->hidden and hidden->hidden masks.
        inp[:, np.newaxis] <= out
        for inp, out in zip(degrees[:-1], degrees[1:])
    ] + [
        # Create hidden->output mask.
        degrees[-1][:, np.newaxis] < degrees[0]
    ]


def _make_masked_initializer(mask: list, 
                             initializer: Any
                            ) -> Any:
    """Returns a masked version of the given initializer."""
    initializer = tf.keras.initializers.get(initializer)
    def masked_initializer(shape: Array, 
                           dtype: DTypeStr = None, 
                           partition_info: Any = None
                          ) -> tf.Tensor:
        # If no `partition_info` is given, then don't pass it to `initializer`, as
        # `initializer` may be a `tf.initializers.Initializer` (which don't accept a
        # `partition_info` argument).
        if partition_info is None:
            x = initializer(shape, 
                            dtype)
        else:
            x = initializer(shape, 
                            dtype, 
                            partition_info)
        return tf.cast(mask, x.dtype) * x
    try:
        return masked_initializer
    except:
        raise Exception("Wrong initializer")


def _make_masked_constraint(mask: list, 
                            constraint: Any = None
                           ) -> Any:
    constraint = tf.keras.constraints.get(constraint)
    def masked_constraint(x: Any) -> tf.Tensor:
        x = tf.convert_to_tensor(x, dtype_hint=tf.float32, name='x')
        if constraint is not None:
            x = constraint(x)
        return tf.cast(mask, x.dtype) * x
    return masked_constraint


#def _validate_bijector_fn(bijector_fn: Callable) -> Callable:
#    """
#    From MaskedAutoregressiveFlow
#    Validates the output of `bijector_fn`."""
#
#    def _wrapper(x, **condition_kwargs):
#        """A wrapper that validates `bijector_fn`."""
#        bijector = bijector_fn(x, **condition_kwargs)
#        if bijector.forward_min_event_ndims != bijector.inverse_min_event_ndims:
#            # Current code won't really work with this, but in principle we could
#            # implement this.
#            raise ValueError('Bijectors which alter `event_ndims` are not supported.')
#        if bijector.forward_min_event_ndims > 0:
#            # Mustn't break auto-regressivity,
#            raise ValueError('Bijectors with `forward_min_event_ndims` > 0 are not supported.')
#        return bijector
#
#    return _wrapper


def _validate_bijector_fn(bijector_fn):
    """
    From RealNVP
    Validates the output of `bijector_fn`."""

    def _wrapper(x, output_units, **condition_kwargs):
        bijector = bijector_fn(x, output_units, **condition_kwargs)
        if bijector.forward_min_event_ndims != bijector.inverse_min_event_ndims:
            # We need to be able to re-combine the state parts.
            raise ValueError('Bijectors which alter `event_ndims` are not supported.')
        if bijector.forward_min_event_ndims > 1:
            # Mostly because we can't propagate this up to the RealNVP bijector.
            raise ValueError('Bijectors with `forward_min_event_ndims` > 1 are not supported.')
        return bijector

    return _wrapper

def _permute_shuffle(ndims):
    """
    """
    arr = np.arange(ndims)
    np.random.shuffle(arr)
    random_shuffle=tf.cast(arr, tf.int32)
    return random_shuffle

def _permute_reverse(ndims):
    """
    """
    arr = np.arange(ndims)
    arr = np.flip(arr)
    perm = tf.cast(arr, tf.int32)
    return perm

def _permute_bi_partition(ndims):
    """
    """
    arr1 = np.arange(int(ndims/2),ndims)
    arr2 = np.arange(0,int(ndims/2))
    arr = np.concatenate((arr1,arr2))
    perm = tf.cast(arr, tf.int32)
    return perm