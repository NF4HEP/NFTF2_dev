# Abstract implementation of Masked Autoregressive Flow
 
__all__ = ["MAFNetwork",
           "MAFBijector",
           "MAFChain"]

import numpy as np
import tensorflow.compat.v1 as tf1
import tensorflow.compat.v2 as tf
from tensorflow.python.keras import Input
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.layers import Layer, Dense
from tensorflow.python.keras.initializers.initializers_v2 import Initializer
from tensorflow.python.keras.regularizers import Regularizer
from tensorflow.python.keras.constraints import Constraint
import tensorflow_probability as tfp
from tensorflow_probability.python.bijectors import Bijector, Chain, Shift, Scale, Permute, BatchNormalization
from tensorflow_probability.python.internal import prefer_static as ps
from tensorflow_probability.python.internal import tensorshape_util

from typing import Union, List, Dict, Callable, Tuple, Optional, NewType, Type, Generic, Any, TypeVar, TYPE_CHECKING
from typing_extensions import TypeAlias
from NF4HEP.utils.custom_types import Array, ArrayInt, ArrayStr, DataType, StrPath, IntBool, StrBool, StrList, StrArray, FigDict, LogPredDict, Number, DTypeStr, DTypeStrList, DictStr
from NF4HEP.utils.verbosity import print, Verbosity
from NF4HEP.utils import utils
from NF4HEP.bijectors.base import BaseNetwork, BaseBijector, BaseChain
from NF4HEP.bijectors.base import _make_dense_autoregressive_masks, _list, _create_input_order, _create_degrees, _create_masks, _make_masked_initializer, _make_masked_constraint, _validate_bijector_fn

header_string_1 = "=============================="
header_string_2 = "------------------------------"


class MAFNetwork(BaseNetwork):
    """
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
        # Attributes type declarations (from parent BaseNetwork class)
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
        self._name: str # this attribute needs to be specified in the inheriting class
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
        # Attributes type declaration (new attributes)
        self._masked_hidden_layers: List[Layer]
        self._conditional_hidden_layers: List[Optional[Layer]]
        # Initialise parent BaseNetwork class
        super().__init__(params = params,
                         event_shape = event_shape,
                         conditional = conditional,
                         conditional_event_shape = conditional_event_shape,
                         conditional_input_layers = conditional_input_layers,
                         input_order = input_order,
                         hidden_degrees = hidden_degrees,
                         hidden_layers = hidden_layers,
                         batch_norm = batch_norm,
                         dropout_rate = dropout_rate,
                         seed = seed,
                         validate_args = validate_args,
                         verbose = verbose,
                         **layer_kwargs)
                    
    @property
    def masked_hidden_layers(self) -> List[Layer]:
        return self._masked_hidden_layers
    
    @property
    def conditional_hidden_layers(self) -> List[Optional[Layer]]:
        return self._conditional_hidden_layers
                    
    def build(self, 
              input_shape: Any):
        """See tfkl.Layer.build."""
        if self._event_shape is None:
            # `event_shape` wasn't specied at __init__, so infer from `input_shape`.
            self._event_shape = [tf.compat.dimension_value(input_shape[-1])] #type: ignore
            self._event_size = self._event_shape[-1]
            self._event_ndims = len(self._event_shape)
            # Should we throw if input_shape has rank > 2?
        if input_shape[-1] != self._event_shape[-1]:
            raise ValueError("Invalid final dimension of `input_shape`. Expected `{!r}`, but got `{!r}`".format(self._event_shape[-1], input_shape[-1]))
        if self.event_size is None:
            raise ValueError("Cannot determine `_event_size`.")
        # Build the un-masked hidden layers
        self.__set_hidden_layers()
        # Read relevant (Dense) hidden layers config to define masked hidden layers
        dense_config: List[Dict[str,Any]] = []
        hidden_units: List[int] = []
        kernel_initializers: List[Optional[Dict[str,Any]]] = []
        kernel_constraints: List[Optional[Dict[str,Any]]] = []
        for layer in self.hidden_layers:
            if "dense" in layer.get_config()["name"]:
                cfg = layer.get_config()
                hidden_units.append(cfg.pop("units"))
                kernel_initializers.append(cfg.pop("kernel_initializer"))
                kernel_constraints.append(cfg.pop("kernel_constraint"))
                dense_config.append(cfg)
        # Construct the masks.
        self._input_order = _create_input_order(input_size = self.event_size,
                                                input_order_input = self.input_order_param,
                                                seed = None)
        self._masks = _make_dense_autoregressive_masks(params = self.params,
                                                       event_size = self.event_size,
                                                       hidden_units = hidden_units,
                                                       input_order_input = self.input_order,
                                                       hidden_degrees = self.hidden_degrees,
                                                       seed = None)
        # Define layers output sizes
        # Input-to-hidden, hidden-to-hidden, and hidden-to-output layers:
        #  [..., self._event_size] -> [..., self._hidden_units[0]].
        #  [..., self._hidden_units[k-1]] -> [..., self._hidden_units[k]].
        #  [..., self._hidden_units[-1]] -> [..., event_size * self._params].
        layer_output_sizes = hidden_units + [self.event_size * self.params]
        # Build the masked and conditional hidden layers
        self._masked_hidden_layers = []
        self._conditional_hidden_layers = []
        for k in range(len(self.masks)):
            cfg = dense_config[k]
            if k + 1 == len(self.masks):
                cfg["activation"] = "linear"
            layer = Dense(layer_output_sizes[k],
                          kernel_initializer=_make_masked_initializer(self._masks[k], kernel_initializers[k]),
                          kernel_constraint=_make_masked_constraint(self._masks[k], kernel_constraints[k]),
                          **cfg)
            if (self.conditional and 
                    ((self.conditional_input_layers == 'all_layers') or 
                        ((self.conditional_input_layers == 'first_layer') and (k == 0)))):
                cfg.pop("use_bias")
                cfg.pop("bias_initializer")
                cfg.pop("bias_regularizer")
                cfg.pop("bias_constraint")
                conditional_layer = Dense(layer_output_sizes[k],
                                          use_bias=False,
                                          kernel_initializer = kernel_initializers[k], # type: ignore
                                          bias_initializer = None, # type: ignore
                                          bias_regularizer = None,
                                          kernel_constraint = kernel_constraints[k],
                                          bias_constraint=None,
                                          **cfg)
            else:
                conditional_layer = None
            self._masked_hidden_layers.append(layer)
            self._conditional_hidden_layers.append(conditional_layer)
            
        x = Input((self.event_size,), dtype=self.dtype)
        y = x
        if self.conditional:
            x_conditional = Input((self.conditional_size,), dtype=self.dtype)
            x = [x, x_conditional]
        else:
            x_conditional = None
        dense_counter = 0
        for layer in self.hidden_layers:
            if "dense" in layer.get_config()["name"]:
                masked_layer = self.masked_hidden_layers[dense_counter]
                conditional_layer = self.conditional_hidden_layers[dense_counter]
                if self.conditional and conditional_layer is not None:
                    y = masked_layer(y)
                    y_conditional = conditional_layer(x_conditional)
                    y = tf.keras.layers.Add()([y,y_conditional])
                else:
                    y = masked_layer(y)
                dense_counter = dense_counter + 1
            else:
                y = layer(y)
        self._network = tf.keras.models.Model(inputs=x,outputs=y)
        # Allow network to be called with inputs of shapes that don't match
        # the specs of the network's input layers.
        self._network.input_spec = None
        # Record that the layer has been built.
        super().build(input_shape)


class MAFBijector(BaseBijector):
    """
    """
    def __init__(self,
                 shift_and_log_scale_fn: Optional[Callable] = None,
                 bijector_fn: Optional[Callable] = None,
                 is_constant_jacobian: bool = False,
                 validate_args: bool = False,
                 unroll_loop: bool = False,
                 event_ndims: int = 1,
                 verbose: Optional[IntBool] = None
                ) -> None:
        # Attributes type declarations (from parent BaseBijector class)
        self._shift_and_log_scale_fn: Optional[Callable]
        self._bijector_fn: Optional[Callable]
        self._is_constant_jacobian: bool
        self._validate_args: bool
        self._event_ndims: int
        # Attributes type declarations
        self._unroll_loop: bool
        # Initialise parent BaseBijector class
        super().__init__(shift_and_log_scale_fn = shift_and_log_scale_fn,
                         bijector_fn = bijector_fn,
                         is_constant_jacobian = is_constant_jacobian,
                         validate_args = validate_args,
                         event_ndims = event_ndims,
                         verbose = verbose)
        # Set verbosity
        verbose, _ = self.get_verbosity(verbose)
        # Initialize object
        print(header_string_1, "\nInitializing MAFBijector object.\n", show = verbose)
        with tf.name_scope(self.__name__) as name:
            # At construction time, we don't know input_depth.
            self._unroll_loop = unroll_loop
            if shift_and_log_scale_fn:
                def _bijector_fn(x, **condition_kwargs):
                    params = shift_and_log_scale_fn(x, **condition_kwargs)
                    if tf.is_tensor(params):
                        shift, log_scale = tf.unstack(params, num=2, axis=-1)
                    else:
                        shift, log_scale = params

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
            
    @property
    def unroll_loop(self) -> bool:
        return self._unroll_loop
    
    def _forward(self, x, **kwargs):
        static_event_size = tensorshape_util.num_elements(tensorshape_util.with_rank_at_least(x.shape, self._event_ndims)[-self._event_ndims:])

        if self._unroll_loop:
            if not static_event_size:
                raise ValueError('The final {} dimensions of `x` must be known at graph '
                'construction time if `unroll_loop=True`. `x.shape: {!r}`'.format(self._event_ndims, x.shape))
            y = tf.zeros_like(x, name='y0')

            for _ in range(static_event_size):
                y = self.bijector_fn(y, **kwargs).forward(x)
            return y

        event_size = ps.reduce_prod(ps.shape(x)[-self._event_ndims:])
        y0 = tf.zeros_like(x, name='y0')
        # call the template once to ensure creation
        if not tf.executing_eagerly():
            _ = self.bijector_fn(y0, **kwargs).forward(y0)
        def _loop_body(y0):
            """While-loop body for autoregression calculation."""
            # Set caching device to avoid re-getting the tf.Variable for every while
            # loop iteration.
            with tf1.variable_scope(tf1.get_variable_scope()) as vs:
              if vs.caching_device is None and not tf.executing_eagerly():
                vs.set_caching_device(lambda op: op.device)
              bijector = self.bijector_fn(y0, **kwargs)
            y = bijector.forward(x)
            return (y,)
        (y,) = tf.while_loop(
            cond=lambda _: True,
            body=_loop_body,
            loop_vars=(y0,),
            maximum_iterations=event_size)
        return y

    def _inverse(self, y, **kwargs):
        bijector = self.bijector_fn(y, **kwargs)
        return bijector.inverse(y)

    def _inverse_log_det_jacobian(self, y, **kwargs):
        return self.bijector_fn(y, **kwargs).inverse_log_det_jacobian(y, event_ndims=self._event_ndims)    
    
    
class MAFChain(BaseChain):
    """
    will have to check is dedicated chain objects are necessary or not
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
        # Attributes type declarations (from parent BaseChain class)
        self._ndims: Optional[int]
        self._nbijectors: int
        self._batch_normalization: bool
        self._permutation: Union[str,ArrayInt,tf.Tensor]
        self._network_kwargs: Dict[str, Any]
        self._bijector_kwargs: Dict[str, Any]
        self._network_name: str
        self._bijector_name: str
        self._bijectors: List[Bijector]
        # Initialise parent BaseChain class
        super().__init__(ndims = ndims,
                         permutation = permutation,
                         nbijectors = nbijectors,
                         batch_normalization = batch_normalization,
                         network_kwargs = network_kwargs,
                         bijector_kwargs = bijector_kwargs,
                         verbose = verbose)