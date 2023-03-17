# Abstract implementation of RealNVP Flow

__all__ = ["RealNVPNetwork",
           "RealNVPBijector",
           "RealNVPChain"]

import numpy as np
import tensorflow as tf
#import tensorflow.compat.v1 as tf1
from tensorflow.python.keras import Input
from tensorflow.python.keras import layers, initializers, regularizers, constraints, callbacks, optimizers, metrics, losses
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.layers import Layer
import tensorflow_probability as tfp #
from tensorflow_probability.python.bijectors import Bijector, Chain, Shift, Scale, Permute, BatchNormalization
from tensorflow_probability.python.internal import tensorshape_util

from typing import Union, List, Dict, Callable, Tuple, Optional, NewType, Type, Generic, Any, TypeVar, TYPE_CHECKING
from typing_extensions import TypeAlias
from NF4HEP.utils.custom_types import Array, ArrayInt, ArrayStr, DataType, StrPath, IntBool, StrBool, StrList, StrArray, FigDict, LogPredDict, Number, DTypeStr, DTypeStrList, DictStr
from NF4HEP.utils.verbosity import print, Verbosity
from NF4HEP.utils import utils
from NF4HEP.bijectors.base import BaseNetwork, BaseBijector, BaseChain
from NF4HEP.bijectors.base import _list, _validate_bijector_fn

header_string_1 = "=============================="
header_string_2 = "------------------------------"

# Singleton object representing "no value", in cases where "None" is meaningful.
UNSPECIFIED = object()

class RealNVPNetwork(BaseNetwork):
    """
    """
    def __init__(self,
                 params: Optional[int] = None,
                 event_shape: Optional[Union[int,List[int]]] = None, # this parameter is determined automatically by the build method if it is not specified
                 hidden_layers: Optional[List[Any]] = None,
                 batch_norm: bool = False,
                 dropout_rate: Union[float,str] = 0.,
                 validate_args: bool = True,
                 seed: Optional[int] = None,
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
        # Attributes type declarations
        
        # Initialise parent BaseNetwork class
        super().__init__(params = params,
                         event_shape = event_shape,
                         hidden_layers = hidden_layers,
                         batch_norm = batch_norm,
                         dropout_rate = dropout_rate,
                         seed = seed,
                         validate_args = validate_args,
                         verbose = verbose,
                         **layer_kwargs)
        # Set verbosity
        verbose, _ = self.get_verbosity(verbose)
        # Set inputs and initialise parent BaseNetwork class
        print(header_string_1, "\nInitializing RealNVPNetwork object.\n", show = verbose)
                
    def __set_output_layers(self,
                            ndims: int,
                            verbose: Optional[IntBool] = None
                           ) -> None:
        """
        """
        verbose, _ = self.get_verbosity(verbose)
        print(header_string_2,"\nSetting output layers\n", show = verbose)
        self._output_layers_string = []
        self._output_layers = []
        shift_layer_string = "layers.Dense("+str(ndims)+", name='shift')"
        log_scale_layer_string = "layers.Dense("+str(ndims)+", activation='tanh', name='log_scale')"
        self._output_layers_string.append(shift_layer_string)
        self._output_layers_string.append(log_scale_layer_string)
        for layer_string in self._output_layers_string:
            try:
                print("Building layer:", layer_string, show = verbose)
                self._output_layers.append(eval(layer_string))
            except:
                print("WARNING: Failed to evaluate:", layer_string, show = True)

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
            raise ValueError('Invalid final dimension of `input_shape`. Expected `{!r}`, but got `{!r}`'.format(self._event_shape[-1], input_shape[-1]))
        # Build the layers
        self.__set_hidden_layers()
        if self.event_size is not None:
            x = Input((self._event_size,), dtype=self.dtype)
            self.__set_output_layers(ndims = self.event_size)
            y = x
            for layer in self.hidden_layers:
                y = layer(y)
            shift = self.output_layers[0](y)
            log_scale = self.output_layers[1](y)
            self._network = Model(inputs=x,outputs=[shift, log_scale])
            # Allow network to be called with inputs of shapes that don't match
            # the specs of the network's input layers.
            self._network.input_spec = None
            # Record that the layer has been built.
            super().build(input_shape)
        else:
            raise ValueError("Cannot determine size of input `_event_size`.")

    def call(self, x):
        """
        """
        # Define and return Model
        if self._network is not None:
            return self._network(x)
        else:
            raise ValueError("The value of `_network` is `None`, i.e. `_network` has not been built.")
        

class RealNVPBijector(BaseBijector):
    """
    Implementation of a Real-NVP for Denisty Estimation. L. Dinh “Density estimation using Real NVP,” 2016.
    """
    def __init__(self,
                 num_masked: Optional[int] = None,
                 fraction_masked: Optional[float] = None,
                 shift_and_log_scale_fn: Optional[Callable] = None,
                 bijector_fn: Optional[Callable] = None,
                 is_constant_jacobian: bool = False,
                 validate_args: bool = False,
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
        self._input_depth: Optional[int]
        self._num_masked: Optional[int]
        self._fraction_masked: Optional[float]
        self._reverse_mask: bool
            #self._rem_dims: int
            #self._tran_ndims: int
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
        print(header_string_1, "\nInitializing RealNVPBijector object.\n", show = verbose)
        with tf.name_scope(self.name) as name:
            # At construction time, we don't know input_depth.
            self._input_depth = None
            if num_masked is not None and fraction_masked is None:
                if int(num_masked) != num_masked:
                    raise TypeError('`num_masked` must be an integer. Got: {} of type {}'.format(num_masked, type(num_masked)))
                self._num_masked = int(num_masked)
                self._fraction_masked = None
                self._reverse_mask = self._num_masked < 0
            elif num_masked is None and fraction_masked is not None:
                if not np.issubdtype(type(fraction_masked), np.floating):
                    raise TypeError('`fraction_masked` must be a float. Got: {} of type {}'.format(fraction_masked, type(fraction_masked)))
                if np.abs(fraction_masked) >= 1.:
                    raise ValueError('`fraction_masked` must be in (-1, 1), but is {}.'.format(fraction_masked))
                self._num_masked = None
                self._fraction_masked = float(fraction_masked)
                self._reverse_mask = self._fraction_masked < 0
            else:
                raise ValueError('Exactly one of `num_masked` and `fraction_masked` should be specified.')
            if shift_and_log_scale_fn:
                def _bijector_fn(x0, input_depth, **condition_kwargs):
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

    @property
    def input_depth(self) -> Optional[int]:
        return self._input_depth
        
    @property
    def num_masked(self) -> Optional[int]:
        return self._num_masked
      
    @property
    def fraction_masked(self) -> Optional[float]:
        return self._fraction_masked
        
    @property
    def reverse_mask(self) -> bool:
            return self._reverse_mask
        
    @property
    def masked_size(self) -> int:
        if self.num_masked is not None:
            masked_size = self.num_masked
        else:
            if self.input_depth is not None and self.fraction_masked is not None:
                masked_size = int(np.round(self.input_depth * self.fraction_masked))
            else:
                raise ValueError("Cannot determine the value of `masked_size`.")
        return masked_size
    
    @property
    def bijector_input_units(self) -> int:
        if self.input_depth is not None:
            return self.input_depth - abs(self.masked_size)
        else:
            raise ValueError("Cannot determine the value of `bijector_input_units`.")
    
    def __cache_input_depth(self, 
                            x: tf.Tensor
                           ) -> None:
        if self.input_depth is None:
            self._input_depth = tf.compat.dimension_value(tensorshape_util.with_rank_at_least(x.shape, 1)[-1])
            if self.input_depth is None:
                raise NotImplementedError('Rightmost dimension must be known prior to graph execution.')
            if abs(self._masked_size) >= self._input_depth:
                raise ValueError('Number of masked units {} must be smaller than the event size {}.'.format(self._masked_size, self._input_depth))

    def _forward(self, x, **condition_kwargs):
        self.__cache_input_depth(x)
        x0, x1 = x[..., :self._masked_size], x[..., self._masked_size:]
        if self.reverse_mask:
            x0, x1 = x1, x0
        y1 = self.bijector_fn(x0, self.bijector_input_units, **condition_kwargs).forward(x1)
        if self.reverse_mask:
            y1, x0 = x0, y1
        y = tf.concat([x0, y1], axis=-1)
        return y

    def _inverse(self, y, **condition_kwargs):
        self.__cache_input_depth(y)
        y0, y1 = y[..., :self._masked_size], y[..., self._masked_size:]
        if self._reverse_mask:
            y0, y1 = y1, y0
        x1 = self.bijector_fn(y0, self.bijector_input_units, **condition_kwargs).inverse(y1)
        if self._reverse_mask:
            x1, y0 = y0, x1
        x = tf.concat([y0, x1], axis=-1)
        return x

    def _forward_log_det_jacobian(self, x, **condition_kwargs):
        self.__cache_input_depth(x)
        x0, x1 = x[..., :self._masked_size], x[..., self._masked_size:]
        if self._reverse_mask:
            x0, x1 = x1, x0
        return self.bijector_fn(x0, self.bijector_input_units, **condition_kwargs).forward_log_det_jacobian(x1, event_ndims=1)

    def _inverse_log_det_jacobian(self, y, **condition_kwargs):
        self.__cache_input_depth(y)
        y0, y1 = y[..., :self._masked_size], y[..., self._masked_size:]
        if self._reverse_mask:
            y0, y1 = y1, y0
        return self.bijector_fn(y0, self._bijector_input_units, **condition_kwargs).inverse_log_det_jacobian(y1, event_ndims=1)


class RealNVPChain(BaseChain):
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