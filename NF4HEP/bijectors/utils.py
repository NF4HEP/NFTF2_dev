__all__ = ["_list"]

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