# Base implementation of Network, Bijector, Chain objects

__all__ = ['BaseNetwork',
           'BaseBijector'
           ]

from abc import ABC, abstractmethod
from numpy import typing as npt
from pathlib import Path

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
from NF4HEP.utils import utils

header_string_1 = "=============================="
header_string_2 = "------------------------------"

 
class BaseNetwork(Layer, Verbosity):
    name: str
    """
    Base class for the Normalizing Flow NN.
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
        # Initialise parent Verbosity class
        Verbosity.__init__(self, verbose)
        # Set verbosity
        verbose, _ = self.get_verbosity(verbose)
        # Initialise parent Layer and Verbosity classes
        Layer.__init__(self)
        self._model_define_inputs = model_define_inputs
        #Layer.__init__(self)
        # Initialize object

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

    def get_config(self):
        config = Layer.get_config(self)
        config.update({"model_define_inputs": self.model_define_inputs,
                       "verbose": self.verbose})
        return config

    #def from_config(self, config):
    #    return self(**config)


class BaseBijector(tfb.Bijector): # type: ignore
    name: str
    """
    Base class for the Normalizing Flow Bijector.
    """
    def __init__(self,
                 nn: Union["ARQSplineNetwork", "CRQSplineNetwork", "MAFNetwork", "RealNVPNetwork"], # type: ignore
                 model_bijector_inputs: Dict[str, Any]
                ) -> None:
        # Attributes type declarations
        self._Model: Model
        self._model_bijector_inputs: Dict[str, Any]
        self._ndims: int
        self._NN: Union["ARQSplineNetwork", "CRQSplineNetwork", "MAFNetwork", "RealNVPNetwork"] # type: ignore
        # Initialize parent Bijector class
        tfb.Bijector.__init__(self, **model_bijector_inputs)
        # Initialize object
        self._NN = nn

    @property
    def Model(self) -> Model:
        return self._Model

    @property
    def bijector_kwargs(self) -> Dict[str, Any]:
        return self._bijector_kwargs

    @property
    def model_bijector_inputs(self) -> Dict[str, Any]:
        return self._model_bijector_inputs

    @property
    def ndims(self) -> int:
        return self._ndims