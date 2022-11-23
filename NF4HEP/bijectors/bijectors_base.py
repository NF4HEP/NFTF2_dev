# Base classes for Flow definitions

from abc import ABC, abstractmethod
import numpy as np
from numpy import typing as npt
from pathlib import Path

from typing import Union, List, Dict, Callable, Tuple, Optional, NewType, Type, Generic, Any, TypeVar, TYPE_CHECKING
from typing_extensions import TypeAlias
from NF4HEP.utils.custom_types import Array, ArrayInt, ArrayStr, DataType, StrPath, IntBool, StrBool, StrList, FigDict, LogPredDict, Number, DTypeStr, DTypeStrList, DictStr
from NF4HEP.utils.verbosity import print, Verbosity

header_string = "=============================="
footer_string = "------------------------------"

 
class AbstractNeuralNetwork(ABC,Verbosity):
    """
    Abstract class for the Normalizing Flow NN.
    """
    def __init__(self,
                 model_define_inputs: Dict,
                 verbose: Optional[IntBool] = None
                ) -> None:
        ABC.__init__(self)
        Verbosity.__init__(self, verbose)
        verbose, verbose_sub = self.get_verbosity(verbose)
        print("Initializing the Neural Network base class", show = verbose)
        self._model_defint_inputs = model_define_inputs

    @abstractmethod
    def define_neural_network(self) -> None:
        pass

class AbstractBijector(ABC,Verbosity):
    """
    Abstract class for the Normalizing Flow Bijector.
    """
    def __init__(self,
                 model_define_inputs: Dict,
                 model_bijector_inputs: Dict,
                 verbose: Optional[IntBool] = None
                ) -> None:
        ABC.__init__(self)
        Verbosity.__init__(self, verbose)
        verbose, verbose_sub = self.get_verbosity(verbose)
        print("Initializing the Neural Network base class", show = verbose)
        self._model_defint_inputs = model_define_inputs
        self._model_bijector_inputs = model_bijector_inputs

    @abstractmethod
    def define_bijector(self) -> None:
        pass

class Flow_base(Verbosity):
    """
    Abstract class for the Normalizing Flow Chain.
    """
    def __init__(self,
                 model_define_inputs: Dict,
                 model_bijector_inputs: Dict,
                 verbose: Optional[IntBool] = None
                ) -> None:
        super().__init__(verbose)
        verbose, verbose_sub = self.get_verbosity(verbose)
        print("Initializing the Bijectors Chain base class", show = verbose)
        self._model_defint_inputs = model_define_inputs
        self._model_bijector_inputs = model_bijector_inputs

    @abstractmethod
    def validate_model_define_inputs(self) -> None:
        pass

    @abstractmethod
    def validate_model_bijector_inputs(self) -> None:
        pass

    @abstractmethod
    def define_flow(self) -> None:
        pass