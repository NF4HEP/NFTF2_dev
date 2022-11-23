# Abstract implementation of Coupling Rational Quadratic Spline Flow
 
__all__ = ['CRQSplineNetwork_default',
           'CRQSplineNetwork_custom',
           'CRQSplineBijector_default',
           'CRQSplineBijector_custom',
           'CRQSplineFlow']

import numpy as np
import tensorflow as tf
import tensorflow as tf
import tensorflow.compat.v1 as tf1 #type: ignore
from tensorflow.keras.models import Model # type: ignore
from tensorflow.keras.layers import Layer #type: ignore
import tensorflow_probability as tfp
tfd = tfp.distributions
tfb = tfp.bijectors

from typing import Union, List, Dict, Callable, Tuple, Optional, NewType, Type, Generic, Any, TypeVar, TYPE_CHECKING
from typing_extensions import TypeAlias
from NF4HEP.utils.custom_types import Array, ArrayInt, ArrayStr, DataType, StrPath, IntBool, StrBool, StrList, FigDict, LogPredDict, Number, DTypeStr, DTypeStrList, DictStr
from NF4HEP.utils.verbosity import print, Verbosity
from NF4HEP.bijectors.bijectors_base import AbstractNeuralNetwork
from NF4HEP.bijectors.bijectors_base import AbstractBijector
from NF4HEP.bijectors.bijectors_base import Flow_base

class CRQSplineNetwork_default(AbstractNeuralNetwork):
    """
    Defines the default CRQSpline NN.
    """
    def __init__(self,
                 model_define_inputs,
                 verbose=True):
        print("Initializing the CRQSpline default NN.")
        super().__init__(model_define_inputs,
                         verbose)
        verbose, verbose_sub = self.get_verbosity(verbose)
        self._model_defint_inputs : Dict

    def define_network(self, verbose=None):
        pass


class CRQSplineNetwork_custom(AbstractNeuralNetwork):
    """
    Defines a custom CRQSpline NN.
    """
    def __init__(self,
                 model_define_inputs,
                 verbose=True):
        print("Initializing the CRQSpline custom NN.")
        super().__init__(model_define_inputs,
                         verbose)
        verbose, verbose_sub = self.get_verbosity(verbose)
        self._model_defint_inputs : Dict

    def define_network(self, verbose=None):
        pass


class CRQSplineBijector_default(AbstractBijector):
    """
    """
    def __init__(self,
                 model_define_inputs,
                 model_bijector_inputs,
                 verbose=True):
        print("Initializing the CRQSpline default Bijector.")
        super().__init__(model_define_inputs,
                         model_bijector_inputs,
                         verbose)
        verbose, verbose_sub = self.get_verbosity(verbose)
        self._model_defint_inputs : Dict
        self._model_bijector_inputs : Dict
        self.NN : Union[CRQSplineNetwork_default,CRQSplineNetwork_custom]

    def define_bijector(self, verbose = None):
        pass


class CRQSplineBijector_custom(AbstractBijector):
    """
    """
    def __init__(self,
                 model_define_inputs,
                 model_bijector_inputs,
                 verbose=True):
        print("Initializing the CRQSpline custom Bijector.")
        super().__init__(model_define_inputs,
                         model_bijector_inputs,
                         verbose)
        verbose, verbose_sub = self.get_verbosity(verbose)
        self._model_defint_inputs : Dict
        self._model_bijector_inputs : Dict
        self.NN : Union[CRQSplineNetwork_default,CRQSplineNetwork_custom]

    def define_bijector(self, verbose = None):
        pass


class CRQSplineChain(Flow_base):
    name = "C-RQSpline"
    """
    """
    def __init__(self,
                 model_define_inputs,
                 model_bijector_inputs,
                 verbose=True):
        print("Initializing the CRQSpline custom Bijector.")
        super().__init__(model_define_inputs,
                         model_bijector_inputs,
                         verbose)
        verbose, verbose_sub = self.get_verbosity(verbose)
        self._model_defint_inputs : Dict
        self._model_bijector_inputs : Dict
        self.NN : Union[CRQSplineNetwork_default,CRQSplineNetwork_custom]
        self.Bijector : Union[CRQSplineBijector_default,CRQSplineBijector_custom]
        self.Flow : tfp.bijectors.Chain

    def define_flow(self, verbose = None):
        pass