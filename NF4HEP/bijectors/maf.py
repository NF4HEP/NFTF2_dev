# Abstract implementation of Masked Autoregressive Flow
 
__all__ = ['MAFNetwork_default',
           'MAFNetwork_custom',
           'MAFBijector_default',
           'MAFBijector_custom',
           'MAFFlow']

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

class MAFNetwork_default(AbstractNeuralNetwork):
    """
    Defines the default MAF NN.
    """
    def __init__(self,
                 model_define_inputs,
                 verbose=True):
        print("Initializing the MAF default NN.")
        super().__init__(model_define_inputs,
                         verbose)
        verbose, verbose_sub = self.get_verbosity(verbose)
        self._model_defint_inputs : Dict

    def define_network(self, verbose=None):
        pass


class MAFNetwork_custom(AbstractNeuralNetwork):
    """
    Defines a custom MAF NN.
    """
    def __init__(self,
                 model_define_inputs,
                 verbose=True):
        print("Initializing the MAF custom NN.")
        super().__init__(model_define_inputs,
                         verbose)
        verbose, verbose_sub = self.get_verbosity(verbose)
        self._model_defint_inputs : Dict

    def define_network(self, verbose=None):
        pass


class MAFBijector_default(AbstractBijector):
    """
    """
    def __init__(self,
                 model_define_inputs,
                 model_bijector_inputs,
                 verbose=True):
        print("Initializing the MAF default Bijector.")
        super().__init__(model_define_inputs,
                         model_bijector_inputs,
                         verbose)
        verbose, verbose_sub = self.get_verbosity(verbose)
        self._model_defint_inputs : Dict
        self._model_bijector_inputs : Dict
        self.NN : Union[MAFNetwork_default,MAFNetwork_custom]

    def define_bijector(self, verbose = None):
        pass


class MAFBijector_custom(AbstractBijector):
    """
    """
    def __init__(self,
                 model_define_inputs,
                 model_bijector_inputs,
                 verbose=True):
        print("Initializing the MAF custom Bijector.")
        super().__init__(model_define_inputs,
                         model_bijector_inputs,
                         verbose)
        verbose, verbose_sub = self.get_verbosity(verbose)
        self._model_defint_inputs : Dict
        self._model_bijector_inputs : Dict
        self.NN : Union[MAFNetwork_default,MAFNetwork_custom]

    def define_bijector(self, verbose = None):
        pass


class MAFChain(Flow_base):
    name = "MAF"
    """
    """
    def __init__(self,
                 model_define_inputs,
                 model_bijector_inputs,
                 verbose=True):
        print("Initializing the MAF custom Bijector.")
        super().__init__(model_define_inputs,
                         model_bijector_inputs,
                         verbose)
        verbose, verbose_sub = self.get_verbosity(verbose)
        self._model_defint_inputs : Dict
        self._model_bijector_inputs : Dict
        self.NN : Union[MAFNetwork_default,MAFNetwork_custom]
        self.Bijector : Union[MAFBijector_default,MAFBijector_custom]
        self.Flow : tfp.bijectors.Chain

    def define_flow(self, verbose = None):
        pass