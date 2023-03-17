__all__ = ['ARQSplineNetwork_default',
           'ARQSplineNetwork_custom',
           'ARQSplineBijector_default',
           'ARQSplineBijector_custom',
           'ARQSplineFlow']

from typing import Union, List, Dict, Callable, Tuple, Optional, NewType, Type, Generic, Any, TypeVar, TYPE_CHECKING
from typing_extensions import TypeAlias
from NF4HEP.utils.custom_types import Array, ArrayInt, ArrayStr, DataType, StrPath, IntBool, StrBool, StrList, StrArray, FigDict, LogPredDict, Number, DTypeStr, DTypeStrList, DictStr

import numpy as np
import tensorflow as tf
import tensorflow as tf
# import tensorflow.compat.v1 as tf1 
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.layers import Layer
import tensorflow_probability as tfp
tfd = tfp.distributions
tfb = tfp.bijectors

from NF4HEP.utils.verbosity import Verbosity, print

from rqs import RationalQuadraticSpline

class ARQSplineNetwork_default(tfp.bijectors.AutoregressiveNetwork, Verbosity):
    """
    """
    def __init__(self,
                 model_define_inputs,
                 verbose=True):
        print("Initializing the ARQSpline default NN.")
        verbose, verbose_sub = self.get_verbosity(verbose)
        self._model_defint_inputs = model_define_inputs
        self.define_network()
        
        self.model_define_inputs = model_define_inputs

    def define_network(self, verbose=None):
        pass


class ARQSplineNetwork_custom(Layer, Verbosity):
    """
    Defines a custom MAF network
    """

    def __init__(self,
                 model_define_inputs,
                 verbose=True):
        pass
        #super(ARQSplineNetwork_custom, self).__init__(**self.model_define_inputs["kwargs"]) ##passing additional keyword erguments to Layer


class ARQSplineBijector_default(tfp.bijectors.MaskedAutoregressiveFlow, Verbosity):
    """
    """
    def __init__(self,
                 model_define_inputs,
                 model_bijector_inputs,
                 verbose=True):
        self.NN : Model

    def define_bijector(self, verbose = None):
        verbose, verbose_sub = self.get_verbosity(verbose)
        self.Bijector = tfb.MaskedAutoregressiveFlow(shift_and_log_scale_fn=self.NN)


class ARQSplineBijector_custom(tfp.bijectors.Bijector, Verbosity):
    """
    """
    def __init__(self,
                 model_define_inputs,
                 model_bijector_inputs,
                 verbose=True):
        self.NN : Model

    def define_bijector(self, verbose = None):
        verbose, verbose_sub = self.get_verbosity(verbose)
        self.Bijector = tfb.MaskedAutoregressiveFlow(shift_and_log_scale_fn=self.NN)


class ARQSplineFlow(tfp.bijectors.Chain, Verbosity):
    """
    """
    def __init__(self,
                 model_define_inputs,
                 model_bijector_inputs,
                 verbose=True):
        self.Bijector : Union[ARQSplineBijector_default,ARQSplineBijector_custom]
        self.Flow : tfp.bijectors.Chain

    def define_flow(self, verbose = None):
        pass