# Abstract implementation of Coupling Rational Quadratic Spline Flow
 
__all__ = ['CRQSplineNetwork',
           'CRQSplineBijector'
           ]

import numpy as np
import tensorflow as tf #type: ignore
import tensorflow.compat.v1 as tf1 #type: ignore
from tensorflow.keras.models import Model # type: ignore
from tensorflow.keras.layers import Layer #type: ignore
import tensorflow_probability as tfp #type: ignore
tfd = tfp.distributions
tfb = tfp.bijectors

from typing import Union, List, Dict, Callable, Tuple, Optional, NewType, Type, Generic, Any, TypeVar, TYPE_CHECKING
from typing_extensions import TypeAlias
from NF4HEP.utils.custom_types import Array, ArrayInt, ArrayStr, DataType, StrPath, IntBool, StrBool, StrList, FigDict, LogPredDict, Number, DTypeStr, DTypeStrList, DictStr
from NF4HEP.utils.verbosity import print, Verbosity
from NF4HEP.utils import utils
from NF4HEP.bijectors.base import BaseNetwork, BaseBijector

header_string_1 = "=============================="
header_string_2 = "------------------------------"

class CRQSplineNetwork(BaseNetwork, Verbosity):
    name = "CRQSplineNetwork"
    """
    """
    def __init__(self,
                 model_define_inputs: Dict[str, Any],
                 verbose: Optional[IntBool] = None
                ) -> None:
        # Attributes type declarations (from parent FileManager class)
        self._batch_norm: StrBool
        self._dropout_rate: Union[np.float_,str]
        self._hidden_layers: List[Any]
        self._layers: List[Layer]
        self._layers_string: List[str]
        self._model_define_inputs: Dict[str, Any]
        self._ndims: int
        # Attributes type declarations
        #
        # Initialise parent Verbosity class
        Verbosity.__init__(self, verbose)
        # Set verbosity
        verbose, _ = self.get_verbosity(verbose)
        # Set inputs and initialise parent BaseNetwork class
        print(header_string_1, "\nInitializing CRQSplineNetwork object.\n", show = verbose)
        self.__set_model_define_inputs(model_define_inputs = model_define_inputs, verbose = verbose)
        super().__init__()
        # Initialize object
        #self.__set_layers()


    def __set_model_define_inputs(self,
                                  model_define_inputs: Dict[str, Any],
                                  verbose: Optional[IntBool] = None
                                 ) -> None:
        verbose, verbose_sub = self.get_verbosity(verbose)

class CRQSplineBijector(BaseBijector, Verbosity):
    name = "CRQSplineBijector"
    """
    """
    def __init__(self,
                 model_define_inputs: Dict[str, Any],
                 model_bijector_inputs: Dict[str, Any],
                 verbose: Optional[IntBool] = None
                ) -> None:
        # Attributes type declarations (from parent FileManager class)
        self._Model: Model
        self._model_bijector_inputs: Dict[str, Any]
        self._ndims: int
        self._NN: CRQSplineNetwork
        # Attributes type declarations
        self._rem_dims: int
        self._tran_ndims: int
        # Initialise parent Verbosity class
        Verbosity.__init__(self, verbose)
        # Set verbosity
        verbose, _ = self.get_verbosity(verbose)
        # Set inputs and initialise parent BaseBijector class
        print(header_string_1, "\nInitializing CRQSplineBijector object.\n", show = verbose)
        self.__set_model_bijector_inputs(model_bijector_inputs = model_bijector_inputs, verbose = verbose)
        nn = CRQSplineNetwork(model_define_inputs)
        super().__init__(nn = nn, model_bijector_inputs = self._bijector_kwargs)
        # Initialize object

    @property
    def NN(self) -> CRQSplineNetwork:
        return self._NN

    @NN.setter
    def NN(self,
           nn: CRQSplineNetwork
          ) -> None:
        self._NN = nn
        self._ndims = self._NN._ndims
        self._Model = None

    def __set_model_bijector_inputs(self,
                                    model_bijector_inputs: Dict[str, Any],
                                    verbose: Optional[IntBool] = None
                                   ) -> None:
        verbose, verbose_sub = self.get_verbosity(verbose)

    def _bijector_fn(self, x):
        pass

    def _forward(self, x):
        pass

    def _inverse(self, y):
        pass

    def _forward_log_det_jacobian(self, x):
        pass

    def _inverse_log_det_jacobian(self, y):
        pass

#class CRQSplineNetwork_default(AbstractNeuralNetwork):
#    """
#    Defines the default CRQSpline NN.
#    """
#    def __init__(self,
#                 model_define_inputs,
#                 verbose=True):
#        print("Initializing the CRQSpline default NN.")
#        super().__init__(model_define_inputs,
#                         verbose)
#        verbose, verbose_sub = self.get_verbosity(verbose)
#        self._model_defint_inputs : Dict
#
#    def define_network(self, verbose=None):
#        pass
#
#
#class CRQSplineNetwork_custom(AbstractNeuralNetwork):
#    """
#    Defines a custom CRQSpline NN.
#    """
#    def __init__(self,
#                 model_define_inputs,
#                 verbose=True):
#        print("Initializing the CRQSpline custom NN.")
#        super().__init__(model_define_inputs,
#                         verbose)
#        verbose, verbose_sub = self.get_verbosity(verbose)
#        self._model_defint_inputs : Dict
#
#    def define_network(self, verbose=None):
#        pass
#
#
#class CRQSplineBijector_default(AbstractBijector):
#    """
#    """
#    def __init__(self,
#                 model_define_inputs,
#                 model_bijector_inputs,
#                 verbose=True):
#        print("Initializing the CRQSpline default Bijector.")
#        super().__init__(model_define_inputs,
#                         model_bijector_inputs,
#                         verbose)
#        verbose, verbose_sub = self.get_verbosity(verbose)
#        self._model_defint_inputs : Dict
#        self._model_bijector_inputs : Dict
#        self.NN : Union[CRQSplineNetwork_default,CRQSplineNetwork_custom]
#
#    def define_bijector(self, verbose = None):
#        pass
#
#
#class CRQSplineBijector_custom(AbstractBijector):
#    """
#    """
#    def __init__(self,
#                 model_define_inputs,
#                 model_bijector_inputs,
#                 verbose=True):
#        print("Initializing the CRQSpline custom Bijector.")
#        super().__init__(model_define_inputs,
#                         model_bijector_inputs,
#                         verbose)
#        verbose, verbose_sub = self.get_verbosity(verbose)
#        self._model_defint_inputs : Dict
#        self._model_bijector_inputs : Dict
#        self.NN : Union[CRQSplineNetwork_default,CRQSplineNetwork_custom]
#
#    def define_bijector(self, verbose = None):
#        pass
