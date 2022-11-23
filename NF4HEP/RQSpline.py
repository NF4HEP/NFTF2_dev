__all__ = ['RQSplineNetwork_default',
           'RQSplineNetwork_custom',
           'RQSplineBijector_default',
           'RQSplineBijector_custom',
           'RQSplineFlow']

import numpy as np
import tensorflow as tf
import tensorflow as tf
import tensorflow.compat.v1 as tf1 #type: ignore
from tensorflow.keras.layers import Layer #type: ignore
import tensorflow_probability as tfp
tfd = tfp.distributions
tfb = tfp.bijectors

from . import inference, utils#, custom_losses
from .verbosity import Verbosity, print

class RQSplineNetwork_default(tfb.AutoregressiveNetwork, Verbosity):
    """
    """

    def define_network(self, verbose=None):
        verbose, verbose_sub = self.set_verbosity(verbose)
        self.NN = tfb.AutoregressiveNetwork(**self.model_define_inputs)

class RQSplineNetwork_custom(Layer, Verbosity):
    """
    Defines a custom MAF network
    """

    def __init__(self,
                 model_define_inputs,
                 verbose=True):
        pass
        #super(RQSplineNetwork_custom, self).__init__(**self.model_define_inputs["kwargs"]) ##passing additional keyword erguments to Layer

class RQSplineBijector_default(tfb.MaskedAutoregressiveFlow, Verbosity):
    """
    """
    def define_bijector(self, verbose = None):
        verbose, verbose_sub = self.set_verbosity(verbose)
        self.Bijector = tfb.MaskedAutoregressiveFlow(shift_and_log_scale_fn=self.NN)

class RQSplineBijector_custom(tfb.Bijector, Verbosity):
    """
    """
    def define_bijector(self, verbose = None):
        verbose, verbose_sub = self.set_verbosity(verbose)
        self.Bijector = tfb.MaskedAutoregressiveFlow(shift_and_log_scale_fn=self.NN)

class RQSplineFlow(tfb.Chain, Verbosity):
    """
    """
    def define_flow(self, verbose = None):
        verbose, verbose_sub = self.set_verbosity(verbose)
        permutation = tf.cast(np.concatenate((np.arange(int(self.ndims/2), self.ndims), np.arange(0, int(self.ndims/2)))), tf.int32)
        Permute = tfb.Permute(permutation=permutation)
        #Permute._dtype = tf.dtypes.as_dtype(self.dtype)
        bijectors = []
        for _ in range(self.num_bijectors):
            if self.batch_norm:
                bijectors.append(tfb.BatchNormalization())
            bijectors.append(self.Bijector)
            bijectors.append(Permute)
        self.Flow = tfb.Chain(list(reversed(bijectors[:-1])), name='maf_chain')