__all__ = ['RQSplineNetwork','RQSplineBijector','RQSplineFlow']

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions
tfb= tfp.bijectors

from . import inference, utils#, custom_losses
from .show_prints import Verbosity, print

class RQSplineNetwork(Verbosity):
    """
    """

    def define_network(self, verbose=None):
        verbose, verbose_sub = self.set_verbosity(verbose)
        self.NN = tfb.AutoregressiveNetwork(**self.model_define_inputs)

class RQSplineBijector(RQSplineNetwork):
    """
    """
    def define_bijector(self, verbose = None):
        verbose, verbose_sub = self.set_verbosity(verbose)
        self.Bijector = tfb.MaskedAutoregressiveFlow(shift_and_log_scale_fn=self.NN)

class RQSplineFlow(RQSplineBijector):
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

class RQSpline(RQSplineFlow):
    """
    """
    def __init__(self,
                 model_define_inputs = None,
                 model_flow_inputs = None,
                 verbose = True):
        self.verbose = verbose
        verbose, verbose_sub = self.set_verbosity(verbose)
        self.__model_define_inputs = model_define_inputs
        self.__model_flow_inputs = model_flow_inputs
        # setting inputs
        self.__set_inputs(verbose = verbose_sub)
        # define network through the __define_network method of MAFNetwork
        self.define_network()
        # ora ho self.NN -> define bijector through the __define_bijector method of MAFBijector
        self.define_bijector()
        # ora ho self.Bijector -> define flow through the __define_flow method of MAFFlow
        self.define_flow()

    def __set_inputs(self, verbose=None):
        self.model_define_inputs = utils.dic_minus_keys(self.__model_define_inputs, ["batch_norm"])
        self.batch_norm = self.__model_define_inputs["batch_norm"]
        self.ndims = self.__model_flow_inputs["ndims"]
        self.num_bijectors = self.__model_flow_inputs["num_bijectors"]