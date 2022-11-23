__all__ = ["Distributions"]

from sklearn import datasets
from sklearn.preprocessing import StandardScaler
import numpy as np
from datetime import datetime
import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np

tfd = tfp.distributions

from .show_prints import Verbosity, print
from . import utils

header_string = "=============================="
footer_string = "------------------------------"

class Distributions(Verbosity):
    """
    """
    def __init__(self,
                 ndims = None,
                 dtype = None,
                 default_dist = "Normal",
                 tf_dist = None,
                 verbose = True):
        """
        """
        self.verbose = verbose
        verbose, verbose_sub = self.set_verbosity(verbose)
        timestamp = "datetime_" + datetime.now().strftime("%Y_%m_%d_%H_%M_%S_%fZ")[:-3]
        self.log = {timestamp: {"action": "created"}}
        self.supported_default_base_distributions = ["Normal"]
        self.ndims = ndims
        if dtype == None:
            self.dtype = "float32"
        else:
            self.dtype = dtype
        self.default_dist = default_dist
        self.tf_dist = tf_dist
        self.__check_inputs()
        if self.default_dist is not None:
            self.__set_base_default_distribution()
        else:
            self.__set_base_tf_distribution()

    def __check_inputs(self):
        if self.ndims is None:
            raise Exception("You have to specify the 'ndims' argument.")
        if self.default_dist is not None and self.default_dist not in self.supported_default_base_distributions:
            raise Exception("The distribution "+self.default_dist+" is not a supported default base distribution.")
        if self.default_dist is None and self.tf_dist is None:
            raise Exception("You have to specify at least one of the 'default_dist' (and related arguments) and 'tf_dist' arguments.")

    def __set_base_default_distribution(self,verbose=None):
        verbose, verbose_sub = self.set_verbosity(verbose)
        print(header_string,"\nSetting base distribution (from default distributions)\n",show=verbose)
        self.base_distribution = None
        default_dist = "self."+ self.default_dist + "()"
        dist_string, dist = eval(default_dist)
        self.base_distribution_string = "tfd.Sample("+dist_string+",sample_shape=["+str(self.ndims)+"])"
        self.base_distribution = tfd.Sample(dist,sample_shape=[self.ndims])
        print("Base distribution set to:", self.base_distribution_string, ".\n", show=verbose)
        timestamp = "datetime_"+datetime.now().strftime("%Y_%m_%d_%H_%M_%S_%fZ")[:-3]
        self.log[timestamp] = {"action": "base distribution set",
                               "distribution": self.base_distribution_string}

    def __set_base_tf_distribution(self,verbose=None):
        verbose, verbose_sub = self.set_verbosity(verbose)
        print(header_string,"\nSetting base distribution (from tfp distributions)\n",show=verbose)
        self.base_distribution = None
        if type(self.tf_dist) == str:
            if "(" in self.tf_dist:
                try:
                    eval("tfd." + self.tf_dist.replace("tfd.", ""))
                    dist_string = "tfd." + self.tf_dist.replace("tfd.", "")
                except:
                    eval(self.tf_dist)
                    dist_string = self.tf_dist
            else:
                try:
                    eval("tfd." + self.tf_dist.replace("tfd.", "") +"()")
                    dist_string = "tfd." + self.tf_dist.replace("tfd.", "") +"()"
                except:
                    eval(self.tf_dist +"()")
                    dist_string = self.tf_dist +"()"
        elif type(self.tf_dist) == dict:
            try:
                name = self.tf_dist["name"]
            except:
                raise Exception("The optimizer ", str(self.tf_dist), " has unspecified name.")
            try:
                args = self.tf_dist["args"]
            except:
                args = []
            try:
                kwargs = self.tf_dist["kwargs"]
            except:
                kwargs = {}
            dist_string = utils.build_method_string_from_dict("tfd", name, args, kwargs)
        else:
            raise Exception("Could not set base distribution. The 'tf_dist' input argument does not have a valid format (str or dict).")
        try:
            dist = eval(dist_string)
            self.base_distribution_string = "tfd.Sample("+dist_string+",sample_shape=["+str(self.ndims)+"])"
            self.base_distribution = tfd.Sample(dist,sample_shape=[self.ndims])
            print("Base distribution set to:", self.base_distribution_string, ".\n", show=verbose)
        except Exception as e:
            print(e)
            raise Exception("Could not set base distribution", dist_string, "\n")
        timestamp = "datetime_"+datetime.now().strftime("%Y_%m_%d_%H_%M_%S_%fZ")[:-3]
        self.log[timestamp] = {"action": "base distribution set",
                               "distribution": self.base_distribution_string}

    def Normal(self):
        dist_string = "tfd.Normal(loc=np.array(0,dtype='"+self.dtype+"'), scale=1, allow_nan_stats=False)"
        dist=eval(dist_string)
        return [dist_string, dist]

    def Normal_mixture(self):
        ### Mix_gauss will be our target distribution.
        dist_string = "tfd.Mixture(cat=tfd.Categorical(probs=[0.3,.7]),components=[tfd.Normal(loc=np.array(3.3,dtype='"+self.dtype+"'), scale=0.4),tfd.Normal(loc=np.array(1.8,dtype='"+self.dtype+"'), scale=0.2)])"
        dist=eval(dist_string)
        return [dist_string, dist]
    




