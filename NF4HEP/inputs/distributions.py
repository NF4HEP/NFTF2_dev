__all__ = ["DefaultDistributions",
           "Distribution"]

from sklearn import datasets # type: ignore
from sklearn.preprocessing import StandardScaler # type: ignore
import numpy as np
from datetime import datetime
import tensorflow as tf # type: ignore
import tensorflow_probability as tfp # type: ignore
import numpy as np
from pathlib import Path
from numpy import typing as npt

tfd = tfp.distributions

from typing import Union, List, Dict, Callable, Tuple, Optional, NewType, Type, Generic, Any, TypeVar, TYPE_CHECKING
from typing_extensions import TypeAlias
from NF4HEP.utils.custom_types import Array, ArrayInt, ArrayStr, DataType, StrPath, IntBool, StrBool, StrList, FigDict, LogPredDict, Number, DTypeStr, DTypeStrList, DictStr
from NF4HEP.base import Name, FileManager, ParsManager, PredictionsManager, FiguresManager, Inference, Plotter
from NF4HEP.utils import mplstyle_path
from NF4HEP import print
from NF4HEP.utils.verbosity import Verbosity
from NF4HEP.utils import corner
from NF4HEP.utils import utils

header_string = "=============================="
footer_string = "------------------------------"

class DefaultDistributions():
    """
    """
    def __init__(self):
        pass

    def MixNormal1(self,
                   n_components: int = 3,
                   n_dimensions: int = 4,
                   seed: Optional[int] = None,
                  ) -> tfp.distributions.Mixture:
        """
        Defines a mixture of 'n_components' Normal distributions in 'n_dimensions' dimensions 
        with means and stddevs given by the tensors 'loc' and 'scale' with shapes 
        '(n_components,n_dimensions)'.
        The components are mixed according to the categorical distribution with probabilities
        'probs' (with shape equal to that of 'loc' and 'scale'). This means that each component in each
        dimension can be assigned a different probability.

        The resulting multivariate distribution has small correlation.

        Note: The functions 'MixNormal1' and 'MixNormal1_indep'
        generate identical samples, different from the samples generated by
        'MixNormal2' and 'MixNormal2_indep' (also identical).
        """
        if seed is None:
            seed = self._seed
        np.random.seed(seed)
        loc = np.random.sample([n_components,n_dimensions])*10
        #loc = [[1.,4.,7.,10.],[2.,5.,8.,11.],[3.,6.,9.,12.]]
        scale = np.random.sample([n_components,n_dimensions])
        #scale = np.full([n_components,n_dimensions],0.1)
        probs = np.random.sample([n_dimensions,n_components])
        #probs = np.full([n_dimensions,n_components],1.)
        components = []
        for i in range(n_components):
            components.append(tfd.Normal(loc=loc[i],scale=scale[i]))
        mix_gauss=tfd.Mixture(
            cat=tfd.Categorical(probs=probs),
            components=components,
            validate_args=True)
        return mix_gauss

    def MixNormal2(self,
                   n_components: int = 3,
                   n_dimensions: int = 4,
                   seed: Optional[int] = None,
                  ) -> tfp.distributions.MixtureSameFamily:
        """
        Defines a mixture of 'n_components' Normal distributions in 'n_dimensions' dimensions 
        with means and stddevs given by the tensors 'loc' and 'scale' with shapes 
        '(n_components,n_dimensions)'.
        The components are mixed according to the categorical distribution with probabilities
        'probs' (with shape equal to 'n_components'). This means that each component in all
        dimension is assigned a single probability.

        The resulting multivariate distribution has small correlation.

        Note: The functions 'MixNormal1' and 'MixNormal1_indep'
        generate identical samples, different from the samples generated by
        'MixNormal2' and 'MixNormal2_indep' (also identical).
        """
        if seed is None:
            seed = self._seed
        np.random.seed(seed)
        loc = np.transpose(np.random.sample([n_components,n_dimensions])*10)
        #loc = np.transpose([[1.,4.,7.,10.],[2.,5.,8.,11.],[3.,6.,9.,12.]])
        scale = np.transpose(np.random.sample([n_components,n_dimensions]))
        #scale = np.transpose(np.full([n_components,n_dimensions],0.1))
        probs = np.random.sample(n_components)
        #probs = np.full(n_components,1.)
        mix_gauss=tfd.MixtureSameFamily(
            mixture_distribution=tfd.Categorical(probs=probs),
            components_distribution=tfd.Normal(loc=loc,scale=scale),
            validate_args=True)
        return mix_gauss

    def MixNormal1_indep(self,
                         n_components: int = 3,
                         n_dimensions: int = 4,
                         seed: Optional[int] = None,
                        ) -> tfp.distributions.Independent:
        """
        Defines a mixture of 'n_components' Normal distributions in 'n_dimensions' dimensions 
        with means and stddevs given by the tensors 'loc' and 'scale' with shapes 
        '(n_components,n_dimensions)'.
        The components are mixed according to the categorical distribution with probabilities
        'probs' (with shape equal to that of 'loc' and 'scale'). This means that each component in each
        dimension can be assigned a different probability.

        The resulting multivariate distribution has small correlation.

        Note: The functions 'MixNormal1' and 'MixNormal1_indep'
        generate identical samples, different from the samples generated by
        'MixNormal2' and 'MixNormal2_indep' (also identical).
        """
        if seed is None:
            seed = self._seed
        np.random.seed(seed)
        loc = np.random.sample([n_components,n_dimensions])*10
        #loc = [[1.,4.,7.,10.],[2.,5.,8.,11.],[3.,6.,9.,12.]]
        scale = np.random.sample([n_components,n_dimensions])
        #scale = np.full([n_components,n_dimensions],0.1)
        probs = np.random.sample([n_dimensions,n_components])
        #probs = np.full([n_dimensions,n_components],1.)
        components = []
        for i in range(n_components):
            components.append(tfd.Normal(loc=loc[i],scale=scale[i]))
        mix_gauss=tfd.Independent(
        distribution=tfd.Mixture(
            cat=tfd.Categorical(probs=probs),
            components=components,
            validate_args=True),
        reinterpreted_batch_ndims=0)
        return mix_gauss

    def MixNormal2_indep(self,
                         n_components: int = 3,
                         n_dimensions: int = 4,
                         seed: Optional[int] = None,
                        ) -> tfp.distributions.Independent:
        """
        Defines a mixture of 'n_components' Normal distributions in 'n_dimensions' dimensions 
        with means and stddevs given by the tensors 'loc' and 'scale' with shapes 
        '(n_components,n_dimensions)'.
        The components are mixed according to the categorical distribution with probabilities
        'probs' (with shape equal to 'n_components'). This means that each component in all
        dimension is assigned a single probability.

        The resulting multivariate distribution has small correlation.

        Note: The functions 'MixNormal1' and 'MixNormal1_indep'
        generate identical samples, different from the samples generated by
        'MixNormal2' and 'MixNormal2_indep' (also identical).
        """
        if seed is None:
            seed = self._seed
        np.random.seed(seed)
        loc = np.transpose(np.random.sample([n_components,n_dimensions])*10)
        #loc = np.transpose([[1.,4.,7.,10.],[2.,5.,8.,11.],[3.,6.,9.,12.]])
        scale = np.transpose(np.random.sample([n_components,n_dimensions]))
        #scale = np.transpose(np.full([n_components,n_dimensions],0.1))
        probs = np.random.sample(n_components)
        #probs = np.full(n_components,1.)
        mix_gauss=tfd.Independent(
            distribution=tfd.MixtureSameFamily(
                mixture_distribution=tfd.Categorical(probs=probs),
                components_distribution=tfd.Normal(loc=loc,scale=scale),
                validate_args=True),
            reinterpreted_batch_ndims=0)
        return mix_gauss

    def MixMultiNormal1(self,
                        n_components: int = 3,
                        n_dimensions: int = 4,
                        seed: Optional[int] = None,
                       ) -> tfp.distributions.Mixture:
        """
        Defines a mixture of 'n_components' Multivariate Normal distributions in 'n_dimensions' dimensions 
        with means and stddevs given by the tensors 'loc' and 'scale' with shapes 
        '(n_components,n_dimensions)'.
        The components are mixed according to the categorical distribution with probabilities
        'probs' (with shape equal to 'n_components'). This means that each Multivariate distribution 
        is assigned a single probability.

        The resulting multivariate distribution has large (random) correlation.

        Note: The functions 'MixMultiNormal1' and 'MixMultiNormal1_indep'
        generate identical samples, different from the samples generated by
        'MixMultiNormal2' and 'MixMultiNormal2_indep' (also identical).
        """
        if seed is None:
            seed = self._seed
        np.random.seed(seed)
        loc = np.random.sample([n_components,n_dimensions])*10
        scale = np.random.sample([n_components,n_dimensions])
        probs = np.random.sample(n_components)
        components = []
        for i in range(n_components):
            components.append(tfd.MultivariateNormalDiag(loc=loc[i],scale_diag=scale[i]))
        mix_gauss=tfd.Mixture(
            cat=tfd.Categorical(probs=probs),
            components=components,
            validate_args=True)
        return mix_gauss

    def MixMultiNormal2(self,
                        n_components: int = 3,
                        n_dimensions: int = 4,
                        seed: Optional[int] = None,
                       ) -> tfp.distributions.MixtureSameFamily:
        """
        Defines a mixture of 'n_components' Multivariate Normal distributions in 'n_dimensions' dimensions 
        with means and stddevs given by the tensors 'loc' and 'scale' with shapes 
        '(n_components,n_dimensions)'.
        The components are mixed according to the categorical distribution with probabilities
        'probs' (with shape equal to 'n_components'). This means that each Multivariate distribution 
        is assigned a single probability.

        The resulting multivariate distribution has large (random) correlation.

        Note: The functions 'MixMultiNormal1' and 'MixMultiNormal1_indep'
        generate identical samples, different from the samples generated by
        'MixMultiNormal2' and 'MixMultiNormal2_indep' (also identical).
        """
        if seed is None:
            seed = self._seed
        np.random.seed(seed)
        loc = np.random.sample([n_components,n_dimensions])*10
        scale = np.random.sample([n_components,n_dimensions])
        probs = np.random.sample(n_components)
        mix_gauss=tfd.MixtureSameFamily(
            mixture_distribution=tfd.Categorical(probs=probs),
            components_distribution=tfd.MultivariateNormalDiag(loc=loc,scale_diag=scale),
            validate_args=True)
        return mix_gauss

    def MixMultiNormal1_indep(self,
                              n_components: int = 3,
                              n_dimensions: int = 4,
                              seed: Optional[int] = None,
                             ) -> tfp.distributions.Independent:
        """
        Defines a mixture of 'n_components' Multivariate Normal distributions in 'n_dimensions' dimensions 
        with means and stddevs given by the tensors 'loc' and 'scale' with shapes 
        '(n_components,n_dimensions)'.
        The components are mixed according to the categorical distribution with probabilities
        'probs' (with shape equal to 'n_components'). This means that each Multivariate distribution 
        is assigned a single probability.

        The resulting multivariate distribution has large (random) correlation.

        Note: The functions 'MixMultiNormal1' and 'MixMultiNormal1_indep'
        generate identical samples, different from the samples generated by
        'MixMultiNormal2' and 'MixMultiNormal2_indep' (also identical).
        """
        if seed is None:
            seed = self._seed
        np.random.seed(seed)
        loc = np.random.sample([n_components,n_dimensions])*10
        scale = np.random.sample([n_components,n_dimensions])
        probs = np.random.sample(n_components)
        components = []
        for i in range(n_components):
            components.append(tfd.MultivariateNormalDiag(loc=loc[i],scale_diag=scale[i]))
        mix_gauss=tfd.Independent(
        distribution=tfd.Mixture(
            cat=tfd.Categorical(probs=probs),
            components=components,
            validate_args=True),
        reinterpreted_batch_ndims=0)
        return mix_gauss

    def MixMultiNormal2_indep(self,
                              n_components: int = 3,
                              n_dimensions: int = 4,
                              seed: Optional[int] = None,
                             ) -> tfp.distributions.Independent:
        """
        Defines a mixture of 'n_components' Multivariate Normal distributions in 'n_dimensions' dimensions 
        with means and stddevs given by the tensors 'loc' and 'scale' with shapes 
        '(n_components,n_dimensions)'.
        The components are mixed according to the categorical distribution with probabilities
        'probs' (with shape equal to 'n_components'). This means that each Multivariate distribution 
        is assigned a single probability.

        The resulting multivariate distribution has large (random) correlation.

        Note: The functions 'MixMultiNormal1' and 'MixMultiNormal1_indep'
        generate identical samples, different from the samples generated by
        'MixMultiNormal2' and 'MixMultiNormal2_indep' (also identical).
        """
        if seed is None:
            seed = self._seed
        np.random.seed(seed)
        loc = np.random.sample([n_components,n_dimensions])*10
        scale = np.random.sample([n_components,n_dimensions])
        probs = np.random.sample(n_components)
        mix_gauss=tfd.Independent(
            distribution=tfd.MixtureSameFamily(
                mixture_distribution=tfd.Categorical(probs=probs),
                components_distribution=tfd.MultivariateNormalDiag(loc=loc,scale_diag=scale),
                validate_args=True),
            reinterpreted_batch_ndims=0)
        return mix_gauss


class Distribution(Verbosity):
    """
    """
    obj_name : str = "Distribution"
    supported_default_base_distributions: List[str] = ["Normal"]
    def __init__(self,
                 ndims: int,
                 seed: Optional[int] = None,
                 dtype: Optional[DTypeStr] = None,
                 default_dist: Optional[str] = "Normal",
                 tf_dist: Optional[DictStr] = None,
                 verbose: IntBool = True
                ):
        """
        """
        # Initialize parent Verbosity class
        super().__init__(verbose)
        # Initialization of verbosity mode
        verbose, verbose_sub = self.get_verbosity(self._verbose)
        # Initialization of object
        timestamp = utils.generate_timestamp()
        # Initialize log
        self.log: LogPredDict = {timestamp: {"action": "created"}}
        print(header_string, "\nInitialize Distributions object.\n", show = verbose)
        self._ndims = ndims
        self.seed = seed if seed is not None else 0
        self.__set_dtype(dtype,verbose=verbose_sub)
        self._default_dist = default_dist if default_dist is not None else ""
        self._tf_dist_str: str = tf_dist if type(tf_dist) is str else ""
        self._tf_dist_dic: dict = tf_dist if type(tf_dist) is dict else {}
        self.__check_inputs()
        if self._default_dist != "":
            self.__set_base_default_distribution()
        else:
            self.__set_base_tf_distribution()

    def __set_dtype(self,
                    dtype: Optional[DTypeStr] = None,
                    verbose: Optional[IntBool] = None
                   ) -> None:
        if dtype is not None:
            try:
                self._dtype = np.dtype(dtype)
            except:
                self._dtype = np.dtype("float32")
                print("Data type",dtype,"is not supported. Data type set to 'float32'.")
        else:
            self._dtype = np.dtype("float32")

    def __check_inputs(self) -> None:
        if self._default_dist != "" and self._default_dist not in self.supported_default_base_distributions:
            raise Exception("The distribution "+self._default_dist+" is not a supported default base distribution.")
        if self._default_dist == "" and self._tf_dist_str == "" and self._tf_dist_dic is {}:
            raise Exception("You have to specify at least one of the 'default_dist' (and related arguments) and 'tf_dist' arguments.")

    def __set_base_default_distribution(self,
                                        verbose: Optional[IntBool] = None
                                       ) -> None:
        verbose, verbose_sub = self.get_verbosity(verbose)
        print(header_string,"\nSetting base distribution (from default distributions)\n", show = verbose)
        self.base_distribution = None
        default_dist = "self."+ self._default_dist + "()"
        dist_string, dist = eval(default_dist)
        self.base_distribution_string = "tfd.Sample("+dist_string+",sample_shape=["+str(self._ndims)+"])"
        self.base_distribution = tfd.Sample(dist,sample_shape=[self._ndims])
        print("Base distribution set to:", self.base_distribution_string, ".\n", show = verbose)
        timestamp = "datetime_"+datetime.now().strftime("%Y_%m_%d_%H_%M_%S_%fZ")[:-3]
        self.log[timestamp] = {"action": "base distribution set",
                               "distribution": self.base_distribution_string}

    def __set_base_tf_distribution(self,
                                   verbose: Optional[IntBool] = None
                                  ) -> None:
        verbose, verbose_sub = self.get_verbosity(verbose)
        print(header_string,"\nSetting base distribution (from tfp distributions)\n", show = verbose)
        self.base_distribution = None
        if self._tf_dist_str != "" and self._tf_dist_dic is {}:
            tf_dist_str = str(self._tf_dist_str)
            if "(" in tf_dist_str:
                try:
                    eval("tfd." + tf_dist_str.replace("tfd.", ""))
                    dist_string = "tfd." + tf_dist_str.replace("tfd.", "")
                except:
                    eval(tf_dist_str)
                    dist_string = tf_dist_str
            else:
                try:
                    eval("tfd." + tf_dist_str.replace("tfd.", "") +"()")
                    dist_string = "tfd." + tf_dist_str.replace("tfd.", "") +"()"
                except:
                    eval(tf_dist_str +"()")
                    dist_string = tf_dist_str +"()"
        elif self._tf_dist_str == "" and self._tf_dist_dic != {}:
            tf_dist_dict = dict(self._tf_dist_dic)
            try:
                name = tf_dist_dict["name"]
            except:
                raise Exception("The distribution ", str(tf_dist_dict), " has unspecified name.")
            try:
                args = tf_dist_dict["args"]
            except:
                args = []
            try:
                kwargs = tf_dist_dict["kwargs"]
            except:
                kwargs = {}
            dist_string = utils.build_method_string_from_dict("tfd", name, args, kwargs)
        else:
            raise Exception("Could not set base distribution. The 'tf_dist' input argument does not have a valid format (str or dict).")
        try:
            dist = eval(dist_string)
            self.base_distribution_string = "tfd.Sample("+dist_string+",sample_shape=["+str(self._ndims)+"])"
            self.base_distribution = tfd.Sample(dist,sample_shape=[self._ndims])
            print("Base distribution set to:", self.base_distribution_string, ".\n", show = verbose)
        except Exception as e:
            print(e)
            raise Exception("Could not set base distribution", dist_string, "\n")
        timestamp = "datetime_"+datetime.now().strftime("%Y_%m_%d_%H_%M_%S_%fZ")[:-3]
        self.log[timestamp] = {"action": "base distribution set",
                               "distribution": self.base_distribution_string}

    def Normal(self) -> tfp.distributions.Normal:
        dist_string = "tfd.Normal(loc=np.array(0,dtype='"+str(self._dtype)+"'), scale=1, allow_nan_stats=False)"
        dist=eval(dist_string)
        return dist

    def Normal_mixture(self) -> tfp.distributions.Mixture:
        ### Mix_gauss will be our target distribution.
        dist_string = "tfd.Mixture(cat=tfd.Categorical(probs=[0.3,.7]),components=[tfd.Normal(loc=np.array(3.3,dtype='"+str(self._dtype)+"'), scale=0.4),tfd.Normal(loc=np.array(1.8,dtype='"+str(self._dtype)+"'), scale=0.2)])"
        dist=eval(dist_string)
        return dist

    def RandCorr(self,
                 ndims: int,
                 seed: Optional[int] = None,
                ) -> npt.NDArray:
        if seed is None:
            seed = self._seed
        np.random.seed(seed)
        V = datasets.make_spd_matrix(ndims,random_state=seed)
        D = np.sqrt(np.diag(np.diag(V)))
        Dinv = np.linalg.inv(D)
        Vnorm = np.matmul(np.matmul(Dinv,V),Dinv)
        return Vnorm

    def RandCov(self,
                std: Array,
                seed: Optional[int] = None,
               ) -> npt.NDArray:
        if seed is None:
            seed = self._seed
        np.random.seed(seed)
        std = np.array(std)
        ndims = len(std)
        corr = self.RandCorr(ndims,seed)
        D = np.diag(std)
        V = np.matmul(np.matmul(D,corr),D)
        return V

    def describe_distributions(self,
                               distribution: tfp.distributions.distribution
                              ) -> None:
        """
        Describes a 'tfp.distributions' object.
        """
        print('\n'.join([str(d) for d in distribution]))



