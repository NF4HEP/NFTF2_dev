from . import inference, utils  # , custom_losses
from .show_prints import Verbosity, print
from .distributions import Distributions
from .resources import Resources
from .data import Data
from .rqspline import RQSplineFlow
from .realnvp import RealNVPFlow
from .cspline import CSplineFlow
from .maf import MAFFlow
from .corner import corner, extend_corner_range, get_1d_hist
__all__ = ["NF"]

import codecs
from statistics import mean, median
import math
from os import path, remove, sep, stat
from timeit import default_timer as timer
from datetime import datetime
import pickle
import time
import random
import deepdish as dd
from decimal import Decimal
import json
import h5py
import pandas as pd
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
from tensorflow.keras import Input
from tensorflow.keras import backend as K
from tensorflow.keras import layers, initializers, regularizers, constraints, callbacks, optimizers, metrics, losses
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.utils import plot_model
import onnx
import tf2onnx
from scipy.stats import anderson_ksamp
from scipy.stats import epps_singleton_2samp
from scipy.stats import wasserstein_distance
from scipy import stats
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions
tfb = tfp.bijectors


header_string = "=============================="
footer_string = "------------------------------"

try:
    from livelossplot import PlotLossesKerasTF as PlotLossesKeras
    from livelossplot.outputs import MatplotlibPlot
except:
    print(header_string, "\nNo module named 'livelossplot's. Continuing without.\nIf you wish to plot the loss in real time please install 'livelossplot'.\n")

sns.set()
kubehelix = sns.color_palette("cubehelix", 30)
reds = sns.color_palette("Reds", 30)
greens = sns.color_palette("Greens", 30)
blues = sns.color_palette("Blues", 30)

mplstyle_path = path.join(path.split(path.realpath(__file__))[
                          0], "matplotlib.mplstyle")


class NF(Resources):
    """
    This class contains the Normalizing Flow object.
    """

    def __init__(self,
                 name=None,
                 flow_type=None,
                 data=None,
                 input_data_file=None,
                 load_on_RAM=False,
                 seed=None,
                 dtype=None,
                 same_data=True,
                 model_data_inputs=None,
                 model_base_dist_inputs=None,
                 model_define_inputs=None,
                 model_chain_inputs=None,
                 model_optimizer_inputs=None,
                 model_compile_inputs=None,
                 model_callbacks_inputs=None,
                 model_train_inputs=None,
                 resources_inputs=None,
                 output_folder=None,
                 ensemble_name=None,
                 input_file=None,
                 verbose=True
                 ):
        self.verbose = verbose
        verbose, verbose_sub = self.set_verbosity(verbose)
        timestamp = "datetime_" + \
            datetime.now().strftime("%Y_%m_%d_%H_%M_%S_%fZ")[:-3]
        self.supported_architectures = [
            "MAF", "RealNVP", "CSpline", "RQSpline"]
        self.output_folder = output_folder
        # Set input files
        self.input_file = input_file
        self.input_data_file = input_data_file
        self.__check_define_input_files(verbose=verbose_sub)
        # Check wheather to create a new NF object from inputs or from files
        if self.input_file == None:
            # Initialize input parameters from arguments
            # Set main inputs
            self.log = {timestamp: {"action": "created"}}
            self.name = name
            self.__check_define_name()
            self.flow_type = flow_type
            self.data = data
            self.load_on_RAM = load_on_RAM
            self.seed = seed
            self.dtype = dtype
            self.same_data = same_data
            self.__model_data_inputs = model_data_inputs
            self.__check_define_model_data_inputs()
            self.__model_base_dist_inputs = model_base_dist_inputs
            self.__model_define_inputs = model_define_inputs
            self.__check_define_model_define_inputs()
            self.__model_chain_inputs = model_chain_inputs
            self.__model_optimizer_inputs = model_optimizer_inputs
            self.__model_compile_inputs = model_compile_inputs
            self.__check_define_model_compile_inputs(verbose=verbose_sub)
            self.__model_callbacks_inputs = model_callbacks_inputs
            self.__model_train_inputs = model_train_inputs
            self.__check_define_model_train_inputs()
            self.npoints_train, self.npoints_val, self.npoints_test = self.__model_data_inputs[
                "npoints"]
            # Set output folder and files
            self.output_folder = output_folder
            self.__check_define_output_files(
                timestamp=timestamp, verbose=verbose_sub)
            # Set ensemble attributes if the Normalizing Flowis part of an ensemble
            self.ensemble_name = ensemble_name
            self.__check_define_ensemble_folder(verbose=verbose_sub)
            # Set model hyperparameters parameters
            self.__set_model_hyperparameters()
        else:
            # Initialize input parameters from file
            # Load summary_log dictionary
            print(header_string, "\nWhen providing NF input folder all arguments but data, load_on_RAM, and dtype are ignored and the object is constructed from saved data.", show=verbose)
            self.__load_json_and_log(verbose=verbose_sub)
            self.data = None
            # Set main inputs and DataSample
            self.load_on_RAM = load_on_RAM
            if dtype != None:
                self.dtype = dtype
            if seed != None:
                self.seed = seed
            # Set name, folders and files names
            self.__check_define_output_files(
                timestamp=timestamp, verbose=verbose_sub)
        # Set resources (__resources_inputs is None for a standalone Normalizing Flow and is passed only if the Normalizing Flow
        # is part of an ensemble)
        self.__resources_inputs = resources_inputs
        self.__set_resources(verbose=verbose_sub)
        # Set additional inputs
        self.__set_seed()
        self.__set_dtype()
        self.__set_data(verbose=verbose_sub)  # also sets self.ndims
        self.__check_define_model_chain_inputs()  # here because it needs self.ndims
        self.__check_define_model_base_dist_inputs()  # here because it needs self.ndims
        # sets base distribution (Distribution object)
        self.__set_base_distriburion(verbose=verbose_sub)
        # optimizer, metrics, callbacks
        self.__set_tf_objects(verbose=verbose_sub)
        # Initialize model, history,scalers, data indices, and predictions
        if self.input_file != None:
            self.__load_data_indices(verbose=verbose_sub)
            self.__load_history(verbose=verbose_sub)
            self.__load_model(verbose=verbose_sub)
            self.__load_predictions(verbose=verbose_sub)
            self.__load_preprocessing(verbose=verbose_sub)
            self.predictions["Figures"] = utils.check_figures_dic(
                self.predictions["Figures"], output_figures_folder=self.output_figures_folder)
        else:
            self.epochs_available = 0
            self.training_time = 0
            self.idx_train, self.idx_val, self.idx_test = [np.array(
                [], dtype="int"), np.array([], dtype="int"), np.array([], dtype="int")]
            self.scalerX, self.rotationX = [None, None]
            self.NN, self.Bijector, self.Flow, self.trainable_distribution, self.log_prob, self.model = [
                None, None, None, None, None, None]
            self.history = {}
            self.predictions = {"Model_evaluation": {},
                                "Bayesian_inference": {},
                                "Frequentist_inference": {},
                                "Figures": {}}
        self.X_train = np.array([[]], dtype=self.dtype)
        self.X_val = np.array([[]], dtype=self.dtype)
        self.X_test = np.array([[]], dtype=self.dtype)
        # Save object
        if self.input_file == None:
            self.save_json(overwrite=False, verbose=verbose_sub)
            self.save_log(overwrite=False, verbose=verbose_sub)
            #self.save(overwrite=False, verbose=verbose_sub)
            #self.save_log(overwrite=False, verbose=verbose_sub)
        else:
            self.save_json(overwrite=True, verbose=verbose_sub)
            self.save_log(overwrite=True, verbose=verbose_sub)

    def __set_resources(self, verbose=None):
        """
        Private method used by the :meth:`NF.__init__ <NF4HEP.NF.__init__>` one to set resources.
        If :attr:`NF.__resources_inputs <NF4HEP.NF.__resources_inputs` is ``None``, it 
        calls the methods 
        :meth:`NF.get_available_cpu <NF4HEP.NF.get_available_cpu` and
        :meth:`NF.set_gpus <NF4HEP.NF.set_gpus` inherited from the
        :class:`Verbosity <NF4HEP.Verbosity>` class, otherwise it sets resources from input arguments.
        The latter method is needed, when the object is a member of an esemble, to pass available resources from the parent
        :class:`NFEnsemble <NF4HEP.NFEnsemble>` object.

        - **Arguments**

            - **verbose**

                See :argument:`verbose <common_methods_arguments.verbose>`.
        """
        verbose, verbose_sub = self.set_verbosity(verbose)
        self.check_tf_gpu(verbose=verbose)
        if self.__resources_inputs is None:
            # self.get_available_gpus(verbose=False)
            self.get_available_cpu(verbose=verbose_sub)
            self.set_gpus(gpus_list="all", verbose=verbose_sub)
        else:
            self.available_gpus = self.__resources_inputs["available_gpus"]
            self.available_cpu = self.__resources_inputs["available_cpu"]
            self.active_gpus = self.__resources_inputs["active_gpus"]
            self.gpu_mode = self.__resources_inputs["gpu_mode"]

    def __check_define_input_files(self, verbose=False):
        """
        Private method used by the :meth:`NF.__init__ <NF4HEP.NF.__init__>` one
        to set the attributes corresponding to input files:

            - :attr:`NF.input_file <NF4HEP.NF.input_file>`
            - :attr:`NF.input_json_file <NF4HEP.NF.input_json_file>`
            - :attr:`NF.input_history_json_file <NF4HEP.NF.input_history_json_file>`
            - :attr:`NF.input_idx_h5_file <NF4HEP.NF.input_idx_h5_file>`
            - :attr:`NF.input_log_file <NF4HEP.NF.input_log_file>`
            - :attr:`NF.input_predictions_h5_file <NF4HEP.NF.input_predictions_h5_file>`
            - :attr:`NF.input_preprocessing_pickle_file <NF4HEP.NF.input_preprocessing_pickle_file>`
            - :attr:`NF.input_tf_model_weights_h5_file <NF4HEP.NF.input_tf_model_weights_h5_file>`
            - :attr:`NF.input_folder <NF4HEP.NF.input_folder>`

        depending on the value of the 
        :attr:`NF.input_file <NF4HEP.NF.input_file>` attribute.
        It also sets the attribute
        :attr:`NF.input_data_file <NF4HEP.NF.input_data_file>` if the object has
        been initialized directly from a :mod:`Data <data>` object.
        """
        verbose, verbose_sub = self.set_verbosity(verbose)
        if self.input_data_file is not None:
            self.input_data_file = path.abspath(
                path.splitext(self.input_data_file)[0])
        if self.input_file == None:
            self.input_json_file = None
            self.input_history_json_file = None
            self.input_idx_h5_file = None
            self.input_log_file = None
            self.input_predictions_h5_file = None
            self.input_preprocessing_pickle_file = None
            self.input_tf_model_weights_h5_file = None
            self.input_folder = None
            print(header_string,
                  "\nNo input files and folders specified.\n", show=verbose)
        else:
            self.input_file = path.abspath(path.splitext(self.input_file)[0])
            self.input_json_file = self.input_file+".json"
            self.input_history_json_file = self.input_file+"_history.json"
            self.input_idx_h5_file = self.input_file+"_idx.h5"
            self.input_log_file = self.input_file+".log"
            self.input_predictions_h5_file = self.input_file+"_predictions.h5"
            self.input_preprocessing_pickle_file = self.input_file+"_preprocessing.pickle"
            self.input_tf_model_weights_h5_file = self.input_file+"_model_weights.h5"
            self.input_folder = path.split(self.input_file)[0]
            print(header_string, "\Input folder set to\n\t",
                  self.input_folder, ".\n", show=verbose)

    def __check_define_output_files(self, timestamp=None, verbose=False):
        """
        Private method used by the :meth:`NF.__init__ <NF4HEP.NF.__init__>` one
        to set the attributes corresponding to output folders

            - :attr:`NF.output_figures_folder <NF4HEP.NF.output_figures_folder>`
            - :attr:`NF.output_folder <NF4HEP.NF.output_folder>`

        and output files

            - :attr:`NF.output_figures_base_file <NF4HEP.NF.output_figures_base_file>`
            - :attr:`NF.output_files_base_name <NF4HEP.NF.output_files_base_name>`
            - :attr:`NF.output_history_json_file <NF4HEP.NF.output_history_json_file>`
            - :attr:`NF.output_idx_h5_file <NF4HEP.NF.output_idx_h5_file>`
            - :attr:`NF.output_log_file <NF4HEP.NF.output_log_file>`
            - :attr:`NF.output_predictions_h5_file <NF4HEP.NF.output_predictions_h5_file>`
            - :attr:`NF.output_preprocessing_pickle_file <NF4HEP.NF.output_preprocessing_pickle_file>`
            - :attr:`NF.output_json_file <NF4HEP.NF.output_json_file>`
            - :attr:`NF.output_tf_model_weights_h5_file <NF4HEP.NF.output_tf_model_weights_h5_file>`

        depending on the value of the 
        :attr:`NF.input_file <NF4HEP.NF.input_files_base_name>` and
        :attr:`NF.output_folder <NF4HEP.NF.output_folder>` attributes.
        It also initializes (to ``None``) the attributes:

            - :attr:`NF.output_checkpoints_files <NF4HEP.NF.output_checkpoints_files>`
            - :attr:`NF.output_checkpoints_folder <NF4HEP.NF.output_checkpoints_folder>`
            - :attr:`NF.output_tensorboard_log_dir <NF4HEP.NF.output_tensorboard_log_dir>`

        and creates the folders
        :attr:`NF.output_folder <NF4HEP.NF.output_folder>`
        and 
        :attr:`NF.output_figures_folder <NF4HEP.NF.output_figures_folder>`
        if they do not exist.
        """
        verbose, verbose_sub = self.set_verbosity(verbose)
        if self.output_folder is not None:
            self.output_folder = path.abspath(self.output_folder)
            if self.input_folder is not None and self.output_folder != self.input_folder:
                utils.copy_and_save_folder(
                    self.input_folder, self.output_folder, timestamp=timestamp, verbose=verbose)
        else:
            if self.input_folder is not None:
                self.output_folder = self.input_folder
            else:
                self.output_folder = path.abspath("")
        self.output_folder = utils.check_create_folder(self.output_folder)
        self.output_figures_folder = utils.check_create_folder(
            path.join(self.output_folder, "figures"))
        self.output_figures_base_file_name = self.name+"_figure"
        self.output_figures_base_file_path = path.join(
            self.output_figures_folder, self.output_figures_base_file_name)
        self.output_files_base_name = path.join(self.output_folder, self.name)
        self.output_history_json_file = self.output_files_base_name+"_history.json"
        self.output_idx_h5_file = self.output_files_base_name+"_idx.h5"
        self.output_log_file = self.output_files_base_name+".log"
        self.output_predictions_h5_file = self.output_files_base_name+"_predictions.h5"
        self.output_predictions_json_file = self.output_files_base_name+"_predictions.json"
        self.output_preprocessing_pickle_file = self.output_files_base_name + \
            "_preprocessing.pickle"
        self.output_h5_file = self.output_files_base_name+".h5"
        self.output_json_file = self.output_files_base_name+".json"
        #self.output_tf_model_graph_pdf_file = self.output_files_base_name+"_model_graph.pdf"
        self.output_tf_model_weights_h5_file = self.output_files_base_name+"_model_weights.h5"
        #self.output_tf_model_onnx_file = self.output_files_base_name+"_model.onnx"
        self.output_checkpoints_files = None
        self.output_checkpoints_folder = None
        self.output_tensorboard_log_dir = None
        print(header_string, "\nNF output folder set to\n\t",
              self.output_folder, ".\n", show=verbose)

    def __check_define_name(self):
        """
        Private method used by the :meth:`NF.__init__ <NF4HEP.NF.__init__>` one
        to define the :attr:`NF.name <NF4HEP.NF.name>` attribute.
        If it is ``None`` it replaces it with 
        ``"model_"+datetime.now().strftime("%Y-%m-%d-%H-%M-%S.%fZ")[:-3]+"_nf"``,
        otherwise it appends the suffix "_nf" 
        (preventing duplication if it is already present).
        """
        if self.name == None:
            timestamp = "datetime_" + \
                datetime.now().strftime("%Y_%m_%d_%H_%M_%S_%fZ")[:-3]
            self.name = "model_"+timestamp+"_nf"
        # else:
        #    self.name = utils.check_add_suffix(self.name, "_nf")

    def __check_npoints(self):
        """
        Private method used by the :meth:`NF.__set_data <NF4HEP.NF._NF__set_data>` one
        to check that the required number of points for train/val/test is less than the total number
        of available points in the :attr:`NF.data <NF4HEP.NF.data>` object.
        """
        self.npoints_available = self.data.npoints
        self.npoints_train_val_available = int(
            (1-self.data.test_fraction)*self.npoints_available)
        self.npoints_test_available = int(
            self.data.test_fraction*self.npoints_available)
        required_points_train_val = self.npoints_train+self.npoints_val
        required_points_test = self.npoints_test
        if required_points_train_val > self.npoints_train_val_available:
            self.data.opened_dataset.close()
            raise Exception("npoints_train+npoints_val larger than the available number of points in data.\
                Please reduce npoints_train+npoints_val or change test_fraction in the :mod:`Data <data>` object.")
        if required_points_test > self.npoints_test_available:
            self.data.opened_dataset.close()
            raise Exception("npoints_test larger than the available number of points in data.\
                Please reduce npoints_test or change test_fraction in the :mod:`Data <data>` object.")

    def __check_define_model_data_inputs(self, verbose=None):
        """
        Private method used by the :meth:`NF.__init__ <NF4HEP.NF.__init__>` one
        to check the private dictionary 
        :attr:`NF.__model_data_inputs <NF4HEP.NF._NF__model_data_inputs>`.
        It checks if the item ``"npoints"`` is correctly specified and if it is not it raises an exception. If valitadion 
        and test number of points are input
        as fractions of the training one, then it converts them to absolute number of points.
        It checks if the items ``"scalerX"`` and ``"rotationX"`` are defined and, if they are not, it sets
        them to their default value ``False``.
        """
        verbose, verbose_sub = self.set_verbosity(verbose)
        try:
            self.__model_data_inputs["npoints"]
        except:
            raise Exception(
                "model_data_inputs dictionary should contain at least a key 'npoints'.")
        if self.__model_data_inputs["npoints"][1] <= 1:
            self.__model_data_inputs["npoints"][1] = round(
                self.__model_data_inputs["npoints"][0]*self.__model_data_inputs["npoints"][1])
        if self.__model_data_inputs["npoints"][2] <= 1:
            self.__model_data_inputs["npoints"][2] = round(
                self.__model_data_inputs["npoints"][0]*self.__model_data_inputs["npoints"][2])
        utils.check_set_dict_keys(self.__model_data_inputs, ["scalerX",
                                                             "rotationX"],
                                  [False, False], verbose=verbose_sub)

    def __check_define_model_base_dist_inputs(self, verbose=None):
        """
        Private method used by the :meth:`NF.__init__ <NF4HEP.NF.__init__>` one
        to check the private dictionary 
        :attr:`NF.__model_base_dist_inputs <NF4HEP.NF._NF__model_base_dist_inputs>`.
        It checks if the input arguments ``default_dist`` and ``tf_dist`` related to the base distribution object
        are set. If they are not, then they are automatically set to their default (Normal distribution).
        The ``ndims`` argument is set from the :attr:`NF.ndims <NF4HEP.NF.ndims>` attribute previously set by the 
        :meth:`NF.__set_data <NF4HEP.NF._NF__set_data>` method.
        """
        verbose, verbose_sub = self.set_verbosity(verbose)
        self.__model_base_dist_inputs["ndims"] = self.ndims
        try:
            if self.__model_base_dist_inputs["tf_dist"] is not None:
                tf_dist_defined = True
            else:
                tf_dist_defined = False
        except:
            tf_dist_defined = False
        try:
            if self.__model_base_dist_inputs["default_dist"] is not None:
                default_dist_defined = True
            else:
                default_dist_defined = False
        except:
            default_dist_defined = False
        if default_dist_defined:
            self.__model_base_dist_inputs["tf_dist"] = None
        elif tf_dist_defined and not default_dist_defined:
            self.__model_base_dist_inputs["default_dist"] = None
        else:
            self.__model_base_dist_inputs["default_dist"] = "Normal"

    def __check_define_model_define_inputs(self, verbose=None):
        """
        .. code-block:: python
            model_define_inputs = {"params": 2, 
                                   "event_shape": None, 
                                   "conditional": False,
                                   "conditional_event_shape": None,
                                   "conditional_input_layers": "all_layers",
                                   "hidden_units": [64,64],
                                   "input_order": "left-to-right",
                                   "hidden_degrees": "equal",
                                   "activation": None,
                                   "use_bias": True, 
                                   "kernel_initializer": "glorot_uniform",
                                   "bias_initializer": "zeros", 
                                   "kernel_regularizer": None,
                                   "bias_regularizer": None, 
                                   "kernel_constraint": None, 
                                   "bias_constraint": None,
                                   "validate_args": False,
                                   "dropout_rate": 0,
                                   "batch_norm": False}
        """
        verbose, verbose_sub = self.set_verbosity(verbose)
        try:
            self.__model_define_inputs["params"]
        except:
            raise Exception(
                "model_define_inputs dictionary should contain at least a key 'params'.")
        utils.check_set_dict_keys(self.__model_define_inputs, ["event_shape",
                                                               "conditional",
                                                               "conditional_event_shape",
                                                               "conditional_input_layers",
                                                               "hidden_units",
                                                               "input_order",
                                                               "hidden_degrees",
                                                               "activation",
                                                               "use_bias",
                                                               "kernel_initializer",
                                                               "bias_initializer",
                                                               "kernel_regularizer",
                                                               "bias_regularizer",
                                                               "kernel_constraint",
                                                               "bias_constraint",
                                                               "validate_args",
                                                               "kwargs",
                                                               "batch_norm"],
                                                               [None,
                                                                False,
                                                                None,
                                                                "all_layers",
                                                                [64, 64, 64],
                                                                "left-to-right",
                                                                "equal",
                                                                None,
                                                                True,
                                                                "glorot_uniform",
                                                                "zeros",
                                                                None,
                                                                None,
                                                                None,
                                                                None,
                                                                False,
                                                                {},
                                                                False],
                                                               verbose=verbose_sub)

    def __check_define_model_chain_inputs(self, verbose=None):
        """
        .. code-block:: python
            model_chain_inputs = {"num_bijectors": 2, 
                                 "spline_knots": 8, 
                                 "range_min": -12}
        """
        verbose, verbose_sub = self.set_verbosity(verbose)
        try:
            pass
        except:
            raise Exception(
                "model_chain_inputs dictionary should contain at least a key xxxx.")
        self.__model_chain_inputs["ndims"] = self.ndims
        utils.check_set_dict_keys(self.__model_chain_inputs, ["num_bijectors"],
                                  [2],
                                  verbose=verbose_sub)
        if self.flow_type == "MAF":
            utils.check_set_dict_keys(self.__model_chain_inputs, ["default_NN",
                                                                 "default_bijector"],
                                      [True,
                                       True],
                                      verbose=verbose_sub)
        elif self.flow_type == "RealNVP":
            utils.check_set_dict_keys(self.__model_chain_inputs, ["spline_knots",
                                                                 "range_min"],
                                      [8,
                                       -12],
                                      verbose=verbose_sub)

    def __check_define_model_compile_inputs(self, verbose=None):
        """
        Private method used by the :meth:`NF.__init__ <NF4HEP.NF.__init__>` one
        to check the private dictionary 
        :attr:`NF.__model_compile_inputs <NF4HEP.NF._NF__model_compile_inputs>`.
        It checks if the attribure exists and, if it does not, it defines it as an empty dictionary.
        It checks if the item ``"metrics"`` is defined and, if it is not, 
        it sets it to its default values ``["kullback_leibler_divergence", "binary_crossentropy"]``.
        It sets the additional attribute 
        :attr:`NF.model_compile_kwargs <NF4HEP.NF.model_compile_kwargs>` equal to the
        private dictionary 
        :attr:`NF.__model_compile_inputs <NF4HEP.NF._DnnLik__model_compile_inputs>` without the
        ``"loss"`` and ``"metrics"`` items.

        - **Arguments**

            - **verbose**

                See :argument:`verbose <common_methods_arguments.verbose>`.
        """
        verbose, verbose_sub = self.set_verbosity(verbose)
        if self.__model_compile_inputs == None:
            self.__model_compile_inputs = {}
        utils.check_set_dict_keys(self.__model_compile_inputs, ["metrics"],
                                  [[]], verbose=verbose_sub)
        self.model_compile_kwargs = utils.dic_minus_keys(
            self.__model_compile_inputs, ["metrics"])

    def __check_define_model_train_inputs(self):
        """
        Private method used by the :meth:`NF.__init__ <NF4HEP.NF.__init__>` one
        to check the private dictionary 
        :attr:`NF.__model_train_inputs <NF4HEP.NF._NF__model_train_inputs>`.
        It checks if the items ``"epochs"`` and ``"batch_size"`` are defined and, if they are not, 
        it raises an exception.
        """
        try:
            self.__model_train_inputs["epochs"]
        except:
            raise Exception(
                "model_train_inputs dictionary should contain at least a keys 'epochs' and 'batch_size'.")
        try:
            self.__model_train_inputs["batch_size"]
        except:
            raise Exception(
                "model_train_inputs dictionary should contain at least a keys 'epochs' and 'batch_size'.")

    def __check_define_ensemble_folder(self, verbose=None):
        """
        Private method used by the :meth:`NF.__init__ <NF4HEP.NF.__init__>` one
        to set the
        :attr:`NF.ensemble_folder <NF4HEP.NF.ensemble_folder>` and 
        :attr:`NF.standalone <NF4HEP.NF.standalone>` attributes. If the object is a member
        of a :class:`DnnLikEnsemble <NF4HEP.NFEnsemble>` object, i.e. of the
        :attr:`NF.ensemble_name <NF4HEP.NF.ensemble_name>` attribute is not ``None``,
        then the two attributes are set to the parent directory of
        :attr:`NF.output_folder <NF4HEP.NF.output_folder>` and to ``False``, respectively, otherwise
        they are set to ``None`` and ``False``, respectively.

        - **Arguments**

            - **verbose**

                See :argument:`verbose <common_methods_arguments.verbose>`.
        """
        verbose, _ = self.set_verbosity(verbose)
        if self.ensemble_name == None:
            self.ensemble_folder = None
            self.standalone = True
            print(header_string, "\nThis is a 'standalone' NF object and does not belong to a NF_ensemble. The attributes 'ensemble_name' and 'ensemble_folder' have therefore been set to None.\n", show=verbose)
        else:
            self.enseble_folder = path.abspath(
                path.join(self.output_folder, ".."))
            self.standalone = False

    def __set_seed(self):
        """
        Private method used by the :meth:`NF.__init__ <NF4HEP.NF.__init__>` one
        to initialize the random state of |numpy_link| and |tf_link| to the value of 
        :attr:`NF.seed <NF4HEP.NF.seed>`.
        """
        if self.seed == None:
            self.seed = 1
        random.seed(self.seed)
        np.random.seed(self.seed)
        tf.random.set_seed(self.seed)

    def __set_dtype(self):
        """
        Private method used by the :meth:`NF.__init__ <NF4HEP.NF.__init__>` one
        to set the dtype of the train/val/test data and of the internal |tf_keras_link| calculations.
        If the :attr:`NF.dtype <NF4HEP.NF.dtype>` attribute is ``None``, then it is
        set to the default value ``"float64"``.
        """
        if self.dtype == None:
            self.dtype = "float32"
        K.set_floatx(self.dtype)
        tf.keras.mixed_precision.set_global_policy(self.dtype)

    def __set_data(self, verbose=None):
        """
        Private method used by the :meth:`NF.__init__ <NF4HEP.NF.__init__>` one
        to initialize the :mod:`Data <data>` object saved in the
        :attr:`NF.data <NF4HEP.NF.data>` attribute and used to provide data to the 
        :class:`NF <NF4HEP.NF>` object.
        Data are set differently depending on the value of the attributes
        :attr:`NF.data <NF4HEP.NF.data>` and 
        :attr:`NF.input_data_file <NF4HEP.NF.input_data_file>`, corresponding to the two
        input class arguments: :argument:`data` and :argument:`input_data_file`, respectively. If both
        are not ``None``, then the former is ignored. If only :attr:`NF.data <NF4HEP.NF.data>`
        is not ``None``, then :attr:`NF.input_data_file <NF4HEP.NF.input_data_file>`
        is set to the :attr:`Data.input_file <NF4HEP.Data.input_file>` attribute of the :mod:`Data <data>` object.
        If :attr:`NF.input_data_file <NF4HEP.NF.input_data_file>` is not ``None`` the 
        :attr:`NF.data <NF4HEP.NF.data>` attribute is set by importing the :class:`Data <NF4HEP.Data>` 
        object from file.
        Once the :mod:`Data <data>` object has been set, the 
        :attr:`NF.ndims <NF4HEP.NF.ndims>` attribute == set from the same attribute of the 
        :mod:`Data <data>` object, and the two private methods
        :meth:`NF.__check_npoints <NF4HEP.NF._NF__check_npoints>` and
        :meth:`NF.__set_pars_info <NF4HEP.NF._NF__set_pars_info>`
        are called.

        - **Arguments**

            - **verbose**

                See :argument:`verbose <common_methods_arguments.verbose>`.
        """
        verbose, verbose_sub = self.set_verbosity(verbose)
        if self.data == None and self.input_data_file == None:
            raise Exception(
                "At least one of the arguments 'data' and 'input_data_file' should be specified.\nPlease specify one and retry.")
        elif self.data != None and self.input_data_file == None:
            self.input_data_file = self.data.input_file
            self.input_data_file = path.abspath(
                path.splitext(self.input_data_file)[0])
        else:
            if self.data != None:
                print(header_string, "\nBoth the arguments 'data' and 'input_data_file' have been specified. 'data' will be ignored and the :mod:`Data <data>` object will be set from 'input_data_file'.\n", show=verbose)
            self.data = Data(name=None,
                             data_X=None,
                             dtype=self.dtype,
                             pars_central=None,
                             pars_pos_poi=None,
                             pars_pos_nuis=None,
                             pars_labels=None,
                             pars_bounds=None,
                             test_fraction=None,
                             load_on_RAM=self.load_on_RAM,
                             output_folder=None,
                             input_file=self.input_data_file,
                             verbose=verbose_sub
                             )
        self.ndims = self.data.ndims
        self.__check_npoints()
        self.__set_pars_info()

    def __set_pars_info(self):
        """
        Private method used by the :meth:`NF.__set_data <NF4HEP.NF._NF__set_data>` one
        to set parameters info. It sets the attributes:

            - :attr:`NF.pars_central <NF4HEP.NF.pars_central>`
            - :attr:`NF.pars_pos_poi <NF4HEP.NF.pars_pos_poi>`
            - :attr:`NF.pars_pos_nuis <NF4HEP.NF.pars_pos_nuis>`
            - :attr:`NF.pars_labels <NF4HEP.NF.pars_labels>`
            - :attr:`NF.pars_labels_auto <NF4HEP.NF.pars_labels_auto>`
            - :attr:`NF.pars_bounds <NF4HEP.NF.pars_bounds>`

        by copying the corresponding attributes of the :mod:`Data <data>` object 
        :attr:`NF.data <NF4HEP.NF.data>`.
        """
        self.pars_central = self.data.pars_central
        self.pars_pos_poi = self.data.pars_pos_poi
        self.pars_pos_nuis = self.data.pars_pos_nuis
        self.pars_labels = self.data.pars_labels
        self.pars_labels_auto = self.data.pars_labels_auto
        self.pars_bounds = self.data.pars_bounds

    def __set_model_hyperparameters(self):
        """
        Private method used by the :meth:`NF.__init__ <NF4HEP.NF.__init__>` one
        to set attributes corresponding to model hyperparameters from the private dictionaries

            - :attr:`NF.__model_data_inputs <NF4HEP.NF._NF__model_data_inputs>`
            - :attr:`NF.__model_define_inputs <NF4HEP.NF._NF__model_define_inputs>`
            - :attr:`NF.__model_train_inputs <NF4HEP.NF._NF__model_train_inputs>`

        The following attributes are set:

            - :attr:`NF.scalerX_bool <NF4HEP.NF.scalerX_bool>`
            - :attr:`NF.rotationX_bool <NF4HEP.NF.rotationX_bool>`
            - :attr:`NF.batch_norm <NF4HEP.NF.batch_norm>`
            - :attr:`NF.epochs_required <NF4HEP.NF.epochs_required>`
            - :attr:`NF.batch_size <NF4HEP.NF.batch_size>`
            - :attr:`NF.model_train_kwargs <NF4HEP.NF.model_train_kwargs>`
        """
        self.scalerX_bool = self.__model_data_inputs["scalerX"]
        self.rotationX_bool = self.__model_data_inputs["rotationX"]
        self.batch_norm = self.__model_define_inputs["batch_norm"]
        self.epochs_required = self.__model_train_inputs["epochs"]
        self.batch_size = self.__model_train_inputs["batch_size"]
        self.model_train_kwargs = utils.dic_minus_keys(
            self.__model_train_inputs, ["epochs", "batch_size"])

    def __set_base_distriburion(self, verbose=None):
        """
        Private method used by the :meth:`NF.__init__ <NF4HEP.NF.__init__>` one
        to set the attributes:

            - :attr:`NF.base_distribution <NF4HEP.NF.base_distribution>`
            - :attr:`NF.base_distribution_string <NF4HEP.NF.base_distribution_string>`

        It works by creating a :class:`Distributions <NF4HEP.Distributions>` object with input
        arguments from the :attr:`NF.__model_base_dist_inputs <NF4HEP.NF._NF__model_base_dist_inputs>`
        attribute and assigning the corresponding 

            - :attr:`Distributions.base_distribution <NF4HEP.Distributions.base_distribution>`
            - :attr:`Distributions.base_distribution_string <NF4HEP.Distributions.base_distribution_string>`

        attributes.

        - **Arguments**

            - **verbose**

                See :argument:`verbose <common_methods_arguments.verbose>`.
        """
        _, verbose_sub = self.set_verbosity(verbose)
        distribution = Distributions(ndims=self.ndims,
                                     dtype=self.dtype,
                                     default_dist=self.__model_base_dist_inputs["default_dist"],
                                     tf_dist=self.__model_base_dist_inputs["tf_dist"],
                                     verbose=verbose_sub)
        self.base_distribution = distribution.base_distribution
        self.base_distribution_string = distribution.base_distribution_string

    def __set_tf_objects(self, verbose=None):
        """
        Private method used by the :meth:`NF.__init__ <NF4HEP.NF.__init__>` one
        to set attributes corresponding to |tf_keras_link| objects by calling the private methods:

            - :meth:`NF.__set_optimizer <NF4HEP.NF._NF__set_optimizer>`
            - :meth:`NF.__set_loss <NF4HEP.NF._NF__set_loss>`
            - :meth:`NF.__set_metrics <NF4HEP.NF._NF__set_metrics>`
            - :meth:`NF.__set_callbacks <NF4HEP.NF._NF__set_callbacks>`

        - **Arguments**

            - **verbose**

                See :argument:`verbose <common_methods_arguments.verbose>`.
        """
        _, verbose_sub = self.set_verbosity(verbose)
        # self.__set_layers(verbose=verbose_sub)
        # this defines the string optimizer_string and object optimizer
        self.__set_optimizer(verbose=verbose_sub)
        # self.__set_loss(verbose=verbose_sub)  # this defines the string loss_string and the object loss
        # this defines the lists metrics_string and metrics
        self.__set_metrics(verbose=verbose_sub)
        # this defines the lists callbacks_strings and callbacks
        self.__set_callbacks(verbose=verbose_sub)

    def __load_json_and_log(self, verbose=None):
        """
        Private method used by the :meth:`NF.__init__ <NF4HEP.NF.__init__>` one 
        to import part of a previously saved
        :class:`NF <NF4HEP.NF>` object from the files 
        :attr:`NF.input_file <NF4HEP.NF.input_file>` and
        :attr:`NF.input_log_file <NF4HEP.NF.input_log_file>`.

        - **Arguments**

            - **verbose**

                See :argument:`verbose <common_methods_arguments.verbose>`.
        """
        verbose, verbose_sub = self.set_verbosity(verbose)
        start = timer()
        with open(self.input_json_file) as json_file:
            dictionary = json.load(json_file)
        self.__dict__.update(dictionary)
        with open(self.input_log_file) as json_file:
            dictionary = json.load(json_file)
        # if self.model_max != {}:
        #    self.model_max["x"] = np.array(self.model_max["x"])
        self.log = dictionary
        end = timer()
        timestamp = "datetime_" + \
            datetime.now().strftime("%Y_%m_%d_%H_%M_%S_%fZ")[:-3]
        self.log[timestamp] = {"action": "loaded object and log json",
                               "files names": [path.split(self.input_json_file)[-1],
                                               path.split(self.input_log_file)[-1]]}
        print(header_string, "\nNF json and log files loaded in",
              str(end-start), ".\n", show=verbose)

    def __load_history(self, verbose=None):
        """
        Private method used by the :meth:`NF.__init__ <NF4HEP.NF.__init__>` one 
        to set the :attr:`NF.history <NF4HEP.NF.history>` attribute
        from the file
        :attr:`NF.input_history_json_file <NF4HEP.NF.input_history_json_file>`.
        Once the attribute is set, it is used to set the 
        :attr:`NF.epochs_available <NF4HEP.NF.epochs_available>` one, determined from the 
        length of the ``"loss"`` item of the :attr:`NF.history <NF4HEP.NF.history>` dictionary.
        If the file is not found the :attr:`NF.history <NF4HEP.NF.history>` and
        :attr:`NF.epochs_available <NF4HEP.NF.epochs_available>` attributes are set to
        an empty dictionary ``{}`` and ``0``, respectively. 

        - **Arguments**

            - **verbose**

                See :argument:`verbose <common_methods_arguments.verbose>`.
        """
        verbose, verbose_sub = self.set_verbosity(verbose)
        start = timer()
        try:
            with open(self.input_history_json_file) as json_file:
                self.history = json.load(json_file)
            self.epochs_available = len(self.history['loss'])
        except:
            print(header_string, "\nNo history file available. The history attribute will be initialized to {}.\n", show=verbose)
            self.history = {}
            self.epochs_available = 0
            return
        end = timer()
        timestamp = "datetime_" + \
            datetime.now().strftime("%Y_%m_%d_%H_%M_%S_%fZ")[:-3]
        self.log[timestamp] = {"action": "loaded history json",
                               "file name": path.split(self.input_history_json_file)[-1]}
        # self.save_log(overwrite=True, verbose=verbose_sub) # log saved at the end of all loadings
        print(header_string, "\nNF history json file loaded in",
              str(end-start), ".\n", show=verbose)

    def __load_model(self, verbose=None):
        """
        Private method used by the :meth:`NF.__init__ <NF4HEP.NF.__init__>` one 
        to set the :attr:`NF.model <NF4HEP.NF.model>` attribute, 
        corresponding to the |tf_keras_model_link|, from the file
        :attr:`NF.input_tf_model_weights_h5_file <NF4HEP.NF.input_tf_model_weights_h5_file>`.
        If the file is not found the attribute is set to ``None``.

        - **Arguments**

            - **verbose**

                See :argument:`verbose <common_methods_arguments.verbose>`.
        """
        verbose, verbose_sub = self.set_verbosity(verbose)
        print(header_string,
              "\nReconstructing tf.keras model and load weights\n", show=verbose)
        start = timer()
        try:
            self.model_define_bijector(verbose=False)
            self.model_define(verbose=False)
        except:
            print(header_string, "\nUnable to re-build tf.keras model.\n", show=verbose)
            self.model = None
            return
        try:
            self.model.load_weights(self.input_tf_model_weights_h5_file)
        except:
            print(header_string,
                  "\nUnable to load tf.keras model weights.\n", show=verbose)
            self.model = None
            return
        if self.model is not None:
            try:
                self.model.history = callbacks.History()
                self.model.history.model = self.model
                self.model.history.history = self.history
                self.model.history.params = {
                    "verbose": 1, "epochs": self.epochs_available}
                self.model.history.epoch = np.arange(
                    self.epochs_available).tolist()
            except:
                print(header_string,
                      "\nNo training history available.\n", show=verbose)
                return
        end = timer()
        timestamp = "datetime_" + \
            datetime.now().strftime("%Y_%m_%d_%H_%M_%S_%fZ")[:-3]
        self.log[timestamp] = {"action": "loaded tf model h5 and tf model history pickle",
                               "file name": path.split(self.input_tf_model_weights_h5_file)[-1]}
        # self.save_log(overwrite=True, verbose=verbose_sub) # log saved at the end of all loadings
        print(header_string, "\nNF tf.keras model reconstructed in",
              str(end-start), ".\n", show=verbose)

    def __load_preprocessing(self, verbose=None):
        """
        Private method used by the :meth:`NF.__init__ <NF4HEP.NF.__init__>` one
        to set the :attr:`NF.scalerX <NF4HEP.NF.scalerX>` and 
        :attr:`NF.rotationX <NF4HEP.NF.rotationX>` attributes from the file
        :attr:`NF.input_preprocessing_pickle_file <NF4HEP.NF.input_preprocessing_pickle_file>`.
        If the file is not found the attributes are set to ``None``.

        - **Arguments**

            - **verbose**

                See :argument:`verbose <common_methods_arguments.verbose>`.
        """
        verbose, verbose_sub = self.set_verbosity(verbose)
        #print(header_string,"\nLoading preprocessing attributes\n",show=verbose)
        start = timer()
        try:
            pickle_in = open(self.input_preprocessing_pickle_file, "rb")
            self.scalerX = pickle.load(pickle_in)
            self.rotationX = pickle.load(pickle_in)
            pickle_in.close()
        except:
            print(header_string, "\nNo scalers file available. The scalerX and rotationX attributes will be initialized to None.\n", show=verbose)
            self.scalerX = None
            self.rotationX = None
            return
        end = timer()
        timestamp = "datetime_" + \
            datetime.now().strftime("%Y_%m_%d_%H_%M_%S_%fZ")[:-3]
        self.log[timestamp] = {"action": "loaded scaler and X rotation h5",
                               "file name": path.split(self.input_preprocessing_pickle_file)[-1]}
        # self.save_log(overwrite=True, verbose=verbose_sub) # log saved at the end of all loadings
        print(header_string, "\nNF preprocessing h5 file loaded in",
              str(end-start), ".\n", show=verbose)

    def __load_data_indices(self, verbose=None):
        """
        Private method used by the :meth:`NF.__init__ <NF4HEP.NF.__init__>` one 
        to set the attributes:

            - :attr:`NF.idx_train <NF4HEP.NF.idx_train>`
            - :attr:`NF.idx_val <NF4HEP.NF.idx_val>`
            - :attr:`NF.idx_test <NF4HEP.NF.idx_test>`

        from the file :attr:`NF.input_idx_h5_file <NF4HEP.NF.input_idx_h5_file>`.
        Once the attributes are set, the items ``"idx_train"``, ``"idx_val"``, and ``"idx_test"``
        of the :attr:`Data.data_dictionary <NF4HEP.Data.data_dictionary>` dictionary attribute
        of the :mod:`Data <data>` object :attr:`NF.data <NF4HEP.NF.data>`
        is updated to match the three index attributes
        If the file is not found the attributes are set to ``None`` and the :mod:`Data <data>` object is
        not touched.

        - **Arguments**

            - **verbose**

                See :argument:`verbose <common_methods_arguments.verbose>`.
        """
        verbose, verbose_sub = self.set_verbosity(verbose)
        #print(header_string,"\nLoading data indices\n",show=verbose)
        start = timer()
        try:
            h5_in = h5py.File(self.input_idx_h5_file, "r")
        except:
            print(header_string, "\nNo data indices file available. The idx_train, idx_val, and idx_test attributes will be initialized to empty arrays.\n")
            self.idx_train, self.idx_val, self.idx_test = [np.array(
                [], dtype="int"), np.array([], dtype="int"), np.array([], dtype="int")]
            return
        data = h5_in.require_group("idx")
        self.idx_train = data["idx_train"][:]
        self.idx_val = data["idx_val"][:]
        self.idx_test = data["idx_test"][:]
        self.data.data_dictionary["idx_train"] = self.idx_train
        self.data.data_dictionary["idx_val"] = self.idx_val
        self.data.data_dictionary["idx_test"] = self.idx_test
        h5_in.close()
        end = timer()
        timestamp = "datetime_" + \
            datetime.now().strftime("%Y_%m_%d_%H_%M_%S_%fZ")[:-3]
        self.log[timestamp] = {"action": "loaded data indices h5",
                               "file name": path.split(self.input_idx_h5_file)[-1]}
        # self.save_log(overwrite=True, verbose=verbose_sub) # log saved at the end of all loadings
        print(header_string, "\nNF data indices h5 file loaded in",
              str(end-start), ".\n", show=verbose)

    def __load_predictions(self, verbose=None):
        """
        Private method used by the :meth:`NF.__init__ <NF4HEP.NF.__init__>` one
        to set the :attr:`NF.predictions <NF4HEP.NF.predictions>` attribute 
        from the file
        :attr:`NF.input_predictions_h5_file <NF4HEP.NF.input_predictions_h5_file>`.
        If the file is not found the attributes is set to an empty dictionary ``{}``.

        - **Arguments**

            - **verbose**

                See :argument:`verbose <common_methods_arguments.verbose>`.
        """
        verbose, verbose_sub = self.set_verbosity(verbose)
        #print(header_string,"\nLoading existing predictions\n",show=verbose)
        start = timer()
        try:
            dictionary = dd.io.load(self.input_predictions_h5_file)
            self.predictions = dictionary
        except:
            print(header_string, "\nNo predictions file available. The predictions attribute will be initialized to {}.\n")
            self.reset_predictions(verbose=verbose_sub)
            return
        end = timer()
        timestamp = "datetime_" + \
            datetime.now().strftime("%Y_%m_%d_%H_%M_%S_%fZ")[:-3]
        self.log[timestamp] = {"action": "loaded predictions h5",
                               "file name": path.split(self.input_predictions_h5_file)[-1]}
        # self.save_log(overwrite=True, verbose=verbose_sub) # log saved at the end of all loadings
        print(header_string, "\nNF predictions h5 file loaded in",
              str(end-start), ".\n", show=verbose)

#    def __set_layers(self, verbose=None):
#        """
#        Method that defines strings representing the |tf_keras_layers_link| that are stored in the
#        :attr:`NF.layers_string <NF4HEP.NF.layers_string>` attribute.
#        These are defined from the attributes
#
#            - :attr:`NF.hidden_layers <NF4HEP.NF.hidden_layers>`
#            - :attr:`NF.batch_norm <NF4HEP.NF.batch_norm>`
#            - :attr:`NF.dropout_rate <NF4HEP.NF.dropout_rate>`
#
#        If |tf_keras_batch_normalization_link| layers are specified in the
#        :attr:`NF.hidden_layers <NF4HEP.NF.hidden_layers>` attribute, then the
#        :attr:`NF.batch_norm <NF4HEP.NF.batch_norm>` attribute is ignored. Otherwise,
#        if :attr:`NF.batch_norm <NF4HEP.NF.batch_norm>` is ``True``, then a
#        |tf_keras_batch_normalization_link| layer is added after the input layer and before
#        each |tf_keras_dense_link| layer.
#
#        If |tf_keras_dropout_link| layers are specified in the
#        :attr:`NF.hidden_layers <NF4HEP.NF.hidden_layers>` attribute, then the
#        :attr:`NF.dropout_rate <NF4HEP.NF.dropout_rate>` attribute is ignored. Otherwise,
#        if :attr:`NF.dropout_rate <NF4HEP.NF.dropout_rate>` is larger than ``0``, then
#        a |tf_keras_dropout_link| layer is added after each |tf_keras_dense_link| layer
#        (but the output layer).
#
#        The method also sets the three attributes:
#
#            - :attr:`NF.layers <NF4HEP.NF.layers>` (set to an empty list ``[]``, filled by the
#                :meth:`NF.model_define <NF4HEP.NF.model_define>` method)
#            - :attr:`NF.model_params <NF4HEP.NF.model_params>`
#            - :attr:`NF.model_trainable_params <NF4HEP.NF.model_trainable_params>`
#            - :attr:`NF.model_non_trainable_params <NF4HEP.NF.model_non_trainable_params>`
#
#        - **Arguments**
#
#            - **verbose**
#
#                See :argument:`verbose <common_methods_arguments.verbose>`.
#
#        - **Produces file**
#
#            - :attr:`NF.output_log_file <NF4HEP.NF.output_log_file>`
#        """
#        verbose, verbose_sub = self.set_verbosity(verbose)
#        print(header_string,"\nSetting hidden layers\n",show=verbose)
#        self.layers_string = []
#        self.layers = []
#        layer_string = "layers.Input(shape=(" + str(self.ndims)+",))"
#        self.layers_string.append(layer_string)
#        print("Added Input layer: ", layer_string, show=verbose)
#        i = 0
#        if "dropout" in str(self.hidden_layers).lower():
#            insert_dropout = False
#            self.dropout_rate = "custom"
#        elif "dropout" not in str(self.hidden_layers).lower() and self.dropout_rate != 0:
#            insert_dropout = True
#        else:
#            insert_dropout = False
#        if "batchnormalization" in str(self.hidden_layers).lower():
#            self.batch_norm = "custom"
#        for layer in self.hidden_layers:
#            if type(layer) == str:
#                if "(" in layer:
#                    layer_string = "layers."+layer
#                else:
#                    layer_string = "layers."+layer+"()"
#            elif type(layer) == dict:
#                try:
#                    name = layer["name"]
#                except:
#                    raise Exception("The layer ", str(layer), " has unspecified name.")
#                try:
#                    args = layer["args"]
#                except:
#                    args = []
#                try:
#                    kwargs = layer["kwargs"]
#                except:
#                    kwargs = {}
#                layer_string = utils.build_method_string_from_dict("layers", name, args, kwargs)
#            elif type(layer) == list:
#                units = layer[0]
#                activation = layer[1]
#                try:
#                    initializer = layer[2]
#                except:
#                    initializer = None
#                if activation == "selu":
#                    layer_string = "layers.Dense(" + str(units) + ", activation='" + activation + "', kernel_initializer='lecun_normal')"
#                elif activation != "selu" and initializer != None:
#                    layer_string = "layers.Dense(" + str(units) + ", activation='" + activation + "')"
#                else:
#                    layer_string = "layers.Dense(" + str(units)+", activation='" + activation + "', kernel_initializer='" + initializer + "')"
#            else:
#                layer_string = None
#                print("Invalid input for layer: ", layer,". The layer will not be added to the model.", show=verbose)
#            if self.batch_norm == True and "dense" in layer_string.lower():
#                self.layers_string.append("layers.BatchNormalization()")
#                print("Added hidden layer: layers.BatchNormalization()", show=verbose)
#                i = i + 1
#            if layer_string is not None:
#                try:
#                    eval(layer_string)
#                    self.layers_string.append(layer_string)
#                    print("Added hidden layer: ", layer_string, show=verbose)
#                except Exception as e:
#                    print(e)
#                    print("Could not add layer", layer_string, "\n", show=verbose)
#                i = i + 1
#            if insert_dropout:
#                try:
#                    act = eval(layer_string+".activation")
#                    if "selu" in str(act).lower():
#                        layer_string = "layers.AlphaDropout("+str(self.dropout_rate)+")"
#                        self.layers_string.append(layer_string)
#                        print("Added hidden layer: ", layer_string, show=verbose)
#                        i = i + 1
#                    elif "linear" not in str(act):
#                        layer_string = "layers.Dropout("+str(self.dropout_rate)+")"
#                        self.layers_string.append(layer_string)
#                        print("Added hidden layer: ", layer_string, show=verbose)
#                        i = i + 1
#                except:
#                    layer_string = "layers.AlphaDropout("+str(self.dropout_rate)+")"
#                    self.layers_string.append(layer_string)
#                    print("Added hidden layer: ", layer_string, show=verbose)
#                    i = i + 1
#        if self.batch_norm == True and "dense" in layer_string.lower():
#            self.layers_string.append("layers.BatchNormalization()")
#            print("Added hidden layer: layers.BatchNormalization()", show=verbose)
#        #outputLayer = layers.Dense(1, activation=self.act_func_out_layer)
#
#        layer_string = "layers.Dense(1, activation='"+str(self.act_func_out_layer)+"')"
#        self.layers_string.append(layer_string)
#        print("Added Output layer: ", layer_string,".\n", show=verbose)
#        timestamp = "datetime_"+datetime.now().strftime("%Y_%m_%d_%H_%M_%S_%fZ")[:-3]
#        self.log[timestamp] = {"action": "layers set",
#                               "metrics": self.layers_string}

    def __set_optimizer(self, verbose=None):
        """
        Private method used by the 
        :meth:`NF.__set_tf_objects <NF4HEP.NF._NF__set_tf_objects>` one
        to set the |tf_keras_optimizers_link| object. It sets the
        :attr:`NF.optimizer_string <NF4HEP.NF.optimizer_string>`
        and :attr:`NF.optimizer <NF4HEP.NF.optimizer>` attributes.
        The former is set from the :attr:`NF.__model_optimizer_inputs <NF4HEP.NF._NF__model_optimizer_inputs>` 
        dictionary.  The latter is set to ``None`` and then updated by the 
        :meth:`NF.model_compile <NF4HEP.NF.model_compile>` method during model compilation). 

        - **Arguments**

            - **verbose**

                See :argument:`verbose <common_methods_arguments.verbose>`.
        """
        verbose, verbose_sub = self.set_verbosity(verbose)
        print(header_string, "\nSetting optimizer\n", show=verbose)
        self.optimizer = None
        if type(self.__model_optimizer_inputs) == str:
            if "(" in self.__model_optimizer_inputs:
                optimizer_string = "optimizers." + \
                    self.__model_optimizer_inputs.replace("optimizers.", "")
            else:
                optimizer_string = "optimizers." + \
                    self.__model_optimizer_inputs.replace(
                        "optimizers.", "") + "()"
        elif type(self.__model_optimizer_inputs) == dict:
            try:
                name = self.__model_optimizer_inputs["name"]
            except:
                raise Exception("The optimizer ", str(
                    self.__model_optimizer_inputs), " has unspecified name.")
            try:
                args = self.__model_optimizer_inputs["args"]
            except:
                args = []
            try:
                kwargs = self.__model_optimizer_inputs["kwargs"]
            except:
                kwargs = {}
            optimizer_string = utils.build_method_string_from_dict(
                "optimizers", name, args, kwargs)
        else:
            raise Exception(
                "Could not set optimizer. The model_optimizer_inputs argument does not have a valid format (str or dict).")
        try:
            optimizer = eval(optimizer_string)
            self.optimizer_string = optimizer_string
            self.optimizer = eval(self.optimizer_string)
            print("Optimizer set to:", self.optimizer_string, "\n", show=verbose)
        except Exception as e:
            print(e)
            raise Exception("Could not set optimizer", optimizer_string, "\n")
        timestamp = "datetime_" + \
            datetime.now().strftime("%Y_%m_%d_%H_%M_%S_%fZ")[:-3]
        self.log[timestamp] = {"action": "optimizer set",
                               "optimizer": self.optimizer_string}

    # def __set_loss(self, verbose=None):
    #    """
    #    Private method used by the
    #    :meth:`NF.__set_tf_objects <NF4HEP.NF._NF__set_tf_objects>` one
    #    to set the loss object (it could be a |tf_keras_losses_link| object or a custom loss defined
    #    in the :mod:`Dnn_likelihood <dnn_likelihood>` module). It sets the
    #    :attr:`NF.loss_string <NF4HEP.NF.loss_string>`
    #    and :attr:`NF.loss <NF4HEP.NF.loss>` attributes. The former is set from the
    #    :attr:`NF.__model_compile_inputs <NF4HEP.NF._NF__model_compile_inputs>`
    #    dictionary, while the latter is set by evaluating the former.
#
    #    - **Arguments**
#
    #        - **verbose**
    #
    #            See :argument:`verbose <common_methods_arguments.verbose>`.
    #    """
    #    verbose, verbose_sub = self.set_verbosity(verbose)
    #    print(header_string,"\nSetting loss\n",show=verbose)
    #    self.loss = None
    #    ls = self.__model_compile_inputs["loss"]
    #    if type(ls) == str:
    #        ls = ls.replace("losses.","")
    #        try:
    #            eval("losses." + ls)
    #            loss_string = "losses." + ls
    #        except:
    #            try:
    #                eval("losses."+losses.deserialize(ls).__name__)
    #                loss_string = "losses."+losses.deserialize(ls).__name__
    #            except:
    #                try:
    #                    eval("self."+ls)
    #                    loss_string = "self."+ls
    #                except:
    #                    try:
    #                        eval("self.custom_losses."+custom_losses.metric_name_unabbreviate(ls))
    #                        loss_string = "self.custom_losses." + custom_losses.metric_name_unabbreviate(ls)
    #                    except:
    #                        loss_string = None
    #    elif type(ls) == dict:
    #        try:
    #            name = ls["name"]
    #        except:
    #            raise Exception("The optimizer ", str(ls), " has unspecified name.")
    #        try:
    #            args = ls["args"]
    #        except:
    #            args = []
    #        try:
    #            kwargs = ls["kwargs"]
    #        except:
    #            kwargs = {}
    #        loss_string = utils.build_method_string_from_dict("losses", name, args, kwargs)
    #    else:
    #        raise Exception("Could not set loss. The model_compile_inputs['loss'] item does not have a valid format (str or dict).")
    #    if loss_string is not None:
    #        try:
    #            eval(loss_string)
    #            self.loss_string = loss_string
    #            self.loss = eval(self.loss_string)
    #            if "self." in loss_string:
    #                print("Custom loss set to:",loss_string.replace("self.",""),".\n",show=verbose)
    #            else:
    #                print("Loss set to:",loss_string,".\n",show=verbose)
    #        except Exception as e:
    #            print(e)
    #            raise Exception("Could not set loss", loss_string, "\n")
    #    else:
    #        raise Exception("Could not set loss", loss_string, "\n")
    #    timestamp = "datetime_"+datetime.now().strftime("%Y_%m_%d_%H_%M_%S_%fZ")[:-3]
    #    self.log[timestamp] = {"action": "loss set",
    #                           "loss": self.loss_string}

    def __set_metrics(self, verbose=None):
        """
        Private method used by the 
        :meth:`NF.__set_tf_objects <NF4HEP.NF._NF__set_tf_objects>` one
        to set the metrics objects (as for the loss, metrics could be |tf_keras_metrics_link|
        objects or a custom metrics defined in the :mod:`Dnn_likelihood <dnn_likelihood>` module). 
        It sets the :attr:`NF.metrics_string <NF4HEP.NF.metrics_string>`
        and :attr:`NF.metrics <NF4HEP.NF.metrics>` attributes. The former is set from the
        :attr:`NF.__model_compile_inputs <NF4HEP.NF._NF__model_compile_inputs>` 
        dictionary, while the latter is set by evaluating each item in the the former.

        - **Arguments**

            - **verbose**

                See :argument:`verbose <common_methods_arguments.verbose>`.
        """
        verbose, verbose_sub = self.set_verbosity(verbose)
        self.metrics_string = []
        self.metrics = []
        print(header_string, "\nSetting metrics\n", show=verbose)
        for met in self.__model_compile_inputs["metrics"]:
            if type(met) == str:
                met = met.replace("metrics.", "")
                try:
                    eval("metrics." + met)
                    metric_string = "metrics." + met
                except:
                    try:
                        eval("metrics."+metrics.deserialize(met).__name__)
                        metric_string = "metrics." + \
                            metrics.deserialize(met).__name__
                    except:
                        try:
                            eval("self."+met)
                            metric_string = "self."+met
                        except:
                            # try:
                            #    eval("custom_losses."+custom_losses.metric_name_unabbreviate(met))
                            #    metric_string = "custom_losses." + custom_losses.metric_name_unabbreviate(met)
                            # except:
                            metric_string = None
            elif type(met) == dict:
                try:
                    name = met["name"]
                except:
                    raise Exception("The metric ", str(
                        met), " has unspecified name.")
                try:
                    args = met["args"]
                except:
                    args = []
                try:
                    kwargs = met["kwargs"]
                except:
                    kwargs = {}
                metric_string = utils.build_method_string_from_dict(
                    "metrics", name, args, kwargs)
            else:
                metric_string = None
                print("Invalid input for metric: ", str(
                    met), ". The metric will not be added to the model.", show=verbose)
            if metric_string is not None:
                try:
                    eval(metric_string)
                    self.metrics_string.append(metric_string)
                    if "self." in metric_string:
                        print("\tAdded custom metric:", metric_string.replace(
                            "self.", ""), show=verbose)
                    else:
                        print("\tAdded metric:", metric_string, show=verbose)
                except Exception as e:
                    print(e)
                    print("Could not add metric",
                          metric_string, "\n", show=verbose)
            else:
                print("Could not add metric", str(met), "\n", show=verbose)
        for metric_string in self.metrics_string:
            self.metrics.append(eval(metric_string))
        print("", show=verbose)
        timestamp = "datetime_" + \
            datetime.now().strftime("%Y_%m_%d_%H_%M_%S_%fZ")[:-3]
        self.log[timestamp] = {"action": "metrics set",
                               "metrics": self.metrics_string}

    def __set_callbacks(self, verbose=None):
        """
        Private method used by the 
        :meth:`NF.__set_tf_objects <NF4HEP.NF._NF__set_tf_objects>` one
        to set the |tf_keras_callbacks_link| objects. It sets the
        :attr:`NF.callbacks_strings <NF4HEP.NF.callbacks_strings>`
        and :attr:`NF.callbacks <NF4HEP.NF.callbacks>` attributes. The former is set from the
        :attr:`NF.__model_callbacks_inputs <NF4HEP.NF._NF__model_callbacks_inputs>` 
        dictionary, while the latter is set by evaluating each item in the the former.

        - **Arguments**

            - **verbose**

                See :argument:`verbose <common_methods_arguments.verbose>`.
        """
        verbose, verbose_sub = self.set_verbosity(verbose)
        print(header_string, "\nSetting callbacks\n", show=verbose)
        self.callbacks_strings = []
        self.callbacks = []
        for cb in self.__model_callbacks_inputs:
            if type(cb) == str:
                if cb == "ModelCheckpoint":
                    self.output_checkpoints_folder = path.join(
                        self.output_folder, "checkpoints")
                    self.output_checkpoints_files = path.join(
                        self.output_checkpoints_folder, self.name+"_checkpoint.{epoch:02d}-{val_loss:.2f}.h5")
                    utils.check_create_folder(self.output_checkpoints_folder)
                    cb_string = "callbacks.ModelCheckpoint(filepath=r'" + \
                        self.output_checkpoints_files+"')"
                elif cb == "TensorBoard":
                    self.output_tensorboard_log_dir = path.join(
                        self.output_folder, "tensorboard_logs")
                    utils.check_create_folder(self.output_tensorboard_log_dir)
                    cb_string = "callbacks.TensorBoard(log_dir=r'" + \
                        self.output_tensorboard_log_dir+"')"
                elif cb == "PlotLossesKeras":
                    self.output_figure_plot_losses_keras_file = path.join(
                        self.output_figures_folder, self.output_figures_base_file_name+"_plot_losses_keras.pdf")
                    utils.check_rename_file(
                        self.output_figure_plot_losses_keras_file)
                    cb_string = "PlotLossesKeras(outputs=[MatplotlibPlot(figpath = r'" + \
                        self.output_figure_plot_losses_keras_file+"')])"
                elif cb == "ModelCheckpoint":
                    self.output_checkpoints_folder = path.join(
                        self.output_folder, "checkpoints")
                    self.output_checkpoints_files = path.join(
                        self.output_checkpoints_folder, self.name+"_checkpoint.{epoch:02d}-{val_loss:.2f}.h5")
                    utils.check_create_folder(self.output_checkpoints_folder)
                    cb_string = "callbacks.ModelCheckpoint(filepath=r'" + \
                        self.output_checkpoints_files+"')"
                else:
                    if "(" in cb:
                        cb_string = "callbacks."+cb.replace("callbacks.", "")
                    else:
                        cb_string = "callbacks." + \
                            cb.replace("callbacks.", "")+"()"
            elif type(cb) == dict:
                try:
                    name = cb["name"]
                except:
                    raise Exception("The layer ", str(
                        cb), " has unspecified name.")
                try:
                    args = cb["args"]
                except:
                    args = []
                try:
                    kwargs = cb["kwargs"]
                except:
                    kwargs = {}
                if name == "ModelCheckpoint":
                    self.output_checkpoints_folder = path.join(
                        self.output_folder, "checkpoints")
                    self.output_checkpoints_files = path.join(
                        self.output_checkpoints_folder, self.name+"_checkpoint.{epoch:02d}-{val_loss:.2f}.h5")
                    utils.check_create_folder(self.output_checkpoints_folder)
                    kwargs["filepath"] = self.output_checkpoints_files
                elif name == "TensorBoard":
                    self.output_tensorboard_log_dir = path.join(
                        self.output_folder, "tensorboard_logs")
                    utils.check_create_folder(self.output_tensorboard_log_dir)
                    #utils.check_create_folder(path.join(self.output_folder, "tensorboard_logs/fit"))
                    kwargs["log_dir"] = self.output_tensorboard_log_dir
                elif name == "PlotLossesKeras":
                    self.output_figure_plot_losses_keras_file = path.join(
                        self.output_figures_folder, self.output_figures_base_file_name+"_plot_losses_keras.pdf")
                    utils.check_rename_file(
                        self.output_figure_plot_losses_keras_file)
                    kwargs["outputs"] = "[MatplotlibPlot(figpath = r'" + \
                        self.output_figure_plot_losses_keras_file+"')]"
                for key, value in kwargs.items():
                    if key == "monitor" and type(value) == str:
                        if "val_" in value:
                            value = value.split("val_")[1]
                        if value == "loss":
                            value = "val_loss"
                        # else:
                        #    value = "val_" + custom_losses.metric_name_unabbreviate(value)
                cb_string = utils.build_method_string_from_dict(
                    "callbacks", name, args, kwargs)
            else:
                cb_string = None
                print("Invalid input for callback: ", cb,
                      ". The callback will not be added to the model.", show=verbose)
            if cb_string is not None:
                try:
                    eval(cb_string)
                    self.callbacks_strings.append(cb_string)
                    print("\tAdded callback:", cb_string, show=verbose)
                except Exception as e:
                    print("Could not set callback",
                          cb_string, "\n", show=verbose)
                    print(e)
            else:
                print("Could not set callback", cb_string, "\n", show=verbose)
        for cb_string in self.callbacks_strings:
            self.callbacks.append(eval(cb_string))
        print("", show=verbose)
        timestamp = "datetime_" + \
            datetime.now().strftime("%Y_%m_%d_%H_%M_%S_%fZ")[:-3]
        self.log[timestamp] = {"action": "callbacks set",
                               "callbacks": self.callbacks_strings}

    def __set_epochs_to_run(self):
        """
        Private method that returns the number of epochs to run computed as the difference between the value of
        :attr:`NF.epochs_required <NF4HEP.NF.epochs_required>` and the value of
        :attr:`NF.epochs_available <NF4HEP.NF.epochs_available>`, i.e. the 
        number of epochs available in 
        :attr:`NF.history <NF4HEP.NF.history>`.
        """
        if self.epochs_required <= self.epochs_available and self.epochs_available > 0:
            epochs_to_run = 0
        else:
            epochs_to_run = self.epochs_required-self.epochs_available
        return epochs_to_run

    def __set_pars_labels(self, pars_labels):
        """
        Private method that returns the ``pars_labels`` choice based on the ``pars_labels`` input.

        - **Arguments**

            - **pars_labels**

                Could be either one of the keyword strings ``"original"`` and ``"generic"`` or a list of labels
                strings with the length of the parameters array. If ``pars_labels="original"`` or ``pars_labels="generic"``
                the function returns the value of :attr:`Sampler.pars_labels <NF4HEP.Sampler.pars_labels>`
                or :attr:`Sampler.pars_labels_auto <NF4HEP.Sampler.pars_labels_auto>`, respectively,
                while if ``pars_labels`` is a list, the function just returns the input.

                    - **type**: ``list`` or ``str``
                    - **shape of list**: ``[ ]``
                    - **accepted strings**: ``"original"``, ``"generic"``
        """
        if pars_labels == "original":
            return self.pars_labels
        elif pars_labels == "generic":
            return self.pars_labels_auto
        else:
            return pars_labels

    def define_rotation(self, verbose=None):
        """
        Method that defines the rotation matrix that diagonalizes the covariance matrix of the 
        ``data_X``, making them uncorrelated.
        Such matrix is defined based on the 
        :attr:`NF.rotationX_bool <NF4HEP.NF.rotationX_bool>` attribute
        When the boolean attribute is ``False`` the matrix is set to the identity matrix.
        The method computes the rotation matrix by calling the corresponding method 
        :meth:`Data.define_rotation <NF4HEP.Data.define_rotation>` method of the 
        :mod:`Data <data>` object :attr:`NF.data <NF4HEP.NF.data>`.

        Note: Data are transformed with the matrix ``V`` through ``np.dot(X,V)`` and transformed back throug
        ``np.dot(X_diag,np.transpose(V))``.

        - **Arguments**

           - **verbose**

                See :argument:`verbose <common_methods_arguments.verbose>`.
        """
        verbose, verbose_sub = self.set_verbosity(verbose)
        print(header_string, "\nDefining data rotation\n", show=verbose)
        self.rotationX = self.data.define_rotation(
            self.X_train, self.rotationX_bool, verbose=verbose_sub)
        timestamp = "datetime_" + \
            datetime.now().strftime("%Y_%m_%d_%H_%M_%S_%fZ")[:-3]
        self.log[timestamp] = {"action": "defined X rotation",
                               "rotation X": self.rotationX_bool}

    def define_scaler(self, verbose=None):
        """
        Method that defines |standard_scaler_link| based on the values of the
        :attr:`NF.scalerX_bool <NF4HEP.NF.scalerX_bool>` attribute.
        When the boolean attribute is ``True`` the scaler is fit to the corresponding training data, otherwise it is set
        equal to the identity.
        The method computes the scaler by calling the corresponding method 
        :meth:`Data.define_scaler <NF4HEP.Data.define_scaler>` method of the 
        :mod:`Data <data>` object :attr:`NF.data <NF4HEP.NF.data>`.

        Note: The X scaler is defined on data rotated with the 
        :attr:`NF.rotationX <NF4HEP.NF.rotationX>` matrix.

        - **Arguments**

           - **verbose**

                See :argument:`verbose <common_methods_arguments.verbose>`.
        """
        verbose, verbose_sub = self.set_verbosity(verbose)
        print(header_string, "\nSetting standard scalers\n", show=verbose)
        self.scalerX = self.data.define_scaler(
            np.dot(self.X_train, self.rotationX), self.scalerX_bool, verbose=verbose_sub)
        timestamp = "datetime_" + \
            datetime.now().strftime("%Y_%m_%d_%H_%M_%S_%fZ")[:-3]
        self.log[timestamp] = {"action": "defined scaler",
                               "scaler X": self.scalerX_bool}
        # self.save_log(overwrite=True, verbose=verbose_sub) #log saved by generate_train_data

    def transform_data_X(self, X, verbose=None):
        """
        Method that transforms X data applying first the rotation 
        :attr:`NF.rotationX <NF4HEP.NF.rotationX>`
        and then the transformation with scalerX
        :attr:`NF.scalerX <NF4HEP.NF.scalerX>` .
        """
        return self.scalerX.transform(np.dot(X, self.rotationX))

    def inverse_transform_data_X(self, X, verbose=None):
        """
        Inverse of the method
        :meth:`Data.transform_data_X <NF4HEP.Data.transform_data_X>`.
        """
        return np.dot(self.scalerX.inverse_transform(X), np.transpose(self.rotationX))

    def generate_train_data(self, verbose=None):
        """
        Method that generates training and validation data corresponding to the attributes

            - :attr:`NF.idx_train <NF4HEP.NF.idx_train>`
            - :attr:`NF.X_train <NF4HEP.NF.X_train>`
            - :attr:`NF.idx_val <NF4HEP.NF.idx_val>`
            - :attr:`NF.X_val <NF4HEP.NF.X_val>`

        Data are generated by calling the methods
        :meth:`Data.update_train_data <NF4HEP.Data.update_train_data>` or
        :meth:`Data.generate_train_data <NF4HEP.Data.generate_train_data>` of the 
        :mod:`Data <data>` object :attr:`NF.data <NF4HEP.NF.data>`
        depending on the value of :attr:`NF.same_data <NF4HEP.NF.same_data>`.

        When the :class:`NF <NF4HEP.NF>` object is not part of a
        :class:`DnnLikEnsemble <NF4HEP.NFEnsemble>` object, that is when the
        :attr:`NF.standalone <NF4HEP.NF.standalone>` attribute is ``True``, 
        or when the :attr:`NF.same_data <NF4HEP.NF.same_data>`
        attribute is ``True``, that means that all members of the ensemble will share the same data (or a
        subset of the same data if they have different number of points), then data are kept up-to-date in the 
        :attr:`Data.data_dictionary <NF4HEP.Data.data_dictionary>` attribute of the 
        :mod:`Data <data>` object :attr:`NF.data <NF4HEP.NF.data>`.
        This means that data are not generated again if they are already available from another member and that
        if the number of points is increased, data are added to the existing ones and are not re-generated from scratch.

        When instead the :class:`NF <NF4HEP.NF>` object is part of a
        :class:`DnnLikEnsemble <NF4HEP.NFEnsemble>` object and the
        :attr:`NF.same_data <NF4HEP.NF.same_data>` attribute is ``False``,
        data are re-generated from scratch at each call of the method.

        The method also generates the attributes

            - :attr:`NF.scalerX <NF4HEP.NF.scalerX>`
            - :attr:`NF.rotationX <NF4HEP.NF.rotationX>`

        by calling the
        :meth:`NF.define_rotation <NF4HEP.NF.define_rotation>` and
        :meth:`Data.define_scaler <NF4HEP.Data.define_scaler>` methods. All transformations
        are defined starting from the training data :attr:`NF.X_train <NF4HEP.NF.X_train>`.

        - **Arguments**

           - **verbose**

                See :argument:`verbose <common_methods_arguments.verbose>`.

        - **Produces file**

            - :attr:`NF.output_log_file <NF4HEP.NF.output_log_file>`
        """
        verbose, verbose_sub = self.set_verbosity(verbose)
        print(header_string, "\nGenerating train/validation data\n", show=verbose)
        # Generate data
        if self.same_data:
            self.data.update_train_data(
                self.npoints_train, self.npoints_val, self.seed, verbose=verbose)
        else:
            self.data.generate_train_data(
                self.npoints_train, self.npoints_val, self.seed, verbose=verbose)
        self.idx_train = self.data.data_dictionary["idx_train"][:self.npoints_train]
        self.X_train = self.data.data_dictionary["X_train"][:self.npoints_train].astype(
            self.dtype)
        self.idx_val = self.data.data_dictionary["idx_val"][:self.npoints_train]
        self.X_val = self.data.data_dictionary["X_val"][:self.npoints_val].astype(
            self.dtype)
        self.pars_bounds_train = np.vstack(
            [np.min(self.X_train, axis=0), np.max(self.X_train, axis=0)]).T
        # Define transformations
        self.define_rotation(verbose=verbose_sub)
        self.define_scaler(verbose=verbose_sub)
        timestamp = "datetime_" + \
            datetime.now().strftime("%Y_%m_%d_%H_%M_%S_%fZ")[:-3]
        self.log[timestamp] = {"action": "generated train data",
                               "data": ["idx_train", "X_train", "idx_val", "X_val"],
                               "npoints train": self.npoints_train,
                               "npoints val": self.npoints_val}
        self.save_log(overwrite=True, verbose=verbose_sub)

    def generate_test_data(self, verbose=None):
        """
        Method that generates test data corresponding to the attributes

            - :attr:`NF.idx_test <NF4HEP.NF.idx_test>`
            - :attr:`NF.X_test <NF4HEP.NF.X_test>`

        Data are generated by calling the methods
        :meth:`Data.update_test_data <NF4HEP.Data.update_test_data>` or
        :meth:`Data.generate_test_data <NF4HEP.Data.generate_test_data>` of the 
        :mod:`Data <data>` object :attr:`NF.data <NF4HEP.NF.data>`
        depending on the value of :attr:`NF.same_data <NF4HEP.NF.same_data>`.

        When the :class:`NF <NF4HEP.NF>` object is not part of a
        :class:`DnnLikEnsemble <NF4HEP.NFEnsemble>` object, that is when the
        :attr:`NF.standalone <NF4HEP.NF.standalone>` attribute is ``True``, 
        or when the :attr:`NF.same_data <NF4HEP.NF.same_data>`
        attribute is ``True``, that means that all members of the ensemble will share the same data (or a
        subset of the same data if they have different number of points), then data are kept up-to-date in the 
        :attr:`Data.data_dictionary <NF4HEP.Data.data_dictionary>` attribute of the 
        :mod:`Data <data>` object :attr:`NF.data <NF4HEP.NF.data>`.
        This means that data are not generated again if they are already available from another member and that
        if the number of points is increased, data are added to the existing ones and are not re-generated from scratch.

        When instead the :class:`NF <NF4HEP.NF>` object is part of a
        :class:`DnnLikEnsemble <NF4HEP.NFEnsemble>` object and the
        :attr:`NF.same_data <NF4HEP.NF.same_data>` attribute is ``False``,
        data are re-generated from scratch at each call of the method.

        - **Arguments**

           - **verbose**

                See :argument:`verbose <common_methods_arguments.verbose>`.

        - **Produces file**

            - :attr:`NF.output_log_file <NF4HEP.NF.output_log_file>`
        """
        verbose, verbose_sub = self.set_verbosity(verbose)
        print(header_string, "\nGenerating test data\n", show=verbose)
        # Generate data
        if self.same_data:
            self.data.update_test_data(
                self.npoints_test, self.seed, verbose=verbose)
        else:
            self.data.generate_test_data(
                self.npoints_test, self.seed, verbose=verbose)
        self.idx_test = self.data.data_dictionary["idx_test"][:self.npoints_train]
        self.X_test = self.data.data_dictionary["X_test"][:self.npoints_test].astype(
            self.dtype)
        timestamp = "datetime_" + \
            datetime.now().strftime("%Y_%m_%d_%H_%M_%S_%fZ")[:-3]
        self.log[timestamp] = {"action": "generated test data",
                               "data": ["idx_test", "X_test"],
                               "npoints test": self.npoints_test}
        self.save_log(overwrite=True, verbose=verbose_sub)

    def model_define_bijector(self, verbose=None):
        """
        Method that defines the Normalizing Flow Bijector using the external class
        corresponding to the required algorithm. The different options are summarized in the following
        table:

        +----------------------------------------------+--------------+-------------------------------------+
        | Algorithm                                    | File         | Object                              |
        +==============================================+==============+=====================================+
        | Masked Autoregressive Flow (MAF)             | maf.py       | :class:`MAF <NF4HEP.MAF>`           |
        +----------------------------------------------+--------------+-------------------------------------+
        | Real-valued Non-Volume Preserving (RealNVP)  | realnvp.py   | :class:`RealNVP <NF4HEP.RealNVP>`   |
        +----------------------------------------------+--------------+-------------------------------------+
        | Coupling Spline (CSpline)                    | cspline.py   | :class:`CSpline <NF4HEP.CSpline>`   |
        +----------------------------------------------+--------------+-------------------------------------+
        | Rational Quadrative Spline (RQSpliine)       | rqspline.py  | :class:`RQSpline <NF4HEP.RQSpline>` |
        +----------------------------------------------+--------------+-------------------------------------+

        The method sets the three attributes:

            - :attr:`NF.NN <NF4HEP.NF.NN>`
            - :attr:`NF.Bijector <NF4HEP.NF.Bijector>`
            - :attr:`NF.Flow <NF4HEP.NF.Flow>`

        depending on the value of the ``"name"`` item of the 
        :attr:`NF.__model_chain_inputs <NF4HEP.NF._NF__model_chain_inputs>` dictionary.
        """
        verbose, verbose_sub = self.set_verbosity(verbose)
        print(header_string, "\nSetting the Normalizing Flow Bijector\n", show=verbose)
        start = timer()
        flow_type = self.flow_type
        if flow_type == "MAF":
            self.Flow = MAFFlow(model_define_inputs=self.__model_define_inputs,
                                model_chain_inputs=self.__model_chain_inputs,
                                verbose=verbose_sub)
        elif flow_type == "RealNVP":
            self.Flow = RealNVPFlow(model_define_inputs=self.__model_define_inputs,
                                    model_chain_inputs=self.__model_chain_inputs,
                                    verbose=verbose_sub)
        elif flow_type == "CSpline":
            self.Flow = CSplineFlow(model_define_inputs=self.__model_define_inputs,
                                    model_chain_inputs=self.__model_chain_inputs,
                                    verbose=verbose_sub)
        elif flow_type == "RQSpline":
            self.Flow = RQSplineFlow(model_define_inputs=self.__model_define_inputs,
                                     model_chain_inputs=self.__model_chain_inputs,
                                     verbose=verbose_sub)
        else:
            raise Exception("The algorithm",)
        self.Bijector = self.Flow.Bijector
        self.NN = self.Flow.Bijector.NN
        end = timer()
        timestamp = "datetime_" + \
            datetime.now().strftime("%Y_%m_%d_%H_%M_%S_%fZ")[:-3]
        self.log[timestamp] = {"action": "defined bijector",
                               "flow_type": flow_type}
        self.save_log(overwrite=True, verbose=verbose_sub)
        print("Bijector for Normalizing Flow", self.name,
              "defined in", str(end-start), "s.\n", show=verbose)

    def model_define(self, verbose=None):
        """
        Method that defines the |tf_keras_model_link| stored in the 
        :attr:`NF.model <NF4HEP.NF.model>` attribute.
        The model is defined from the attribute

            - :attr:`NF.layers_string <NF4HEP.NF.layers_string>`

        created by the method :meth:`NF.__set_layers <NF4HEP.NF._NF__set_layers>`
        during initialization. Each layer string is evaluated and appended to the list attribute
        :attr:`NF.layers <NF4HEP.NF.layers>`

        The method also sets the three attributes:

            - :attr:`NF.model_params <NF4HEP.NF.model_params>`
            - :attr:`NF.model_trainable_params <NF4HEP.NF.model_trainable_params>`
            - :attr:`NF.model_non_trainable_params <NF4HEP.NF.model_non_trainable_params>`

        - **Arguments**

            - **verbose**

                See :argument:`verbose <common_methods_arguments.verbose>`.

        - **Produces file**

            - :attr:`NF.output_log_file <NF4HEP.NF.output_log_file>`
        """
        verbose, verbose_sub = self.set_verbosity(verbose)
        print(header_string, "\nDefining Keras model\n", show=verbose)
        start = timer()
        if self.Flow == None:
            self.model_define_bijector(verbose=verbose_sub)
        x = Input(shape=(self.ndims,), dtype=self.dtype)
        self.trainable_distribution = tfd.TransformedDistribution(
            self.base_distribution, self.Flow)
        self.log_prob = self.trainable_distribution.log_prob(x)
        self.model = Model(x, self.log_prob)
        self.model_params = int(self.model.count_params())
        self.model_trainable_params = int(
            np.sum([K.count_params(p) for p in self.model.trainable_weights]))
        self.model_non_trainable_params = int(
            np.sum([K.count_params(p) for p in self.model.non_trainable_weights]))
        end = timer()
        timestamp = "datetime_" + \
            datetime.now().strftime("%Y_%m_%d_%H_%M_%S_%fZ")[:-3]
        summary_list = []
        self.model.summary(
            print_fn=lambda x: summary_list.append(x.replace("\"", "'")))
        self.log[timestamp] = {"action": "defined tf model",
                               "model summary": summary_list}
        self.save_log(overwrite=True, verbose=verbose_sub)
        print("Model for Normalizing Flow", self.name,
              "defined in", str(end-start), "s.\n", show=verbose)

        def print_fn(string):
            print(string, show=verbose)
        self.model.summary(print_fn=print_fn)

    def model_compile(self, verbose=None):
        """
        Method that compiles the |tf_keras_model_link| stored in the 
        :attr:`NF.model <NF4HEP.NF.model>` attribute.
        The model is compiled by calling the |tf_keras_model_compile_link| method and passing it the attributes

            - :attr:`NF.loss <NF4HEP.NF.loss>`
            - :attr:`NF.optimizer <NF4HEP.NF.optimizer>`
            - :attr:`NF.metrics <NF4HEP.NF.metrics>`
            - :attr:`DnnLik.model_compile_kwargs <DNNLikelihood.DnnLik.model_compile_kwargs>`

        The first attribute is constructed as a ``lambda`` function from the 
        :attr:`NF.log_prob <NF4HEP.NF.log_prob>` attribute (it is set to minus the log-probability),
        while the second and third attributes are set from the corresponding string attributes

            - :attr:`NF.optimizer_string <NF4HEP.NF.optimizer_string>`
            - :attr:`NF.metrics_string <NF4HEP.NF.metrics_string>`

        - **Arguments**

            - **verbose**

                See :argument:`verbose <common_methods_arguments.verbose>`.

        - **Produces file**

            - :attr:`NF.output_log_file <NF4HEP.NF.output_log_file>`
        """
        verbose, verbose_sub = self.set_verbosity(verbose)
        print(header_string, "\nCompiling Keras model\n", show=verbose)
        # Compile model
        start = timer()
        log_prob = self.log_prob
        self.loss = lambda _, log_prob: -log_prob
        self.optimizer = eval(self.optimizer_string)
        self.metrics = []
        for metric_string in self.metrics_string:
            self.metrics.append(eval(metric_string))
        self.model.compile(loss=self.loss, optimizer=self.optimizer,
                           metrics=self.metrics, **self.model_compile_kwargs)
        end = timer()
        timestamp = "datetime_" + \
            datetime.now().strftime("%Y_%m_%d_%H_%M_%S_%fZ")[:-3]
        self.log[timestamp] = {"action": "compiled tf model"}
        self.save_log(overwrite=True, verbose=verbose_sub)
        print("Model for Normalizing Flow", self.name,
              "compiled in", str(end-start), "s.\n", show=verbose)

    def model_build(self, gpu="auto", force=False, verbose=None):
        """
        Method that calls the methods

            - :meth:`NF.model_define <NF4HEP.NF.model_define>`
            - :meth:`NF.model_compile <NF4HEP.NF.model_compile>`

        on a specific GPU by using the |tf_distribute_onedevicestrategy_link| class.
        Using this method different :class:`NF <NF4HEP.NF>` members of a
        :class:`NFEnsemble <NF4HEP.NFEnsemble>` object can be compiled and run in parallel 
        on different GPUs (when available).
        Notice that, in case the model has already been created and compiled, the method does not re-builds the
        model on a different GPU unless the ``force`` flag is set to ``True`` (default is ``False``).

        - **Arguments**

            - **gpu**

                GPU number (e.g. 0,1,etc..) of the GPU where the model should be built.
                The available GPUs are listed in the 
                :attr:`NF.active_gpus <NF4HEP.NF.active_gpus>`.
                If ``gpu="auto"`` the first GPU, corresponding to number ``0`` is automatically set.

                    - **type**: ``int`` or ``str``
                    - **default**: ``auto`` (``0``)

            - **force**

                If set to ``True`` the model is re-built even if it was already 
                available.

                    - **type**: ``bool``
                    - **default**: ``False``

            - **verbose**

                See :argument:`verbose <common_methods_arguments.verbose>`.

        - **Produces file**

            - :attr:`NF.output_log_file <NF4HEP.NF.output_log_file>`
        """
        verbose, verbose_sub = self.set_verbosity(verbose)
        print(header_string, "\nDefining and compiling Keras model\n", show=verbose)
        try:
            self.model
            create = False
            if self.model._is_compiled:
                compile = False
            else:
                compile = True
        except:
            create = True
            compile = True
        if force:
            create = True
            compile = True
        if not create and not compile:
            print("Model already built.", show=verbose)
            return
        if self.gpu_mode:
            if gpu == "auto":
                gpu = 0
            elif gpu > len(self.available_gpus):
                print(
                    "gpu", gpu, "does not exist. Continuing on first gpu.\n", show=verbose)
                gpu = 0
            self.training_device = self.available_gpus[gpu]
            device_id = self.training_device[0]
        else:
            if gpu != "auto":
                print(
                    "GPU mode selected without any active GPU. Proceeding with CPU support.\n", show=verbose)
            self.training_device = self.available_cpu
            device_id = self.training_device[0]
        strategy = tf.distribute.OneDeviceStrategy(device=device_id)
        print("Building tf model for Normalizing Flow", self.name,
              "on device", self.training_device, ".\n", show=verbose)
        with strategy.scope():
            if create:
                self.model_define(verbose=verbose_sub)
            if compile:
                self.model_compile(verbose=verbose_sub)
        timestamp = "datetime_" + \
            datetime.now().strftime("%Y_%m_%d_%H_%M_%S_%fZ")[:-3]
        self.log[timestamp] = {"action": "built tf model",
                               "gpu mode": self.gpu_mode,
                               "device id": device_id}
        self.save_log(overwrite=True, verbose=verbose_sub)

    def model_train(self, verbose=None):
        """
        Method that trains the |tf_keras_model_link| stored in the 
        :attr:`NF.model <NF4HEP.NF.model>` attribute.
        The model is trained by calling the |tf_keras_model_fit_link| method and passing it the inputs 

            - X_train (attribute :attr:`NF.X_train <NF4HEP.NF.X_train>` scaled with :attr:`NF.scalerX <NF4HEP.NF.scalerX>`)
            - X_val (attribute :attr:`NF.X_val <NF4HEP.NF.X_val>` scaled with :attr:`NF.scalerX <NF4HEP.NF.scalerX>`)
            - epochs_to_run (difference between :attr:`NF.epochs_required <NF4HEP.NF.epochs_required>` and :attr:`NF.epochs_available <NF4HEP.NF.epochs_available>`)
            - :attr:`NF.batch_size <NF4HEP.NF.batch_size>`
            - :attr:`NF.callbacks <NF4HEP.NF.callbacks>`

        After training the method updates the attributes

            - :attr:`NF.model <NF4HEP.NF.model>`
            - :attr:`NF.history <NF4HEP.NF.history>`
            - :attr:`NF.epochs_available <NF4HEP.NF.epochs_available>`

        - **Arguments**

            - **verbose**

                See :argument:`verbose <common_methods_arguments.verbose>`.
                Also see the documentation of |tf_keras_model_fit_link| for the available verbosity modes.

        - **Produces file**

            - :attr:`NF.output_log_file <NF4HEP.NF.output_log_file>`
        """
        verbose, verbose_sub = self.set_verbosity(verbose)
        print(header_string, "\nTraining Keras model\n", show=verbose)
        # Reset random state
        self.__set_seed()
        # Scale data
        start = timer()
        epochs_before_run = self.epochs_available
        epochs_to_run = self.__set_epochs_to_run()
        print("Required a total of", self.epochs_required, "epochs.", self.epochs_available,
              "epochs already available. Training for a maximum of", epochs_to_run, "epochs.\n", show=verbose)
        if not self.model._is_compiled:
            self.model_compile(verbose=verbose_sub)
        if epochs_to_run == 0:
            print(
                "Please increase epochs_required to train for more epochs.\n", show=verbose)
        else:
            if len(self.X_train) <= 1:
                self.generate_train_data(verbose=verbose_sub)
            print("Scaling training/val data.\n", show=verbose)
            X_train = self.transform_data_X(self.X_train)
            X_val = self.transform_data_X(self.X_val)
            Y_train = np.zeros((X_train.shape[0], ), dtype=self.dtype)
            Y_val = np.zeros((X_val.shape[0], ), dtype=self.dtype)
            if "PlotLossesKeras" in str(self.callbacks_strings):
                plt.style.use(mplstyle_path)
            # for callback_string in self.callbacks_strings:
            #    self.callbacks.append(eval(callback_string))
            # Train model
            print("Start training of model for Normalizing Flow",
                  self.name, ".\n", show=verbose)
            history = self.model.fit(x=X_train,
                                     y=Y_train,
                                     initial_epoch=self.epochs_available,
                                     epochs=self.epochs_required,
                                     batch_size=self.batch_size,
                                     verbose=verbose_sub,
                                     validation_data=(X_val, Y_val),
                                     callbacks=self.callbacks,
                                     **self.model_train_kwargs)
            end = timer()
            history = history.history
            for k, v in history.items():
                history[k] = list(np.array(v, dtype=self.dtype))
            if self.history == {}:
                print("No existing history. Setting new history.\n", show=verbose)
                self.history = history
            else:
                print("Found existing history. Appending new history.\n", show=verbose)
                for k, v in self.history.items():
                    self.history[k] = v + history[k]
            self.epochs_available = len(self.history["loss"])
            epochs_current_run = self.epochs_available-epochs_before_run
            training_time_current_run = (end - start)
            self.training_time = self.training_time + training_time_current_run
            print("Updating model.history and model.epoch attribute.\n", show=verbose)
            self.model.history.history = self.history
            self.model.history.params["epochs"] = self.epochs_available
            self.model.history.epoch = np.arange(
                self.epochs_available).tolist()
            if "PlotLossesKeras" in str(self.callbacks_strings):
                plt.close()
            timestamp = "datetime_" + \
                datetime.now().strftime("%Y_%m_%d_%H_%M_%S_%fZ")[:-3]
            self.log[timestamp] = {"action": "trained tf model",
                                   "epochs to run": epochs_to_run,
                                   "epochs run": epochs_current_run,
                                   "epochs total": self.epochs_available,
                                   "batch size": self.batch_size,
                                   "training time for current run": training_time_current_run,
                                   "total training time": self.training_time}
            self.save_log(overwrite=True, verbose=verbose_sub)
            print("Model for Normalizing Flow", self.name, "successfully trained for", epochs_current_run, "epochs in",
                  training_time_current_run, "s (", training_time_current_run/epochs_current_run, "s/epoch).\n", show=verbose)

    def check_x_bounds(self, pars_val, pars_bounds):
        res = []
        for i in range(len(pars_val)):
            tmp = []
            if pars_bounds[i][0] == -np.inf:
                tmp.append(True)
            else:
                if pars_val[i] >= pars_bounds[i][0]:
                    tmp.append(True)
                else:
                    tmp.append(False)
            if pars_bounds[i][1] == np.inf:
                tmp.append(True)
            else:
                if pars_val[i] <= pars_bounds[i][1]:
                    tmp.append(True)
                else:
                    tmp.append(False)
            res.append(tmp)
        return np.all(res)

    def generate_fig_base_title(self):
        """
        Generates a common title for figures including information on the model and saved in the 
        :attr:`NF.fig_base_title <NF4HEP.NF.fig_base_title>` attribute.
        """
        title = "Ndim: " + str(self.ndims) + " - "
        title = title + "Nevt: " + \
            "%.E" % Decimal(str(self.npoints_train)) + " - "
        title = title + "Layers: " + str(len(self.hidden_layers)) + " - "
        #title = title + "Nodes: " + str(self.hidden_layers[0][0]) + " - "
        title = title.replace("+", "") + "Loss: " + str(self.loss_string)
        self.fig_base_title = title

    def update_figures(self, figure_file=None, timestamp=None, overwrite=False, verbose=None):
        """
        Method that generates new file names and renames old figure files when new ones are produced with the argument ``overwrite=False``. 
        When ``overwrite=False`` it calls the :func:`utils.check_rename_file <NF4HEP.utils.check_rename_file>` function and, if 
        ``figure_file`` already existed in the :attr:`NF.predictions <NF4HEP.NF.predictions>` dictionary, then it
        updates the dictionary by appennding to the old figure name the timestamp corresponding to its generation timestamp 
        (that is the key of the :attr:`NF.predictions["Figures"] <NF4HEP.NF.predictions>` dictionary).
        When ``overwrite="dump"`` it calls the :func:`utils.generate_dump_file_name <NF4HEP.utils.generate_dump_file_name>` function
        to generate the dump file name.
        It returns the new figure_file.

        - **Arguments**

            - **figure_file**

                Figure file path. If the figure already exists in the 
                :meth:`NF.predictions <NF4HEP.NF.predictions>` dictionary, then its name is updated with the corresponding timestamp.

            - **overwrite**

                The method updates file names and :attr:`NF.predictions <NF4HEP.NF.predictions>` dictionary only if
                ``overwrite=False``. If ``overwrite="dump"`` the method generates and returns the dump file path. 
                If ``overwrite=True`` the method just returns ``figure_file``.

            - **verbose**

                See :argument:`verbose <common_methods_arguments.verbose>`.

        - **Returns**

            - **new_figure_file**

                String identical to the input string ``figure_file`` unless ``verbose="dump"``.

        - **Creates/updates files**

            - Updates ``figure_file`` file name.
        """
        verbose, verbose_sub = self.set_verbosity(verbose)
        print("Checking and updating figures dictionary", show=verbose)
        if figure_file is None:
            raise Exception(
                "figure_file input argument of update_figures method needs to be specified while it is None.")
        else:
            new_figure_file = figure_file
            if type(overwrite) == bool:
                if not overwrite:
                    # search figure
                    timestamp = None
                    for k, v in self.predictions["Figures"].items():
                        if figure_file in v:
                            timestamp = k
                    old_figure_file = utils.check_rename_file(path.join(
                        self.output_figures_folder, figure_file), timestamp=timestamp, return_value="file_name", verbose=verbose_sub)
                    if timestamp is not None:
                        self.predictions["Figures"][timestamp] = [
                            f.replace(figure_file, old_figure_file) for f in v]
            elif overwrite == "dump":
                new_figure_file = utils.generate_dump_file_name(
                    figure_file, timestamp=timestamp)
        return new_figure_file

    def plot_training_history(self, metrics=["loss"], yscale="log", show_plot=False, timestamp=None, overwrite=False, verbose=None):
        verbose, verbose_sub = self.set_verbosity(verbose)
        print(header_string, "\nPlotting training history\n", show=verbose)
        if timestamp is None:
            timestamp = "datetime_" + \
                datetime.now().strftime("%Y_%m_%d_%H_%M_%S_%fZ")[:-3]
        plt.style.use(mplstyle_path)
        metrics = np.unique(metrics)
        for metric in metrics:
            start = timer()
            #metric = custom_losses.metric_name_unabbreviate(metric)
            val_metric = "val_" + metric
            plt.plot(self.history[metric])
            plt.plot(self.history[val_metric])
            plt.yscale(yscale)
            #plt.grid(linestyle="--", dashes=(5, 5))
            plt.title(r"%s" % self.fig_base_title, fontsize=10)
            plt.xlabel(r"epoch")
            ylabel = (metric.replace("_", "-"))
            plt.ylabel(r"%s" % ylabel)
            plt.legend([r"training", r"validation"])
            plt.tight_layout()
            ax = plt.axes()
            #x1, x2, y1, y2 = plt.axis()
            plt.text(0.967, 0.2, r"%s" % self.summary_text, fontsize=7, bbox=dict(facecolor="green", alpha=0.15,
                     edgecolor="black", boxstyle="round,pad=0.5"), ha="right", ma="left", transform=ax.transAxes)
            figure_file_name = self.update_figures(
                figure_file=self.output_figures_base_file_name+"_training_history_" + metric+".pdf", timestamp=timestamp, overwrite=overwrite)
            utils.savefig(
                r"%s" % (path.join(self.output_figures_folder, figure_file_name)))
            utils.check_set_dict_keys(self.predictions["Figures"], [
                                      timestamp], [[]], verbose=False)
            utils.append_without_duplicate(
                self.predictions["Figures"][timestamp], figure_file_name)
            if show_plot:
                plt.show()
            plt.close()
            end = timer()
            self.log[timestamp] = {"action": "saved figure",
                                   "file name": figure_file_name}
            print("\n"+header_string+"\nFigure file\n\t", r"%s" % (figure_file_name),
                  "\ncreated and saved in", str(end-start), "s.\n", show=verbose)
        # self.save_log(overwrite=True, verbose=verbose_sub) #log saved at the end of predictions

    def plot_corners_1samp(self, X, W=None, pars=None, max_points=None, nbins=50, pars_labels="original",
                           HPDI_dic={"sample": "train", "type": "true"},
                           ranges_extend=None, title=None, color="green",
                           plot_title="Params contours", legend_labels=None, legend_loc="upper right",
                           figure_file_name=None, show_plot=False, timestamp=None, overwrite=False, verbose=None):
        verbose, verbose_sub = self.set_verbosity(verbose)
        print(header_string,
              "\nPlotting 2d posterior distributions for single sample\n", show=verbose)
        if timestamp is None:
            timestamp = "datetime_" + \
                datetime.now().strftime("%Y_%m_%d_%H_%M_%S_%fZ")[:-3]
        plt.style.use(mplstyle_path)
        start = timer()
        linewidth = 1.3
        intervals = inference.CI_from_sigma([1, 2, 3])
        if ranges_extend == None:
            ranges = extend_corner_range(X, X, pars, 0)
        else:
            ranges = extend_corner_range(X, X, pars, ranges_extend)
        pars_labels = self.__set_pars_labels(pars_labels)
        labels = np.array(pars_labels)[pars].tolist()
        nndims = len(pars)
        if max_points != None:
            if type(max_points) == list:
                nnn = np.min([len(X), max_points[0]])
            else:
                nnn = np.min([len(X), max_points])
        else:
            nnn = len(X)
        np.random.seed(self.seed)
        rnd_idx = np.random.choice(np.arange(len(X)), nnn, replace=False)
        samp = X[rnd_idx][:, pars]
        if W is not None:
            W = W[rnd_idx]
        try:
            HPDI = [[self.predictions["Bayesian_inference"]['HPDI'][timestamp][par][HPDI_dic["type"]]
                     [HPDI_dic["sample"]][interval]["Intervals"] for interval in intervals] for par in pars]
        except:
            print("HPDI not present in predictions. Computing them.\n", show=verbose)
            HPDI = [[inference.HPDI(samp[:, i], intervals=intervals, weights=W, nbins=nbins, print_hist=False, optimize_binning=True)[
                interval]["Intervals"] for i in range(nndims)] for interval in intervals]
        levels = np.array([[np.sort(inference.HPD_quotas(samp[:, [i, j]], nbins=nbins, intervals=inference.CI_from_sigma(
            [1, 2, 3]), weights=W)).tolist() for j in range(nndims)] for i in range(nndims)])
        fig, axes = plt.subplots(nndims, nndims, figsize=(3*nndims, 3*nndims))
        figure = corner(samp, bins=nbins, weights=W, labels=[r"%s" % s for s in labels],
                        fig=fig, max_n_ticks=6, color=color, plot_contours=True, smooth=True,
                        smooth1d=True, range=ranges, plot_datapoints=True, plot_density=False,
                        fill_contours=False, normalize1d=True, hist_kwargs={"color": color, "linewidth": "1.5"},
                        label_kwargs={"fontsize": 16}, show_titles=False, title_kwargs={"fontsize": 18},
                        levels_lists=levels, data_kwargs={"alpha": 1},
                        contour_kwargs={"linestyles": ["dotted", "dashdot", "dashed"][:len(
                            HPDI[0])], "linewidths": [linewidth, linewidth, linewidth][:len(HPDI[0])]},
                        no_fill_contours=False, contourf_kwargs={"colors": ["white", "lightgreen", color], "alpha": 1})
        # , levels=(0.393,0.68,)) ,levels=[300],levels_lists=levels1)#,levels=[120])
        axes = np.array(figure.axes).reshape((nndims, nndims))
        for i in range(nndims):
            title_i = ""
            ax = axes[i, i]
            #ax.axvline(value1[i], color="green",alpha=1)
            #ax.axvline(value2[i], color="red",alpha=1)
            ax.grid(True, linestyle="--", linewidth=1, alpha=0.3)
            ax.tick_params(axis="both", which="major", labelsize=16)
            hists_1d = get_1d_hist(i, samp, nbins=nbins, ranges=ranges, weights=W, normalize1d=True)[
                0]  # ,intervals=HPDI681)
            HPDI68 = HPDI[i][0]
            HPDI95 = HPDI[i][1]
            HPDI3s = HPDI[i][2]
            for j in HPDI3s:
                #ax.fill_between(hists_1d_1[0], 0, hists_1d_1[1], where=(hists_1d_1[0]>=j[0])*(hists_1d_1[0]<=j[1]), facecolor="lightgreen", alpha=0.2)#facecolor=(255/255,89/255,71/255,.4))#
                ax.axvline(hists_1d[0][hists_1d[0] >= j[0]][0], color=color,
                           alpha=1, linestyle=":", linewidth=linewidth)
                ax.axvline(hists_1d[0][hists_1d[0] <= j[1]][-1],
                           color=color, alpha=1, linestyle=":", linewidth=linewidth)
            for j in HPDI95:
                #ax.fill_between(hists_1d_1[0], 0, hists_1d_1[1], where=(hists_1d_1[0]>=j[0])*(hists_1d_1[0]<=j[1]), facecolor="lightgreen", alpha=0.2)#facecolor=(255/255,89/255,71/255,.4))#
                ax.axvline(hists_1d[0][hists_1d[0] >= j[0]][0], color=color,
                           alpha=1, linestyle="-.", linewidth=linewidth)
                ax.axvline(hists_1d[0][hists_1d[0] <= j[1]][-1], color=color,
                           alpha=1, linestyle="-.", linewidth=linewidth)
            for j in HPDI68:
                #ax.fill_between(hists_1d_1[0], 0, hists_1d_1[1], where=(hists_1d_1[0]>=j[0])*(hists_1d_1[0]<=j[1]), facecolor="white", alpha=1)#facecolor=(0,1,0,.5))#
                ax.axvline(hists_1d[0][hists_1d[0] >= j[0]][0], color=color,
                           alpha=1, linestyle="--", linewidth=linewidth)
                ax.axvline(hists_1d[0][hists_1d[0] <= j[1]][-1], color=color,
                           alpha=1, linestyle="--", linewidth=linewidth)
                title_i = r"%s" % title + \
                    ": ["+"{0:1.2e}".format(j[0])+"," + \
                    "{0:1.2e}".format(j[1])+"]"
            if i == 0:
                x1, x2, _, _ = ax.axis()
                ax.set_xlim(x1*1.3, x2)
            ax.set_title(title_i, fontsize=10)
        for yi in range(nndims):
            for xi in range(yi):
                ax = axes[yi, xi]
                if xi == 0:
                    x1, x2, _, _ = ax.axis()
                    ax.set_xlim(x1*1.3, x2)
                ax.grid(True, linestyle="--", linewidth=1)
                ax.tick_params(axis="both", which="major", labelsize=16)
        fig.subplots_adjust(top=0.85, wspace=0.25, hspace=0.25)
        fig.suptitle(r"%s" %
                     (plot_title+"\n\n"+self.fig_base_title), fontsize=26)
        #fig.text(0.5 ,1, r"%s" % plot_title, fontsize=26)
        colors = [color, "black", "black", "black"]
        red_patch = matplotlib.patches.Patch(
            color=colors[0])  # , label="The red data")
        # blue_patch = matplotlib.patches.Patch(color=colors[1])  # , label="The blue data")
        line1 = matplotlib.lines.Line2D(
            [0], [0], color=colors[0], lw=int(7+2*nndims))
        line2 = matplotlib.lines.Line2D(
            [0], [0], color=colors[1], linewidth=3, linestyle="--")
        line3 = matplotlib.lines.Line2D(
            [0], [0], color=colors[2], linewidth=3, linestyle="-.")
        line4 = matplotlib.lines.Line2D(
            [0], [0], color=colors[3], linewidth=3, linestyle=":")
        lines = [line1, line2, line3, line4]
        # (1/nndims*1.05,1/nndims*1.1))#transform=axes[0,0].transAxes)# loc=(0.53, 0.8))
        fig.legend(lines, legend_labels, fontsize=int(7+2*nndims),
                   loc=legend_loc, bbox_to_anchor=(0.95, 0.8))
        # plt.tight_layout()
        if figure_file_name is not None:
            figure_file_name = self.update_figures(
                figure_file=figure_file_name, timestamp=timestamp, overwrite=overwrite)
        else:
            figure_file_name = self.update_figures(figure_file=self.output_figures_base_file_name+"_corner_posterior_1samp_pars_" + "_".join([
                                                   str(i) for i in pars]) + ".pdf", timestamp=timestamp, overwrite=overwrite)
        utils.savefig(
            r"%s" % (path.join(self.output_figures_folder, figure_file_name)), dpi=50)
        utils.check_set_dict_keys(self.predictions["Figures"], [
                                  timestamp], [[]], verbose=False)
        utils.append_without_duplicate(
            self.predictions["Figures"][timestamp], figure_file_name)
        #self.predictions["Figures"] = utils.check_figures_dic(self.predictions["Figures"],output_figures_folder=self.output_figures_folder)
        if show_plot:
            plt.show()
        plt.close()
        end = timer()
        self.log[timestamp] = {"action": "saved figure",
                               "file name": figure_file_name}
        print("\n"+header_string+"\nFigure file\n\t", r"%s" % (figure_file_name),
              "\ncreated and saved in", str(end-start), "s.\n", show=verbose)

    def plot_corners_2samp(self, X1, X2, W1=None, W2=None, pars=None, max_points=None, nbins=50, pars_labels=None,
                           HPDI1_dic={"sample": "train", "type": "true"}, HPDI2_dic={"sample": "test", "type": "true"},
                           ranges_extend=None, title1=None, title2=None,
                           color1="green", color2="red",
                           plot_title="Params contours", legend_labels=None, legend_loc="upper right",
                           figure_file_name=None, show_plot=False, timestamp=None, overwrite=False, verbose=None):
        verbose, verbose_sub = self.set_verbosity(verbose)
        print(header_string, "\nPlotting 2d posterior distributions for two samples comparison\n", show=verbose)
        if timestamp is None:
            timestamp = "datetime_" + \
                datetime.now().strftime("%Y_%m_%d_%H_%M_%S_%fZ")[:-3]
        plt.style.use(mplstyle_path)
        start = timer()
        linewidth = 1.3
        intervals = inference.CI_from_sigma([1, 2, 3])
        if ranges_extend == None:
            ranges = extend_corner_range(X1, X2, pars, 0)
        else:
            ranges = extend_corner_range(X1, X2, pars, ranges_extend)
        pars_labels = self.__set_pars_labels(pars_labels)
        labels = np.array(pars_labels)[pars].tolist()
        nndims = len(pars)
        if max_points != None:
            if type(max_points) == list:
                nnn1 = np.min([len(X1), max_points[0]])
                nnn2 = np.min([len(X2), max_points[1]])
            else:
                nnn1 = np.min([len(X1), max_points])
                nnn2 = np.min([len(X2), max_points])
        else:
            nnn1 = len(X1)
            nnn2 = len(X2)
        np.random.seed(self.seed)
        rnd_idx_1 = np.random.choice(np.arange(len(X1)), nnn1, replace=False)
        rnd_idx_2 = np.random.choice(np.arange(len(X2)), nnn2, replace=False)
        samp1 = X1[rnd_idx_1][:, pars]
        samp2 = X2[rnd_idx_2][:, pars]
        if W1 is not None:
            W1 = W1[rnd_idx_1]
        if W2 is not None:
            W2 = W2[rnd_idx_2]
        try:
            HPDI1 = [[self.predictions["Bayesian_inference"]['HPDI'][timestamp][par][HPDI1_dic["type"]]
                      [HPDI1_dic["sample"]][interval]["Intervals"] for interval in intervals] for par in pars]
            HPDI2 = [[self.predictions["Bayesian_inference"]['HPDI'][timestamp][par][HPDI2_dic["type"]]
                      [HPDI2_dic["sample"]][interval]["Intervals"] for interval in intervals] for par in pars]
            # print(np.shape(HPDI1),np.shape(HPDI2))
        except:
            print("HPDI not present in predictions. Computing them.\n", show=verbose)
            HPDI1 = [[inference.HPDI(samp1[:, i], intervals=intervals, weights=W1, nbins=nbins, print_hist=False, optimize_binning=True)[
                interval]["Intervals"] for i in range(nndims)] for interval in intervals]
            HPDI2 = [[inference.HPDI(samp2[:, i], intervals=intervals, weights=W2, nbins=nbins, print_hist=False, optimize_binning=True)[
                interval]["Intervals"] for i in range(nndims)] for interval in intervals]
        levels1 = np.array([[np.sort(inference.HPD_quotas(samp1[:, [i, j]], nbins=nbins, intervals=inference.CI_from_sigma(
            [1, 2, 3]), weights=W1)).tolist() for j in range(nndims)] for i in range(nndims)])
        levels2 = np.array([[np.sort(inference.HPD_quotas(samp2[:, [i, j]], nbins=nbins, intervals=inference.CI_from_sigma(
            [1, 2, 3]), weights=W2)).tolist() for j in range(nndims)] for i in range(nndims)])
        fig, axes = plt.subplots(nndims, nndims, figsize=(3*nndims, 3*nndims))
        figure1 = corner(samp1, bins=nbins, weights=W1, labels=[r"%s" % s for s in labels],
                         fig=fig, max_n_ticks=6, color=color1, plot_contours=True, smooth=True,
                         smooth1d=True, range=ranges, plot_datapoints=True, plot_density=False,
                         fill_contours=False, normalize1d=True, hist_kwargs={"color": color1, "linewidth": "1.5"},
                         label_kwargs={"fontsize": 16}, show_titles=False, title_kwargs={"fontsize": 18},
                         levels_lists=levels1, data_kwargs={"alpha": 1},
                         contour_kwargs={"linestyles": ["dotted", "dashdot", "dashed"][:len(
                             HPDI1[0])], "linewidths": [linewidth, linewidth, linewidth][:len(HPDI1[0])]},
                         no_fill_contours=False, contourf_kwargs={"colors": ["white", "lightgreen", color1], "alpha": 1})
        # , levels=(0.393,0.68,)) ,levels=[300],levels_lists=levels1)#,levels=[120])
        figure2 = corner(samp2, bins=nbins, weights=W2, labels=[r"%s" % s for s in labels],
                         fig=fig, max_n_ticks=6, color=color2, plot_contours=True, smooth=True,
                         range=ranges, smooth1d=True, plot_datapoints=True, plot_density=False,
                         fill_contours=False, normalize1d=True, hist_kwargs={"color": color2, "linewidth": "1.5"},
                         label_kwargs={"fontsize": 16}, show_titles=False, title_kwargs={"fontsize": 18}, levels_lists=levels2, data_kwargs={"alpha": 1},
                         contour_kwargs={"linestyles": ["dotted", "dashdot", "dashed"][0:len(
                             HPDI2[0])], "linewidths": [linewidth, linewidth, linewidth][:len(HPDI2[0])]},
                         no_fill_contours=False, contourf_kwargs={"colors": ["white", "tomato", color2], "alpha": 1})
        # , quantiles = (0.16, 0.84), levels=(0.393,0.68,)), levels=[300],levels_lists=levels2)#,levels=[120])
        axes = np.array(figure1.axes).reshape((nndims, nndims))
        for i in range(nndims):
            ax = axes[i, i]
            title = ""
            #ax.axvline(value1[i], color="green",alpha=1)
            #ax.axvline(value2[i], color="red",alpha=1)
            ax.grid(True, linestyle="--", linewidth=1, alpha=0.3)
            ax.tick_params(axis="both", which="major", labelsize=16)
            HPDI681 = HPDI1[i][0]
            HPDI951 = HPDI1[i][1]
            HPDI3s1 = HPDI1[i][2]
            HPDI682 = HPDI2[i][0]
            HPDI952 = HPDI2[i][1]
            HPDI3s2 = HPDI2[i][2]
            hists_1d_1 = get_1d_hist(i, samp1, nbins=nbins, ranges=ranges, weights=W1, normalize1d=True)[
                0]  # ,intervals=HPDI681)
            hists_1d_2 = get_1d_hist(i, samp2, nbins=nbins, ranges=ranges, weights=W2, normalize1d=True)[
                0]  # ,intervals=HPDI682)
            for j in HPDI3s1:
                #ax.fill_between(hists_1d_1[0], 0, hists_1d_1[1], where=(hists_1d_1[0]>=j[0])*(hists_1d_1[0]<=j[1]), facecolor="lightgreen", alpha=0.2)#facecolor=(255/255,89/255,71/255,.4))#
                ax.axvline(hists_1d_1[0][hists_1d_1[0] >= j[0]][0],
                           color=color1, alpha=1, linestyle=":", linewidth=linewidth)
                ax.axvline(hists_1d_1[0][hists_1d_1[0] <= j[1]][-1],
                           color=color1, alpha=1, linestyle=":", linewidth=linewidth)
            for j in HPDI3s2:
                #ax.fill_between(hists_1d_2[0], 0, hists_1d_2[1], where=(hists_1d_2[0]>=j[0])*(hists_1d_2[0]<=j[1]), facecolor="tomato", alpha=0.2)#facecolor=(255/255,89/255,71/255,.4))#
                ax.axvline(hists_1d_2[0][hists_1d_2[0] >= j[0]][0],
                           color=color2, alpha=1, linestyle=":", linewidth=linewidth)
                ax.axvline(hists_1d_2[0][hists_1d_2[0] <= j[1]][-1],
                           color=color2, alpha=1, linestyle=":", linewidth=linewidth)
            for j in HPDI951:
                #ax.fill_between(hists_1d_1[0], 0, hists_1d_1[1], where=(hists_1d_1[0]>=j[0])*(hists_1d_1[0]<=j[1]), facecolor="lightgreen", alpha=0.2)#facecolor=(255/255,89/255,71/255,.4))#
                ax.axvline(hists_1d_1[0][hists_1d_1[0] >= j[0]][0],
                           color=color1, alpha=1, linestyle="-.", linewidth=linewidth)
                ax.axvline(hists_1d_1[0][hists_1d_1[0] <= j[1]][-1],
                           color=color1, alpha=1, linestyle="-.", linewidth=linewidth)
            for j in HPDI952:
                #ax.fill_between(hists_1d_2[0], 0, hists_1d_2[1], where=(hists_1d_2[0]>=j[0])*(hists_1d_2[0]<=j[1]), facecolor="tomato", alpha=0.2)#facecolor=(255/255,89/255,71/255,.4))#
                ax.axvline(hists_1d_2[0][hists_1d_2[0] >= j[0]][0],
                           color=color2, alpha=1, linestyle="-.", linewidth=linewidth)
                ax.axvline(hists_1d_2[0][hists_1d_2[0] <= j[1]][-1],
                           color=color2, alpha=1, linestyle="-.", linewidth=linewidth)
            for j in HPDI681:
                #ax.fill_between(hists_1d_1[0], 0, hists_1d_1[1], where=(hists_1d_1[0]>=j[0])*(hists_1d_1[0]<=j[1]), facecolor="white", alpha=1)#facecolor=(0,1,0,.5))#
                ax.axvline(hists_1d_1[0][hists_1d_1[0] >= j[0]][0],
                           color=color1, alpha=1, linestyle="--", linewidth=linewidth)
                ax.axvline(hists_1d_1[0][hists_1d_1[0] <= j[1]][-1],
                           color=color1, alpha=1, linestyle="--", linewidth=linewidth)
                title = title+r"%s" % title1 + \
                    ": ["+"{0:1.2e}".format(j[0])+"," + \
                    "{0:1.2e}".format(j[1])+"]"
            title = title+"\n"
            for j in HPDI682:
                #ax.fill_between(hists_1d_2[0], 0, hists_1d_2[1], where=(hists_1d_2[0]>=j[0])*(hists_1d_2[0]<=j[1]), facecolor="white", alpha=1)#facecolor=(1,0,0,.4))#
                ax.axvline(hists_1d_2[0][hists_1d_2[0] >= j[0]][0],
                           color=color2, alpha=1, linestyle="--", linewidth=linewidth)
                ax.axvline(hists_1d_2[0][hists_1d_2[0] <= j[1]][-1],
                           color=color2, alpha=1, linestyle="--", linewidth=linewidth)
                title = title+r"%s" % title2 + \
                    ": ["+"{0:1.2e}".format(j[0])+"," + \
                    "{0:1.2e}".format(j[1])+"]"
            if i == 0:
                x1, x2, _, _ = ax.axis()
                ax.set_xlim(x1*1.3, x2)
            ax.set_title(title, fontsize=10)
        for yi in range(nndims):
            for xi in range(yi):
                ax = axes[yi, xi]
                if xi == 0:
                    x1, x2, _, _ = ax.axis()
                    ax.set_xlim(x1*1.3, x2)
                ax.grid(True, linestyle="--", linewidth=1)
                ax.tick_params(axis="both", which="major", labelsize=16)
        fig.subplots_adjust(top=0.85, wspace=0.25, hspace=0.25)
        fig.suptitle(r"%s" %
                     (plot_title+"\n\n"+self.fig_base_title), fontsize=26)
        #fig.text(0.5 ,1, r"%s" % plot_title, fontsize=26)
        colors = [color1, color2, "black", "black", "black"]
        red_patch = matplotlib.patches.Patch(
            color=colors[0])  # , label="The red data")
        blue_patch = matplotlib.patches.Patch(
            color=colors[1])  # , label="The blue data")
        line1 = matplotlib.lines.Line2D(
            [0], [0], color=colors[0], lw=int(7+2*nndims))
        line2 = matplotlib.lines.Line2D(
            [0], [0], color=colors[1], lw=int(7+2*nndims))
        line3 = matplotlib.lines.Line2D(
            [0], [0], color=colors[2], linewidth=3, linestyle="--")
        line4 = matplotlib.lines.Line2D(
            [0], [0], color=colors[3], linewidth=3, linestyle="-.")
        line5 = matplotlib.lines.Line2D(
            [0], [0], color=colors[4], linewidth=3, linestyle=":")
        lines = [line1, line2, line3, line4, line5]
        # (1/nndims*1.05,1/nndims*1.1))#transform=axes[0,0].transAxes)# loc=(0.53, 0.8))
        fig.legend(lines, legend_labels, fontsize=int(
            7+2*nndims), loc=legend_loc, bbox_to_anchor=(0.95, 0.8))
        # plt.tight_layout()
        if figure_file_name is not None:
            figure_file_name = self.update_figures(
                figure_file=figure_file_name, timestamp=timestamp, overwrite=overwrite)
        else:
            figure_file_name = self.update_figures(figure_file=self.output_figures_base_file_name+"_corner_posterior_2samp_pars_" + "_".join([
                                                   str(i) for i in pars]) + ".pdf", timestamp=timestamp, overwrite=overwrite)
        utils.savefig(
            r"%s" % (path.join(self.output_figures_folder, figure_file_name)), dpi=50)
        utils.check_set_dict_keys(self.predictions["Figures"], [
                                  timestamp], [[]], verbose=False)
        utils.append_without_duplicate(
            self.predictions["Figures"][timestamp], figure_file_name)
        #self.predictions["Figures"] = utils.check_figures_dic(self.predictions["Figures"],output_figures_folder=self.output_figures_folder)
        if show_plot:
            plt.show()
        plt.close()
        end = timer()
        timestamp = "datetime_" + \
            datetime.now().strftime("%Y_%m_%d_%H_%M_%S_%fZ")[:-3]
        self.log[timestamp] = {"action": "saved figure",
                               "file name": figure_file_name}
        print("\n"+header_string+"\nFigure file\n\t", r"%s" % (figure_file_name),
              "\ncreated and saved in", str(end-start), "s.\n", show=verbose)

    def reset_predictions(self, delete_figures=False, verbose=None):
        """
        Re-initializes the :attr:`Lik.predictions <NF4HEP.Lik.predictions>` dictionary to

         .. code-block:: python

            predictions = {"Model_evaluation": {},
                           "Bayesian_inference": {},
                           "Frequentist_inference": {},
                           "Figures": figs}

        Where ``figs`` may be either an empty dictionary or the present value of the corresponding one,
        depending on the value of the ``delete_figures`` argument.

        - **Arguments**

            - **delete_figures**

                If ``True`` all files in the :attr:`Lik.output_figures_folder <NF4HEP.Lik.output_figures_folder>` 
                folder are deleted and the ``"Figures"`` item is reset to an empty dictionary.

                    - **type**: ``bool``
                    - **default**: ``True`` 

            - **verbose**

                See :argument:`verbose <common_methods_arguments.verbose>`.

        """
        verbose, verbose_sub = self.set_verbosity(verbose)
        try:
            if delete_figures:
                utils.check_delete_all_files_in_path(
                    self.output_figures_folder)
                figs = {}
                print(header_string, "\nAll predictions and figures have been deleted and the 'predictions' attribute has been initialized.\n")
            else:
                figs = utils.check_figures_dic(
                    self.predictions["Figures"], output_figures_folder=self.output_figures_folder)
                print(header_string, "\nAll predictions have been deleted and the 'predictions' attribute has been initialized. No figure file has been deleted.\n")
            self.predictions = {"Model_evaluation": {},
                                "Bayesian_inference": {},
                                "Frequentist_inference": {},
                                "Figures": figs}
        except:
            self.predictions = {"Model_evaluation": {},
                                "Bayesian_inference": {},
                                "Frequentist_inference": {},
                                "Figures": {}}

    def model_compute_predictions(self):
        pass

    def generate_summary_text(self, model_predictions={}, timestamp=None, verbose=None):
        verbose, verbose_sub = self.set_verbosity(verbose)
        if timestamp is None:
            timestamp = "datetime_" + \
                datetime.now().strftime("%Y_%m_%d_%H_%M_%S_%fZ")[:-3]
        utils.check_set_dict_keys(model_predictions, ["Model_evaluation",
                                                      "Bayesian_inference",
                                                      "Frequentist_inference"],
                                  [False, False, False], verbose=verbose_sub)
        summary_text = "Sample file: " + \
            str(path.split(self.input_data_file)
                [1].replace("_", r"$\_$")) + "\n"
        #layers_string = [x.replace("layers.","") for x in self.layers_string[1:-1]]
        #summary_text = summary_text + "Layers: " + utils.string_add_newline_at_char(str(layers_string),",") + "\n"
        summary_text = summary_text + "Pars: " + str(self.ndims) + "\n"
        summary_text = summary_text + "NF type: " + self.flow_type + "\n"
        summary_text = summary_text + "Trainable pars: " + \
            str(self.model_trainable_params) + "\n"
        summary_text = summary_text + "Scaled X: " + \
            str(self.scalerX_bool) + "\n"
        summary_text = summary_text + "Rotated X: " + \
            str(self.rotationX_bool) + "\n"
        summary_text = summary_text + "Batch norm: " + \
            str(self.batch_norm) + "\n"
        optimizer_string = self.optimizer_string.replace("optimizers.", "")
        summary_text = summary_text + "Optimizer: " + \
            utils.string_add_newline_at_char(
                optimizer_string, ",").replace("_", r"$\_$") + "\n"
        summary_text = summary_text + "Batch size: " + \
            str(self.batch_size) + "\n"
        summary_text = summary_text + "Epochs: " + \
            str(self.epochs_available) + "\n"
        summary_text = summary_text + \
            "GPU(s): " + utils.string_add_newline_at_char(str(self.training_device), ",") + "\n"
        if model_predictions["Model_evaluation"]:
            try:
                metrics_scaled = self.predictions["Model_evaluation"]["Metrics_on_scaled_data"][timestamp]
                summary_text = summary_text + "Best losses: " + "[" + "{0:1.2e}".format(metrics_scaled["loss_best"]) + "," + \
                                                                      "{0:1.2e}".format(metrics_scaled["val_loss_best"]) + "," + \
                                                                      "{0:1.2e}".format(
                                                                          metrics_scaled["test_loss_best"]) + "]" + "\n"
            except:
                pass
            try:
                metrics_unscaled = self.predictions["Model_evaluation"]["Metrics_on_unscaled_data"][timestamp]
                summary_text = summary_text + "Best losses scaled: " + "[" + "{0:1.2e}".format(metrics_unscaled["loss_best_unscaled"]) + "," + \
                                                                             "{0:1.2e}".format(metrics_unscaled["val_loss_best_unscaled"]) + "," + \
                                                                             "{0:1.2e}".format(
                                                                                 metrics_unscaled["test_loss_best_unscaled"]) + "]" + "\n"
            except:
                pass
        if model_predictions["Bayesian_inference"]:
            try:
                ks_medians = self.predictions["Bayesian_inference"]["KS_medians"][timestamp]
                summary_text = summary_text + "KS $p$-median: " + "[" + "{0:1.2e}".format(ks_medians["Test_vs_pred_on_train"]) + "," + \
                    "{0:1.2e}".format(ks_medians["Test_vs_pred_on_val"]) + "," + \
                    "{0:1.2e}".format(ks_medians["Val_vs_pred_on_test"]) + "," + \
                    "{0:1.2e}".format(
                    ks_medians["Train_vs_pred_on_train"]) + "]" + "\n"
            except:
                pass
        if model_predictions["Frequentist_inference"]:
            summary_text = summary_text + "Average error on tmu: "
        # if FREQUENTISTS_RESULTS:
        #    summary_text = summary_text + "Mean error on tmu: "+ str(summary_log["Frequentist mean error on tmu"]) + "\n"
        summary_text = summary_text + "Training time: " + \
            str(round(self.training_time, 1)) + "s" + "\n"
        if model_predictions["Model_evaluation"] or model_predictions["Bayesian_inference"]:
            try:
                summary_text = summary_text + "Pred time per point: " + \
                    str(round(
                        self.predictions["Model_evaluation"][timestamp]["Prediction_time"], 1)) + "s"
            except:
                pass
        self.summary_text = summary_text

    def save_log(self, timestamp=None, overwrite=False, verbose=None):
        """
        Bla bla
        """
        verbose, verbose_sub = self.set_verbosity(verbose)
        start = timer()
        if timestamp is None:
            timestamp = "datetime_" + \
                datetime.now().strftime("%Y_%m_%d_%H_%M_%S_%fZ")[:-3]
        if type(overwrite) == bool:
            output_log_file = self.output_log_file
            if not overwrite:
                utils.check_rename_file(output_log_file, verbose=verbose_sub)
        elif overwrite == "dump":
            output_log_file = utils.generate_dump_file_name(
                self.output_log_file, timestamp=timestamp)
        dictionary = utils.convert_types_dict(self.log)
        with codecs.open(output_log_file, "w", encoding="utf-8") as f:
            json.dump(dictionary, f, separators=(",", ":"), indent=4)
        end = timer()
        if type(overwrite) == bool:
            if overwrite:
                print(header_string, "\nNF log file\n\t", output_log_file,
                      "\nupdated (or saved if it did not exist) in", str(end-start), "s.\n", show=verbose)
            else:
                print(header_string, "\nNF log file\n\t", output_log_file,
                      "\nsaved in", str(end-start), "s.\n", show=verbose)
        elif overwrite == "dump":
            print(header_string, "\nNF log file dump\n\t", output_log_file,
                  "\nsaved in", str(end-start), "s.\n", show=verbose)

    def save_data_indices(self, timestamp=None, overwrite=False, verbose=None):
        """ Save indices to member_n_idx.h5 as h5 file
        """
        verbose, verbose_sub = self.set_verbosity(verbose)
        start = timer()
        if timestamp is None:
            timestamp = "datetime_" + \
                datetime.now().strftime("%Y_%m_%d_%H_%M_%S_%fZ")[:-3]
        if type(overwrite) == bool:
            output_idx_h5_file = self.output_idx_h5_file
            if not overwrite:
                utils.check_rename_file(
                    output_idx_h5_file, verbose=verbose_sub)
        elif overwrite == "dump":
            output_idx_h5_file = utils.generate_dump_file_name(
                self.output_idx_h5_file, timestamp=timestamp)
        # self.close_opened_dataset(verbose=verbose_sub)
        utils.check_delete_files(output_idx_h5_file)
        h5_out = h5py.File(output_idx_h5_file, "w")
        h5_out.require_group(self.name)
        data = h5_out.require_group("idx")
        data["idx_train"] = self.idx_train
        data["idx_val"] = self.idx_val
        data["idx_test"] = self.idx_test
        h5_out.close()
        end = timer()
        self.log[timestamp] = {"action": "saved indices",
                               "file name": path.split(output_idx_h5_file)[-1]}
        # self.save_log(overwrite=True, verbose=verbose_sub) # log saved by save
        if type(overwrite) == bool:
            if overwrite:
                print(header_string, "\nIdx h5 file\n\t", output_idx_h5_file,
                      "\nupdated (or saved if it did not exist) in", str(end-start), "s.\n", show=verbose)
            else:
                print(header_string, "\nIdx h5 file\n\t", output_idx_h5_file,
                      "\nsaved in", str(end-start), "s.\n", show=verbose)
        elif overwrite == "dump":
            print(header_string, "\nIdx h5 file dump\n\t", output_idx_h5_file,
                  "\nsaved in", str(end-start), "s.\n", show=verbose)

    # def save_model_json(self, timestamp=None, overwrite=False, verbose=None):
    #    """ Save model to json
    #    """
    #    verbose, verbose_sub = self.set_verbosity(verbose)
    #    start = timer()
    #    if timestamp is None:
    #        timestamp = "datetime_"+datetime.now().strftime("%Y_%m_%d_%H_%M_%S_%fZ")[:-3]
    #    if type(overwrite) == bool:
    #        output_tf_model_json_file = self.output_tf_model_json_file
    #        if not overwrite:
    #            utils.check_rename_file(output_tf_model_json_file, verbose=verbose_sub)
    #    elif overwrite == "dump":
    #        output_tf_model_json_file = utils.generate_dump_file_name(self.output_tf_model_json_file, timestamp=timestamp)
    #    try:
    #        model_json = self.model.to_json()
    #    except:
    #        print(header_string,"\nModel not defined. No file is saved.\n")
    #        return
    #    with open(output_tf_model_json_file, "w") as json_file:
    #        json_file.write(model_json)
    #    end = timer()
    #
    #    self.log[timestamp] = {"action": "saved tf model json",
    #                           "file name": path.split(output_tf_model_json_file)[-1]}
    #    #self.save_log(overwrite=True, verbose=verbose_sub) # log saved by save
    #    if type(overwrite) == bool:
    #        if overwrite:
    #            print(header_string,"\nModel json file\n\t", output_tf_model_json_file, "\nupdated (or saved if it did not exist) in", str(end-start), "s.\n", show=verbose)
    #        else:
    #            print(header_string,"\nModel json file\n\t", output_tf_model_json_file, "\nsaved in", str(end-start), "s.\n", show=verbose)
    #    elif overwrite == "dump":
    #        print(header_string,"\nModel json file dump\n\t", output_tf_model_json_file, "\nsaved in", str(end-start), "s.\n", show=verbose)

    def save_model_weights_h5(self, timestamp=None, overwrite=False, verbose=None):
        """ Save model to h5
        """
        verbose, verbose_sub = self.set_verbosity(verbose)
        start = timer()
        if timestamp is None:
            timestamp = "datetime_" + \
                datetime.now().strftime("%Y_%m_%d_%H_%M_%S_%fZ")[:-3]
        if type(overwrite) == bool:
            output_tf_model_weights_h5_file = self.output_tf_model_weights_h5_file
            if not overwrite:
                utils.check_rename_file(
                    output_tf_model_weights_h5_file, verbose=verbose_sub)
        elif overwrite == "dump":
            output_tf_model_weights_h5_file = utils.generate_dump_file_name(
                self.output_tf_model_weights_h5_file, timestamp=timestamp)
        try:
            self.model.save_weights(output_tf_model_weights_h5_file)
        except:
            print(header_string, "\nModel not defined. No file is saved.\n")
            return
        end = timer()
        self.log[timestamp] = {"action": "saved tf model weights h5",
                               "file name": path.split(output_tf_model_weights_h5_file)[-1]}
        # self.save_log(overwrite=True, verbose=verbose_sub) # log saved by save
        if type(overwrite) == bool:
            if overwrite:
                print(header_string, "\nModel weights h5 file\n\t", output_tf_model_weights_h5_file,
                      "\nupdated (or saved if it did not exist) in", str(end-start), "s.\n", show=verbose)
            else:
                print(header_string, "\nModel weights h5 file\n\t", output_tf_model_weights_h5_file,
                      "\nsaved in", str(end-start), "s.\n", show=verbose)
        elif overwrite == "dump":
            print(header_string, "\nModel weights h5 file dump\n\t",
                  output_tf_model_weights_h5_file, "\nsaved in", str(end-start), "s.\n", show=verbose)

    # def save_model_onnx(self, timestamp=None, overwrite=False, verbose=None):
    #    """ Save model to onnx
    #    """
    #    verbose, verbose_sub = self.set_verbosity(verbose)
    #    start = timer()
    #    if timestamp is None:
    #        timestamp = "datetime_"+datetime.now().strftime("%Y_%m_%d_%H_%M_%S_%fZ")[:-3]
    #    if type(overwrite) == bool:
    #        output_tf_model_onnx_file = self.output_tf_model_onnx_file
    #        if not overwrite:
    #            utils.check_rename_file(output_tf_model_onnx_file, verbose=verbose_sub)
    #    elif overwrite == "dump":
    #        output_tf_model_onnx_file = utils.generate_dump_file_name(self.output_tf_model_onnx_file, timestamp=timestamp)
    #    try:
    #        onnx_model = tf2onnx.convert.from_keras(self.model)#, self.name)
    #    except:
    #        print(header_string,"\nModel not defined. No file is saved.\n")
    #        return
    #    onnx.save_model(onnx_model, output_tf_model_onnx_file)
    #    end = timer()
    #    self.log[timestamp] = {"action": "saved tf model onnx",
    #                           "file name": path.split(output_tf_model_onnx_file)[-1]}
    #    #self.save_log(overwrite=True, verbose=verbose_sub) # log saved by save
    #    if type(overwrite) == bool:
    #        if overwrite:
    #            print(header_string,"\nModel onnx file\n\t", output_tf_model_onnx_file, "\nupdated (or saved if it did not exist) in", str(end-start), "s.\n", show=verbose)
    #        else:
    #            print(header_string,"\nModel onnx file\n\t", output_tf_model_onnx_file, "\nsaved in", str(end-start), "s.\n", show=verbose)
    #    elif overwrite == "dump":
    #        print(header_string,"\nModel onnx file dump\n\t", output_tf_model_onnx_file, "\nsaved in", str(end-start), "s.\n", show=verbose)

    def save_history_json(self, timestamp=None, overwrite=False, verbose=None):
        """ Save training history to json
        """
        verbose, verbose_sub = self.set_verbosity(verbose)
        start = timer()
        if timestamp is None:
            timestamp = "datetime_" + \
                datetime.now().strftime("%Y_%m_%d_%H_%M_%S_%fZ")[:-3]
        if type(overwrite) == bool:
            output_history_json_file = self.output_history_json_file
            if not overwrite:
                utils.check_rename_file(
                    output_history_json_file, verbose=verbose_sub)
        elif overwrite == "dump":
            output_history_json_file = utils.generate_dump_file_name(
                self.output_history_json_file, timestamp=timestamp)
        # for key in list(history.keys()):
        #    self.history[utils.metric_name_abbreviate(key)] = self.history.pop(key)
        dictionary = utils.convert_types_dict(self.history)
        with codecs.open(output_history_json_file, "w", encoding="utf-8") as f:
            json.dump(dictionary, f, separators=(
                ",", ":"), sort_keys=True, indent=4)
        end = timer()
        self.log[timestamp] = {"action": "saved history json",
                               "file name": path.split(output_history_json_file)[-1]}
        # self.save_log(overwrite=True, verbose=verbose_sub) # log saved by save
        if type(overwrite) == bool:
            if overwrite:
                print(header_string, "\nModel history file\n\t", output_history_json_file,
                      "\nupdated (or saved if it did not exist) in", str(end-start), "s.\n", show=verbose)
            else:
                print(header_string, "\nModel history file\n\t", output_history_json_file,
                      "\nsaved in", str(end-start), "s.\n", show=verbose)
        elif overwrite == "dump":
            print(header_string, "\nModel history file dump\n\t",
                  output_history_json_file, "\nsaved in", str(end-start), "s.\n", show=verbose)

    def save_json(self, timestamp=None, overwrite=False, verbose=None):
        """ Save object to json
        """
        verbose, verbose_sub = self.set_verbosity(verbose)
        start = timer()
        if timestamp is None:
            timestamp = "datetime_" + \
                datetime.now().strftime("%Y_%m_%d_%H_%M_%S_%fZ")[:-3]
        if type(overwrite) == bool:
            output_json_file = self.output_json_file
            if not overwrite:
                utils.check_rename_file(output_json_file, verbose=verbose_sub)
        elif overwrite == "dump":
            output_json_file = utils.generate_dump_file_name(
                self.output_json_file, timestamp=timestamp)
        dictionary = utils.dic_minus_keys(self.__dict__, ["_NF__resources_inputs",
                                                          "callbacks", "data", "history",
                                                          "idx_test", "idx_train", "idx_val",
                                                          "input_files_base_name", "input_folder", "input_json_file",
                                                          "input_history_json_file", "input_idx_h5_file", "input_log_file",
                                                          "input_predictions_h5_file",
                                                          "input_preprocessing_pickle_file", "input_file",
                                                          "input_tf_model_weights_h5_file", "input_data_file",
                                                          "layers", "load_on_RAM",
                                                          "log", "loss", "metrics", "model", "optimizer",
                                                          "NN", "Bijector", "Flow", "log_prob",
                                                          "base_distribution", "trainable_distribution",
                                                          "output_folder", "output_figures_folder",
                                                          "output_figures_base_file_name", "output_figures_base_file_path",
                                                          "output_files_base_name", "output_history_json_file",
                                                          "output_idx_h5_file", "output_h5_file", "output_json_file",
                                                          "output_log_file", "output_predictions_h5_file", "output_predictions_json_file",
                                                          "output_preprocessing_pickle_file",
                                                          "output_tf_model_graph_pdf_file", "output_tf_model_weights_h5_file",
                                                          "output_tf_model_onnx_file",
                                                          "output_checkpoints_files",
                                                          "output_checkpoints_folder", "output_figure_plot_losses_keras_file",
                                                          "output_tensorboard_log_dir", "predictions",
                                                          "rotationX", "scalerX", "verbose",
                                                          "X_test", "X_train", "X_val"])
        dictionary = utils.convert_types_dict(dictionary)
        with codecs.open(output_json_file, "w", encoding="utf-8") as f:
            json.dump(dictionary, f, separators=(",", ":"), indent=4)
        end = timer()
        self.log[timestamp] = {"action": "saved object json",
                               "file name": path.split(output_json_file)[-1]}
        # self.save_log(overwrite=True, verbose=verbose_sub) # log saved by save
        if type(overwrite) == bool:
            if overwrite:
                print(header_string, "\nJson file\n\t", output_json_file,
                      "\nupdated (or saved if it did not exist) in", str(end-start), "s.\n", show=verbose)
            else:
                print(header_string, "\nJson file\n\t", output_json_file,
                      "\nsaved in", str(end-start), "s.\n", show=verbose)
        elif overwrite == "dump":
            print(header_string, "\n\nJson file\n\t dump\n\t", output_json_file,
                  "\nsaved in", str(end-start), "s.\n", show=verbose)

    def save_predictions_json(self, timestamp=None, overwrite=False, verbose=None):
        """ Save predictions json
        """
        verbose, verbose_sub = self.set_verbosity(verbose)
        start = timer()
        if timestamp is None:
            timestamp = "datetime_" + \
                datetime.now().strftime("%Y_%m_%d_%H_%M_%S_%fZ")[:-3]
        if type(overwrite) == bool:
            output_predictions_json_file = self.output_predictions_json_file
            if not overwrite:
                utils.check_rename_file(
                    output_predictions_json_file, verbose=verbose_sub)
        elif overwrite == "dump":
            output_predictions_json_file = utils.generate_dump_file_name(
                self.output_predictions_json_file, timestamp=timestamp)
        dictionary = utils.convert_types_dict(self.predictions)
        with codecs.open(output_predictions_json_file, "w", encoding="utf-8") as f:
            json.dump(dictionary, f, separators=(",", ":"), indent=4)
        end = timer()
        self.log[timestamp] = {"action": "saved predictions json",
                               "file name": path.split(output_predictions_json_file)[-1]}
        # self.save_log(overwrite=True, verbose=verbose_sub) # log saved by save
        if type(overwrite) == bool:
            if overwrite:
                print(header_string, "\nPredictions json file\n\t", output_predictions_json_file,
                      "\nupdated (or saved if it did not exist) in", str(end-start), "s.\n", show=verbose)
            else:
                print(header_string, "\nPredictions json file\n\t", output_predictions_json_file,
                      "\nsaved in", str(end-start), "s.\n", show=verbose)
        elif overwrite == "dump":
            print(header_string, "\nPredictions json file dump\n\t",
                  output_predictions_json_file, "\nsaved in", str(end-start), "s.\n", show=verbose)

    def save_predictions_h5(self, timestamp=None, overwrite=False, verbose=None):
        """ Save predictions h5
        """
        verbose, verbose_sub = self.set_verbosity(verbose)
        start = timer()
        if timestamp is None:
            timestamp = "datetime_" + \
                datetime.now().strftime("%Y_%m_%d_%H_%M_%S_%fZ")[:-3]
        if type(overwrite) == bool:
            output_predictions_h5_file = self.output_predictions_h5_file
            if not overwrite:
                utils.check_rename_file(
                    output_predictions_h5_file, verbose=verbose_sub)
        elif overwrite == "dump":
            output_predictions_h5_file = utils.generate_dump_file_name(
                self.output_predictions_h5_file, timestamp=timestamp)
        dictionary = dict(self.predictions)
        dd.io.save(output_predictions_h5_file, dictionary)
        end = timer()
        self.log[timestamp] = {"action": "saved predictions json",
                               "file name": path.split(output_predictions_h5_file)[-1]}
        if type(overwrite) == bool:
            if overwrite:
                print(header_string, "\nPredictions h5 file\n\t", output_predictions_h5_file,
                      "\nupdated (or saved if it did not exist) in", str(end-start), "s.\n", show=verbose)
            else:
                print(header_string, "\nPredictions h5 file\n\t", output_predictions_h5_file,
                      "\nsaved in", str(end-start), "s.\n", show=verbose)
        elif overwrite == "dump":
            print(header_string, "\nPredictions h5 file dump\n\t",
                  output_predictions_h5_file, "\nsaved in", str(end-start), "s.\n", show=verbose)

    def save_predictions(self, timestamp=None, overwrite=False, verbose=None):
        """ Save predictions h5 and json
        """
        verbose, verbose_sub = self.set_verbosity(verbose)
        if timestamp is None:
            timestamp = "datetime_" + \
                datetime.now().strftime("%Y_%m_%d_%H_%M_%S_%fZ")[:-3]
        self.save_predictions_json(
            timestamp=timestamp, overwrite=overwrite, verbose=verbose)
        self.save_predictions_h5(
            timestamp=timestamp, overwrite=overwrite, verbose=verbose)

    def save_preprocessing(self, timestamp=None, overwrite=False, verbose=None):
        """ 
        Save X scaler and rotation to pickle
        """
        verbose, verbose_sub = self.set_verbosity(verbose)
        start = timer()
        if timestamp is None:
            timestamp = "datetime_" + \
                datetime.now().strftime("%Y_%m_%d_%H_%M_%S_%fZ")[:-3]
        if type(overwrite) == bool:
            output_preprocessing_pickle_file = self.output_preprocessing_pickle_file
            if not overwrite:
                utils.check_rename_file(
                    output_preprocessing_pickle_file, verbose=verbose_sub)
        elif overwrite == "dump":
            output_preprocessing_pickle_file = utils.generate_dump_file_name(
                self.output_preprocessing_pickle_file, timestamp=timestamp)
        pickle_out = open(output_preprocessing_pickle_file, "wb")
        pickle.dump(self.scalerX, pickle_out, protocol=pickle.HIGHEST_PROTOCOL)
        pickle.dump(self.rotationX, pickle_out,
                    protocol=pickle.HIGHEST_PROTOCOL)
        pickle_out.close()
        end = timer()
        self.log[timestamp] = {"action": "saved scalers and X rotation h5",
                               "file name": path.split(output_preprocessing_pickle_file)[-1]}
        # self.save_log(overwrite=True, verbose=verbose_sub) # log saved by save
        if type(overwrite) == bool:
            if overwrite:
                print(header_string, "\nPreprocessing pickle file\n\t", output_preprocessing_pickle_file,
                      "\nupdated (or saved if it did not exist) in", str(end-start), "s.\n", show=verbose)
            else:
                print(header_string, "\nPreprocessing pickle file\n\t",
                      output_preprocessing_pickle_file, "\nsaved in", str(end-start), "s.\n", show=verbose)
        elif overwrite == "dump":
            print(header_string, "\nPreprocessing pickle file dump\n\t",
                  output_preprocessing_pickle_file, "\nsaved in", str(end-start), "s.\n", show=verbose)

    # def save_model_graph_pdf(self, timestamp=None, overwrite=False, verbose=None):
    #    """ Save model graph to pdf
    #    """
    #    verbose, verbose_sub = self.set_verbosity(verbose)
    #    start = timer()
    #    if timestamp is None:
    #        timestamp = "datetime_"+datetime.now().strftime("%Y_%m_%d_%H_%M_%S_%fZ")[:-3]
    #    if type(overwrite) == bool:
    #        output_tf_model_graph_pdf_file = self.output_tf_model_graph_pdf_file
    #        if not overwrite:
    #            utils.check_rename_file(output_tf_model_graph_pdf_file, verbose=verbose_sub)
    #    elif overwrite == "dump":
    #        output_tf_model_graph_pdf_file = utils.generate_dump_file_name(self.output_tf_model_graph_pdf_file, timestamp=timestamp)
    #    png_file = path.splitext(output_tf_model_graph_pdf_file)[0]+".png"
    #    try:
    #        plot_model(self.model, show_shapes=True, show_layer_names=True, to_file=png_file)
    #    except:
    #        print(header_string,"\nModel not defined. No file is saved.\n")
    #        return
    #    utils.make_pdf_from_img(png_file)
    #    try:
    #        remove(png_file)
    #    except:
    #        try:
    #            time.sleep(1)
    #            remove(png_file)
    #        except:
    #            print(header_string,"\nCannot remove png file",png_file,".\n", show=verbose)
    #    end = timer()
    #    self.log[timestamp] = {"action": "saved model graph pdf",
    #                           "file name": path.split(output_tf_model_graph_pdf_file)[-1]}
    #    #self.save_log(overwrite=True, verbose=verbose_sub) # log saved by save
    #    if type(overwrite) == bool:
    #        if overwrite:
    #            print(header_string,"\nModel graph pdf file\n\t", output_tf_model_graph_pdf_file, "\nupdated (or saved if it did not exist) in", str(end-start), "s.\n", show=verbose)
    #        else:
    #            print(header_string,"\nModel graph pdf file\n\t", output_tf_model_graph_pdf_file, "\nsaved in", str(end-start), "s.\n", show=verbose)
    #    elif overwrite == "dump":
    #        print(header_string,"\nModel graph pdf file dump\n\t", output_tf_model_graph_pdf_file, "\nsaved in", str(end-start), "s.\n", show=verbose)

    def save(self, timestamp=None, overwrite=False, verbose=None):
        """ Save all model information
        - data indices as hdf5 dataset
        - model in json format
        - model in h5 format (with weights)
        - model in onnx format
        - history, including summary log as json
        - scalers to jlib file
        - model graph to pdf
        """
        verbose, _ = self.set_verbosity(verbose)
        if timestamp is None:
            timestamp = "datetime_" + \
                datetime.now().strftime("%Y_%m_%d_%H_%M_%S_%fZ")[:-3]
        self.save_data_indices(timestamp=timestamp,
                               overwrite=overwrite, verbose=verbose)
        # self.save_model_json(timestamp=timestamp,overwrite=overwrite,verbose=verbose)
        self.save_model_weights_h5(
            timestamp=timestamp, overwrite=overwrite, verbose=verbose)
        # elf.save_model_onnx(timestamp=timestamp,overwrite=overwrite,verbose=verbose)
        self.save_history_json(timestamp=timestamp,
                               overwrite=overwrite, verbose=verbose)
        self.save_predictions(timestamp=timestamp,
                              overwrite=overwrite, verbose=verbose)
        self.save_json(timestamp=timestamp,
                       overwrite=overwrite, verbose=verbose)
        self.save_preprocessing(timestamp=timestamp,
                                overwrite=overwrite, verbose=verbose)
        # self.save_model_graph_pdf(timestamp=timestamp,overwrite=overwrite,verbose=verbose)
        self.save_log(timestamp=timestamp,
                      overwrite=overwrite, verbose=verbose)
