__all__ = ["Data"]

import builtins
import codecs
import json
import time
from datetime import datetime
from os import path
from timeit import default_timer as timer

import deepdish as dd
import h5py
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from . import inference, utils
from .corner import corner, extend_corner_range, get_1d_hist
from .show_prints import Verbosity, print

sns.set()
kubehelix = sns.color_palette("cubehelix", 30)
reds = sns.color_palette("Reds", 30)
greens = sns.color_palette("Greens", 30)
blues = sns.color_palette("Blues", 30)

mplstyle_path = path.join(path.split(path.realpath(__file__))[0],"matplotlib.mplstyle")

header_string = "=============================="
footer_string = "------------------------------"

class Data(Verbosity):
    """
    This class contains the ``Data`` object representing the dataset used for training, validating and testing
    the NF4HEP. It can be creaded both feeding X and Y data or by loading an existing ``Data`` object.
    """
    def __init__(self,
                 name = None,
                 data_X = None,
                 dtype = None,
                 pars_central=None,
                 pars_pos_poi=None,
                 pars_pos_nuis=None,
                 pars_labels=None,
                 pars_bounds=None,
                 test_fraction = None,
                 load_on_RAM=False,
                 output_folder = None,
                 input_file=None,
                 verbose = True
                 ):
        """
        The :mod:`Data <data>` object can be initialized in two different ways, depending on the value of
        the :argument:`input_file` argument.

        - :argument:`input_file` is ``None`` (default)

            All other arguments are parsed and saved in corresponding attributes. If no :argument:`name` is given,
            then one is created.
            The object is saved upon creation through the :meth:`Data.save <NF4HEP.Data.save>` method.

        - :argument:`input_file` is not ``None``

            The object is reconstructed from the input files through the private method
            :meth:`Data.__load <NF4HEP.Data._Data__load>`.
            If the :argument:`load_on_RAM` argument is ``True``, then all samples are loaded into RAM, otherwise the dataset is
            kept open and data are retrieved on demand.
            Depending on the value of the input argument :argument:`output_folder` the :meth:`Data.__init__ <NF4HEP.Data.__init__>` method behaves as follows:

                - If :argument:`output_folder` is ``None`` (default)
                    
                    The attribute :attr:`Data.output_folder <NF4HEP.Data.output_folder>`
                    is set from the :attr:`Data.input_folder <NF4HEP.Data.input_folder>` one.
                - If :argument:`output_folder` corresponds to a path different from that stored in the loaded :attr:`Data.output_folder <NF4HEP.Data.output_folder>` attribute
                    
                    - if path stored in the loaded :attr:`Data.output_folder <NF4HEP.Data.output_folder>` attribute exists, then its content is copied to the new ``output_folder`` (if the new ``output_foler`` already exists it is renamed by adding a timestamp);
                    - if path stored in the loaded :attr:`Data.output_folder <NF4HEP.Data.output_folder>` does not exists, then the content of the path :attr:`Data.input_folder <NF4HEP.Data.input_folder>` is copied to the new ``output_folder``.
                - If :argument:`output_folder` corresponds to the same path stored in the loaded :attr:`Data.output_folder <NF4HEP.Data.output_folder>` attribute
                    
                    Output folder, files, and path attributes are not updated and everything is read from the loaded object.

        - **Arguments**

            See class :ref:`Arguments documentation <data_arguments>`.

        - **Creates files**

            - :attr:`Data.output_log_file <NF4HEP.Data.output_log_file>`
            - :attr:`Data.output_h5_file <NF4HEP.Data.output_h5_file>` (only the first time the object is created, i.e. if :argument:`input_file` is ``None``)
            - :attr:`Data.output_samples_h5_file <NF4HEP.Data.output_samples_h5_file>` (only the first time the object is created, i.e. if :argument:`input_file` is ``None``)
        """
        self.verbose = verbose
        verbose, verbose_sub = self.set_verbosity(verbose)
        timestamp = "datetime_"+datetime.now().strftime("%Y_%m_%d_%H_%M_%S_%fZ")[:-3]
        print(header_string, "\nInitialize Data object.\n", show=verbose)
        self.output_folder = output_folder
        self.input_file = input_file
        self.__check_define_input_files(verbose=verbose_sub)
        if self.input_file == None:
            self.log = {timestamp: {"action": "created"}}
            self.name = name
            self.__check_define_name()
            self.__check_define_output_files(timestamp=timestamp,verbose=verbose_sub)
            self.dtype = dtype
            self.data_X = data_X
            self.__check_define_data()
            self.pars_pos_poi = pars_pos_poi
            self.pars_pos_nuis = pars_pos_nuis
            self.pars_central = pars_central
            self.ndims = self.data_X.shape[1]
            self.npoints = self.data_X.shape[0]
            self.pars_labels = pars_labels
            self.pars_bounds = pars_bounds
            self.__check_define_pars(verbose=verbose_sub)
            self.test_fraction = test_fraction
            self.__define_test_fraction()
            self.load_on_RAM = load_on_RAM
            self.predictions = {"Figures": {}}
            self.save(overwrite=False, verbose=verbose_sub)
        else:
            self.load_on_RAM = load_on_RAM
            if dtype != None:
                if type(dtype) == str:
                    self.dtype_required = dtype
                elif type(dtype) == list:
                    self.dtype_required = dtype[1]
            else:
                self.dtype_required = "float64"
            self.__load(verbose=verbose_sub)
            self.__check_define_output_files(timestamp=timestamp,verbose=verbose_sub)
            self.__define_test_fraction()
            try:
                self.predictions["Figures"]
            except:
                self.reset_predictions(delete_figures=True, verbose=verbose_sub)
            self.predictions["Figures"] = utils.check_figures_dic(self.predictions["Figures"],output_figures_folder=self.output_figures_folder)
            self.save_log(overwrite=True, verbose=verbose_sub)
        self.data_dictionary = {"X_train": np.array([[]], dtype=self.dtype_required), 
                                "X_val": np.array([[]], dtype=self.dtype_required),
                                "X_test": np.array([[]], dtype=self.dtype_required),
                                "idx_train": np.array([], dtype="int"), 
                                "idx_val": np.array([], dtype="int"), 
                                "idx_test": np.array([], dtype="int")}

    def __check_define_input_files(self,verbose=False):
        """
        Private method used by the :meth:`Data.__init__ <NF4HEP.Data.__init__>` one
        to set the attributes corresponding to input files:

            - :attr:`Data.input_log_file <NF4HEP.Data.input_log_file>`
            - :attr:`Data.input_h5_file <NF4HEP.Data.input_h5_file>`
            - :attr:`Data.input_samples_h5_file <NF4HEP.Data.input_samples_h5_file>`

        depending on the value of the 
        :attr:`Data.input_file <NF4HEP.Data.input_file>` attribute.
        """
        verbose, verbose_sub = self.set_verbosity(verbose)
        if self.input_file == None:
            self.input_h5_file = None
            self.input_samples_h5_file = None
            self.input_log_file = None
            self.input_folder = None
            print(header_string,"\nNo Data input files and folders specified.\n", show=verbose)
        else:
            self.input_file = path.abspath(path.splitext(self.input_file)[0])
            self.input_h5_file = self.input_file+".h5"
            self.input_samples_h5_file = self.input_file+"_samples.h5"
            self.input_log_file = self.input_file+".log"
            self.input_folder = path.split(self.input_file)[0]
            print(header_string,"\nData input folder set to\n\t", self.input_folder,".\n",show=verbose)

    def __check_define_output_files(self,timestamp=None,verbose=False):
        """
        Private method used by the :meth:`Data.__init__ <NF4HEP.Data.__init__>` one
        to set the attributes corresponding to output folders
        
            - :attr:`NF.output_figures_folder <NF4HEP.NF.output_figures_folder>`
            - :attr:`NF.output_folder <NF4HEP.NF.output_folder>`
        
        and output files

            - :attr:`NF.output_figures_base_file <NF4HEP.NF.output_figures_base_file>`
            - :attr:`Data.output_log_file <NF4HEP.Data.output_log_file>`
            - :attr:`Data.output_predictions_json_file <NF4HEP.Data.output_predictions_json_file>`
            - :attr:`Data.output_h5_file <NF4HEP.Data.output_h5_file>`
            - :attr:`Data.output_samples_h5_file <NF4HEP.Data.output_samples_h5_file>`

        depending on the values of the 
        :attr:`Data.output_folder <NF4HEP.Data.output_folder>` attribute.
        It also creates the folder 
        :attr:`Data.output_folder <NF4HEP.Data.output_folder>` if 
        it does not exist.
        """
        verbose, verbose_sub = self.set_verbosity(verbose)
        if self.output_folder is not None:
            self.output_folder = path.abspath(self.output_folder)
            if self.input_folder is not None and self.output_folder != self.input_folder:
                utils.copy_and_save_folder(self.input_folder, self.output_folder, timestamp=timestamp, verbose=verbose)
        else:
            if self.input_folder is not None:
                self.output_folder = self.input_folder
            else:
                self.output_folder = path.abspath("")
        self.output_folder = utils.check_create_folder(self.output_folder)
        self.output_figures_folder =  utils.check_create_folder(path.join(self.output_folder, "figures"))
        self.output_figures_base_file_name = self.name+"_figure"
        self.output_figures_base_file_path = path.join(self.output_figures_folder, self.output_figures_base_file_name)
        self.output_h5_file = path.join(self.output_folder, self.name+".h5")
        self.output_samples_h5_file = path.join(self.output_folder, self.name+"_samples.h5")
        self.output_log_file = path.join(self.output_folder, self.name+".log")
        self.output_predictions_json_file = path.join(self.output_folder, self.name+"_predictions.json")
        print(header_string,"\nData output folder set to\n\t", self.output_folder,".\n",show=verbose)
        
    def __check_define_name(self):
        """
        Private method used by the :meth:`Data.__init__ <NF4HEP.Data.__init__>` one
        to define the :attr:`Data.name <NF4HEP.Data.name>` attribute.
        If it is ``None`` it replaces it with
        ``"model_"+datetime.now().strftime("%Y-%m-%d-%H-%M-%S.%fZ")[:-3]+"_data"``,
        otherwise it appends the suffix "_data" (preventing duplication if it is already present).
        """
        if self.name == None:
            timestamp = "datetime_"+datetime.now().strftime("%Y_%m_%d_%H_%M_%S_%fZ")[:-3]
            self.name = "model_"+timestamp+"_data"
        #else:
        #    self.name = utils.check_add_suffix(self.name, "_data")

    def __check_define_data(self):
        """ 
        Private method used by the :meth:`Data.__init__ <NF4HEP.Data.__init__>` one
        to set the data type attributes
        :attr:`Data.dtype_required <NF4HEP.Data.dtype_required>` and
        :attr:`Data.dtype_stored <NF4HEP.Data.dtype_stored>` (defaults "float64"), and to convert
        :attr:`Data.data_X <NF4HEP.Data.data_X>` into the dtype corresponding to 
        :attr:`Data.dtype_stored <NF4HEP.Data.dtype_stored>`.
        """
        if self.dtype == None:
            self.dtype_stored = "float64"
            self.dtype_required = "float64"
        elif type(self.dtype) == str:
            self.dtype_stored = self.dtype
            self.dtype_required = self.dtype
        elif type(self.dtype) == list:
            self.dtype_stored = self.dtype[0]
            self.dtype_required = self.dtype[1]
        del(self.dtype)
        self.data_X = self.data_X.astype(self.dtype_stored)

    def __check_define_pars(self, verbose=None):
        """
        Private method used by the :meth:`Data.__init__ <NF4HEP.Data.__init__>` one
        to check the consistency of the attributes

            - :attr:`Data.pars_pos_nuis <NF4HEP.Data.pars_pos_nuis>`
            - :attr:`Data.pars_pos_poi <NF4HEP.Data.pars_pos_poi>`
            - :attr:`Data.pars_central <NF4HEP.Data.pars_pos_poi>`
            - :attr:`Data.pars_labels <NF4HEP.Data.pars_labels>`
            - :attr:`Data.pars_bounds <NF4HEP.Data.pars_bounds>` 

        and to convert the attributes

            - :attr:`Data.pars_pos_nuis <NF4HEP.Data.pars_pos_nuis>`
            - :attr:`Data.pars_pos_poi <NF4HEP.Data.pars_pos_poi>`
            - :attr:`Data.pars_central <NF4HEP.Data.pars_pos_poi>`
            - :attr:`Data.pars_bounds <NF4HEP.Data.pars_bounds>` 
        
        to |numpy_link| arrays.
        If no parameters positions are specified, all parameters are assumed to be parameters of interest.
        If only the position of the parameters of interest or of the nuisance parameters is specified,
        the other is automatically generated by matching dimensions.
        If labels are not provided then :attr:`Data.pars_labels <NF4HEP.Data.pars_labels>`
        is set to the value of :attr:`Data.pars_labels_auto <NF4HEP.Data.pars_labels_auto>`.
        If parameters bounds are not provided, they are set to ``(-np.inf,np.inf)``.
        A check is performed on the length of the four attributes and an Exception is raised if the length
        does not match :attr:`Data.ndims <NF4HEP.Data.ndims>`.

        - **Arguments**

            - **verbose**
            
                See :argument:`verbose <common_methods_arguments.verbose>`.
        """
        verbose, _ = self.set_verbosity(verbose)
        if self.pars_central is not None:
            self.pars_central = np.array(self.pars_central)
            if len(self.pars_central) != self.ndims:
                raise Exception("The length of the parameters central values array does not match the number of dimensions.")
        else:
            self.pars_central = np.zeros(self.ndims)
            print(header_string,"\nNo central values for the parameters 'pars_central' has been specified. They have been set to zero for all\
                parameters. If they are known it == better to build the object providing parameters central values.\n", show=verbose)
        if self.pars_pos_nuis is not None and self.pars_pos_poi is not None:
            if len(self.pars_pos_poi)+len(self.pars_pos_nuis) == self.ndims:
                self.pars_pos_nuis = np.array(self.pars_pos_nuis)
                self.pars_pos_poi = np.array(self.pars_pos_poi)
            else:
                raise Exception("The number of parameters positions do not match the number of dimensions.")
        elif self.pars_pos_nuis is None and self.pars_pos_poi is None:
            print(header_string,"\nThe positions of the parameters of interest (pars_pos_poi) and of the nuisance parameters (pars_pos_nuis) have not been specified. Assuming all parameters are parameters of interest.\n", show=verbose)
            self.pars_pos_nuis = np.array([])
            self.pars_pos_poi = np.array(list(range(self.ndims)))
        elif self.pars_pos_nuis is not None and self.pars_pos_poi is None:
            print(header_string,"\nOnly the positions of the nuisance parameters have been specified. Assuming all other parameters are parameters of interest.\n", show=verbose)
            self.pars_pos_poi = np.setdiff1d(np.array(range(self.ndims)), np.array(self.pars_pos_nuis))
        elif self.pars_pos_nuis is None and self.pars_pos_poi is not None:
            print(header_string,"\nOnly the positions of the parameters of interest have been specified. Assuming all other parameters are nuisance parameters.\n", show=verbose)
            self.pars_pos_nuis = np.setdiff1d(np.array(range(self.ndims)), np.array(self.pars_pos_poi))
        self.pars_labels_auto = utils.define_pars_labels_auto(self.pars_pos_poi, self.pars_pos_nuis)
        if self.pars_labels is None:
            self.pars_labels = self.pars_labels_auto
        elif len(self.pars_labels) != self.ndims:
            raise Exception("The number of parameters labels do not match the number of dimensions.")
        if self.pars_bounds is not None:
            self.pars_bounds = np.array(self.pars_bounds)
        else:
            self.pars_bounds = np.vstack([np.full(self.ndims, -np.inf), np.full(self.ndims, np.inf)]).T
        if len(self.pars_bounds) != self.ndims:
            raise Exception("The length of the parameters bounds array does not match the number of dimensions.")

    def __load(self,verbose=None):
        """
        Private method used by the :meth:`Data.__init__ <NF4HEP.Data.__init__>` one 
        to load a previously saved
        :mod:`Data <data>` object from the files 
        
            - :attr:`Data.input_samples_h5_file <NF4HEP.Data.input_samples_h5_file>`
            - :attr:`Data.input_h5_file <NF4HEP.Data.input_h5_file>`
            - :attr:`Data.input_log_file <NF4HEP.Data.input_log_file>`

        If :attr:`Data.load_on_RAM <NF4HEP.Data.load_on_RAM>` is ``True``, then all samples are loaded into RAM, 
        otherwise the dataset is kept open and data are retrieved on demand.

        - **Arguments**

            - **verbose**
            
                See :argument:`verbose <common_methods_arguments.verbose>`.
        """
        verbose, verbose_sub = self.set_verbosity(verbose)
        start = timer()
        dictionary = dd.io.load(self.input_h5_file)
        self.__dict__.update(dictionary)
        with open(self.input_log_file) as json_file:
            dictionary = json.load(json_file)
        self.log = dictionary
        self.opened_dataset = h5py.File(self.input_samples_h5_file, "r")
        data = self.opened_dataset["data"]
        self.data_X = data.get("X")
        if self.load_on_RAM:
            self.data_X = self.data_X[:].astype(self.dtype_required)
            self.opened_dataset.close()
        end = timer()
        timestamp = "datetime_"+datetime.now().strftime("%Y_%m_%d_%H_%M_%S_%fZ")[:-3]
        self.log[timestamp] = {"action": "loaded", 
                               "files names": [path.split(self.input_samples_h5_file)[-1],
                                               path.split(self.input_log_file)[-1],
                                               path.split(self.input_h5_file)[-1]]}
        print(header_string,"\nData object loaded in", str(end-start), ".\n",show=verbose)
        if self.load_on_RAM:
            print(header_string,"\nSamples loaded on RAM.\n", show=verbose)

    def __define_test_fraction(self):
        """ 
        Private method used by the :meth:`Data.__init__ <NF4HEP.Data.__init__>` one
        to set the :attr:`Data.test_fraction <NF4HEP.Data.test_fraction>` attribute and, from this, the
        two attributes :attr:`Data.train_range <NF4HEP.Data.train_range>` and 
        :attr:`Data.test_range <NF4HEP.Data.test_range>` containing the range of indices of
        the train (and validation) and test sets given by
        :attr:`Data.data_X <NF4HEP.Data.data_X>`.
        """
        if self.test_fraction == None:
            self.test_fraction = 0
        self.train_range = range(int(round(self.npoints*(1-self.test_fraction))))
        self.test_range = range(int(round(self.npoints*(1-self.test_fraction))),self.npoints)

    def __check_sync_data_dictionary(self,verbose=None):
        """
        Private method used to check that indices and data are synced in the 
        :attr:`Data.data_dictionary <NF4HEP.Data.data_dictionary>`
        dictionary. It is called by all methods generating or updating data.
        This is needed when the :mod:`Data <data>` object is used by the 
        :class:`NF <NF4HEP.NF>` one. In particular, when the latter is imported from files
        only indices are loaded and properly put in the corresponding 
        :attr:`Data.data_dictionary <NF4HEP.Data.data_dictionary>`, while, to gain speed, data are not.
        Whenever the :class:`NF <NF4HEP.NF>` needs data, and calls any of its method that
        generate them, the method :meth:`Data.__check_sync_data_dictionary <NF4HEP.Data._Data__check_sync_data_dictionary>`
        automatically loads the data corresponding to the existing indices in the 
        :attr:`Data.data_dictionary <NF4HEP.Data.data_dictionary>` of the corresponding 
        :mod:`Data <data>` object. See the documentation of 
        the :mod:`DNNLikelihood <dnn_likelihood>` object for more details.
        """
        if len(self.data_dictionary["idx_train"]) != 0 and len(self.data_dictionary["idx_train"]) != len(self.data_dictionary["X_train"]):
            self.data_dictionary["X_train"] = self.data_X[self.data_dictionary["idx_train"]].astype(self.dtype_required)
            print(header_string,"\nLoaded train data corresponding to existing indices.\n")
        if len(self.data_dictionary["idx_val"]) != 0 and len(self.data_dictionary["idx_val"]) != len(self.data_dictionary["X_val"]):
            self.data_dictionary["X_val"] = self.data_X[self.data_dictionary["idx_val"]].astype(self.dtype_required)
            print(header_string,"\nLoaded val data corresponding to existing indices.\n")
        if len(self.data_dictionary["idx_test"]) != 0 and len(self.data_dictionary["idx_test"]) != len(self.data_dictionary["X_test"]):
            self.data_dictionary["X_test"] = self.data_X[self.data_dictionary["idx_test"]].astype(self.dtype_required)
            print(header_string,"\nLoaded test data corresponding to existing indices.\n")

    def __set_pars_labels(self, pars_labels):
        """
        Private method that returns the ``pars_labels`` choice based on the ``pars_labels`` input.

        - **Arguments**

            - **pars_labels**
            
                Could be either one of the keyword strings ``"original"`` and ``"generic"`` or a list of labels
                strings with the length of the parameters array. If ``pars_labels="original"`` or ``pars_labels="generic"``
                the function returns the value of :attr:`Sampler.pars_labels <NF4HEP.Data.pars_labels>`
                or :attr:`Data.pars_labels_auto <NF4HEP.Data.pars_labels_auto>`, respectively,
                while if ``pars_labels`` == a list, the function just returns the input.

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

    def define_test_fraction(self,verbose=None):
        """ 
        Method analog to the private method :meth:`Data.__define_test_fraction <NF4HEP.Data._Data__define_test_fraction>`
        useful to change the available fraction of train/valifation vs test data.
        Differently from the analog private method, this one updates the files corresponding to the
        :attr:`Data.output_h5_file <NF4HEP.Data.output_h5_file>` and
        :attr:`Data.output_log_file <NF4HEP.Data.output_log_file>` attributes.

        - **Arguments**

            - **verbose**
            
                See :argument:`verbose <common_methods_arguments.verbose>`.

        - **Updates files**

            - :attr:`Data.output_log_file <NF4HEP.Data.output_log_file>`
            - :attr:`Data.output_h5_file <NF4HEP.Data.output_h5_file>`
        """
        _, verbose_sub = self.set_verbosity(verbose)
        if self.test_fraction == None:
            self.test_fraction = 0
        self.train_range = range(int(round(self.npoints*(1-self.test_fraction))))
        self.test_range = range(int(round(self.npoints*(1-self.test_fraction))),self.npoints)
        self.save(overwrite=True, verbose=verbose_sub)

    def reset_predictions(self, 
                          delete_figures=False, 
                          verbose=None):
        """
        Re-initializes the :attr:`Lik.predictions <NF4HEP.Lik.predictions>` dictionary to

         .. code-block:: python

            predictions = {"logpdf_max": {},
                           "logpdf_profiled_max": {},
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
        if delete_figures:
            utils.check_delete_all_files_in_path(self.output_figures_folder)
            figs = {}
            print(header_string,"\nAll predictions and figures have been deleted and the 'predictions' attribute has been initialized.\n")
        else:
            figs = utils.check_figures_dic(self.predictions["Figures"],output_figures_folder=self.output_figures_folder)
            print(header_string,"\nAll predictions have been deleted and the 'predictions' attribute has been initialized. No figure file has been deleted.\n")
        self.predictions = {"Figures": figs}

    def close_opened_dataset(self,verbose=None):
        """ 
        Closes the opened h5py datasets 
        :attr:`Data.opened_dataset <NF4HEP.Data.opened_dataset>`.

        - **Arguments**

            - **verbose**
            
                See :argument:`verbose <common_methods_arguments.verbose>`.
        """
        verbose, _ = self.set_verbosity(verbose)
        try:
            self.opened_dataset.close()
            del(self.opened_dataset)
            print(header_string,"\nClosed", self.input_file,".\n",show=verbose)
        except:
            print(header_string,"\nNo dataset to close.\n", show=verbose)

    def save_log(self, overwrite=False, verbose=None):
        """
        Saves the content of the :attr:`Data.log <NF4HEP.Data.log>` attribute in the file
        :attr:`Data.output_log_file <NF4HEP.Data.output_log_file>`

        This method is called by the methods
        
        - :meth:`Data.__init__ <NF4HEP.Data.__init__>` with ``overwrite=True`` and ``verbose=verbose_sub``  with ``overwrite=True`` and ``verbose=verbose_sub`` if :attr:`Data.input_file <NF4HEP.Data.input_file>` is not ``None``, and with ``overwrite=True`` and ``verbose=verbose_sub`` otherwise
        - :meth:`Data.save <NF4HEP.Data.save>` with ``overwrite=overwrite`` and ``verbose=verbose``
        - :meth:`Data.generate_train_indices <NF4HEP.Data.generate_train_indices>` with ``overwrite=True`` and ``verbose=verbose_sub``
        - :meth:`Data.generate_train_data <NF4HEP.Data.generate_train_data>` with ``overwrite=True`` and ``verbose=verbose_sub``
        - :meth:`Data.update_train_indices <NF4HEP.Data.update_train_indices>` with ``overwrite=True`` and ``verbose=verbose_sub``
        - :meth:`Data.update_train_data <NF4HEP.Data.update_train_data>` with ``overwrite=True`` and ``verbose=verbose_sub``
        - :meth:`Data.generate_test_indices <NF4HEP.Data.generate_test_indices>` with ``overwrite=True`` and ``verbose=verbose_sub``
        - :meth:`Data.generate_test_data <NF4HEP.Data.generate_test_data>` with ``overwrite=True`` and ``verbose=verbose_sub``
        - :meth:`Data.compute_sample_weights <NF4HEP.Data.compute_sample_weights>` with ``overwrite=True`` and ``verbose=verbose_sub``
        - :meth:`Data.define_scaler <NF4HEP.Data.define_scaler>` with ``overwrite=True`` and ``verbose=verbose_sub``

        This method is called by the
        :meth:`Data.save <NF4HEP.Data.save>` method to save the entire object.

        - **Arguments**

            - **overwrite**
            
                See :argument:`overwrite <common_methods_arguments.overwrite>`.

            - **verbose**
            
                See :argument:`verbose <common_methods_arguments.verbose>`.

        - **Creates/updates file**

            - :attr:`Data.output_log_file <NF4HEP.Data.output_log_file>`
        """
        verbose, verbose_sub = self.set_verbosity(verbose)
        start = timer()
        if not overwrite:
            utils.check_rename_file(self.output_log_file, verbose=verbose_sub)
        dictionary = utils.convert_types_dict(self.log)
        with codecs.open(self.output_log_file, "w", encoding="utf-8") as f:
            json.dump(dictionary, f, separators=(",", ":"), indent=4)
        end = timer()
        if overwrite:
            print(header_string,"\nData log file\n\t", self.output_log_file,"\nupdated (or saved if it did not exist) in", str(end-start), "s.\n",show=verbose)
        else:
            print(header_string,"\nData log file\n\t", self.output_log_file, "\nsaved in", str(end-start), "s.\n", show=verbose)

    def save_predictions_json(self, timestamp=None, overwrite=False, verbose=None):
        """ Save predictions json
        """
        verbose, verbose_sub = self.set_verbosity(verbose)
        start = timer()
        if timestamp is None:
            timestamp = "datetime_"+datetime.now().strftime("%Y_%m_%d_%H_%M_%S_%fZ")[:-3]
        if type(overwrite) == bool:
            output_predictions_json_file = self.output_predictions_json_file
            if not overwrite:
                utils.check_rename_file(output_predictions_json_file, verbose=verbose_sub)
        elif overwrite == "dump":
            output_predictions_json_file = utils.generate_dump_file_name(self.output_predictions_json_file, timestamp=timestamp)
        dictionary = utils.convert_types_dict(self.predictions)
        with codecs.open(output_predictions_json_file, "w", encoding="utf-8") as f:
            json.dump(dictionary, f, separators=(",", ":"), indent=4)
        end = timer()
        self.log[timestamp] = {"action": "saved predictions json",
                               "file name": path.split(output_predictions_json_file)[-1]}
        #self.save_log(overwrite=True, verbose=verbose_sub) # log saved by save
        if type(overwrite) == bool:
            if overwrite:
                print(header_string,"\nPredictions json file\n\t", output_predictions_json_file, "\nupdated (or saved if it did not exist) in", str(end-start), "s.\n", show=verbose)
            else:
                print(header_string,"\nPredictions json file\n\t", output_predictions_json_file, "\nsaved in", str(end-start), "s.\n", show=verbose)
        elif overwrite == "dump":
            print(header_string,"\nPredictions json file dump\n\t", output_predictions_json_file, "\nsaved in", str(end-start), "s.\n", show=verbose)

    def save_object_h5(self, overwrite=False, verbose=True):
        """
        The :class:`Lik <NF4HEP.Data>` object is saved to the HDF5 file
        :attr:`Data.output_h5_file <NF4HEP.Data.output_h5_file>`, the dataset corresponding to the 
        :attr:`Data.data_X <NF4HEP.Data.data_X>` attribute
        is saved to the file :attr:`Data.output_samples_h5_file <NF4HEP.Data.output_samples_h5_file>`, and the object 
        log is saved to the json file :attr:`Lik.output_log_file <NF4HEP.Lik.output_log_file>`.
        
        This method saves the content of the :attr:``Lik.__dict__ <NF4HEP.Lik.__dict__>`` 
        attribute in the h5 file :attr:`Data.output_h5_file <NF4HEP.Data.output_h5_file>`
        using the |deepdish_link| package. The following attributes are excluded from the saved
        dictionary:

            - :attr:`Data.data_dictionary <NF4HEP.Data.data_dictionary>`
            - :attr:`Data.data_X <NF4HEP.Data.data_X>` (saved to the file :attr:`Data.output_samples_h5_file <NF4HEP.Data.output_samples_h5_file>`)
            - :attr:`Data.dtype_required <NF4HEP.Data.dtype_required>`
            - :attr:`Data.input_file <NF4HEP.Data.input_file>`
            - :attr:`Data.input_folder <NF4HEP.Data.input_folder>`
            - :attr:`Data.input_samples_h5_file <NF4HEP.Data.input_samples_h5_file>`
            - :attr:`Data.input_h5_file <NF4HEP.Data.input_h5_file>`
            - :attr:`Data.input_log_file <NF4HEP.Data.input_log_file>`
            - :attr:`Data.load_on_RAM <NF4HEP.Data.load_on_RAM>`
            - :attr:`Data.log <NF4HEP.Data.log>` (saved to the file :attr:`Data.output_log_file <NF4HEP.Data.output_log_file>`)
            - :attr:`Data.opened_dataset <NF4HEP.Data.opened_dataset>`
            - :attr:`Data.output_folder <NF4HEP.Data.output_folder>`
            - :attr:`Data.output_figures_folder <NF4HEP.Data.output_figures_folder>`
            - :attr:`Data.output_figures_base_file_name <NF4HEP.Data.output_figures_base_file_name>`
            - :attr:`Data.output_figures_base_file_path <NF4HEP.Data.output_figures_base_file_path>`
            - :attr:`Data.output_h5_file <NF4HEP.Data.output_h5_file>`
            - :attr:`Data.output_samples_h5_file <NF4HEP.Data.output_samples_h5_file>`
            - :attr:`Data.output_log_file <NF4HEP.Data.output_log_file>`
            - :attr:`Data.test_range <NF4HEP.Data.test_range>`
            - :attr:`Data.train_range <NF4HEP.Data.train_range>`
            - :attr:`Data.verbose <NF4HEP.Data.verbose>`

        This method is called by the
        :meth:`Data.save <NF4HEP.Data.save>` method to save the entire object.

        - **Arguments**

            - **overwrite**
            
                See :argument:`overwrite <common_methods_arguments.overwrite>`.

            - **verbose**
            
                See :argument:`verbose <common_methods_arguments.verbose>`.

        - **Produces file**

            - :attr:`Data.output_h5_file <NF4HEP.Data.output_h5_file>`
        """
        verbose, verbose_sub = self.set_verbosity(verbose)
        start = timer()
        if not overwrite:
            utils.check_rename_file(self.output_h5_file, verbose=verbose_sub)
        dictionary = utils.dic_minus_keys(self.__dict__, ["data_dictionary",
                                                          "data_X",
                                                          "dtype_required",
                                                          "input_file",
                                                          "input_folder",
                                                          "input_samples_h5_file",
                                                          "input_h5_file",
                                                          "input_log_file",
                                                          "load_on_RAM",
                                                          "log",
                                                          "opened_dataset",
                                                          "output_folder",
                                                          "output_figures_folder",
                                                          "output_figures_base_file_name",
                                                          "output_figures_base_file_path",
                                                          "output_h5_file",
                                                          "output_samples_h5_file",
                                                          "output_log_file",
                                                          "test_range",
                                                          "train_range",
                                                          "verbose"])
        dd.io.save(self.output_h5_file, dictionary)
        end = timer()
        timestamp = "datetime_"+datetime.now().strftime("%Y_%m_%d_%H_%M_%S_%fZ")[:-3]
        self.log[timestamp] = {"action": "saved",
                               "file name": path.split(self.output_h5_file)[-1]}
        print(header_string,"\nData object saved to file\n\t", self.output_h5_file,"in", str(end-start), "s.\n",show=verbose)

    def save_samples_h5(self, overwrite=False, verbose=None):
        """
        The :class:`Lik <NF4HEP.Data>` object is saved to the HDF5 file
        :attr:`Data.output_h5_file <NF4HEP.Data.output_h5_file>`, the dataset corresponding to the 
        :attr:`Data.data_X <NF4HEP.Data.data_X>` attribute
        is saved to the file :attr:`Data.output_samples_h5_file <NF4HEP.Data.output_samples_h5_file>`, and the object 
        log is saved to the json file :attr:`Lik.output_log_file <NF4HEP.Lik.output_log_file>`.
        
        This method saves the content of the :attr:`Data.data_X <NF4HEP.Data.data_X>` attribute, 
        and an additional entry with its shape in the h5 file 
        :attr:`Data.output_samples_h5_file <NF4HEP.Data.output_samples_h5_file>`
        using the |h5py_link| package.
        
        This method is called by the :meth:`Data.save <NF4HEP.Data.save>` method to save the entire object.

        - **Arguments**

            - **overwrite**
            
                See :argument:`overwrite <common_methods_arguments.overwrite>`.

            - **verbose**
            
                See :argument:`verbose <common_methods_arguments.verbose>`.

        - **Produces file**

            - :attr:`Data.output_samples_h5_file <NF4HEP.Data.output_samples_h5_file>`
        """
        verbose, verbose_sub = self.set_verbosity(verbose)
        start = timer()
        if not overwrite:
            utils.check_rename_file(self.output_samples_h5_file,verbose=verbose_sub)
        start = timer()     
        h5_out = h5py.File(self.output_samples_h5_file, "w")
        data = h5_out.create_group("data")
        data["shape"] = np.shape(self.data_X)
        data["X"] = self.data_X.astype(self.dtype_stored)
        h5_out.close()
        end = timer()
        timestamp = "datetime_"+datetime.now().strftime("%Y_%m_%d_%H_%M_%S_%fZ")[:-3]
        self.log[timestamp] = {"action": "saved",
                               "file name": path.split(self.output_samples_h5_file)[-1]}
        print(header_string,"\n"+str(self.npoints), "data samples (data_X) saved to file\n\t", self.output_samples_h5_file,"\nin", end-start, "s.\n",show=verbose)

    def save(self, overwrite=False, verbose=None):
        """
        The :class:`Lik <NF4HEP.Data>` object is saved to the HDF5 file
        :attr:`Data.output_h5_file <NF4HEP.Data.output_h5_file>`, the dataset corresponding to the 
        :attr:`Data.data_X <NF4HEP.Data.data_X>` attribute
        is saved to the file :attr:`Data.output_samples_h5_file <NF4HEP.Data.output_samples_h5_file>`, and the object 
        log is saved to the json file :attr:`Lik.output_log_file <NF4HEP.Lik.output_log_file>`.

        This method calls in order the three methods
        
            - :meth:`Data.save_log <NF4HEP.Data.save_json>`
            - :meth:`Data.save_predictions_json <NF4HEP.Data.save_predictions_json>`
            - :meth:`Data.save_object_h5 <NF4HEP.Data.save_object_h5>`
            - :meth:`Data.save_samples_h5 <NF4HEP.Data.save_samples_h5>`

        to save the full object.

        - **Arguments**
            
            Same arguments as the called methods.

        - **Produces files**

            - :attr:`Data.output_log_file <NF4HEP.Data.output_log_file>`
            - :attr:`Data.output_predictions_json_file <NF4HEP.Data.output_predictions_json_file>`
            - :attr:`Data.output_h5_file <NF4HEP.Data.output_h5_file>`
            - :attr:`Data.output_samples_h5_file <NF4HEP.Data.output_samples_h5_file>` 
        """
        verbose, _ = self.set_verbosity(verbose)
        self.save_object_h5(overwrite=overwrite, verbose=verbose)
        self.save_samples_h5(overwrite=overwrite, verbose=verbose)
        self.save_predictions_json(overwrite=overwrite, verbose=verbose)
        self.save_log(overwrite=overwrite, verbose=verbose)
        
    def generate_train_indices(self, npoints_train, npoints_val, seed, verbose=None):
        """
        Method used to generate ``npoints_train`` train and ``npoints_val`` validation data indices 
        (integer positions in the 
        :attr:`Data.data_X <NF4HEP.Data.data_X>` array). It updates the 
        ``"idx_train"`` and ``"idx_val"`` items of the 
        :attr:`Data.data_dictionary <NF4HEP.Data.data_dictionary>` dictionary.

        - **Arguments**

            - **npoints_train**

                Required number of 
                training points.

                    - **type**: ``int``
            
            - **npoints_val**

                Required number of 
                validation points.

                    - **type**: ``int``

            - **seed**

                Seed of the random number generator. It is used to initialize 
                the |numpy_link| random state.

                    - **type**: ``int``

            - **verbose**
            
                See :argument:`verbose <common_methods_arguments.verbose>`.
        """
        _, verbose_sub = self.set_verbosity(verbose)
        np.random.seed(seed)
        idx_train = np.random.choice(self.train_range, npoints_train+npoints_val, replace=False)
        idx_train, idx_val = train_test_split(idx_train, train_size=npoints_train, test_size=npoints_val, shuffle=False)
        self.data_dictionary["idx_train"] = np.sort(idx_train)
        self.data_dictionary["idx_val"] = np.sort(idx_val)
        timestamp = "datetime_"+datetime.now().strftime("%Y_%m_%d_%H_%M_%S_%fZ")[:-3]
        self.log[timestamp] = {"action": "updated data dictionary",
                               "data": ["idx_train", "idx_val"],
                               "npoints train": npoints_train,
                               "npoints val": npoints_val}
        #self.save_log(overwrite=True, verbose=verbose_sub) # log is saved by generate_train_data

    def generate_train_data(self, npoints_train, npoints_val, seed, verbose=None):
        """
        Method used to generate ``npoints_train`` train and ``npoints_val`` validation data.
        In order to extract the required number of points from the :attr:`Data.data_X <NF4HEP.Data.data_X>`
        array (which may not be available on RAM),
        it first generates indices with the :meth:`Data.generate_train_indices <NF4HEP.Data.generate_train_indices>`
        method and then slices the data with the generated indices.
        It updates the ``"X_train"`` and ``"X_val"`` items of the
        :attr:`Data.data_dictionary <NF4HEP.Data.data_dictionary>` dictionary.

        - **Arguments**

            - **npoints_train**

                Required number of 
                training points.

                    - **type**: ``int``
            
            - **npoints_val**

                Required number of 
                validation points.

                    - **type**: ``int``

            - **seed**

                Seed of the random number generator. It is used to initialize 
                the |numpy_link| random state.

                    - **type**: ``int``

            - **verbose**
            
                See :argument:`verbose <common_methods_arguments.verbose>`.
        """
        verbose, verbose_sub = self.set_verbosity(verbose)
        start = timer()
        self.__check_sync_data_dictionary(verbose=verbose_sub)
        self.generate_train_indices(npoints_train, npoints_val, seed)
        self.data_dictionary["X_train"] = self.data_X[self.data_dictionary["idx_train"]].astype(self.dtype_required)
        self.data_dictionary["X_val"] = self.data_X[self.data_dictionary["idx_val"]].astype(self.dtype_required)
        end = timer()
        timestamp = "datetime_"+datetime.now().strftime("%Y_%m_%d_%H_%M_%S_%fZ")[:-3]
        self.log[timestamp] = {"action": "updated data dictionary",
                               "data": ["X_train", "X_val"],
                               "npoints train": npoints_train,
                               "npoints val": npoints_val}
        self.save_log(overwrite=True, verbose=verbose_sub)
        print(header_string,"\nGenerated", str(npoints_train), "X_train samples and ", str(npoints_val),"X_val samples in", end-start,"s.\n",show=verbose)

    def update_train_indices(self, npoints_train, npoints_val, seed, verbose=None):
        """
        Analog of :meth:`Data.generate_train_indices <NF4HEP.Data.generate_train_indices>`, but it only updates
        indices adding more points if any are already available in the 
        :attr:`Data.data_dictionary <NF4HEP.Data.data_dictionary>` dictionary.
        It updates the  ``"idx_train"`` and ``"idx_val"`` items of the 
        :attr:`Data.data_dictionary <NF4HEP.Data.data_dictionary>` dictionary.

        - **Arguments**

            - **npoints_train**

                Required number of 
                training points.

                    - **type**: ``int``
            
            - **npoints_val**

                Required number of 
                validation points.

                    - **type**: ``int``

            - **seed**

                Seed of the random number generator. It is used to initialize 
                the |numpy_link| random state.

                    - **type**: ``int``

            - **verbose**
            
                See :argument:`verbose <common_methods_arguments.verbose>`.
        """
        _, verbose_sub = self.set_verbosity(verbose)
        existing_train = np.sort(np.concatenate((self.data_dictionary["idx_train"], self.data_dictionary["idx_val"])))
        np.random.seed(seed)
        idx_train = np.random.choice(np.setdiff1d(np.array(self.train_range), existing_train), npoints_train+npoints_val, replace=False)
        if np.size(idx_train) != 0:
            idx_train, idx_val = [np.sort(idx) for idx in train_test_split(idx_train, train_size=npoints_train, test_size=npoints_val, shuffle=False)]
        else:
            idx_val = idx_train
        self.data_dictionary["idx_train"] = np.sort(np.concatenate((self.data_dictionary["idx_train"],idx_train)))
        self.data_dictionary["idx_val"] = np.sort(np.concatenate((self.data_dictionary["idx_val"],idx_val)))
        timestamp = "datetime_"+datetime.now().strftime("%Y_%m_%d_%H_%M_%S_%fZ")[:-3]
        self.log[timestamp] = {"action": "updated data dictionary",
                               "data": ["idx_train", "idx_val"],
                               "npoints train": npoints_train,
                               "npoints val": npoints_val}
        #self.save_log(overwrite=True, verbose=verbose_sub) # log is saved by update_train_data
        return [idx_train, idx_val]

    def update_train_data(self, npoints_train, npoints_val, seed, verbose=None):
        """
        Analog of :meth:`Data.generate_train_data <NF4HEP.Data.generate_train_data>`, but it only updates
        data adding more points if any are already available in the 
        :attr:`Data.data_dictionary <NF4HEP.Data.data_dictionary>` dictionary.
        It updates the  ``"X_train"`` and ``"X_val"`` items of the 
        :attr:`Data.data_dictionary <NF4HEP.Data.data_dictionary>` dictionary.

        - **Arguments**

            - **npoints_train**

                Required number of 
                training points.

                    - **type**: ``int``
            
            - **npoints_val**

                Required number of 
                validation points.

                    - **type**: ``int``

            - **seed**

                Seed of the random number generator. It is used to initialize 
                the |numpy_link| random state.

                    - **type**: ``int``

            - **verbose**
            
                See :argument:`verbose <common_methods_arguments.verbose>`.
        """
        verbose, verbose_sub = self.set_verbosity(verbose)
        start = timer()
        self.__check_sync_data_dictionary(verbose=verbose_sub)
        [npoints_train, npoints_val] = [(i > 0) * i for i in [npoints_train-len(self.data_dictionary["idx_train"]), 
                                                                            npoints_val-len(self.data_dictionary["idx_val"])]]
        idx_train, idx_val = self.update_train_indices(npoints_train, npoints_val, seed, verbose=verbose)
        if idx_train != []:
            if self.data_dictionary["X_train"].size == 0:
                self.data_dictionary["X_train"] = self.data_X[idx_train].astype(self.dtype_required)
            else:
                self.data_dictionary["X_train"] = np.concatenate((self.data_dictionary["X_train"],self.data_X[idx_train])).astype(self.dtype_required)
        if idx_val != []:
            if self.data_dictionary["X_val"].size == 0:
                self.data_dictionary["X_val"] = self.data_X[idx_val].astype(self.dtype_required)
            else:
                self.data_dictionary["X_val"] = np.concatenate((self.data_dictionary["X_val"], self.data_X[idx_val])).astype(self.dtype_required)
        end = timer()
        timestamp = "datetime_"+datetime.now().strftime("%Y_%m_%d_%H_%M_%S_%fZ")[:-3]
        self.log[timestamp] = {"action": "updated data dictionary",
                               "data": ["X_train", "X_val"],
                               "npoints train": npoints_train,
                               "npoints val": npoints_val}
        self.save_log(overwrite=True, verbose=verbose_sub)
        print(header_string,"\nAdded", str(npoints_train), "X_train samples and", str(npoints_val),"X_val samples in", end-start,"s.\n",show=verbose)

    def generate_test_indices(self, npoints_test, seed, verbose=None):
        """
        Method used to generate ``npoints_test`` test data indices 
        (integer positions in the :attr:`Data.data_X <NF4HEP.Data.data_X>` array). It updates the 
        ``"idx_test"`` item of the 
        :attr:`Data.data_dictionary <NF4HEP.Data.data_dictionary>` dictionary.

        - **Arguments**

            - **npoints_test**

                Required number of 
                test points.

                    - **type**: ``int``
            
            - **seed**

                Seed of the random number generator. It is used to initialize 
                the |numpy_link| random state.

                    - **type**: ``int``

            - **verbose**
            
                See :argument:`verbose <common_methods_arguments.verbose>`.
        """
        _, verbose_sub = self.set_verbosity(verbose)
        np.random.seed(seed)
        idx_test = np.random.choice(self.test_range, npoints_test, replace=False)
        self.data_dictionary["idx_test"] = np.sort(idx_test)
        timestamp = "datetime_"+datetime.now().strftime("%Y_%m_%d_%H_%M_%S_%fZ")[:-3]
        self.log[timestamp] = {"action": "updated data dictionary",
                               "data": ["idx_test"],
                               "npoints test": npoints_test}
        #self.save_log(overwrite=True, verbose=verbose_sub) # log is saved by generate_test_data

    def generate_test_data(self, npoints_test, seed, verbose=None):
        """
        Method used to generate ``npoints_test`` test data.
        In order to extract the required number of points from the :attr:`Data.data_X <NF4HEP.Data.data_X>`
        array (which may not be available on RAM),
        it first generates indices with the :meth:`Data.generate_test_indices <NF4HEP.Data.generate_test_indices>`
        method and then slices the data with the generated indices.
        It updates the ``"X_test"`` item of the
        :attr:`Data.data_dictionary <NF4HEP.Data.data_dictionary>` dictionary.

        - **Arguments**

            - **npoints_test**

                Required number of
                test points.

                    - **type**: ``int``

            - **seed**

                Seed of the random number generator. It is used to initialize 
                the |numpy_link| random state.

                    - **type**: ``int``

            - **verbose**
            
                See :argument:`verbose <common_methods_arguments.verbose>`.
        """
        verbose, verbose_sub = self.set_verbosity(verbose)
        start = timer()
        self.__check_sync_data_dictionary(verbose=verbose_sub)
        self.generate_test_indices(npoints_test, seed)
        self.data_dictionary["X_test"] = self.data_X[self.data_dictionary["idx_test"]].astype(self.dtype_required)
        end = timer()
        timestamp = "datetime_"+datetime.now().strftime("%Y_%m_%d_%H_%M_%S_%fZ")[:-3]
        self.log[timestamp] = {"action": "updated data dictionary",
                               "data": ["X_test"],
                               "npoints test": npoints_test}
        self.save_log(overwrite=True, verbose=verbose_sub)
        print(header_string,"\nGenerated", str(npoints_test), "X_test samples in", end-start,"s.\n",show=verbose)

    def update_test_indices(self, npoints_test, seed, verbose=None):
        """
        Analog of :meth:`Data.generate_test_indices <NF4HEP.Data.generate_test_indices>`, but it only updates
        indices adding more points if any are already available in the 
        :attr:`Data.data_dictionary <NF4HEP.Data.data_dictionary>` dictionary.
        It updates the  ``"idx_test"`` item of the 
        :attr:`Data.data_dictionary <NF4HEP.Data.data_dictionary>` dictionary.

        - **Arguments**

            - **npoints_test**

                Required number of 
                test points.

                    - **type**: ``int``

            - **seed**

                Seed of the random number generator. It is used to initialize 
                the |numpy_link| random state.

                    - **type**: ``int``

            - **verbose**
            
                See :argument:`verbose <common_methods_arguments.verbose>`.
        """
        _, verbose_sub = self.set_verbosity(verbose)
        existing_test = self.data_dictionary["idx_test"]
        np.random.seed(seed)
        idx_test = np.sort(np.random.choice(np.setdiff1d(np.array(self.test_range), existing_test), npoints_test, replace=False))
        self.data_dictionary["idx_test"] = np.sort(np.concatenate((self.data_dictionary["idx_test"],idx_test)))
        timestamp = "datetime_"+datetime.now().strftime("%Y_%m_%d_%H_%M_%S_%fZ")[:-3]
        self.log[timestamp] = {"action": "updated data dictionary",
                               "data": ["idx_test"],
                               "npoints test": npoints_test}
        #self.save_log(overwrite=True, verbose=verbose_sub) # log is saved by update_test_data
        return idx_test

    def update_test_data(self, npoints_test, seed, verbose=None):
        """
        Analog of :meth:`Data.generate_test_data <NF4HEP.Data.generate_test_data>`, but it only updates
        data adding more points if any are already available in the 
        :attr:`Data.data_dictionary <NF4HEP.Data.data_dictionary>` dictionary.
        It updates the  ``"X_test"`` item of the 
        :attr:`Data.data_dictionary <NF4HEP.Data.data_dictionary>` dictionary.

        - **Arguments**

            - **npoints_test**

                Required number of 
                test points.

                    - **type**: ``int``
            
            - **seed**

                Seed of the random number generator. It is used to initialize 
                the |numpy_link| random state.

                    - **type**: ``int``

            - **verbose**
            
                See :argument:`verbose <common_methods_arguments.verbose>`.
        """
        verbose, verbose_sub = self.set_verbosity(verbose)
        start = timer()
        self.__check_sync_data_dictionary(verbose=verbose_sub)
        npoints_test = npoints_test-len(self.data_dictionary["idx_test"])
        idx_test = self.update_test_indices(npoints_test, seed, verbose=verbose)
        if idx_test != []:
            if self.data_dictionary["X_test"].size == 0:
                self.data_dictionary["X_test"] = self.data_X[idx_test].astype(self.dtype_required)
            else:
                self.data_dictionary["X_test"] = np.concatenate((self.data_dictionary["X_test"],self.data_X[idx_test])).astype(self.dtype_required)
        end = timer()
        timestamp = "datetime_"+datetime.now().strftime("%Y_%m_%d_%H_%M_%S_%fZ")[:-3]
        self.log[timestamp] = {"action": "updated data dictionary",
                               "data": ["X_test"],
                               "npoints test": npoints_test}
        self.save_log(overwrite=True, verbose=verbose_sub)
        print(header_string,"\nAdded", str(npoints_test), "X_test samples in", end-start,"s.\n",show=verbose)

    def compute_sample_weights(self, sample, nbins=100, power=1, verbose=None):
        """
        Method that computes weights of points given their distribution. Sample weights are used to weigth data as a function of 
        their frequency, obtained by binning the data in ``nbins`` and assigning weight equal to the inverse of the bin count
        to the power ``power``. In order to avoid too large weights for bins with very few counts, all bins with less than 5 counts
        are assigned frequency equal to 1/5 to the power ``power``.
        
        - **Arguments**

            - **sample**

                Distribution of points that 
                need to be weighted.

                    - **type**: ``numpy.ndarray``
                    - **shape**: ``(len(sample),)``

            - **nbins**

                Number of bins to histogram the 
                sample data

                    - **type**: ``int``
                    - **default**: ``100``

            - **power**

                Exponent of the inverse of the bin count used
                to assign weights.

                    - **type**: ``float``
                    - **default**: ``1``

            - **verbose**
            
                See :argument:`verbose <common_methods_arguments.verbose>`.

        - **Returns**

            |Numpy_link| array of the same length of ``sample`` containing
            the required weights.
            
                - **type**: ``numpy.ndarray``
                - **shape**: ``(len(sample),)``
        """
        verbose, verbose_sub = self.set_verbosity(verbose)
        start = timer()
        hist, edges = np.histogram(sample, bins=nbins)
        hist = np.where(hist < 5, 5, hist)
        tmp = np.digitize(sample, edges, right=True)
        W = 1/np.power(hist[np.where(tmp == nbins, nbins-1, tmp)], power)
        W = W/np.sum(W)*len(sample)
        end = timer()
        timestamp = "datetime_"+datetime.now().strftime("%Y_%m_%d_%H_%M_%S_%fZ")[:-3]
        self.log[timestamp] = {"action": "computed sample weights"}
        self.save_log(overwrite=True, verbose=verbose_sub)
        print(header_string,"\nSample weights computed in", end-start, "s.\n",show=verbose)
        return W

    def define_scaler(self, data_X, scalerX_bool, verbose=None):
        """
        Method that defines |standard_scalers_link| fit to the ``data_X`` data.
        The scalers is defined based on the boolean flag ``scalerX_bool``. When the flag is ``False``
        the scaler is defined with the arguemtns ``with_mean=False`` and ``with_std=False`` which correspond to identity
        transformations.

        - **Arguments**

            - **data_X**

                ``X`` data to fit
                the |standard_scaler_link| ``scalerX``.

                    - **type**: ``numpy.ndarray``
                    - **shape**: ``(npoints,ndim)``

            - **scalerX_bool**

                If ``True`` the ``X`` scaler is fit to the ``data_X`` data, otherwise it is set
                to the identity transformation.

                    - **type**: ``bool``

            - **verbose**
            
                See :argument:`verbose <common_methods_arguments.verbose>`.

        - **Returns**

            ``scalerX`` containing 
            the ``X`` scaler object.
            
                - **type**: |standard_scalers_link|
        """
        _, verbose_sub = self.set_verbosity(verbose)
        start = timer()
        if scalerX_bool:
            scalerX = StandardScaler(with_mean=True, with_std=True)
            scalerX.fit(data_X)
        else:
            scalerX = StandardScaler(with_mean=False, with_std=False)
            scalerX.fit(data_X)
        end = timer()
        timestamp = "datetime_"+datetime.now().strftime("%Y_%m_%d_%H_%M_%S_%fZ")[:-3]
        self.log[timestamp] = {"action": "defined scaler",
                               "scaler X": scalerX_bool}
        self.save_log(overwrite=True, verbose=verbose_sub)
        print(header_string,"\nStandard scaler defined in", end-start, "s.\n", show=verbose)
        return scalerX

    def define_rotation(self, data_X, rotationX_bool, verbose=None):
        """
        Method that defines the rotation matrix that diagonalizes the covariance matrix of the ``data_X``,
        making them uncorrelated.
        Such matrix is defined based on the boolean flag ``rotationX_bool``. When the flag is ``False``
        the matrix is set to the identity matrix.
        
        Note: Data are transformed with the matrix ``V`` through ``np.dot(X,V)`` and transformed back throug
        ``np.dot(X_diag,np.transpose(V))``.
        
        - **Arguments**

            - **data_X**

                ``X`` data to compute the 
                rotation matrix that diagonalizes the covariance matrix.

                    - **type**: ``numpy.ndarray``
                    - **shape**: ``(npoints,ndim)``

            - **rotationX_bool**

                If ``True`` the rotation matrix is set to the identity matrix.

                    - **type**: ``bool``

            - **verbose**
            
                See :argument:`verbose <common_methods_arguments.verbose>`.

        - **Returns**

            Matrix in the form of a 2D |Numpy_link| 
            array.
            
                - **type**: ``numpy.ndarray``
                - **shape**: ``(ndim,ndim)``
        """
        _, verbose_sub = self.set_verbosity(verbose)
        start = timer()
        if rotationX_bool:
            cov_matrix = np.cov(data_X, rowvar=False)
            w, V = np.linalg.eig(cov_matrix)
        else:
            V = np.identity(self.ndims)
        end = timer()
        timestamp = "datetime_" + \
            datetime.now().strftime("%Y_%m_%d_%H_%M_%S_%fZ")[:-3]
        self.log[timestamp] = {"action": "defined covariance rotation matrix",
                               "rotation X": rotationX_bool}
        self.save_log(overwrite=True, verbose=verbose_sub)
        print(header_string, "\nMatrix that rotates the correlation matrix defined in",end-start, "s.\n", show=verbose)
        return V

    def update_figures(self,figure_file=None,timestamp=None,overwrite=False,verbose=None):
        """
        Method that generates new file names and renames old figure files when new ones are produced with the argument ``overwrite=False``. 
        When ``overwrite=False`` it calls the :func:`utils.check_rename_file <NF4HEP.utils.check_rename_file>` function and, if 
        ``figure_file`` already existed in the :attr:`Data.predictions <NF4HEP.Data.predictions>` dictionary, then it
        updates the dictionary by appennding to the old figure name the timestamp corresponding to its generation timestamp 
        (that is the key of the :attr:`Data.predictions["Figures"] <NF4HEP.Data.predictions>` dictionary).
        When ``overwrite="dump"`` it calls the :func:`utils.generate_dump_file_name <NF4HEP.utils.generate_dump_file_name>` function
        to generate the dump file name.
        It returns the new figure_file.

        - **Arguments**

            - **figure_file**

                Figure file path. If the figure already exists in the 
                :meth:`Data.predictions <NF4HEP.Data.predictions>` dictionary, then its name is updated with the corresponding timestamp.

            - **overwrite**

                The method updates file names and :attr:`Data.predictions <NF4HEP.Data.predictions>` dictionary only if
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
        print("Checking and updating figures dictionary",show=verbose)
        if figure_file is None:
            raise Exception("figure_file input argument of update_figures method needs to be specified while it is None.")
        else:
            new_figure_file = figure_file
            if type(overwrite) == bool:
                if not overwrite:
                    # search figure
                    timestamp=None
                    for k, v in self.predictions["Figures"].items():
                        if figure_file in v:
                            timestamp = k
                    old_figure_file = utils.check_rename_file(path.join(self.output_figures_folder,figure_file),timestamp=timestamp,return_value="file_name",verbose=verbose_sub)
                    if timestamp is not None:
                        self.predictions["Figures"][timestamp] = [f.replace(figure_file,old_figure_file) for f in v] 
            elif overwrite == "dump":
                new_figure_file = utils.generate_dump_file_name(figure_file, timestamp=timestamp)
        return new_figure_file

    def data_description(self,
                         X=None,
                         pars_labels="original",
                         timestamp=None,
                         overwrite=False,
                         verbose=None):
        """
        Gives a description of data by calling the |Pandas_dataframe_describe|
        method.

        - **Arguments**

            - **X**

                X data to use for the plot. If ``None`` is given the 
                :attr:`Data.data_X <NF4HEP.Data.data_X>` dataset is used.

                    - **type**: ``list`` or ``numpy.ndarray``
                    - **shape**: ``(npoints,ndims)``
                    - **default**: ``None``

            - **pars_labels**

                    Argument that is passed to the :meth:`Data.__set_pars_labels < NF4HEP.Data._Lik__set_pars_labels>`
                    method to set the parameters labels to be used in the plots.

                        - **type**: ``list`` or ``str``
                        - **shape of list**: ``[]``
                        - **accepted strings**: ``"original"``, ``"generic"``
                        - **default**: ``original``

            - **timestamp**
            
                See :argument:`timestamp <common_methods_arguments.timestamp>`.

            - **overwrite**
            
                See :argument:`overwrite <common_methods_arguments.overwrite>`.

            - **verbose**
            
                See :argument:`verbose <common_methods_arguments.verbose>`.

        - **Updates file**

            - :attr:`Data.output_log_file <NF4HEP.Data.output_log_file>`
        """
        verbose, verbose_sub = self.set_verbosity(verbose)
        print(header_string,"\nGenerating data description", show=verbose)
        if timestamp is None:
            timestamp = "datetime_"+datetime.now().strftime("%Y_%m_%d_%H_%M_%S_%fZ")[:-3]
        start = timer()
        if X is None:
            X = self.data_X
        else:
            X = np.array(X)
        pars_labels = self.__set_pars_labels(pars_labels)
        df = pd.DataFrame(X,columns=pars_labels)
        df_description = pd.DataFrame(df.describe())
        end = timer()
        print("\n"+header_string+"\nData description generated in", str(end-start), "s.\n", show=verbose)
        return df_description

    def plot_X_distributions_summary(self,
                                     X=None,
                                     max_points=None,
                                     nbins=50,
                                     pars_labels="original", 
                                     color="green",
                                     figure_file_name=None,
                                     show_plot=False,
                                     timestamp=None,
                                     overwrite=False,
                                     verbose=None,
                                     **step_kwargs):
        """
        Plots a summary of all 1D distributions of the parameters in 
        the :attr:`Data.data_X <NF4HEP.Data.data_X>` dataset.

        - **Arguments**

            - **X**

                X data to use for the plot. If ``None`` is given the 
                :attr:`Data.data_X <NF4HEP.Data.data_X>` dataset is used.

                    - **type**: ``list`` or ``numpy.ndarray``
                    - **shape**: ``(npoints,ndims)``
                    - **default**: ``None``
            
            - **max_points**

                Maximum number of points used to make
                the plot. If the numnber is smaller than the total
                number of available points, then a random subset is taken.
                If ``None`` then all available points are used.

                    - **type**: ``int`` or ``None``
                    - **default**: ``None``

            - **nbins**

                Number of bins used to make 
                the histograms.

                    - **type**: ``int``
                    - **default**: ``50``

            - **pars_labels**

                Argument that is passed to the :meth:`Data.__set_pars_labels < NF4HEP.Data._Lik__set_pars_labels>`
                method to set the parameters labels to be used in the plots.

                    - **type**: ``list`` or ``str``
                    - **shape of list**: ``[]``
                    - **accepted strings**: ``"original"``, ``"generic"``
                    - **default**: ``original``

            - **color**

                Plot 
                color.

                    - **type**: ``str``
                    - **default**: ``"green"``

            - **figure_file_name**

                File name for the generated figure. If it is ``None`` (default),
                it is automatically generated.

                    - **type**: ``str`` or ``None``
                    - **default**: ``None``

            - **show_plot**
            
                See :argument:`show_plot <common_methods_arguments.show_plot>`.

            - **timestamp**
            
                See :argument:`timestamp <common_methods_arguments.timestamp>`.

            - **overwrite**
            
                See :argument:`overwrite <common_methods_arguments.overwrite>`.

            - **verbose**
            
                See :argument:`verbose <common_methods_arguments.verbose>`.

            - **step_kwargs**

                Additional keyword arguments to pass to the ``plt.step``function.

                    - **type**: ``dict``

        - **Updates file**

            - :attr:`Data.output_log_file <NF4HEP.Data.output_log_file>`
        """
        verbose, verbose_sub = self.set_verbosity(verbose)
        print(header_string,"\nPlotting 1D distributions summary", show=verbose)
        if timestamp is None:
            timestamp = "datetime_"+datetime.now().strftime("%Y_%m_%d_%H_%M_%S_%fZ")[:-3]
        plt.style.use(mplstyle_path)
        start = timer()
        if X is None:
            X = self.data_X
        else:
            X = np.array(X)
        pars_labels = self.__set_pars_labels(pars_labels)
        labels = np.array(pars_labels).tolist()
        if max_points != None:
            nnn = np.min([len(X), max_points])
        else:
            nnn = len(X)
        rnd_indices = np.random.choice(np.arange(len(X)),size=nnn,replace=False)
        sqrt_n_plots = int(np.ceil(np.sqrt(len(X[1, :]))))
        plt.rcParams["figure.figsize"] = (sqrt_n_plots*3, sqrt_n_plots*3)
        for i in range(len(X[1,:])):
            plt.subplot(sqrt_n_plots,sqrt_n_plots,i+1)
            counts, bins = np.histogram(X[rnd_indices,i], nbins)
            integral = 1
            plt.step(bins[:-1], counts/integral, where='post',color = color,**step_kwargs)
            plt.xlabel(pars_labels[i],fontsize=11)
            plt.xticks(fontsize=11, rotation=90)
            plt.yticks(fontsize=11, rotation=90)
            x1,x2,y1,y2 = plt.axis()
            plt.tight_layout()
        plt.subplots_adjust(top=0.92, bottom=0.08, left=0.10, right=0.95, hspace=0.4, wspace=0.4)
        plt.tight_layout()
        if figure_file_name is not None:
            figure_file_name = self.update_figures(figure_file=figure_file_name,timestamp=timestamp,overwrite=overwrite)
        else:
            figure_file_name = self.update_figures(figure_file=self.output_figures_base_file_name+"_pars_summary.pdf",timestamp=timestamp,overwrite=overwrite) 
        utils.savefig(r"%s" % (path.join(self.output_figures_folder, figure_file_name)), dpi=50)
        utils.check_set_dict_keys(self.predictions["Figures"],[timestamp],[[]],verbose=False)
        utils.append_without_duplicate(self.predictions["Figures"][timestamp], figure_file_name)
        if show_plot:
            plt.show()
        plt.close()
        end = timer()
        self.log[timestamp] = {"action": "saved figure",
                               "file name": figure_file_name}
        print("\n"+header_string+"\nFigure file\n\t", r"%s" % (figure_file_name), "\ncreated and saved in", str(end-start), "s.\n", show=verbose)
        self.save_log(overwrite=overwrite, verbose=verbose_sub)

    def plot_X_distributions(self,
                             X=None,
                             pars=None,
                             max_points=None,
                             nbins=50,
                             pars_labels="original", 
                             color="green",
                             figure_file_name=None,
                             show_plot=False,
                             timestamp=None,
                             overwrite=False,
                             verbose=None,
                             **step_kwargs):
        """
        Plots 1D distributions of the parameters ``pars`` in 
        the :attr:`Data.data_X <NF4HEP.Data.data_X>` dataset.

        - **Arguments**

            - **X**

                X data to use for the plot. If ``None`` is given the 
                :attr:`Data.data_X <NF4HEP.Data.data_X>` dataset is used.

                    - **type**: ``list`` or ``numpy.ndarray``
                    - **shape**: ``(npoints,ndims)``
                    - **default**: ``None``
            
            - **pars**

                List of parameters 
                for which the plots are produced.

                    - **type**: ``list`` or ``None``
                    - **shape of list**: ``[ ]``
                    - **default**: ``None``

            - **max_points**

                Maximum number of points used to make
                the plot. If the numnber is smaller than the total
                number of available points, then a random subset is taken.
                If ``None`` then all available points are used.

                    - **type**: ``int`` or ``None``
                    - **default**: ``None``

            - **nbins**

                Number of bins used to make 
                the histograms.

                    - **type**: ``int``
                    - **default**: ``50``

            - **pars_labels**

                Argument that is passed to the :meth:`Data.__set_pars_labels < NF4HEP.Data._Lik__set_pars_labels>`
                method to set the parameters labels to be used in the plots.

                    - **type**: ``list`` or ``str``
                    - **shape of list**: ``[]``
                    - **accepted strings**: ``"original"``, ``"generic"``
                    - **default**: ``original``

            - **color**

                Plot 
                color.

                    - **type**: ``str``
                    - **default**: ``"green"``

            - **figure_file_name**

                File name for the generated figure. If it is ``None`` (default),
                it is automatically generated.

                    - **type**: ``str`` or ``None``
                    - **default**: ``None``

            - **show_plot**
            
                See :argument:`show_plot <common_methods_arguments.show_plot>`.

            - **timestamp**
            
                See :argument:`timestamp <common_methods_arguments.timestamp>`.

            - **overwrite**
            
                See :argument:`overwrite <common_methods_arguments.overwrite>`.

            - **verbose**
            
                See :argument:`verbose <common_methods_arguments.verbose>`.

            - **step_kwargs**

                Additional keyword arguments to pass to the ``plt.step`` function.

                    - **type**: ``dict``

        - **Updates file**

            - :attr:`Data.output_log_file <NF4HEP.Data.output_log_file>`
        """
        verbose, verbose_sub = self.set_verbosity(verbose)
        print(header_string,"\nPlotting 1D distributions summary", show=verbose)
        if timestamp is None:
            timestamp = "datetime_"+datetime.now().strftime("%Y_%m_%d_%H_%M_%S_%fZ")[:-3]
        plt.style.use(mplstyle_path)
        start = timer()
        if X is None:
            X = self.data_X
        else:
            X = np.array(X)
        pars_labels = self.__set_pars_labels(pars_labels)
        if max_points != None:
            nnn = np.min([len(X), max_points])
        else:
            nnn = len(X)
        rnd_indices = np.random.choice(np.arange(len(X)),size=nnn,replace=False)
        for par in pars:
            counts, bins = np.histogram(X[rnd_indices,par], nbins)
            integral = 1
            plt.step(bins[:-1], counts/integral, where='post',color = color,**step_kwargs)
            plt.xlabel(pars_labels[par])
            plt.ylabel(r"number of samples")
            x1,x2,y1,y2 = plt.axis()
            plt.tight_layout()
            if figure_file_name is not None:
                figure_file_name = self.update_figures(figure_file=figure_file_name,timestamp=timestamp,overwrite=overwrite)
            else:
                figure_file_name = self.update_figures(figure_file=self.output_figures_base_file_name+"_1D_distr_par_"+str(par)+".pdf",timestamp=timestamp,overwrite=overwrite) 
            utils.savefig(r"%s" % (path.join(self.output_figures_folder, figure_file_name)), dpi=50)
            utils.check_set_dict_keys(self.predictions["Figures"],[timestamp],[[]],verbose=False)
            utils.append_without_duplicate(self.predictions["Figures"][timestamp], figure_file_name)
            self.log[timestamp] = {"action": "saved figure",
                                   "file name": figure_file_name}
            end = timer()
            print("\n"+header_string+"\nFigure file\n\t", r"%s" % (figure_file_name), "\ncreated and saved in", str(end-start), "s.\n", show=verbose)
            if show_plot:
                plt.show()
            plt.close()
        self.save_log(overwrite=overwrite, verbose=verbose_sub)

    def plot_corners_1samp(self,
                           X=None,
                           intervals=inference.CI_from_sigma([1, 2, 3]), 
                           weights=None, 
                           pars=None, 
                           max_points=None, 
                           nbins=50, 
                           pars_labels="original",
                           ranges_extend=None, 
                           title = "", 
                           color="green",
                           plot_title="Corner plot", 
                           legend_labels=None, 
                           figure_file_name=None, 
                           show_plot=False, 
                           timestamp=None, 
                           overwrite=False, 
                           verbose=None, 
                           **corner_kwargs):
        """
        Plots the 1D and 2D distributions (corner plot) of the distribution of the parameters ``pars`` in the ``X`` array.

        - **Arguments**

            - **X**

                X data to use for the plot. If ``None`` is given the 
                :attr:`Data.data_X <NF4HEP.Data.data_X>` dataset is used.

                    - **type**: ``list`` or ``numpy.ndarray``
                    - **shape**: ``(npoints,ndims)``
                    - **default**: ``None``
                    
            - **intervals**

                Probability intervals for which 
                contours are drawn.

                    - **type**: ``list`` or ``numpy.ndarray``
                    - **shape**: ``(nintervals,)``
                    - **default**: ``numpy.array([0.68268949, 0.95449974, 0.9973002])`` (corresponding to 1,2, and 3 sigmas for a 1D Gaussian distribution)

            - **weights**

                List or |Numpy_link| array with the 
                Weights correspomnding to the ``X`` points

                    - **type**: ``list`` or ``numpy.ndarray`` or ``None``
                    - **shape**: ``(npoints,)``
                    - **default**: ``None``

            - **pars**

                List of parameters 
                for which the plots are produced.

                    - **type**: ``list`` or ``None``
                    - **shape of list**: ``[ ]``
                    - **default**: ``None``

            - **max_points**

                Maximum number of points used to make
                the plot. If the numnber is smaller than the total
                number of available points, then a random subset is taken.
                If ``None`` then all available points are used.

                    - **type**: ``int`` or ``None``
                    - **default**: ``None``

            - **nbins**

                Number of bins used to make 
                the histograms.

                    - **type**: ``int``
                    - **default**: ``50``

            - **pars_labels**

                Argument that is passed to the :meth:`Data.__set_pars_labels < NF4HEP.Data._Lik__set_pars_labels>`
                method to set the parameters labels to be used in the plots.

                    - **type**: ``list`` or ``str``
                    - **shape of list**: ``[]``
                    - **accepted strings**: ``"original"``, ``"generic"``
                    - **default**: ``original``

            - **ranges_extend**

                Percent increase or reduction of the range of the plots with respect
                to the range automatically determined from the points values.

                    - **type**: ``int`` or ``float`` or ``None``

            - **title**

                Subplot title to which the 
                68% HPDI values are appended.

                    - **type**: ``str`` or ``None``
                    - **default**: ``None``

            - **color**

                Plot 
                color.

                    - **type**: ``str``
                    - **default**: ``"green"``

            - **plot_title**

                Title of the corner 
                plot.

                    - **type**: ``str``
                    - **default**: ``"Corner plot"``

            - **legend_labels**

                List of strings. Labels for the contours corresponding to the 
                ``intervals`` to show in the legend.
                If ``None`` the legend automatically reports the intervals.

                    - **type**: ``str`` or ``None``
                    - **default**: ``None``

            - **figure_file_name**

                File name for the generated figure. If it is ``None`` (default),
                it is automatically generated.

                    - **type**: ``str`` or ``None``
                    - **default**: ``None``

            - **show_plot**
            
                See :argument:`show_plot <common_methods_arguments.show_plot>`.

            - **timestamp**
            
                See :argument:`timestamp <common_methods_arguments.timestamp>`.

            - **overwrite**
            
                See :argument:`overwrite <common_methods_arguments.overwrite>`.

            - **verbose**
            
                See :argument:`verbose <common_methods_arguments.verbose>`.

            - **corner_kwargs**

                Additional keyword arguments to pass to the ``corner`` function.

                    - **type**: ``dict``

        - **Updates file**

            - :attr:`Data.output_log_file <NF4HEP.Data.output_log_file>`
        """
        verbose, verbose_sub = self.set_verbosity(verbose)
        print(header_string,"\nPlotting 2D marginal posterior density", show=verbose)
        if timestamp is None:
            timestamp = "datetime_"+datetime.now().strftime("%Y_%m_%d_%H_%M_%S_%fZ")[:-3]
        if legend_labels != None:
            if len(legend_labels) != len(intervals):
                raise Exception("Legend labels should either be None or a list of strings with the same length as intervals.")
        plt.style.use(mplstyle_path)
        start = timer()
        if X is None:
            X = self.data_X
        else:
            X = np.array(X)
        weigths = np.array(weights)
        if title == None:
            title = ""
        linewidth = 1.3
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
        rnd_idx = np.random.choice(np.arange(len(X)), nnn, replace=False)
        samp = X[rnd_idx][:,pars]
        if weights is not None:
            weights = weights[rnd_idx]
        print("Computing HPDIs.\n", show=verbose)
        HPDI = [inference.HPDI(samp[:,i], intervals = intervals, weights=weights, nbins=nbins, print_hist=False, optimize_binning=False) for i in range(nndims)]
        levels = np.array([[np.sort(inference.HPD_quotas(samp[:,[i,j]], nbins=nbins, intervals = intervals, weights=weights)).tolist() for j in range(nndims)] for i in range(nndims)])
        corner_kwargs_default = {"labels":  [r"%s" % s for s in labels],
                                 "max_n_ticks": 6, 
                                 "color": color,
                                 "plot_contours": True,
                                 "smooth": True, 
                                 "smooth1d": True,
                                 "range": ranges,
                                 "plot_datapoints": True, 
                                 "plot_density": False, 
                                 "fill_contours": False, 
                                 "normalize1d": True,
                                 "hist_kwargs": {"color": color, "linewidth": "1.5"}, 
                                 "label_kwargs": {"fontsize": 16}, 
                                 "show_titles": False,
                                 "title_kwargs": {"fontsize": 18}, 
                                 "levels_lists": levels,
                                 "data_kwargs": {"alpha": 1},
                                 "contour_kwargs": {"linestyles": ["dotted", "dashdot", "dashed"][:len(HPDI[0])], "linewidths": [linewidth, linewidth, linewidth][:len(HPDI[0])]},
                                 "no_fill_contours": False, 
                                 "contourf_kwargs": {"colors": ["white", "lightgreen", color], "alpha": 1}}
        corner_kwargs_default = utils.dic_minus_keys(corner_kwargs_default, list(corner_kwargs.keys()))
        corner_kwargs = {**corner_kwargs,**corner_kwargs_default}
        fig, axes = plt.subplots(nndims, nndims, figsize=(3*nndims, 3*nndims))
        figure = corner(samp, bins=nbins, weights=weights, fig=fig, **corner_kwargs)
                        # , levels=(0.393,0.68,)) ,levels=[300],levels_lists=levels1)#,levels=[120])
        #figure = corner(samp, bins=nbins, weights=weights, labels=[r"%s" % s for s in labels],
        #                fig=fig, max_n_ticks=6, color=color, plot_contours=True, smooth=True,
        #                smooth1d=True, range=ranges, plot_datapoints=True, plot_density=False,
        #                fill_contours=False, normalize1d=True, hist_kwargs={"color": color, "linewidth": "1.5"},
        #                label_kwargs={"fontsize": 16}, show_titles=False, title_kwargs={"fontsize": 18},
        #                levels_lists=levels, data_kwargs={"alpha": 1},
        #                contour_kwargs={"linestyles": ["dotted", "dashdot", "dashed"][:len(
        #                    HPDI[0])], "linewidths": [linewidth, linewidth, linewidth][:len(HPDI[0])]},
        #                no_fill_contours=False, contourf_kwargs={"colors": ["white", "lightgreen", color], "alpha": 1}, **kwargs)
        #                # , levels=(0.393,0.68,)) ,levels=[300],levels_lists=levels1)#,levels=[120])
        axes = np.array(figure.axes).reshape((nndims, nndims))
        lines_array = list(matplotlib.lines.lineStyles.keys())
        linestyles = (lines_array[0:4]+lines_array[0:4]+lines_array[0:4])[0:len(intervals)]
        intervals_str = [r"${:.2f}".format(i*100)+"\%$ HPDI" for i in intervals]
        for i in range(nndims):
            title_i = ""
            ax = axes[i, i]
            #ax.axvline(value1[i], color="green",alpha=1)
            #ax.axvline(value2[i], color="red",alpha=1)
            ax.grid(True, linestyle="--", linewidth=1, alpha=0.3)
            ax.tick_params(axis="both", which="major", labelsize=16)
            hists_1d = get_1d_hist(i, samp, nbins=nbins, ranges=ranges, weights=weights, normalize1d=True)[0]  # ,intervals=HPDI681)
            for q in range(len(intervals)):
                for j in HPDI[i][intervals[q]]["Intervals"]:
                    ax.axvline(hists_1d[0][hists_1d[0] >= j[0]][0], color=color, alpha=1, linestyle=linestyles[q], linewidth=linewidth)
                    ax.axvline(hists_1d[0][hists_1d[0] <= j[1]][-1], color=color, alpha=1, linestyle=linestyles[q], linewidth=linewidth)
                title_i = r"%s"%title + ": ["+"{0:1.2e}".format(HPDI[i][intervals[0]]["Intervals"][0][0])+","+"{0:1.2e}".format(HPDI[i][intervals[0]]["Intervals"][0][1])+"]"
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
        fig.subplots_adjust(top=0.85,wspace=0.25, hspace=0.25)
        fig.suptitle(r"%s" % (plot_title), fontsize=26)
        #fig.text(0.5 ,1, r"%s" % plot_title, fontsize=26)
        colors = [color, "black", "black", "black"]
        red_patch = matplotlib.patches.Patch(color=colors[0])  # , label="The red data")
        #blue_patch = matplotlib.patches.Patch(color=colors[1])  # , label="The blue data")
        lines = [matplotlib.lines.Line2D([0], [0], color=colors[1], linewidth=3, linestyle=l) for l in linestyles]
        if legend_labels is None:
            legend_labels = [intervals_str[i] for i in range(len(intervals))]
        fig.legend(lines, legend_labels, fontsize=int(7+2*nndims), loc="upper right")#(1/nndims*1.05,1/nndims*1.1))#transform=axes[0,0].transAxes)# loc=(0.53, 0.8))
        #plt.tight_layout()
        if figure_file_name is not None:
            figure_file_name = self.update_figures(figure_file=figure_file_name,timestamp=timestamp,overwrite=overwrite)
        else:
            figure_file_name = self.update_figures(figure_file=self.output_figures_base_file_name+"_corner_posterior_1samp_pars_" + "_".join([str(i) for i in pars]) +".pdf",timestamp=timestamp,overwrite=overwrite) 
        utils.savefig(r"%s" % (path.join(self.output_figures_folder, figure_file_name)), dpi=50)
        utils.check_set_dict_keys(self.predictions["Figures"],[timestamp],[[]],verbose=False)
        utils.append_without_duplicate(self.predictions["Figures"][timestamp], figure_file_name)
        if show_plot:
            plt.show()
        plt.close()
        end = timer()
        timestamp = "datetime_"+datetime.now().strftime("%Y_%m_%d_%H_%M_%S_%fZ")[:-3]
        self.log[timestamp] = {"action": "saved figure",
                               "file name": figure_file_name}
        print("\n"+header_string+"\nFigure file\n\t", r"%s" % (figure_file_name), "\ncreated and saved in", str(end-start), "s.\n", show=verbose)

    def plot_correlation_matrix(self, 
                                X=None,
                                pars_labels="original",
                                title = None,
                                figure_file_name=None, 
                                show_plot=False, 
                                timestamp=None, 
                                overwrite=False, 
                                verbose=None, 
                                **matshow_kwargs):
        """
        Plots the correlation matrix of the  
        :attr:`Data.data_X <NF4HEP.Data.data_X>` dataset.

        - **Arguments**

            - **X**

                X data to use for the plot. If ``None`` is given the 
                :attr:`Data.data_X <NF4HEP.Data.data_X>` dataset is used.

                    - **type**: ``list`` or ``numpy.ndarray``
                    - **shape**: ``(npoints,ndims)``
                    - **default**: ``None``
                    
            - **pars_labels**

                Argument that is passed to the :meth:`Data.__set_pars_labels < NF4HEP.Data._Lik__set_pars_labels>`
                method to set the parameters labels to be used in the plots.

                    - **type**: ``list`` or ``str``
                    - **shape of list**: ``[]``
                    - **accepted strings**: ``"original"``, ``"generic"``
                    - **default**: ``original``

            - **title**

                Subplot title to which the 
                68% HPDI values are appended.

                    - **type**: ``str`` or ``None``
                    - **default**: ``None``

            - **figure_file_name**

                File name for the generated figure. If it is ``None`` (default),
                it is automatically generated.

                    - **type**: ``str`` or ``None``
                    - **default**: ``None``

            - **show_plot**
            
                See :argument:`show_plot <common_methods_arguments.show_plot>`.

            - **timestamp**
            
                See :argument:`timestamp <common_methods_arguments.timestamp>`.

            - **overwrite**
            
                See :argument:`overwrite <common_methods_arguments.overwrite>`.

            - **verbose**
            
                See :argument:`verbose <common_methods_arguments.verbose>`.

            - **step_kwargs**

                Additional keyword arguments to pass to the ``plt.matshow`` function.

                    - **type**: ``dict``

        - **Updates file**

            - :attr:`Data.output_log_file <NF4HEP.Data.output_log_file>`
        """
        verbose, verbose_sub = self.set_verbosity(verbose)
        print(header_string,"\nPlotting X data correlation matrix", show=verbose)
        if timestamp is None:
            timestamp = "datetime_"+datetime.now().strftime("%Y_%m_%d_%H_%M_%S_%fZ")[:-3]
        plt.style.use(mplstyle_path)
        start = timer()
        if title == None:
            title = "Correlation Matrix"
        if X is None:
            X = self.data_X
        else:
            X = np.array(X)
        pars_labels = self.__set_pars_labels(pars_labels)
        df = pd.DataFrame(X)
        f = plt.figure(figsize=(18, 18))
        plt.matshow(df.corr(), fignum=f.number, **matshow_kwargs)
        plt.xticks(range(df.select_dtypes(['number']).shape[1]), pars_labels, fontsize=10, rotation=45)
        plt.yticks(range(df.select_dtypes(['number']).shape[1]), pars_labels, fontsize=10)
        cb = plt.colorbar()
        cb.ax.tick_params(labelsize=11)
        plt.title('Correlation Matrix', fontsize=13)
        plt.grid(False)
        if figure_file_name is not None:
            figure_file_name = self.update_figures(figure_file=figure_file_name,timestamp=timestamp,overwrite=overwrite)
        else:
            figure_file_name = self.update_figures(figure_file=self.output_figures_base_file_name+"_correlation_matrix.pdf",timestamp=timestamp,overwrite=overwrite) 
        utils.savefig(r"%s" % (path.join(self.output_figures_folder, figure_file_name)), dpi=50)
        utils.check_set_dict_keys(self.predictions["Figures"],[timestamp],[[]],verbose=False)
        utils.append_without_duplicate(self.predictions["Figures"][timestamp], figure_file_name)
        if show_plot:
            plt.show()
        plt.close()
        end = timer()
        timestamp = "datetime_"+datetime.now().strftime("%Y_%m_%d_%H_%M_%S_%fZ")[:-3]
        self.log[timestamp] = {"action": "saved figure",
                               "file name": figure_file_name}
        print("\n"+header_string+"\nFigure file\n\t", r"%s" % (figure_file_name), "\ncreated and saved in", str(end-start), "s.\n", show=verbose)
