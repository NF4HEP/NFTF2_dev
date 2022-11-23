__all__ = ["DataFileManager",
           "DataParsManager",
           "DataSamples",
           "DataMain",
           "DataManager",
           "DataPredictionsManager",
           "DataFiguresManager",
           "DataInference",
           "DataPlotter"]

import builtins
import codecs
from email.headerregistry import Group
import json
import time
from datetime import datetime
from os import path
from timeit import default_timer as timer

import deepdish as dd # type: ignore
import h5py # type: ignore
import matplotlib # type: ignore
import matplotlib.pyplot as plt #type: ignore
import numpy as np
from numpy import typing as npt
from pathlib import Path

import pandas as pd # type: ignore
import seaborn as sns # type: ignore
from sklearn.model_selection import train_test_split # type: ignore
from sklearn.preprocessing import StandardScaler #type: ignore

from typing import Union, List, Dict, Callable, Tuple, Optional, NewType, Type, Generic, Any, TypeVar, TYPE_CHECKING
from typing_extensions import TypeAlias
from NF4HEP.utils.custom_types import Array, ArrayInt, ArrayStr, DataType, StrPath, IntBool, StrBool, StrList, FigDict, LogPredDict, Number, DTypeStr, DTypeStrList, DictStr
from NF4HEP.base import Name, FileManager, PredictionsManager, FiguresManager, Inference, Plotter, InvalidInput
from NF4HEP.utils import mplstyle_path
from NF4HEP import print
from NF4HEP.utils.verbosity import Verbosity
from NF4HEP.utils import corner
from NF4HEP.utils import utils

sns.set()
kubehelix = sns.color_palette("cubehelix", 30)
reds = sns.color_palette("Reds", 30)
greens = sns.color_palette("Greens", 30)
blues = sns.color_palette("Blues", 30)

header_string_1 = "=============================="
header_string_2 = "------------------------------"

class DataFileManager(FileManager):
    """
    """
    managed_object: str = "Data"
    def __init__(self,
                 name: Optional[str] = None,
                 input_file: Optional[StrPath] = None, 
                 output_folder: Optional[StrPath] = None,
                 load_on_RAM: bool = False,
                 verbose: Optional[IntBool] = None
                ) -> None:
        # Attributes type declarations (from parent FileManager class)
        self._input_file: Optional[Path]
        self._name: Name
        self._output_folder: Path
        self._ManagedObject: "DataMain"
        # Attributes type declarations
        self._load_on_RAM: bool
        self._opened_dataset: h5py.File
        # Initialize parent FileManager class
        super().__init__(name=name,
                         input_file=input_file,
                         output_folder=output_folder,
                         verbose=verbose)
        # Set verbosity
        verbose, verbose_sub = self.get_verbosity(verbose)
        # Initialize object
        self.load_on_RAM = load_on_RAM

    @property
    def ManagedObject(self) -> "DataMain":
        return self._ManagedObject

    @ManagedObject.setter
    def ManagedObject(self,
                      managed_object: "DataMain"
                     ) -> None:
        try:
            self._ManagedObject
            raise Exception("The 'ManagedObject' attribute is automatically set when initialising the DataMain object and cannot be replaced.")
        except:
            self._ManagedObject = managed_object

    @property
    def input_idx_h5_file(self) -> Optional[Path]:
        if self._input_file is not None:
            return self._input_file.parent.joinpath(self._input_file.stem + "_idx.h5")
        else:
            return None

    @property
    def input_preprocessing_h5_file(self) -> Optional[Path]:
        if self._input_file is not None:
            return self._input_file.parent.joinpath(self._input_file.stem + "_preprocessing.h5")
        else:
            return None

    @property
    def input_samples_h5_file(self) -> Optional[Path]:
        if self._input_file is not None:
            return self._input_file.parent.joinpath(self._input_file.stem + "_samples.h5")
        else:
            return None

    @property
    def load_on_RAM(self) -> bool:
        return self._load_on_RAM

    @load_on_RAM.setter
    def load_on_RAM(self,
                    load_on_RAM: bool) -> None:
        self._load_on_RAM = load_on_RAM

    @property
    def opened_dataset(self) -> h5py.File:
        return self._opened_dataset

    @property
    def output_idx_h5_file(self) -> Path:
        return self._output_folder.joinpath(self.name+"_idx.h5")

    @property
    def output_idx_json_file(self) -> Path:
        return self._output_folder.joinpath(self.name+"_idx.json")

    @property
    def output_preprocessing_h5_file(self) -> Path:
        return self._output_folder.joinpath(self.name+"_preprocessing.h5")

    @property
    def output_samples_h5_file(self) -> Path:
        return self._output_folder.joinpath(self.name+"_samples.h5")

    def close_opened_dataset(self,
                             verbose: Optional[IntBool] = None
                            ) -> None:
        """ 
        """
        verbose, _ = self.get_verbosity(verbose)
        try:
            self._opened_dataset.close()
            del(self._opened_dataset)
            print(header_string_2,"\nClosed", self.input_samples_h5_file,".\n", show = verbose)
        except:
            print(header_string_2,"\nNo dataset to close.\n", show = verbose)

    def load(self, verbose: Optional[IntBool] = None) -> None:
        """
        """
        verbose, _ = self.get_verbosity(verbose)
        self.load_log(verbose = verbose)
        self.load_object(verbose = verbose)
        self.load_data_idx(verbose = verbose)
        self.load_data_preprocessing(verbose = verbose)
        self.load_predictions(verbose = verbose)

    def load_object(self, verbose: Optional[IntBool] = None) -> None:
        verbose, verbose_sub = self.get_verbosity(verbose)
        if self.input_object_h5_file is not None:
            log: LogPredDict
            dictionary: Dict
            [log, dictionary] = self._FileManager__load_h5(input_h5_file = self.input_object_h5_file, # type: ignore
                                                           verbose = verbose
                                                           )
            # Load log
            self.ManagedObject._log = {**self.ManagedObject._log, **log}
            # Load DataMain main object attributes
            dict_to_load_main = dictionary["Main"]
            self.ManagedObject.__dict__.update(dict_to_load_main)
            utils.reset_random_seeds(self.ManagedObject.seed)
            self._name._Name__check_define_name(name = self.ManagedObject._name) # type: ignore
            if self.load_on_RAM != self.ManagedObject.load_on_RAM:
                self.load_on_RAM = self.ManagedObject._load_on_RAM
                print(header_string_2,"\nWARNING: The 'load_on_RAM' attribute from saved object differs from the one specified in the FileManager. Using the loaded one (",str(self.load_on_RAM),").\n", show = True)
            # Load arguments and re-build ParsManager object
            dict_to_load_pars_manager = dictionary["ParsManager"]
            self.ManagedObject.ParsManager = DataParsManager(ndims = dict_to_load_pars_manager["_ndims"],
                                                             pars_central = dict_to_load_pars_manager["_pars_central"],
                                                             pars_bounds = dict_to_load_pars_manager["_pars_bounds"],
                                                             pars_labels = dict_to_load_pars_manager["_pars_labels"],
                                                             pars_pos_nuis = dict_to_load_pars_manager["_pars_pos_nuis"],
                                                             pars_pos_poi = dict_to_load_pars_manager["_pars_pos_poi"],
                                                             verbose = verbose_sub)
            # Load arguments and re-build DataSamples and DataManager objects
            self.load_input_data(verbose = verbose)
            dict_to_load_data_manager: Dict = dictionary["DataManager"]
            self.ManagedObject.DataManager = DataManager(data_main = self.ManagedObject,
                                                         npoints = [dict_to_load_data_manager["_npoints_train"], # type: ignore
                                                                    dict_to_load_data_manager["_npoints_val"],
                                                                    dict_to_load_data_manager["_npoints_test"]], # list with [n_train, n_val, n_test]
                                                         preprocessing = [dict_to_load_data_manager["_scalerX_bool"], # type: ignore
                                                                          dict_to_load_data_manager["_scalerY_bool"],
                                                                          dict_to_load_data_manager["_rotationX_bool"]], # list with [scalerX_bool, scalerY_bool, rotationX_bool]s
                                                         seed = dict_to_load_data_manager["_seed"],
                                                         verbose = verbose_sub)
        else:
            raise Exception("Input file not defined.")

    def load_data_idx(self, verbose: Optional[IntBool] = None) -> None:
        verbose, _ = self.get_verbosity(verbose)
        if self.input_idx_h5_file is not None:
            [log, dictionary] = self._FileManager__load_h5(input_h5_file = self.input_idx_h5_file, # type: ignore
                                                           verbose = verbose
                                                           )
            self.ManagedObject._log = {**self.ManagedObject._log, **log}
            self.ManagedObject.DataManager.__dict__.update(dictionary)
        else:
            raise Exception("Input file not defined.")

    def load_data_preprocessing(self, verbose: Optional[IntBool] = None) -> None:
        verbose, _ = self.get_verbosity(verbose)
        if self.input_preprocessing_h5_file is not None:
            [log, dictionary] = self._FileManager__load_h5(input_h5_file = self.input_preprocessing_h5_file, # type: ignore
                                                           verbose = verbose
                                                           )
            self.ManagedObject._log = {**self.ManagedObject._log, **log}
            self.ManagedObject.DataManager.__dict__.update(dictionary)
        else:
            raise Exception("Input file not defined.")

    def load_input_data(self,
                        dtype_required: Optional[DTypeStr] = None,
                        verbose: Optional[IntBool] = None
                       ) -> None:
        """
        """
        verbose, verbose_sub = self.get_verbosity(verbose)
        start = timer()
        if self.input_samples_h5_file is not None:
            input_file = self.input_samples_h5_file
        else:
            raise Exception("Input file not defined.")
        self._opened_dataset = h5py.File(input_file, "r")
        data = h5py.Group(self._opened_dataset["input_data"].id)
        data_X = h5py.Dataset(data["X"].id)
        data_Y = h5py.Dataset(data["Y"].id)
        if dtype_required is None:
            dtype_required = np.dtype(h5py.Datatype(data["dtype_required"].id)).type
        else:
            dtype_required = np.dtype(dtype_required).type
        dtype_stored = np.dtype(h5py.Datatype(data["dtype_stored"].id)).type
        if self._load_on_RAM:
            data_X = np.array(data_X[:]).astype(dtype_stored)
            data_Y = np.array(data_Y[:]).astype(dtype_stored)
            self._opened_dataset.close()
        self.ManagedObject.InputData = DataSamples(data_X = data_X,
                                                   data_Y = data_Y,
                                                   dtype = [dtype_stored,dtype_required],
                                                   verbose = verbose_sub)
        end = timer()
        self.ManagedObject._log[utils.generate_timestamp()] = {"action": "loaded DataSamples",
                                                               "files names": [input_file.name]}
        print(header_string_2,"\nData samples file\n",input_file,"\nloaded in", str(end-start), ".\n", show = verbose)
        if self._load_on_RAM:
            print(header_string_2,"\nSamples loaded on RAM.\n", show = verbose)

    def save(self,
             timestamp: Optional[str] = None,
             overwrite: StrBool = False,
             verbose: Optional[IntBool] = None
            ) -> None:
        """
        """
        if timestamp is None:
            timestamp = utils.generate_timestamp()
        self.save_object(timestamp = timestamp, overwrite = overwrite, verbose = verbose)
        self.save_data_idx(timestamp = timestamp, overwrite = overwrite, verbose = verbose)
        self.save_data_preprocessing(timestamp = timestamp, overwrite = overwrite, verbose = verbose)
        self.save_input_data(timestamp = timestamp, overwrite = overwrite, verbose = verbose) 
        self.save_predictions(timestamp = timestamp, overwrite = overwrite, verbose = verbose)
        self.save_log(timestamp = timestamp, overwrite = overwrite, verbose = verbose)

    def save_input_data(self,
                        timestamp: Optional[str] = None,
                        overwrite: StrBool = False,
                        verbose: Optional[IntBool] = None
                       ) -> None:
        """
        Save samples to h5 dataset
        """
        verbose, verbose_sub = self.get_verbosity(verbose)
        input_data = self.ManagedObject.InputData
        if timestamp is None:
            timestamp = utils.generate_timestamp()
        start = timer()
        output_h5_file = self.get_target_file_overwrite(input_file=self.output_samples_h5_file,
                                                        timestamp = timestamp,
                                                        overwrite=overwrite,
                                                        verbose=verbose_sub)
        h5_out = h5py.File(output_h5_file, "w")
        try:
            data = h5_out.create_group("input_data")
            data["shape"] = np.shape(input_data._data_X)
            data["X"] = input_data._data_X.astype(input_data._dtype_stored)
            data["Y"] = input_data._data_Y.astype(input_data._dtype_stored)
            data["dtype_required"] = np.dtype(input_data._dtype_required)
            data["dtype_stored"] = np.dtype(input_data._dtype_stored)
            h5_out.close()
        except:
            h5_out.close()
            raise Exception("Failed to save data. The file\n",output_h5_file,"\nhas been safely closed.")
        self.ManagedObject._log[utils.generate_timestamp()] = {"action": "saved samples h5",
                                                               "file names": [output_h5_file.name]}
        end = timer()
        self.print_save_info(filename = output_h5_file,
                             time = str(end-start),
                             overwrite = overwrite,
                             verbose = verbose)

    def save_object(self, 
                    timestamp: Optional[str] = None,
                    overwrite: StrBool = False,
                    verbose: Optional[IntBool] = None
                   ) -> None:
        if timestamp is None:
            timestamp = utils.generate_timestamp()
        dict_to_save_main = utils.dic_minus_keys(self.ManagedObject.__dict__,self.ManagedObject.excluded_attributes)
        dict_to_save_pars_manager = self.ManagedObject.ParsManager.__dict__
        dict_to_save_data_manager = utils.dic_minus_keys(self.ManagedObject.DataManager.__dict__,self.ManagedObject.DataManager.excluded_attributes)
        dict_to_save_data_manager["_train_val_range"] = [self.ManagedObject.DataManager._train_val_range[0],
                                                         self.ManagedObject.DataManager._train_val_range[-1]+1]
        dict_to_save_data_manager["_test_range"] = [self.ManagedObject.DataManager._test_range[0],
                                                    self.ManagedObject.DataManager._test_range[-1]+1]
        dict_to_save = {"Main": dict_to_save_main, "ParsManager": dict_to_save_pars_manager, "DataManager":  dict_to_save_data_manager}
        log = self._FileManager__save_dict_h5_json(dict_to_save = dict_to_save, # type: ignore
                                                   output_file = self.output_object_h5_file,
                                                   overwrite = overwrite,
                                                   verbose = verbose)
        self.ManagedObject._log = {**self.ManagedObject._log, **log}

    def save_data_idx(self, 
                    timestamp: Optional[str] = None,
                    overwrite: StrBool = False,
                    verbose: Optional[IntBool] = None
                   ) -> None:
        if timestamp is None:
            timestamp = utils.generate_timestamp()
        data_manager_dict = self.ManagedObject.DataManager.__dict__
        dict_to_save_idx = {}
        dict_to_save_idx["_idx_train"] = data_manager_dict["_idx_train"]
        dict_to_save_idx["_idx_val"] = data_manager_dict["_idx_val"]
        dict_to_save_idx["_idx_test"] = data_manager_dict["_idx_test"]
        log = self._FileManager__save_dict_h5_json(dict_to_save = dict_to_save_idx, # type: ignore
                                                   output_file = self.output_idx_h5_file,
                                                   overwrite = overwrite,
                                                   verbose = verbose)
        self.ManagedObject._log = {**self.ManagedObject._log, **log}

    def save_data_preprocessing(self, 
                                timestamp: Optional[str] = None,
                                overwrite: StrBool = False,
                                verbose: Optional[IntBool] = None
                               ) -> None:
        if timestamp is None:
            timestamp = utils.generate_timestamp()
        data_manager_dict = self.ManagedObject.DataManager.__dict__
        dict_to_save_preprocessing = {}
        dict_to_save_preprocessing["_scalerX"] = data_manager_dict["_scalerX"]
        dict_to_save_preprocessing["_scalerY"] = data_manager_dict["_scalerY"]
        dict_to_save_preprocessing["_rotationX"] = data_manager_dict["_rotationX"]
        log = self._FileManager__save_dict_h5(dict_to_save = dict_to_save_preprocessing, # type: ignore
                                              output_file = self.output_preprocessing_h5_file,
                                              overwrite = overwrite,
                                              verbose = verbose)
        self.ManagedObject._log = {**self.ManagedObject._log, **log}


class DataParsManager(Verbosity):
    """
    Class that defines all parameters properties in objects :obj:`DataMain <NF4HEP.DataMain>` and :obj:`NF <NF4HEP.NF>`.
    """
    def __init__(self,
                 ndims: Optional[int] = None,
                 pars_bounds: Optional[Array] = None,
                 pars_central: Optional[Array] = None,
                 pars_labels: Optional[ArrayStr] = None,
                 pars_pos_nuis: Optional[ArrayInt] = None,
                 pars_pos_poi: Optional[ArrayInt] = None,
                 verbose: Optional[IntBool] = None
                ) -> None:
        """
        """
        # Attributes type declatation
        self._ndims: int
        self._pars_bounds: npt.NDArray[np.float_]
        self._pars_central: npt.NDArray[np.float_]
        self._pars_labels: npt.NDArray[np.str_]
        self._pars_labels_auto: npt.NDArray[np.str_]
        self._pars_pos_nuis: npt.NDArray[np.int_]
        self._pars_pos_poi: npt.NDArray[np.int_]
        # Initialize parent Verbosity class
        super().__init__(verbose)
        # Set verbosity
        verbose, verbose_sub = self.get_verbosity(verbose)
        # Initialize object
        print(header_string_1, "\nInitializing ParsManager object.\n", show = verbose)
        self._ndims = ndims if ndims is not None else 0
        self._pars_bounds = np.array(pars_bounds, dtype=np.float_) if pars_bounds is not None else np.array([], dtype=np.float_)
        self._pars_central = np.array(pars_central, dtype=np.float_) if pars_central is not None else np.array([], dtype=np.float_)
        self._pars_labels = np.array(pars_labels, dtype=np.str_) if pars_labels is not None else np.array([],dtype=np.str_)
        self._pars_pos_nuis = np.array(pars_pos_nuis, dtype=np.int_) if pars_pos_nuis is not None else np.array([],dtype=np.int_)
        self._pars_pos_poi = np.array(pars_pos_poi, dtype=np.int_) if pars_pos_poi is not None else np.array([],dtype=np.int_)
        self.__check_define_pars(verbose = verbose_sub)
    
    @property
    def ndims(self) -> int:
        return self._ndims

    @property
    def pars_bounds(self) -> npt.NDArray[np.float_]:
        return self._pars_bounds

    @property
    def pars_central(self) -> npt.NDArray[np.float_]:
        return self._pars_central

    @property
    def pars_labels(self) -> npt.NDArray[np.str_]:
        return self._pars_labels

    @property
    def pars_labels_auto(self) -> npt.NDArray[np.str_]:
        return self._pars_labels_auto

    @property
    def pars_pos_nuis(self) -> npt.NDArray[np.int_]:
        return self._pars_pos_nuis

    @property
    def pars_pos_poi(self) -> npt.NDArray[np.int_]:
        return self._pars_pos_poi

    def __check_define_pars(self,
                            verbose: Optional[IntBool] = None
                           ) -> None:
        """
        Private method used by the :meth:`ParsManager.__init__ <NF4HEP.ParsManager.__init__>` one
        to set and check the consistency of the attributes

            - :attr:`ParsManager._pars_central <NF4HEP.ParsManager._pars_central>`
            - :attr:`ParsManager._pars_bounds <NF4HEP.ParsManager._pars_bounds>` 
            - :attr:`ParsManager._pars_pos_poi <NF4HEP.ParsManager._pars_pos_poi>`
            - :attr:`ParsManager._pars_pos_nuis <NF4HEP.ParsManager._pars_pos_nuis>`
            - :attr:`ParsManager._pars_labels <NF4HEP.ParsManager._pars_labels>`
            - :attr:`ParsManager._ndims <NF4HEP.ParsManager._ndims>`
            
        If no parameters positions are specified, all parameters are assumed to be parameters of interest.
        If only the position of the parameters of interest or of the nuisance parameters is specified,
        the other is automatically generated by matching dimensions.
        If labels are not provided then :attr:`ParsManager._pars_labels <NF4HEP.ParsManager._pars_labels>`
        is set to the value of :attr:`ParsManager._pars_labels_auto <NF4HEP.ParsManager._pars_labels_auto>`.
        If parameters bounds are not provided, they are set to ``(-np.inf,np.inf)``.
        A check is performed on the length of the first five attributes and an InvalidInput exception is raised 
        if the length does not match :attr:`ParsManager._ndims <NF4HEP.ParsManager._ndims>`.

        - **Arguments**

            - **verbose**
            
                See :argument:`verbose <common_methods_arguments.verbose>`.
        """
        verbose, _ = self.get_verbosity(verbose)
        if len(self._pars_central) != 0:
            if self._ndims != 0:
                if len(self._pars_central) != self._ndims:
                    raise InvalidInput("Length of array with parameter central values different than declared number of dimensions.")
            else:
                self._ndims = len(self.pars_central)
        elif len(self._pars_central) == 0 and self._ndims != 0:
            self._pars_central = np.zeros(self._ndims)
            print(header_string_2,"\nNo central values for the parameters 'pars_central' have been specified. The central values have been set to zero for all. If they are known it is suggested to build the object providing parameters central values.\n", show = verbose)
        else:
            raise InvalidInput("Impossible to determine the number of parameters/dimensions and the parameters central values. Please specify at least one of the input parameters 'ndims' and 'pars_central'.")
        if len(self._pars_pos_nuis) != 0 and len(self._pars_pos_poi) != 0:
            if len(self._pars_pos_poi)+len(self._pars_pos_nuis) != self._ndims:
                raise InvalidInput("The number of parameters positions do not match the number of dimensions.")
        elif len(self._pars_pos_nuis) == 0 and len(self._pars_pos_poi) == 0:
            self._pars_pos_poi = np.arange(self._ndims)
            print(header_string_2,"\nThe positions of the parameters of interest (pars_pos_poi) and of the nuisance parameters (pars_pos_nuis) have not been specified. Assuming all parameters are parameters of interest.\n", show = verbose)
        elif len(self._pars_pos_nuis) != 0 and len(self._pars_pos_poi) == 0:
            self._pars_pos_poi = np.setdiff1d(np.arange(self._ndims), self._pars_pos_nuis)
            print(header_string_2,"\nOnly the positions of the nuisance parameters have been specified. Assuming all other parameters are parameters of interest.\n", show = verbose)
        elif len(self._pars_pos_nuis) == 0 and len(self._pars_pos_poi) != 0:
            self._pars_pos_nuis = np.setdiff1d(np.arange(self._ndims), self._pars_pos_poi)
            print(header_string_2,"\nOnly the positions of the parameters of interest have been specified. Assuming all other parameters are nuisance parameters.\n", show = verbose)
        self._pars_labels_auto = self.__get_pars_labels_auto()
        if len(self._pars_labels) == 0:
            self._pars_labels = self._pars_labels_auto
        elif len(self._pars_labels) != self._ndims:
            raise InvalidInput("The number of parameters labels do not match the number of dimensions.")
        if len(self._pars_bounds) == 0:
            self._pars_bounds = np.vstack([np.full(self._ndims, -np.inf), np.full(self._ndims, np.inf)]).T
            print(header_string_2,"\nNo bounds for the parameters 'pars_bounds' have been specified. The bounds have been set to [-inf,inf] for all parameters. If they are known it is suggested to build the object providing parameters bounds.\n", show = verbose)
        else:
            if len(self._pars_bounds) != self._ndims:
                raise InvalidInput("The length of the parameters bounds array does not match the number of dimensions.")

    def __get_pars_labels_auto(self) -> npt.NDArray[np.str_]:
        """
        """
        pars_labels_auto: List[str] = []
        i_poi: int = 1
        i_nuis: int = 1
        for i in range(len(self._pars_pos_poi)+len(self._pars_pos_nuis)):
            if i in self._pars_pos_poi:
                pars_labels_auto.append(r"$\theta_{%d}$" % i_poi)
                i_poi = i_poi+1
            else:
                pars_labels_auto.append(r"$\nu_{%d}$" % i_nuis)
                i_nuis = i_nuis+1
        return np.array(pars_labels_auto)

    def __set_pars_labels(self, 
                          pars_labels: Union[str,ArrayStr]
                         ) -> npt.NDArray[np.str_]:
        """
        """
        if type(pars_labels) == str:
            if pars_labels == "original":
                value = self._pars_labels
            elif pars_labels == "auto":
                value = self._pars_labels_auto
            else:
                raise InvalidInput("The 'pars_labels' argument should be one of the 'original' or 'auto' strings or a list (or array) of strings.")
        elif type(pars_labels) != str and len(pars_labels) != 0:
            if len(pars_labels) == self._ndims:
                value = np.array(pars_labels)
            else:
                raise InvalidInput("The 'pars_labels' array length does not match the nummber of dimensions.")
        else:
            raise InvalidInput("The 'pars_labels' argument should be one of the 'original' or 'auto' strings or a list (or array) of strings.")
        return value


class DataSamples(Verbosity):
    """
    Container of data from which DataManager inherits
    """
    def __init__(self,
                 data_X: Optional[DataType] = None,
                 data_Y: Optional[DataType] = None,
                 dtype: Optional[DTypeStrList] = None,
                 verbose: Optional[IntBool] = None
                ) -> None:
        """
        """
        # Attributes type declatation (from parent ParsManagers class)
        self._data_X: DataType
        self._data_Y: DataType
        self._dtype_required: npt.DTypeLike
        self._dtype_stored: npt.DTypeLike
        self._npoints: int
        self._ndims: int
        # Initialize parent Verbosity class
        super().__init__(verbose)
        # Set verbosity
        verbose, verbose_sub = self.get_verbosity(verbose)
        # Initialize object
        print(header_string_1, "\nInitializing DataSamples object.\n", show = verbose)
        self._data_X = np.array([[]]) if data_X is None else data_X
        self._data_Y = np.array([]) if data_Y is None else data_Y
        self._npoints = 0 if data_X is None else self._data_X.shape[0]
        self._ndims = 0 if data_X is None else self._data_X.shape[1]
        self.__check_data()
        self.set_dtype(dtype = dtype, verbose = verbose_sub)

    def set_dtype(self,
                  dtype: Optional[DTypeStrList] = None,
                  verbose: Optional[IntBool] = None
                 ) -> None:
        """ 
        """
        verbose, verbose_sub = self.get_verbosity(verbose)
        if dtype is None:
            self._dtype_stored = np.dtype("float32").type
            self._dtype_required = np.dtype("float32").type
        elif type(dtype) == str:
            try:
                self._dtype_stored = np.dtype(dtype).type
                self._dtype_required = np.dtype(dtype).type
            except:
                self._dtype_stored = np.dtype("float32").type
                self._dtype_required = np.dtype("float32").type
                print(header_string_2,"\nWarning: invalid data type",dtype,". Data types set to 'float32'.\n", show = True)
        elif type(dtype) == list:
            dtype_list = list(dtype) # type: ignore
            if len(dtype_list) == 2: 
                try:
                    self._dtype_stored = np.dtype(dtype_list[0]).type
                    self._dtype_required = np.dtype(dtype_list[1]).type
                except:
                    self._dtype_stored = np.dtype("float32").type
                    self._dtype_required = np.dtype("float32").type
                    print(header_string_2,"\nWarning: invalid data type list",dtype,". Data types set to 'float32'.\n", show = True)
            else:
                self._dtype_stored = np.dtype("float32").type
                self._dtype_required = np.dtype("float32").type
                print(header_string_2,"\nWarning: invalid data type list",dtype,". Data types set to 'float32'.\n", show = True)

    def __check_data(self) -> None:
        """ 
        """
        if np.shape(self._data_X) == (1,0) and np.shape(self._data_Y) == (0,):
            pass
        elif len(self._data_X) == len(self._data_Y):
            pass
        else:
            raise Exception("data_X and data_Y have different length.")

    @property
    def data_X(self) -> npt.NDArray:
        return np.array(self._data_X).astype(self._dtype_required)

    @property
    def data_Y(self) -> npt.NDArray:
        return np.array(self._data_Y).astype(self._dtype_required)

    @property
    def dtype_stored(self) -> npt.DTypeLike:
        return self._dtype_stored

    @property
    def dtype_required(self) -> npt.DTypeLike:
        return self._dtype_required

    @property
    def npoints(self) -> int:
        return self._npoints

    @property
    def ndims(self) -> int:
        return self._ndims


class DataMain(Verbosity):
    """
    This class contains the ``Data`` object representing the dataset used for training, validating and testing
    the DNNLikelihood. It can be creaded both feeding X and Y data or by loading an existing ``Data`` object.
    """
    managed_object: str = "DataMain"
    def __init__(self,
                 file_manager: DataFileManager,
                 pars_manager: Optional[DataParsManager] = None,
                 input_data: Optional[DataSamples] = None,
                 npoints: Optional[List[int]] = None, # list with [n_train, n_val, n_test]
                 preprocessing: Optional[List[bool]] = None, # list with [scalerX_bool, scalerY_bool, rotationX_bool]s
                 seed: Optional[int] = None,
                 verbose: IntBool = True
                 ) -> None:
        """
        """
        # Attributes type declatation
        self._load_on_RAM: bool
        self._log: LogPredDict
        self._seed: int
        self._DataManager: DataManager
        self._Figures: DataFiguresManager
        self._FileManager: DataFileManager
        self._InputData: DataSamples
        self._ParsManager: DataParsManager
        self._Predictions: DataPredictionsManager
        self._Inference: DataInference
        self._Plotter: DataPlotter
        # Initialize parent Verbosity class
        super().__init__(verbose)
        # Set verbosity
        verbose, verbose_sub = self.get_verbosity(verbose)
        # Initialize object
        timestamp = utils.generate_timestamp()
        print(header_string_1, "\nInitializing DataMain object.\n", show = verbose)
        self._log = {}
        print(header_string_2,"\nSetting FileManager.\n", show = verbose)
        self.FileManager = file_manager # also sets the DataMain managed object FileManager attribute and load_on_RAM DataMain attribute
        self.Predictions = DataPredictionsManager(data_main = self)
        print(header_string_2,"\nSetting Predictions.\n", show = self.verbose)
        self.Figures = DataFiguresManager(data_main = self)
        print(header_string_2,"\nSetting Figures.\n", show = verbose)
        if self.FileManager.input_file is None:
            self.seed = seed if seed is not None else 0
            print(header_string_2,"\nInitializing new DataMain object.\n", show = verbose)
            if pars_manager is not None:
                self.ParsManager = pars_manager
            else:
                raise Exception("When no input file is specified the 'pars_manager' argument needs to be different from 'None'.")
            if input_data is not None:
                self.InputData = input_data
            else:
                raise Exception("When no input file is specified the 'input_data' argument needs to be different from 'None'.")
            self.DataManager = DataManager(data_main = self,
                                           npoints = npoints,
                                           preprocessing = preprocessing,
                                           seed = seed,
                                           verbose = verbose)
        else:
            print(header_string_2,"\nLoading existing DataMain object.\n", show = verbose)
            for attr in [pars_manager,input_data,npoints,seed]:
                if attr is not None:
                    print(header_string_2,"\nWarning: an input file was specified and the argument '",attr,"' will be ignored. The related attribute will be set from the input file.\n", show = True)
            self.FileManager.load(verbose = verbose)
        self.Inference = DataInference(data_main = self)
        print(header_string_2,"\nSetting Inference.\n", show = verbose)
        self.Plotter = DataPlotter(data_main = self)
        print(header_string_2,"\nSetting Plotter.\n", show = verbose)
        if self.FileManager.input_file is None:
            self._log[timestamp] = {"action": "object created from input arguments"}
            #self.FileManager.save(timestamp = timestamp, overwrite = False, verbose = verbose_sub)
        else:
            self._log[utils.generate_timestamp()] = {"action": "object reconstructed from loaded files"}
            #self.FileManager.save_log(timestamp = timestamp, overwrite = True, verbose = verbose_sub)
        self.FileManager.save_log(timestamp = timestamp, overwrite = bool(self.FileManager.input_file), verbose = verbose_sub)

    @property
    def excluded_attributes(self) -> list:
        tmp = ["_log",
               "_verbose",
               "_DataManager",
               "_DataSamples",
               "_Figures",
               "_FileManager",
               "_Inference",
               "_ParsManager",
               "_Plotter",
               "_Predictions"
              ]
        return tmp

    @property
    def FileManager(self) -> DataFileManager:
        return self._FileManager

    @FileManager.setter
    def FileManager(self,
                    file_manager: DataFileManager,
                   ) -> None:
        try:
            self._FileManager
            raise Exception("The 'FileManager' attribute is automatically set when initialising the DataMain object and cannot be manually set.")
        except:
            self._FileManager = file_manager
            self._FileManager.ManagedObject = self
            self._load_on_RAM = self._FileManager.load_on_RAM

    @property
    def load_on_RAM(self) -> bool:
        return self._load_on_RAM

    @property
    def name(self) -> str:
        return self._FileManager.name

    @property
    def Predictions(self) -> "DataPredictionsManager":
        return self._Predictions

    @Predictions.setter
    def Predictions(self,
                    predictions: "DataPredictionsManager"
                   ) -> None:
        self._Predictions = predictions

    @property
    def Figures(self) -> "DataFiguresManager":
        return self._Figures

    @Figures.setter
    def Figures(self,
                figures: "DataFiguresManager"
               ) -> None:
        self._Figures = figures

    @property
    def ParsManager(self) -> DataParsManager:
        return self._ParsManager

    @ParsManager.setter
    def ParsManager(self,
                    pars_manager: DataParsManager
                   ) -> None:
        try:
            self._ParsManager
            raise Exception("The 'ParsManager' attribute is automatically set when initialising the DataMain object and cannot be manually set.")
        except:
            self._ParsManager = pars_manager
        print(header_string_2,"\nSetting ParsManager.\n", show = self._verbose)

    @property
    def InputData(self) -> DataSamples:
        return self._InputData

    @InputData.setter
    def InputData(self,
                  input_data: DataSamples
                 ) -> None:
        try:
            self._InputData
            raise Exception("The 'InputData' attribute is automatically set when initialising the DataMain object and cannot be manually set.")
        except:
            self._InputData = input_data
        print(header_string_2,"\nSetting DataSamples.\n", show = self.verbose)

    @property
    def DataManager(self) -> "DataManager":
        return self._DataManager

    @DataManager.setter
    def DataManager(self,
                    data_manager: "DataManager"
                   ) -> None:
        try:
            self._DataManager
            raise Exception("The 'DataManager' attribute is automatically set when initialising the DataMain object and cannot be manually set.")
        except:
            self._DataManager = data_manager
        print(header_string_2,"\nSetting DataManager.\n", show = self.verbose)

    @property
    def Inference(self) -> "DataInference":
        return self._Inference

    @Inference.setter
    def Inference(self,
                  inference: "DataInference"
                 ) -> None:
        try:
            self._Inference
            raise Exception("The 'Inference' attribute is automatically set when initialising the DataMain object and cannot be manually set.")
        except:
            self._Inference = inference
        print(header_string_2,"\nSetting Inference.\n", show = self.verbose)

    @property
    def Plotter(self) -> "DataPlotter":
        return self._Plotter

    @Plotter.setter
    def Plotter(self,
                plotter: "DataPlotter"
               ) -> None:
        try:
            self._Plotter
            raise Exception("The 'Inference' attribute is automatically set when initialising the DataMain object and cannot be manually set.")
        except:
            self._Plotter = plotter
        print(header_string_2,"\nSetting Plotter.\n", show = self.verbose)

    @property
    def seed(self) -> int:
        return self._seed

    @seed.setter
    def seed(self,
             seed: int
            ) -> None:
        self._seed = seed
        utils.reset_random_seeds(self._seed)

    #def update_figures(self,figure_file=None,timestamp=None,overwrite=False,verbose=None):
    #    """
    #    Method that generates new file names and renames old figure files when new ones are produced with the argument ``overwrite=False``. 
    #    When ``overwrite=False`` it calls the :func:`utils.check_rename_file <DNNLikelihood.utils.check_rename_file>` function and, if 
    #    ``figure_file`` already existed in the :attr:`Data.predictions <DNNLikelihood.Data.predictions>` dictionary, then it
    #    updates the dictionary by appennding to the old figure name the timestamp corresponding to its generation timestamp 
    #    (that is the key of the :attr:`Data.predictions["Figures"] <DNNLikelihood.Data.predictions>` dictionary).
    #    When ``overwrite="dump"`` it calls the :func:`utils.generate_dump_file_name <DNNLikelihood.utils.generate_dump_file_name>` function
    #    to generate the dump file name.
    #    It returns the new figure_file.
#
    #    - **Arguments**
#
    #        - **figure_file**
#
    #            Figure file path. If the figure already exists in the 
    #            :meth:`Data.predictions <DNNLikelihood.Data.predictions>` dictionary, then its name is updated with the corresponding timestamp.
#
    #        - **overwrite**
#
    #            The method updates file names and :attr:`Data.predictions <DNNLikelihood.Data.predictions>` dictionary only if
    #            ``overwrite=False``. If ``overwrite="dump"`` the method generates and returns the dump file path. 
    #            If ``overwrite=True`` the method just returns ``figure_file``.
#
    #        - **verbose**
    #        
    #            See :argument:`verbose <common_methods_arguments.verbose>`.
    #    
    #    - **Returns**
#
    #        - **new_figure_file**
    #            
    #            String identical to the input string ``figure_file`` unless ``verbose="dump"``.
#
    #    - **Creates/updates files**
#
    #        - Updates ``figure_file`` file name.
    #    """
    #    verbose, verbose_sub = self.get_verbosity(verbose)
    #    print("Checking and updating figures dictionary", show = verbose)
    #    if figure_file is None:
    #        raise Exception("figure_file input argument of update_figures method needs to be specified while it is None.")
    #    else:
    #        new_figure_file = figure_file
    #        if type(overwrite) == bool:
    #            if not overwrite:
    #                # search figure
    #                timestamp=None
    #                for k, v in self.predictions["Figures"].items():
    #                    if figure_file in v:
    #                        timestamp = k
    #                old_figure_file = utils.check_rename_file(path.join(self.output_figures_folder,figure_file),timestamp=timestamp,return_value="file_name",verbose=verbose_sub)
    #                if timestamp is not None:
    #                    self.predictions["Figures"][timestamp] = [f.replace(figure_file,old_figure_file) for f in v] 
    #        elif overwrite == "dump":
    #            new_figure_file = utils.generate_dump_file_name(figure_file, timestamp=timestamp)
    #    return new_figure_file
#
    #def data_description(self,
    #                     X=None,
    #                     pars_labels="original",
    #                     timestamp=None,
    #                     overwrite=False,
    #                     verbose=None):
    #    """
    #    Gives a description of data by calling the |Pandas_dataframe_describe|
    #    method.
#
    #    - **Arguments**
#
    #        - **X**
#
    #            X data to use for the plot. If ``None`` is given the 
    #            :attr:`Data.data_X <DNNLikelihood.Data.data_X>` dataset is used.
#
    #                - **type**: ``list`` or ``numpy.ndarray``
    #                - **shape**: ``(npoints,ndims)``
    #                - **default**: ``None``
#
    #        - **pars_labels**
#
    #                Argument that is passed to the :meth:`Data.__set_pars_labels < DNNLikelihood.Data._Lik__set_pars_labels>`
    #                method to set the parameters labels to be used in the plots.
#
    #                    - **type**: ``list`` or ``str``
    #                    - **shape of list**: ``[]``
    #                    - **accepted strings**: ``"original"``, ``"generic"``
    #                    - **default**: ``original``
#
    #        - **timestamp**
    #        
    #            See :argument:`timestamp <common_methods_arguments.timestamp>`.
#
    #        - **overwrite**
    #        
    #            See :argument:`overwrite <common_methods_arguments.overwrite>`.
#
    #        - **verbose**
    #        
    #            See :argument:`verbose <common_methods_arguments.verbose>`.
#
    #    - **Updates file**
#
    #        - :attr:`Data.output_log_file <DNNLikelihood.Data.output_log_file>`
    #    """
    #    verbose, verbose_sub = self.get_verbosity(verbose)
    #    print(header_string,"\nGenerating data description", show = verbose)
    #    if timestamp is None:
    #        timestamp = utils.generate_timestamp()
    #    start = timer()
    #    if X is None:
    #        X = self.data.data_X
    #    else:
    #        X = np.array(X)
    #    pars_labels = self.__set_pars_labels(pars_labels)
    #    df = pd.DataFrame(X,columns=pars_labels)
    #    df_description = pd.DataFrame(df.describe())
    #    end = timer()
    #    print("\n"+header_string+"\nData description generated in", str(end-start), "s.\n", show = verbose)
    #    return df_description
#
    #def plot_X_distributions_summary(self,
    #                                 X=None,
    #                                 max_points=None,
    #                                 nbins=50,
    #                                 pars_labels="original", 
    #                                 color="green",
    #                                 figure_file_name=None,
    #                                 show_plot=False,
    #                                 timestamp=None,
    #                                 overwrite=False,
    #                                 verbose=None,
    #                                 **step_kwargs):
    #    """
    #    Plots a summary of all 1D distributions of the parameters in 
    #    the :attr:`Data.data_X <DNNLikelihood.Data.data_X>` dataset.
#
    #    - **Arguments**
#
    #        - **X**
#
    #            X data to use for the plot. If ``None`` is given the 
    #            :attr:`Data.data_X <DNNLikelihood.Data.data_X>` dataset is used.
#
    #                - **type**: ``list`` or ``numpy.ndarray``
    #                - **shape**: ``(npoints,ndims)``
    #                - **default**: ``None``
    #        
    #        - **max_points**
#
    #            Maximum number of points used to make
    #            the plot. If the numnber is smaller than the total
    #            number of available points, then a random subset is taken.
    #            If ``None`` then all available points are used.
#
    #                - **type**: ``int`` or ``None``
    #                - **default**: ``None``
#
    #        - **nbins**
#
    #            Number of bins used to make 
    #            the histograms.
#
    #                - **type**: ``int``
    #                - **default**: ``50``
#
    #        - **pars_labels**
#
    #            Argument that is passed to the :meth:`Data.__set_pars_labels < DNNLikelihood.Data._Lik__set_pars_labels>`
    #            method to set the parameters labels to be used in the plots.
#
    #                - **type**: ``list`` or ``str``
    #                - **shape of list**: ``[]``
    #                - **accepted strings**: ``"original"``, ``"generic"``
    #                - **default**: ``original``
#
    #        - **color**
#
    #            Plot 
    #            color.
#
    #                - **type**: ``str``
    #                - **default**: ``"green"``
#
    #        - **figure_file_name**
#
    #            File name for the generated figure. If it is ``None`` (default),
    #            it is automatically generated.
#
    #                - **type**: ``str`` or ``None``
    #                - **default**: ``None``
#
    #        - **show_plot**
    #        
    #            See :argument:`show_plot <common_methods_arguments.show_plot>`.
#
    #        - **timestamp**
    #        
    #            See :argument:`timestamp <common_methods_arguments.timestamp>`.
#
    #        - **overwrite**
    #        
    #            See :argument:`overwrite <common_methods_arguments.overwrite>`.
#
    #        - **verbose**
    #        
    #            See :argument:`verbose <common_methods_arguments.verbose>`.
#
    #        - **step_kwargs**
#
    #            Additional keyword arguments to pass to the ``plt.step``function.
#
    #                - **type**: ``dict``
#
    #    - **Updates file**
#
    #        - :attr:`Data.output_log_file <DNNLikelihood.Data.output_log_file>`
    #    """
    #    verbose, verbose_sub = self.get_verbosity(verbose)
    #    print(header_string,"\nPlotting 1D distributions summary", show = verbose)
    #    if timestamp is None:
    #        timestamp = utils.generate_timestamp()
    #    plt.style.use(mplstyle_path)
    #    start = timer()
    #    if X is None:
    #        X = self.data.data_X
    #    else:
    #        X = np.array(X)
    #    pars_labels = self.__set_pars_labels(pars_labels)
    #    labels = np.array(pars_labels).tolist()
    #    if max_points is not None:
    #        nnn = np.min([len(X), max_points])
    #    else:
    #        nnn = len(X)
    #    rnd_indices = np.random.choice(np.arange(len(X)),size=nnn,replace=False)
    #    sqrt_n_plots = int(np.ceil(np.sqrt(len(X[1, :]))))
    #    plt.rcParams["figure.figsize"] = (sqrt_n_plots*3, sqrt_n_plots*3)
    #    for i in range(len(X[1,:])):
    #        plt.subplot(sqrt_n_plots,sqrt_n_plots,i+1)
    #        counts, bins = np.histogram(X[rnd_indices,i], nbins)
    #        integral = 1
    #        plt.step(bins[:-1], counts/integral, where='post',color = color,**step_kwargs)
    #        plt.xlabel(pars_labels[i],fontsize=11)
    #        plt.xticks(fontsize=11, rotation=90)
    #        plt.yticks(fontsize=11, rotation=90)
    #        x1,x2,y1,y2 = plt.axis()
    #        plt.tight_layout()
    #    plt.subplots_adjust(top=0.92, bottom=0.08, left=0.10, right=0.95, hspace=0.4, wspace=0.4)
    #    plt.tight_layout()
    #    if figure_file_name is not None:
    #        figure_file_name = self.update_figures(figure_file=figure_file_name,timestamp=timestamp,overwrite=overwrite)
    #    else:
    #        figure_file_name = self.update_figures(figure_file=self.output_figures_base_file_name+"_pars_summary.pdf",timestamp=timestamp,overwrite=overwrite) 
    #    utils.savefig(r"%s" % (path.join(self.output_figures_folder, figure_file_name)), dpi=50)
    #    utils.check_set_dict_keys(self.predictions["Figures"],[timestamp],[[]],verbose=False)
    #    utils.append_without_duplicate(self.predictions["Figures"][timestamp], figure_file_name)
    #    if show_plot:
    #        plt.show()
    #    plt.close()
    #    end = timer()
    #    self.log[utils.generate_timestamp()] = {"action": "saved figure",
    #                           "file name": figure_file_name}
    #    print("\n"+header_string+"\nFigure file\n\t", r"%s" % (figure_file_name), "\ncreated and saved in", str(end-start), "s.\n", show = verbose)
    #    self.save_log(overwrite=overwrite, verbose=verbose_sub)
#
    #def plot_X_distributions(self,
    #                         X=None,
    #                         pars=None,
    #                         max_points=None,
    #                         nbins=50,
    #                         pars_labels="original", 
    #                         color="green",
    #                         figure_file_name=None,
    #                         show_plot=False,
    #                         timestamp=None,
    #                         overwrite=False,
    #                         verbose=None,
    #                         **step_kwargs):
    #    """
    #    Plots 1D distributions of the parameters ``pars`` in 
    #    the :attr:`Data.data_X <DNNLikelihood.Data.data_X>` dataset.
#
    #    - **Arguments**
#
    #        - **X**
#
    #            X data to use for the plot. If ``None`` is given the 
    #            :attr:`Data.data_X <DNNLikelihood.Data.data_X>` dataset is used.
#
    #                - **type**: ``list`` or ``numpy.ndarray``
    #                - **shape**: ``(npoints,ndims)``
    #                - **default**: ``None``
    #        
    #        - **pars**
#
    #            List of parameters 
    #            for which the plots are produced.
#
    #                - **type**: ``list`` or ``None``
    #                - **shape of list**: ``[ ]``
    #                - **default**: ``None``
#
    #        - **max_points**
#
    #            Maximum number of points used to make
    #            the plot. If the numnber is smaller than the total
    #            number of available points, then a random subset is taken.
    #            If ``None`` then all available points are used.
#
    #                - **type**: ``int`` or ``None``
    #                - **default**: ``None``
#
    #        - **nbins**
#
    #            Number of bins used to make 
    #            the histograms.
#
    #                - **type**: ``int``
    #                - **default**: ``50``
#
    #        - **pars_labels**
#
    #            Argument that is passed to the :meth:`Data.__set_pars_labels < DNNLikelihood.Data._Lik__set_pars_labels>`
    #            method to set the parameters labels to be used in the plots.
#
    #                - **type**: ``list`` or ``str``
    #                - **shape of list**: ``[]``
    #                - **accepted strings**: ``"original"``, ``"generic"``
    #                - **default**: ``original``
#
    #        - **color**
#
    #            Plot 
    #            color.
#
    #                - **type**: ``str``
    #                - **default**: ``"green"``
#
    #        - **figure_file_name**
#
    #            File name for the generated figure. If it is ``None`` (default),
    #            it is automatically generated.
#
    #                - **type**: ``str`` or ``None``
    #                - **default**: ``None``
#
    #        - **show_plot**
    #        
    #            See :argument:`show_plot <common_methods_arguments.show_plot>`.
#
    #        - **timestamp**
    #        
    #            See :argument:`timestamp <common_methods_arguments.timestamp>`.
#
    #        - **overwrite**
    #        
    #            See :argument:`overwrite <common_methods_arguments.overwrite>`.
#
    #        - **verbose**
    #        
    #            See :argument:`verbose <common_methods_arguments.verbose>`.
#
    #        - **step_kwargs**
#
    #            Additional keyword arguments to pass to the ``plt.step`` function.
#
    #                - **type**: ``dict``
#
    #    - **Updates file**
#
    #        - :attr:`Data.output_log_file <DNNLikelihood.Data.output_log_file>`
    #    """
    #    verbose, verbose_sub = self.get_verbosity(verbose)
    #    print(header_string,"\nPlotting 1D distributions summary", show = verbose)
    #    if timestamp is None:
    #        timestamp = utils.generate_timestamp()
    #    plt.style.use(mplstyle_path)
    #    start = timer()
    #    if X is None:
    #        X = self.data.data_X
    #    else:
    #        X = np.array(X)
    #    pars_labels = self.__set_pars_labels(pars_labels)
    #    if max_points is not None:
    #        nnn = np.min([len(X), max_points])
    #    else:
    #        nnn = len(X)
    #    rnd_indices = np.random.choice(np.arange(len(X)),size=nnn,replace=False)
    #    for par in pars:
    #        counts, bins = np.histogram(X[rnd_indices,par], nbins)
    #        integral = 1
    #        plt.step(bins[:-1], counts/integral, where='post',color = color,**step_kwargs)
    #        plt.xlabel(pars_labels[par])
    #        plt.ylabel(r"number of samples")
    #        x1,x2,y1,y2 = plt.axis()
    #        plt.tight_layout()
    #        if figure_file_name is not None:
    #            figure_file_name = self.update_figures(figure_file=figure_file_name,timestamp=timestamp,overwrite=overwrite)
    #        else:
    #            figure_file_name = self.update_figures(figure_file=self.output_figures_base_file_name+"_1D_distr_par_"+str(par)+".pdf",timestamp=timestamp,overwrite=overwrite) 
    #        utils.savefig(r"%s" % (path.join(self.output_figures_folder, figure_file_name)), dpi=50)
    #        utils.check_set_dict_keys(self.predictions["Figures"],[timestamp],[[]],verbose=False)
    #        utils.append_without_duplicate(self.predictions["Figures"][timestamp], figure_file_name)
    #        self.log[utils.generate_timestamp()] = {"action": "saved figure",
    #                               "file name": figure_file_name}
    #        end = timer()
    #        print("\n"+header_string+"\nFigure file\n\t", r"%s" % (figure_file_name), "\ncreated and saved in", str(end-start), "s.\n", show = verbose)
    #        if show_plot:
    #            plt.show()
    #        plt.close()
    #    self.save_log(overwrite=overwrite, verbose=verbose_sub)
#
    #def plot_Y_distribution(self,
    #                        Y=None,
    #                        max_points=None,
    #                        log=True,
    #                        nbins=50,
    #                        color="green",
    #                        figure_file_name=None,
    #                        show_plot=False,
    #                        timestamp=None,
    #                        overwrite=False,
    #                        verbose=None,
    #                        **step_kwargs):
    #    """
    #    Plots the distribution of the data in
    #    the :attr:`Data.data_Y <DNNLikelihood.Data.data_Y>` dataset.
#
    #    - **Arguments**
#
    #        - **Y**
#
    #            Y data to use for the plot. If ``None`` is given the 
    #            :attr:`Data.data_Y <DNNLikelihood.Data.data_Y>` dataset is used.
#
    #                - **type**: ``list`` or ``numpy.ndarray``
    #                - **shape**: ``(npoints,ndims)``
    #                - **default**: ``None``
    #                
    #        - **max_points**
#
    #            Maximum number of points used to make
    #            the plot. If the numnber is smaller than the total
    #            number of available points, then a random subset is taken.
    #            If ``None`` then all available points are used.
#
    #                - **type**: ``int`` or ``None``
    #                - **default**: ``None``
#
    #        - **log**
#
    #            If ``True`` the plot is made in
    #            log scale
#
    #                - **type**: ``bool``
    #                - **default**: ``True``
#
    #        - **nbins**
#
    #            Number of bins used to make 
    #            the histograms.
#
    #                - **type**: ``int``
    #                - **default**: ``50``
#
    #        - **color**
#
    #            Plot 
    #            color.
#
    #                - **type**: ``str``
    #                - **default**: ``"green"``
#
    #        - **figure_file_name**
#
    #            File name for the generated figure. If it is ``None`` (default),
    #            it is automatically generated.
#
    #                - **type**: ``str`` or ``None``
    #                - **default**: ``None``
#
    #        - **show_plot**
    #        
    #            See :argument:`show_plot <common_methods_arguments.show_plot>`.
#
    #        - **timestamp**
    #        
    #            See :argument:`timestamp <common_methods_arguments.timestamp>`.
#
    #        - **overwrite**
    #        
    #            See :argument:`overwrite <common_methods_arguments.overwrite>`.
#
    #        - **verbose**
    #        
    #            See :argument:`verbose <common_methods_arguments.verbose>`.
#
    #        - **step_kwargs**
#
    #            Additional keyword arguments to pass to the ``plt.step`` function.
#
    #                - **type**: ``dict``
#
    #    - **Updates file**
#
    #        - :attr:`Data.output_log_file <DNNLikelihood.Data.output_log_file>`
    #    """
    #    verbose, verbose_sub = self.get_verbosity(verbose)
    #    print(header_string, "\nPlotting 1D distributions summary", show = verbose)
    #    if timestamp is None:
    #        timestamp = "datetime_" + \
    #            datetime.now().strftime("%Y_%m_%d_%H_%M_%S_%fZ")[:-3]
    #    plt.style.use(mplstyle_path)
    #    start = timer()
    #    if Y is None:
    #        Y = self.data.data_Y
    #    else:
    #        Y = np.array(Y)
    #    if max_points is not None:
    #        nnn = np.min([len(Y), max_points])
    #    else:
    #        nnn = len(Y)
    #    rnd_indices = np.random.choice(np.arange(len(Y)), size=nnn, replace=False)
    #    counts, bins = np.histogram(Y[rnd_indices], nbins)
    #    integral = 1
    #    plt.step(bins[:-1], counts/integral, where='post', color=color, **step_kwargs)
    #    plt.xlabel(r"Y data")
    #    plt.ylabel(r"number of samples")
    #    x1, x2, y1, y2 = plt.axis()
    #    if log:
    #        plt.yscale('log')
    #    plt.tight_layout()
    #    if figure_file_name is not None:
    #        figure_file_name = self.update_figures(figure_file=figure_file_name, timestamp=timestamp, overwrite=overwrite)
    #    else:
    #        figure_file_name = self.update_figures(figure_file=self.output_figures_base_file_name+"_Y_distribution.pdf", timestamp=timestamp, overwrite=overwrite)
    #    utils.savefig(r"%s" % (path.join(self.output_figures_folder, figure_file_name)), dpi=50)
    #    utils.check_set_dict_keys(self.predictions["Figures"], [timestamp], [[]], verbose=False)
    #    utils.append_without_duplicate(
    #        self.predictions["Figures"][timestamp], figure_file_name)
    #    self.log[utils.generate_timestamp()] = {"action": "saved figure",
    #                           "file name": figure_file_name}
    #    end = timer()
    #    print("\n"+header_string+"\nFigure file\n\t", r"%s" % (figure_file_name),"\ncreated and saved in", str(end-start), "s.\n", show = verbose)
    #    if show_plot:
    #        plt.show()
    #    plt.close()
    #    self.save_log(overwrite=overwrite, verbose=verbose_sub)
#
    #def plot_corners_1samp(self,
    #                       X=None,
    #                       intervals=self.inference.CI_from_sigma([1, 2, 3]), 
    #                       weights=None, 
    #                       pars=None, 
    #                       max_points=None, 
    #                       nbins=50, 
    #                       pars_labels="original",
    #                       ranges_extend=None, 
    #                       title = "", 
    #                       color="green",
    #                       plot_title="Corner plot", 
    #                       legend_labels=None, 
    #                       figure_file_name=None, 
    #                       show_plot=False, 
    #                       timestamp=None, 
    #                       overwrite=False, 
    #                       verbose=None, 
    #                       **corner_kwargs):
    #    """
    #    Plots the 1D and 2D distributions (corner plot) of the distribution of the parameters ``pars`` in the ``X`` array.
#
    #    - **Arguments**
#
    #        - **X**
#
    #            X data to use for the plot. If ``None`` is given the 
    #            :attr:`Data.data_X <DNNLikelihood.Data.data_X>` dataset is used.
#
    #                - **type**: ``list`` or ``numpy.ndarray``
    #                - **shape**: ``(npoints,ndims)``
    #                - **default**: ``None``
    #                
    #        - **intervals**
#
    #            Probability intervals for which 
    #            contours are drawn.
#
    #                - **type**: ``list`` or ``numpy.ndarray``
    #                - **shape**: ``(nintervals,)``
    #                - **default**: ``numpy.array([0.68268949, 0.95449974, 0.9973002])`` (corresponding to 1,2, and 3 sigmas for a 1D Gaussian distribution)
#
    #        - **weights**
#
    #            List or |Numpy_link| array with the 
    #            Weights correspomnding to the ``X`` points
#
    #                - **type**: ``list`` or ``numpy.ndarray`` or ``None``
    #                - **shape**: ``(npoints,)``
    #                - **default**: ``None``
#
    #        - **pars**
#
    #            List of parameters 
    #            for which the plots are produced.
#
    #                - **type**: ``list`` or ``None``
    #                - **shape of list**: ``[ ]``
    #                - **default**: ``None``
#
    #        - **max_points**
#
    #            Maximum number of points used to make
    #            the plot. If the numnber is smaller than the total
    #            number of available points, then a random subset is taken.
    #            If ``None`` then all available points are used.
#
    #                - **type**: ``int`` or ``None``
    #                - **default**: ``None``
#
    #        - **nbins**
#
    #            Number of bins used to make 
    #            the histograms.
#
    #                - **type**: ``int``
    #                - **default**: ``50``
#
    #        - **pars_labels**
#
    #            Argument that is passed to the :meth:`Data.__set_pars_labels < DNNLikelihood.Data._Lik__set_pars_labels>`
    #            method to set the parameters labels to be used in the plots.
#
    #                - **type**: ``list`` or ``str``
    #                - **shape of list**: ``[]``
    #                - **accepted strings**: ``"original"``, ``"generic"``
    #                - **default**: ``original``
#
    #        - **ranges_extend**
#
    #            Percent increase or reduction of the range of the plots with respect
    #            to the range automatically determined from the points values.
#
    #                - **type**: ``int`` or ``float`` or ``None``
#
    #        - **title**
#
    #            Subplot title to which the 
    #            68% HPDI values are appended.
#
    #                - **type**: ``str`` or ``None``
    #                - **default**: ``None``
#
    #        - **color**
#
    #            Plot 
    #            color.
#
    #                - **type**: ``str``
    #                - **default**: ``"green"``
#
    #        - **plot_title**
#
    #            Title of the corner 
    #            plot.
#
    #                - **type**: ``str``
    #                - **default**: ``"Corner plot"``
#
    #        - **legend_labels**
#
    #            List of strings. Labels for the contours corresponding to the 
    #            ``intervals`` to show in the legend.
    #            If ``None`` the legend automatically reports the intervals.
#
    #                - **type**: ``str`` or ``None``
    #                - **default**: ``None``
#
    #        - **figure_file_name**
#
    #            File name for the generated figure. If it is ``None`` (default),
    #            it is automatically generated.
#
    #                - **type**: ``str`` or ``None``
    #                - **default**: ``None``
#
    #        - **show_plot**
    #        
    #            See :argument:`show_plot <common_methods_arguments.show_plot>`.
#
    #        - **timestamp**
    #        
    #            See :argument:`timestamp <common_methods_arguments.timestamp>`.
#
    #        - **overwrite**
    #        
    #            See :argument:`overwrite <common_methods_arguments.overwrite>`.
#
    #        - **verbose**
    #        
    #            See :argument:`verbose <common_methods_arguments.verbose>`.
#
    #        - **corner_kwargs**
#
    #            Additional keyword arguments to pass to the ``corner`` function.
#
    #                - **type**: ``dict``
#
    #    - **Updates file**
#
    #        - :attr:`Data.output_log_file <DNNLikelihood.Data.output_log_file>`
    #    """
    #    verbose, verbose_sub = self.get_verbosity(verbose)
    #    print(header_string,"\nPlotting 2D marginal posterior density", show = verbose)
    #    if timestamp is None:
    #        timestamp = utils.generate_timestamp()
    #    if legend_labels is not None:
    #        if len(legend_labels) != len(intervals):
    #            raise Exception("Legend labels should either be None or a list of strings with the same length as intervals.")
    #    plt.style.use(mplstyle_path)
    #    start = timer()
    #    if X is None:
    #        X = self.data.data_X
    #    else:
    #        X = np.array(X)
    #    weigths = np.array(weights)
    #    if title is None:
    #        title = ""
    #    linewidth = 1.3
    #    if ranges_extend is None:
    #        ranges = extend_corner_range(X, X, pars, 0)
    #    else:
    #        ranges = extend_corner_range(X, X, pars, ranges_extend)
    #    pars_labels = self.__set_pars_labels(pars_labels)
    #    labels = np.array(pars_labels)[pars].tolist()
    #    nndims = len(pars)
    #    if max_points is not None:
    #        if type(max_points) == list:
    #            nnn = np.min([len(X), max_points[0]])
    #        else:
    #            nnn = np.min([len(X), max_points])
    #    else:
    #        nnn = len(X)
    #    rnd_idx = np.random.choice(np.arange(len(X)), nnn, replace=False)
    #    samp = X[rnd_idx][:,pars]
    #    if weights is not None:
    #        weights = weights[rnd_idx]
    #    print("Computing HPDIs.\n", show = verbose)
    #    HPDI = [self.inference.HPDI(samp[:,i], intervals = intervals, weights=weights, nbins=nbins, print_hist=False, optimize_binning=False) for i in range(nndims)]
    #    levels = np.array([[np.sort(self.inference.HPD_quotas(samp[:,[i,j]], nbins=nbins, intervals = intervals, weights=weights)).tolist() for j in range(nndims)] for i in range(nndims)])
    #    corner_kwargs_default = {"labels":  [r"%s" % s for s in labels],
    #                             "max_n_ticks": 6, 
    #                             "color": color,
    #                             "plot_contours": True,
    #                             "smooth": True, 
    #                             "smooth1d": True,
    #                             "range": ranges,
    #                             "plot_datapoints": True, 
    #                             "plot_density": False, 
    #                             "fill_contours": False, 
    #                             "normalize1d": True,
    #                             "hist_kwargs": {"color": color, "linewidth": "1.5"}, 
    #                             "label_kwargs": {"fontsize": 16}, 
    #                             "show_titles": False,
    #                             "title_kwargs": {"fontsize": 18}, 
    #                             "levels_lists": levels,
    #                             "data_kwargs": {"alpha": 1},
    #                             "contour_kwargs": {"linestyles": ["dotted", "dashdot", "dashed"][:len(HPDI[0])], "linewidths": [linewidth, linewidth, linewidth][:len(HPDI[0])]},
    #                             "no_fill_contours": False, 
    #                             "contourf_kwargs": {"colors": ["white", "lightgreen", color], "alpha": 1}}
    #    corner_kwargs_default = utils.dic_minus_keys(corner_kwargs_default, list(corner_kwargs.keys()))
    #    corner_kwargs = {**corner_kwargs,**corner_kwargs_default}
    #    fig, axes = plt.subplots(nndims, nndims, figsize=(3*nndims, 3*nndims))
    #    figure = corner(samp, bins=nbins, weights=weights, fig=fig, **corner_kwargs)
    #                    # , levels=(0.393,0.68,)) ,levels=[300],levels_lists=levels1)#,levels=[120])
    #    #figure = corner(samp, bins=nbins, weights=weights, labels=[r"%s" % s for s in labels],
    #    #                fig=fig, max_n_ticks=6, color=color, plot_contours=True, smooth=True,
    #    #                smooth1d=True, range=ranges, plot_datapoints=True, plot_density=False,
    #    #                fill_contours=False, normalize1d=True, hist_kwargs={"color": color, "linewidth": "1.5"},
    #    #                label_kwargs={"fontsize": 16}, show_titles=False, title_kwargs={"fontsize": 18},
    #    #                levels_lists=levels, data_kwargs={"alpha": 1},
    #    #                contour_kwargs={"linestyles": ["dotted", "dashdot", "dashed"][:len(
    #    #                    HPDI[0])], "linewidths": [linewidth, linewidth, linewidth][:len(HPDI[0])]},
    #    #                no_fill_contours=False, contourf_kwargs={"colors": ["white", "lightgreen", color], "alpha": 1}, **kwargs)
    #    #                # , levels=(0.393,0.68,)) ,levels=[300],levels_lists=levels1)#,levels=[120])
    #    axes = np.array(figure.axes).reshape((nndims, nndims))
    #    lines_array = list(matplotlib.lines.lineStyles.keys())
    #    linestyles = (lines_array[0:4]+lines_array[0:4]+lines_array[0:4])[0:len(intervals)]
    #    intervals_str = [r"${:.2f}".format(i*100)+"\%$ HPDI" for i in intervals]
    #    for i in range(nndims):
    #        title_i = ""
    #        ax = axes[i, i]
    #        #ax.axvline(value1[i], color="green",alpha=1)
    #        #ax.axvline(value2[i], color="red",alpha=1)
    #        ax.grid(True, linestyle="--", linewidth=1, alpha=0.3)
    #        ax.tick_params(axis="both", which="major", labelsize=16)
    #        hists_1d = get_1d_hist(i, samp, nbins=nbins, ranges=ranges, weights=weights, normalize1d=True)[0]  # ,intervals=HPDI681)
    #        for q in range(len(intervals)):
    #            for j in HPDI[i][intervals[q]]["Intervals"]:
    #                ax.axvline(hists_1d[0][hists_1d[0] >= j[0]][0], color=color, alpha=1, linestyle=linestyles[q], linewidth=linewidth)
    #                ax.axvline(hists_1d[0][hists_1d[0] <= j[1]][-1], color=color, alpha=1, linestyle=linestyles[q], linewidth=linewidth)
    #            title_i = r"%s"%title + ": ["+"{0:1.2e}".format(HPDI[i][intervals[0]]["Intervals"][0][0])+","+"{0:1.2e}".format(HPDI[i][intervals[0]]["Intervals"][0][1])+"]"
    #        if i == 0:
    #            x1, x2, _, _ = ax.axis()
    #            ax.set_xlim(x1*1.3, x2)
    #        ax.set_title(title_i, fontsize=10)
    #    for yi in range(nndims):
    #        for xi in range(yi):
    #            ax = axes[yi, xi]
    #            if xi == 0:
    #                x1, x2, _, _ = ax.axis()
    #                ax.set_xlim(x1*1.3, x2)
    #            ax.grid(True, linestyle="--", linewidth=1)
    #            ax.tick_params(axis="both", which="major", labelsize=16)
    #    fig.subplots_adjust(top=0.85,wspace=0.25, hspace=0.25)
    #    fig.suptitle(r"%s" % (plot_title), fontsize=26)
    #    #fig.text(0.5 ,1, r"%s" % plot_title, fontsize=26)
    #    colors = [color, "black", "black", "black"]
    #    red_patch = matplotlib.patches.Patch(color=colors[0])  # , label="The red data")
    #    #blue_patch = matplotlib.patches.Patch(color=colors[1])  # , label="The blue data")
    #    lines = [matplotlib.lines.Line2D([0], [0], color=colors[1], linewidth=3, linestyle=l) for l in linestyles]
    #    if legend_labels is None:
    #        legend_labels = [intervals_str[i] for i in range(len(intervals))]
    #    fig.legend(lines, legend_labels, fontsize=int(7+2*nndims), loc="upper right")#(1/nndims*1.05,1/nndims*1.1))#transform=axes[0,0].transAxes)# loc=(0.53, 0.8))
    #    #plt.tight_layout()
    #    if figure_file_name is not None:
    #        figure_file_name = self.update_figures(figure_file=figure_file_name,timestamp=timestamp,overwrite=overwrite)
    #    else:
    #        figure_file_name = self.update_figures(figure_file=self.output_figures_base_file_name+"_corner_posterior_1samp_pars_" + "_".join([str(i) for i in pars]) +".pdf",timestamp=timestamp,overwrite=overwrite) 
    #    utils.savefig(r"%s" % (path.join(self.output_figures_folder, figure_file_name)), dpi=50)
    #    utils.check_set_dict_keys(self.predictions["Figures"],[timestamp],[[]],verbose=False)
    #    utils.append_without_duplicate(self.predictions["Figures"][timestamp], figure_file_name)
    #    if show_plot:
    #        plt.show()
    #    plt.close()
    #    end = timer()
    #    timestamp = utils.generate_timestamp()
    #    self.log[utils.generate_timestamp()] = {"action": "saved figure",
    #                           "file name": figure_file_name}
    #    print("\n"+header_string+"\nFigure file\n\t", r"%s" % (figure_file_name), "\ncreated and saved in", str(end-start), "s.\n", show = verbose)
#
    #def plot_correlation_matrix(self, 
    #                            X=None,
    #                            pars_labels="original",
    #                            title = None,
    #                            figure_file_name=None, 
    #                            show_plot=False, 
    #                            timestamp=None, 
    #                            overwrite=False, 
    #                            verbose=None, 
    #                            **matshow_kwargs):
    #    """
    #    Plots the correlation matrix of the  
    #    :attr:`Data.data_X <DNNLikelihood.Data.data_X>` dataset.
#
    #    - **Arguments**
#
    #        - **X**
#
    #            X data to use for the plot. If ``None`` is given the 
    #            :attr:`Data.data_X <DNNLikelihood.Data.data_X>` dataset is used.
#
    #                - **type**: ``list`` or ``numpy.ndarray``
    #                - **shape**: ``(npoints,ndims)``
    #                - **default**: ``None``
    #                
    #        - **pars_labels**
#
    #            Argument that is passed to the :meth:`Data.__set_pars_labels < DNNLikelihood.Data._Lik__set_pars_labels>`
    #            method to set the parameters labels to be used in the plots.
#
    #                - **type**: ``list`` or ``str``
    #                - **shape of list**: ``[]``
    #                - **accepted strings**: ``"original"``, ``"generic"``
    #                - **default**: ``original``
#
    #        - **title**
#
    #            Subplot title to which the 
    #            68% HPDI values are appended.
#
    #                - **type**: ``str`` or ``None``
    #                - **default**: ``None``
#
    #        - **figure_file_name**
#
    #            File name for the generated figure. If it is ``None`` (default),
    #            it is automatically generated.
#
    #                - **type**: ``str`` or ``None``
    #                - **default**: ``None``
#
    #        - **show_plot**
    #        
    #            See :argument:`show_plot <common_methods_arguments.show_plot>`.
#
    #        - **timestamp**
    #        
    #            See :argument:`timestamp <common_methods_arguments.timestamp>`.
#
    #        - **overwrite**
    #        
    #            See :argument:`overwrite <common_methods_arguments.overwrite>`.
#
    #        - **verbose**
    #        
    #            See :argument:`verbose <common_methods_arguments.verbose>`.
#
    #        - **step_kwargs**
#
    #            Additional keyword arguments to pass to the ``plt.matshow`` function.
#
    #                - **type**: ``dict``
#
    #    - **Updates file**
#
    #        - :attr:`Data.output_log_file <DNNLikelihood.Data.output_log_file>`
    #    """
    #    verbose, verbose_sub = self.get_verbosity(verbose)
    #    print(header_string,"\nPlotting X data correlation matrix", show = verbose)
    #    if timestamp is None:
    #        timestamp = utils.generate_timestamp()
    #    plt.style.use(mplstyle_path)
    #    start = timer()
    #    if title is None:
    #        title = "Correlation Matrix"
    #    if X is None:
    #        X = self.data.data_X
    #    else:
    #        X = np.array(X)
    #    pars_labels = self.__set_pars_labels(pars_labels)
    #    df = pd.DataFrame(X)
    #    f = plt.figure(figsize=(18, 18))
    #    plt.matshow(df.corr(), fignum=f.number, **matshow_kwargs)
    #    plt.xticks(range(df.select_dtypes(['number']).shape[1]), pars_labels, fontsize=10, rotation=45)
    #    plt.yticks(range(df.select_dtypes(['number']).shape[1]), pars_labels, fontsize=10)
    #    cb = plt.colorbar()
    #    cb.ax.tick_params(labelsize=11)
    #    plt.title('Correlation Matrix', fontsize=13)
    #    plt.grid(False)
    #    if figure_file_name is not None:
    #        figure_file_name = self.update_figures(figure_file=figure_file_name,timestamp=timestamp,overwrite=overwrite)
    #    else:
    #        figure_file_name = self.update_figures(figure_file=self.output_figures_base_file_name+"_correlation_matrix.pdf",timestamp=timestamp,overwrite=overwrite) 
    #    utils.savefig(r"%s" % (path.join(self.output_figures_folder, figure_file_name)), dpi=50)
    #    utils.check_set_dict_keys(self.predictions["Figures"],[timestamp],[[]],verbose=False)
    #    utils.append_without_duplicate(self.predictions["Figures"][timestamp], figure_file_name)
    #    if show_plot:
    #        plt.show()
    #    plt.close()
    #    end = timer()
    #    timestamp = utils.generate_timestamp()
    #    self.log[utils.generate_timestamp()] = {"action": "saved figure",
    #                           "file name": figure_file_name}
    #    print("\n"+header_string+"\nFigure file\n\t", r"%s" % (figure_file_name), "\ncreated and saved in", str(end-start), "s.\n", show = verbose)

class DataManager(Verbosity):
    """
    Manages input data
    """
    def __init__(self,
                 data_main: DataMain,
                 npoints: Optional[List[int]] = None, # list with [n_train, n_val, n_test]
                 preprocessing: Optional[List[bool]] = None, # list with [scalerX_bool, scalerY_bool, rotationX_bool]s
                 seed: Optional[int] = None,
                 verbose: Optional[IntBool] = None
                ) -> None:
        """
        """
        # Attributes type declatation
        self._idx_train: npt.NDArray[np.int_]
        self._idx_val: npt.NDArray[np.int_]
        self._idx_test: npt.NDArray[np.int_]
        self._ndims: int
        self._npoints_available: int
        self._npoints_required: int
        self._npoints_test: int
        self._npoints_train: int
        self._npoints_val: int
        self._rotationX: npt.NDArray[np.float_]
        self._rotationX_bool: bool
        self._seed: int
        self._scalerX: StandardScaler
        self._scalerX_bool: bool
        self._scalerY: StandardScaler
        self._scalerY_bool: bool
        self._test_fraction: float
        self._train_val_range: range
        self._test_range: range
        self.DataMain: DataMain
        self.TestData: DataSamples
        self.TrainData: DataSamples
        self.ValData: DataSamples
        # Initialize parent Verbosity class
        super().__init__(verbose)
        # Set verbosity
        verbose, verbose_sub = self.get_verbosity(verbose)
        # Initialize object
        print(header_string_1,"\nInitializing DataManager.\n", show = verbose)
        self.DataMain = data_main
        self._ndims = self.DataMain.InputData._ndims
        self.__check_set_npoints(npoints = npoints)
        self.__check_set_preprocessing(preprocessing = preprocessing)
        self.__define_test_fraction()
        self._seed = seed if seed is not None else 0
        self.__init_train_data(verbose = verbose_sub)
        self.__init_val_data(verbose = verbose_sub)
        self.__init_test_data(verbose = verbose_sub)

    @property
    def excluded_attributes(self) -> list:
        tmp = ["excluded_attributes",
               "_idx_train",
               "_idx_val",
               "_idx_test",
               "_test_range",
               "_train_val_range",
               "_verbose",
               "_rotationX",
               "_scalerX",
               "_scalerY",
               "DataMain",
               "TrainData",
               "ValData",
               "TestData",]
        return tmp

    @property
    def idx_test(self) -> npt.NDArray[np.int_]:
        return self._idx_test

    @property
    def idx_train(self) -> npt.NDArray[np.int_]:
        return self._idx_train

    @property
    def idx_val(self) -> npt.NDArray[np.int_]:
        return self._idx_val

    @property
    def ndims(self) -> int:
        return self._ndims

    @property
    def npoints_available(self) -> int:
        return self._npoints_available

    @property
    def npoints_required(self) -> int:
        return self._npoints_required

    @property
    def npoints_test(self) -> int:
        return self._npoints_test

    @property
    def npoints_train(self) -> int:
        return self._npoints_train

    @property
    def npoints_val(self) -> int:
        return self._npoints_val

    @property
    def rotationX(self) -> npt.NDArray[np.float_]:
        return self._rotationX

    @property
    def rotationX_bool(self) -> bool:
        return self._rotationX_bool

    @property
    def seed(self) -> int:
        return self._seed

    @property
    def scalerX(self) -> StandardScaler:
        return self._scalerX

    @property
    def scalerX_bool(self) -> bool:
        return self._scalerX_bool

    @property
    def scalerY(self) -> StandardScaler:
        return self._scalerY

    @property
    def scalerY_bool(self) -> bool:
        return self._scalerY_bool

    @property
    def test_fraction(self) -> float:
        return self._test_fraction

    @property
    def test_range(self) -> range:
        return self._test_range

    @property
    def train_val_range(self) -> range:
        return self._train_val_range

    def __define_test_fraction(self) -> None:
        """ 
        """
        self._test_fraction = self._npoints_test/(self._npoints_train+self._npoints_val)
        self._train_val_range = range(int(round(self.DataMain.InputData._npoints*(1-self.test_fraction))))
        self._test_range = range(int(round(self.DataMain.InputData._npoints*(1-self.test_fraction))),self.DataMain.InputData._npoints)        

    def __check_set_npoints(self,
                        npoints: Optional[List[int]] = None, # list with [n_train, n_val, n_test]
                        verbose: Optional[IntBool] = None
                       ) -> None:
        verbose, _ = self.get_verbosity(verbose)
        if npoints is None:
            npoints = [0,0,0]
        if len(npoints) != 3:
            raise InvalidInput("The 'npoints' argument should be a list of three integers of the form [n_train, n_val, n_test].")
        self._npoints_required = np.sum(npoints)
        self._npoints_available = self.DataMain.InputData._npoints
        if self._npoints_required > self._npoints_available:
            raise InvalidInput("The total requires number of points is larger than the available points (",self._npoints_available,").")
        self._npoints_test = npoints[2]
        self._npoints_train = npoints[0]
        self._npoints_val = npoints[1]
        self.DataMain._log[utils.generate_timestamp()] = {"action": "set npoints",
                                                          "npoints train": [self._npoints_train],
                                                          "npoints val": [self._npoints_train],
                                                          "npoints test": [self._npoints_val]}
        print(header_string_2,"\nSet the number of required points: train (",self._npoints_train,"); val (",self._npoints_val,"); test (",self._npoints_test,").\n", show = verbose)

    def __check_set_preprocessing(self,
                                  preprocessing: Optional[List[bool]] = None, # list with [scalerX_bool, scalerY_bool, rotationX_bool]
                                  verbose: Optional[IntBool] = None
                                 ) -> None:
        verbose, _ = self.get_verbosity(verbose)
        if preprocessing is None:
            preprocessing = [False,False,False]
        if len(preprocessing) != 3:
            raise InvalidInput("The 'preprocessing' argument should be a list of three boolean of the form [scalerX_bool, scalerY_bool, rotationX_bool].")
        self._scalerX_bool = preprocessing[0]
        self._scalerY_bool = preprocessing[1]
        self._rotationX_bool = preprocessing[2]
        print(header_string_2,"\nSet the preprocessing flags: scalerX_bool (",self._scalerX_bool,"); scalerY_bool (",self._scalerY_bool,"); rotationX_bool (",self._rotationX_bool,").\n", show = verbose)
        self.__init_preprocessing(verbose = verbose)

    def __init_preprocessing(self,
                             verbose: Optional[IntBool] = None) -> None:
        """
        """
        verbose, _ = self.get_verbosity(verbose)
        print(header_string_2,"\nInitializing preprocessing objects: scalerX, scalerY, rotationX.\n", show = verbose)
        self._scalerX = StandardScaler(with_mean=self._scalerX_bool, with_std=self._scalerX_bool)
        self._scalerY = StandardScaler(with_mean=self._scalerY_bool, with_std=self._scalerY_bool)
        self._rotationX = np.identity(self._ndims)

    def __init_train_data(self, 
                          verbose: Optional[IntBool] = None
                         ) -> None:
        verbose, _ = self.get_verbosity(verbose)
        print(header_string_2,"\nInitializing TrainData object.\n", show = verbose)
        self._idx_train = np.array([],dtype=np.int_)
        self.TrainData = DataSamples(dtype = [self.DataMain.InputData.dtype_stored,
                                            self.DataMain.InputData.dtype_required],
                                   verbose = False)
        
    def __init_val_data(self, 
                        verbose: Optional[IntBool] = None
                       ) -> None:
        verbose, _ = self.get_verbosity(verbose)
        print(header_string_2,"\nInitializing TrainData object.\n", show = verbose)
        self._idx_val = np.array([],dtype=np.int_)
        self.ValData = DataSamples(dtype = [self.DataMain.InputData.dtype_stored,
                                          self.DataMain.InputData.dtype_required],
                                 verbose = False)
        
    def __init_test_data(self, verbose: Optional[IntBool] = None) -> None:
        verbose, verbose_sub = self.get_verbosity(verbose)
        print(header_string_2,"\nInitializing TestData object.\n", show = verbose)
        self._idx_test = np.array([],dtype=np.int_)
        self.TestData = DataSamples(dtype = [self.DataMain.InputData.dtype_stored,
                                           self.DataMain.InputData.dtype_required],
                                  verbose = False)

    def compute_sample_weights(self, 
                               data_X: npt.NDArray,
                               nbins: int = 100, 
                               power: float = 1., 
                               verbose: Optional[IntBool] = None
                              ) -> npt.NDArray:
        """
        Method that computes weights of points given their distribution. Sample weights are used to weigth data as a function of 
        their frequency, obtained by binning the data in ``nbins`` and assigning weight equal to the inverse of the bin count
        to the power ``power``. In order to avoid too large weights for bins with very few counts, all bins with less than 5 counts
        are assigned frequency equal to 1/5 to the power ``power``.
        
        - **Arguments**

            - **data_X**

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
        verbose, verbose_sub = self.get_verbosity(verbose)
        start = timer()
        print(header_string_2, "\nComputing sample weights\n", show = verbose)
        hist, edges = np.histogram(data_X, bins=nbins)
        hist = np.where(hist < 5, 5, hist)
        tmp = np.digitize(data_X, edges, right=True)
        W = 1/np.power(hist[np.where(tmp == nbins, nbins-1, tmp)], power)
        W = W/np.sum(W)*len(data_X)
        end = timer()
        #self.log[utils.generate_timestamp()] = {"action": "computed data_X weights"}
        #self.save_log(overwrite=True, verbose=verbose_sub)
        print(header_string_2,"\nSample weights computed in", end-start, "s.\n", show = verbose)
        return W

    def define_data(self,
                    idx: npt.NDArray[np.int_],
                    verbose: Optional[IntBool] = None
                   ) -> List[npt.NDArray[np.float_]]:
        """
        """
        verbose, verbose_sub = self.get_verbosity(verbose)
        npoints = len(idx)
        if npoints > 0:
            data_X = np.array(self.DataMain.InputData._data_X[idx]).astype(self.DataMain.InputData._dtype_required)
            data_Y = np.array(self.DataMain.InputData._data_Y[idx]).astype(self.DataMain.InputData._dtype_required)
            print(header_string_2,"\nLoaded required data from dataset.\n", show = verbose)
        else:
            data_X = np.array([[]],dtype=np.float_)
            data_Y = np.array([],dtype=np.float_)
        return [data_X, data_Y]

    def define_rotationX(self, 
                         data_X: npt.NDArray,
                         rotationX_bool: IntBool = False,
                         verbose: Optional[IntBool] = None
                        ) -> npt.NDArray:
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
        _, verbose_sub = self.get_verbosity(verbose)
        start = timer()
        if rotationX_bool:
            cov_matrix = np.cov(data_X, rowvar=False)
            w, V = np.linalg.eig(cov_matrix)
        else:
            V = np.identity(len(data_X[0]))
        end = timer()
        self.DataMain._log[utils.generate_timestamp()] = {"action": "defined covariance rotation matrix",
                                                          "rotationX_bool": rotationX_bool}
        print(header_string_2, "\nMatrix that rotates the correlation matrix defined in",end-start, "s.\n", show = verbose)
        return V

    def define_scalers(self, 
                       data_X: npt.NDArray,
                       data_Y: npt.NDArray,
                       scalerX_bool: IntBool = False,
                       scalerY_bool: IntBool = False,
                       verbose: Optional[IntBool] = None
                      ) -> List[StandardScaler]:
        """
        Method that defines |standard_scalers_link| fit to the ``data_X`` and ``data_Y`` data.
        Scalers are defined based on the boolean flags ``scalerX_bool`` and ``scalerY_bool``. When the flags are ``False``
        the scalers are defined with the arguemtns ``with_mean=False`` and ``with_std=False`` which correspond to identity
        transformations.

        - **Arguments**

            - **data_X**

                ``X`` data to fit
                the |standard_scaler_link| ``scalerX``.

                    - **type**: ``numpy.ndarray``
                    - **shape**: ``(npoints,ndim)``

            - **data_Y**

                ``Y`` data to fit
                the |standard_scaler_link| ``scalerY``.

                    - **type**: ``numpy.ndarray``
                    - **shape**: ``(npoints,)``

            - **scalerX_bool**

                If ``True`` the ``X`` scaler is fit to the ``data_X`` data, otherwise it is set
                to the identity transformation.

                    - **type**: ``bool``

            - **scalerY_bool**

                If ``True`` the ``Y`` scaler is fit to the ``data_Y`` data, otherwise it is set
                to the identity transformation.

                    - **type**: ``bool``

            - **verbose**
            
                See :argument:`verbose <common_methods_arguments.verbose>`.

        - **Returns**

            List of the form ``[scalerX,scalerY]`` containing 
            the ``X`` and ``Y`` scalers.
            
                - **type**: ``list``
                - **shape**: ``[scalerX,scalerY]``
        """
        _, verbose_sub = self.get_verbosity(verbose)
        start = timer()
        if scalerX_bool:
            scalerX = StandardScaler(with_mean=True, with_std=True)
            scalerX.fit(data_X)
        else:
            scalerX = StandardScaler(with_mean=False, with_std=False)
            scalerX.fit(data_X)
        if scalerY_bool:
            scalerY = StandardScaler(with_mean=True, with_std=True)
            scalerY.fit(data_Y.reshape(-1, 1))
        else:
            scalerY = StandardScaler(with_mean=False, with_std=False)
            scalerY.fit(data_Y.reshape(-1, 1))
        end = timer()
        self.DataMain._log[utils.generate_timestamp()] = {"action": "defined standard scalers",
                                                          "scaler X": scalerX_bool,
                                                          "scaler Y": scalerY_bool}
        print(header_string_2,"\nStandard scalers defined in", end-start, "s.\n", show = verbose)
        return [scalerX, scalerY]

    def generate_test_data(self, 
                           verbose: Optional[IntBool] = None
                          ) -> None:
        verbose, verbose_sub = self.get_verbosity(verbose)
        start = timer()
        if len(self._idx_test) == self._npoints_test:
            print(header_string_2,"\nLoading test data corresponding to existing indices.\n", show = verbose)
            action = "Loaded existing test data"
        else:
            print(header_string_2,"\nGenerating test data corresponding to randomly generated indices.\n", show = verbose)
            np.random.seed(self._seed)
            idx_test = np.random.choice(self._test_range, self._npoints_test, replace=False)
            self._idx_test = np.sort(idx_test)
            action = "Generated new test data"
        [self.TestData._data_X, self.TestData._data_Y] = self.define_data(self._idx_test)
        end = timer()
        self.DataMain._log[utils.generate_timestamp()] = {"action": action,
                                                          "npoints test": self._npoints_test}
        print(header_string_2,"\nGenerated/loaded", str(self._npoints_test), "(X_test, Y_test) data in", end-start,"s.\n", show = verbose)

    def generate_train_val_data(self, 
                                verbose: Optional[IntBool] = None
                               ) -> None:
        verbose, verbose_sub = self.get_verbosity(verbose)
        start = timer()
        if len(self._idx_train) == self._npoints_train and len(self._idx_val) == self._npoints_val:
            print(header_string_2,"\nLoading train/val data corresponding to existing indices.\n", show = verbose)
            action = "Loaded existing train/val data"
        else:
            print(header_string_2,"\nGenerating train/val data corresponding to randomly generated indices.\n", show = verbose)
            np.random.seed(self._seed)
            idx_train = np.random.choice(self._train_val_range, self._npoints_train+self._npoints_val, replace=False)
            idx_train, idx_val = train_test_split(idx_train, train_size=self._npoints_train, test_size=self._npoints_val, shuffle=False)
            self._idx_train = np.sort(idx_train)
            self._idx_val = np.sort(idx_val)
            action = "Generated new train/val data"
        [self.TrainData._data_X, self.TrainData._data_Y] = self.define_data(self._idx_train)
        [self.ValData._data_X, self.ValData._data_Y] = self.define_data(self._idx_val)
        end = timer()
        self.DataMain._log[utils.generate_timestamp()] = {"action": action,
                                                          "npoints train": self._npoints_train,
                                                          "npoints val": self._npoints_val}
        print(header_string_2,"\nGenerated/loaded", str(self._npoints_train), "(X_train, Y_train) data and ", str(self._npoints_val),"(X_val, Y_val) data in", end-start,"s.\n", show = verbose)
        [self._scalerX, self._scalerY] = self.define_scalers(data_X = self.TrainData._data_X,
                                                             data_Y = self.TrainData._data_Y,
                                                             scalerX_bool = self._scalerX_bool,
                                                             scalerY_bool = self._scalerY_bool,
                                                             verbose = verbose_sub)
        self._rotationX = self.define_rotationX(data_X = self.TrainData._data_X,
                                                rotationX_bool = self._rotationX_bool,
                                                verbose = verbose_sub)

    def inverse_transform_data_X(self, 
                                 data_X: npt.NDArray,
                                 rotation_X: npt.NDArray,
                                 scaler_X: StandardScaler
                                ) -> npt.NDArray:
        """
        Inverse of the method
        :meth:`Data.transform_data_X <NF4HEP.Data.transform_data_X>`.
        """
        return np.dot(scaler_X.inverse_transform(data_X), np.transpose(rotation_X))

    def transform_data_X(self, 
                         data_X: npt.NDArray,
                         rotation_X: npt.NDArray,
                         scaler_X: StandardScaler
                        ) -> npt.NDArray:
        """
        Method that transforms X data applying first the rotation 
        :attr:`NF.rotationX <NF4HEP.NF.rotationX>`
        and then the transformation with scalerX
        :attr:`NF.scalerX <NF4HEP.NF.scalerX>` .
        """
        return np.array(scaler_X.transform(np.dot(data_X, rotation_X)))

    def update_npoints(self,
                       npoints: List[int], # list with [n_train, n_val, n_test]
                       verbose: Optional[IntBool] = None
                      ) -> None:
        verbose, verbose_sub = self.get_verbosity(verbose)
        if len(npoints) != 3:
            raise InvalidInput("The 'npoints' argument should be a list of three integers of the form [n_train, n_val, n_test].")
        self._npoints_required = np.sum(npoints)
        if self._npoints_required > self._npoints_available:
            raise InvalidInput("The total requires number of points is larger than the available points (",self._npoints_available,").")
        old_test = self._npoints_test
        old_train = self._npoints_train
        old_val = self._npoints_val
        self._npoints_test = npoints[2]
        self._npoints_train = npoints[0]
        self._npoints_val = npoints[1]
        self.DataMain._log[utils.generate_timestamp()] = {"action": "updated npoints",
                                                          "npoints train [old,new]": [old_train, self._npoints_train],
                                                          "npoints val [old,new]": [old_val, self._npoints_train],
                                                          "npoints test [old, new]": [old_test, self._npoints_val]}
        print(header_string_2,"\nUpdated the number of required points: train (",old_train,"->",self._npoints_train,"); val (",old_val,"->",self._npoints_val,"); test (",old_test,"->",self._npoints_test,").\n", show = verbose)

    def update_test_data(self,
                         new_npoints_test: int,
                         verbose: Optional[IntBool] = None
                        ) -> None:
        """
        """
        verbose, verbose_sub = self.get_verbosity(verbose)
        start = timer()
        old_test = self._npoints_test
        npoints_test = [(i > 0) * i for i in [new_npoints_test-old_test]][0]
        existing_test = self._idx_test
        np.random.seed(self._seed)
        idx_test = np.random.choice(np.setdiff1d(np.array(self._test_range), existing_test), new_npoints_test, replace=False)
        self._idx_test = np.sort(np.concatenate((self._idx_test,idx_test)))
        [self.TestData._data_X, self.TestData._data_Y] = self.define_data(self._idx_test)
        self.update_npoints(npoints = [self._npoints_train, self._npoints_val, new_npoints_test],
                            verbose = verbose_sub)
        end = timer()
        self.DataMain._log[utils.generate_timestamp()] = {"action": "updated test data",
                                                          "npoints train": self._npoints_test,
                                                          "npoints val": self._npoints_val}
        print(header_string_2,"\nAdded", str(npoints_test), "(X_test, Y_test) data in", end-start,"s.\n", show = verbose)

    def update_train_val_data(self,
                              new_npoints_train: int,
                              new_npoints_val: int,
                              verbose: Optional[IntBool] = None
                             ) -> None:
        """
        """
        verbose, verbose_sub = self.get_verbosity(verbose)
        start = timer()
        [npoints_train, npoints_val] = [(i > 0) * i for i in [new_npoints_train-len(self._idx_train), 
                                                              new_npoints_val-len(self._idx_val)]]
        existing_train_val = np.sort(np.concatenate((self._idx_train, self._idx_val)))
        np.random.seed(self._seed)
        idx_train_val = np.random.choice(np.setdiff1d(np.array(self._train_val_range), existing_train_val), new_npoints_train+new_npoints_val, replace=False)
        if np.size(idx_train_val) != 0:
            idx_train, idx_val = [np.sort(idx) for idx in train_test_split(idx_train_val, train_size=new_npoints_train, test_size=new_npoints_train, shuffle=False)]
        else:
            idx_train = idx_train_val
            idx_val = idx_train_val
        self._idx_train = np.sort(np.concatenate((self._idx_train,idx_train)))
        self._idx_val = np.sort(np.concatenate((self._idx_val,idx_val)))
        [self.TrainData._data_X, self.TrainData._data_Y] = self.define_data(self._idx_train)
        [self.ValData._data_X, self.ValData._data_Y] = self.define_data(self._idx_val)
        self.update_npoints(npoints = [new_npoints_train, new_npoints_val, self._npoints_test],
                            verbose = verbose_sub)
        end = timer()
        self.DataMain._log[utils.generate_timestamp()] = {"action": "updated train/val data",
                                                          "npoints train": self._npoints_train,
                                                          "npoints val": self._npoints_val}
        print(header_string_2,"\nAdded", str(npoints_train), "(X_train, Y_train) data and", str(npoints_val),"(X_val, Y_val) data in", end-start,"s.\n", show = verbose)


class DataPredictionsManager(PredictionsManager):
    """
    """
    def __init__(self,
                 data_main: DataMain
                ) -> None:
        super().__init__(obj = data_main)

    def init_predictions(self):
        pass

    def reset_predictions(self):
        pass

    def validate_predictions(self):
        pass


class DataFiguresManager(FiguresManager):
    """
    """
    def __init__(self,
                 data_main: DataMain
                ) -> None:
        super().__init__(obj = data_main)


class DataInference(Inference):
    """
    """
    def __init__(self,
                 data_main: DataMain,
                ) -> None:
        super().__init__(obj = data_main)


class DataPlotter(Plotter):
    """
    """
    def __init__(self,
                 data_main: DataMain
                ) -> None:
        super().__init__(obj = data_main)