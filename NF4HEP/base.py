__all__ = ["InputFileNotFoundError",
           "InvalidPredictions",
           "Name",
           "FileManager",
           "ParsInput",
           "ParsManager",
           "Predictions",
           "Figures",
           "Inference",
           "Plots"]

from argparse import ArgumentError
import h5py # type: ignore
import json
import os
import re
import shutil
import codecs
import sys
import numpy as np
from matplotlib import pyplot as plt # type:  ignore
from numpy import typing as npt
from abc import abstractmethod
from scipy import stats, optimize # type: ignore

from typing import Union, List, Dict, Callable, Tuple, Optional, NewType, Type, Generic, Any, TypeVar, TYPE_CHECKING

from datetime import datetime
from pathlib import Path
from timeit import default_timer as timer
import deepdish as dd # type: ignore

#from .histfactory import HistfactoryMain
#from .likelihood import LikMain
#from .sampler import SamplerMain
#from .data import DataMain
#from .dnn_likelihood import DnnLikMain
from .verbosity import print
from . import utils
from .verbosity import Verbosity

Array = Union[list, npt.NDArray]
ArrayInt = Union[List[int], npt.NDArray[np.int_]]
ArrayStr = Union[List[str], npt.NDArray[np.str_]]
DataType = Union[npt.NDArray[Any],h5py.Dataset]
StrPath = Union[str,Path]
IntBool = Union[int, bool]
StrBool = Union[str, bool]
StrList = Union[str, List[str]]
LogPredDict = Dict[str,Dict[str,Any]]
FigDict = Dict[str,List[Path]]
Number = Union[int,float]

header_string = "=============================="
footer_string = "------------------------------"

class InputFileNotFoundError(FileNotFoundError):
    pass


class InvalidPredictions(Exception):
    pass


class Name:
    """
    """
    def __init__(self,
                 obj_name: str,
                 name: Optional[str] = None
                ) -> None:
        self.obj_name = obj_name
        self._name = name
        self.__check_define_name()

    def __check_define_name(self) -> str:
        """
        """
        self.name_str: str
        if self._name is None:
            timestamp = utils.generate_timestamp()
            self.name_str = "model_"+timestamp+"_"+self.obj_name.lower()
        else:
            self.name_str = utils.check_add_suffix(self._name, "_"+self.obj_name.lower())  
        return self.name_str

class FileManager(Verbosity):
    """
    """
    obj_name: str
    allowed_objects: List[str] = ["HistfactoryMain", "LikMain", "SamplerMain", "DataMain", "DnnLikMain"]
    #allowed_types = Union["HistfactoryMain", "LikMain", "SamplerMain", "DataMain", "DnnLikMain"]
    def __init__(self,
                 name: Union[str,None] = None,
                 input_file: Optional[StrPath] = None, 
                 output_folder: Optional[StrPath] = None, 
                 verbose: Optional[IntBool] = None
                ) -> None:
        super().__init__(verbose)
        verbose, verbose_sub = self.set_verbosity(verbose)
        self.timestamp = utils.generate_timestamp()
        self.name = Name(self.obj_name,name)
        self.name_str = self.name.name_str
        self._input_file = input_file
        self._output_folder = output_folder
        # Define self.input_file, self.input_h5_file, self.input_log_file, self.input_folder
        self.__define_base_input_files_folder(verbose=verbose) 
        # Define self.output_folder, self.output_h5_file, self.output_json_file, self.output_log_file
        self.__define_base_output_files_folder(verbose=verbose)
        # Define self.output_figures_folder, self.output_figures_base_file_name, self.output_figures_base_file_path
        self.__define_predictions_files()

    def __define_base_input_files_folder(self,
                                         verbose: Optional[IntBool] = None
                                        ) -> None:
        """
        """
        verbose, verbose_sub = self.set_verbosity(verbose)
        if self._input_file is None:
            print(header_string,"\nNo input files and folders specified.\n", show=verbose)
        else:
            try:
                self.input_file = Path(self._input_file).absolute()
                self.input_h5_file = self.input_file.with_suffix(".h5")
                self.input_log_file = self.input_file.with_suffix(".log")
                self.input_folder = self.input_file.parent
                if self.input_h5_file.exists() and self.input_log_file.exists():
                    print(header_string, "\nInput folder set to\n\t",self.input_folder, ".\n", show=verbose)
                else:
                    raise InputFileNotFoundError("The file",self.input_h5_file,"has not been found.")
            except InputFileNotFoundError:
                raise InputFileNotFoundError("The file",self.input_h5_file,"has not been found.")

    def __define_base_output_files_folder(self,
                                          timestamp: Optional[str] = None,
                                          verbose: Optional[IntBool] = None
                                         ) -> None:
        """
        """
        verbose, verbose_sub = self.set_verbosity(verbose)
        input_folder = Path(self._input_file).absolute().parent if self._input_file is not None else None
        if self._output_folder is not None:
            self.output_folder = Path(self._output_folder).absolute()
            if input_folder is not None:
                self.copy_and_save_folder(input_folder, self.output_folder, timestamp=timestamp, verbose=verbose)
        else:
            if input_folder is not None:
                self.output_folder = input_folder
            else:
                self.output_folder = Path("").absolute()
        self.output_folder = self.check_create_folder(self.output_folder)
        self.output_h5_file = self.output_folder.joinpath(self.name_str+".h5")
        self.output_json_file = self.output_folder.joinpath(self.name_str+".json")
        self.output_log_file = self.output_folder.joinpath(self.name_str+".log")
        print(header_string,"\nOutput folder set to\n\t", self.output_folder,".\n",show=verbose)

    def __define_predictions_files(self) -> None:
        """
        """
        self.output_figures_folder = self.check_create_folder(self.output_folder.joinpath("figures"))
        self.output_figures_base_file_name = self.name_str+"_figure"
        self.output_figures_base_file_path = self.output_figures_folder.joinpath(self.output_figures_base_file_name)
        self.output_predictions_json_file = self.output_folder.joinpath(self.name_str+"_predictions.json")

    @abstractmethod
    def __load(self):
        return

    #def __load(self,
    #           verbose: Optional[IntBool] = None
    #          ) -> List[Dict]:
    #    verbose, verbose_sub = self.set_verbosity(verbose)
    #    start = timer()
    #    obj = dd.io.load(self.input_h5_file)
    #    with self.input_log_file.open() as json_file:
    #        log = json.load(json_file)
    #    end = timer()
    #    timestamp = utils.generate_timestamp()
    #    log[timestamp] = {"action": "loaded",
    #                      "files names": [path.split(self.input_h5_file)[-1],
    #                                      path.split(self.input_log_file)[-1]]}
    #    print(header_string, "\n",self.obj_name," object loaded in", str(end-start), ".\n", show=verbose)
    #    return [obj, log]

    def save_h5(self,
                dict_to_save: Dict,
                log: LogPredDict,
                overwrite: StrBool = False,
                verbose: Optional[IntBool] = None
               ) -> None:
        """
        """
        verbose, verbose_sub = self.set_verbosity(verbose)
        timestamp = utils.generate_timestamp()
        start = timer()
        output_h5_file = self.get_target_file_overwrite(input_file=self.output_h5_file,
                                                        timestamp = timestamp,
                                                        overwrite=overwrite,
                                                        verbose=verbose_sub)
        dd.io.save(output_h5_file, dict_to_save)
        log[timestamp] = {"action": "saved h5",
                          "file name": output_h5_file.name}
        end = timer()
        self.print_save_info(filename=output_h5_file,
                             time= str(end-start),
                             overwrite = overwrite,
                             verbose = verbose)

    def save_json(self,
                  dict_to_save: Dict,
                  log: LogPredDict,
                  overwrite: StrBool = False,
                  verbose: Optional[IntBool] = None
                  ) -> None:
        """
        """
        verbose, verbose_sub = self.set_verbosity(verbose)
        timestamp = utils.generate_timestamp()
        start = timer()
        output_json_file = self.get_target_file_overwrite(input_file=self.output_json_file,
                                                          timestamp=timestamp,
                                                          overwrite=overwrite,
                                                          verbose=verbose_sub)
        with codecs.open(str(output_json_file), "w", encoding="utf-8") as f:
            json.dump(dict_to_save, f, separators=(",", ":"), indent=4)
        log[timestamp] = {"action": "saved json",
                          "file name": output_json_file.name}
        end = timer()
        self.print_save_info(filename=output_json_file,
                             time= str(end-start),
                             overwrite = overwrite,
                             verbose = verbose)

    def save_log(self,
                 log: LogPredDict,
                 overwrite: StrBool = False,
                 verbose: Optional[IntBool] = None
                 ) -> None:
        """
        """
        verbose, verbose_sub = self.set_verbosity(verbose)
        timestamp = utils.generate_timestamp()
        start = timer()
        output_log_file = self.get_target_file_overwrite(input_file = self.output_log_file,
                                                         timestamp = timestamp,
                                                         overwrite=overwrite,
                                                         verbose = verbose_sub)
        dict_to_save = utils.convert_types_dict(log)
        with codecs.open(str(output_log_file), "w", encoding="utf-8") as f:
            json.dump(dict_to_save, f, separators=(",", ":"), indent=4)
        end = timer()
        self.print_save_info(filename = output_log_file,
                             time= str(end-start),
                             overwrite = overwrite,
                             verbose = verbose)
    
    def check_create_folder(self, 
                            folder_path: StrPath, 
                           ) -> Path:
        """
        """
        folder_path = Path(folder_path).absolute()
        folder_path.mkdir(exist_ok=True)
        return folder_path

    def check_delete_all_files_in_path(self, 
                                       folder_path: StrPath,
                                      ) -> None:
        """
        """
        folder_path = Path(folder_path).absolute()
        items = [folder_path.joinpath(q) for q in folder_path.iterdir() if q.is_file()]
        self.check_delete_files_folders(items)

    def check_delete_all_folders_in_path(self, 
                                         folder_path: StrPath,
                                        ) -> None:
        """
        """
        folder_path = Path(folder_path).absolute()
        items = [folder_path.joinpath(q) for q in folder_path.iterdir() if q.is_dir()]
        self.check_delete_files_folders(items)

    def check_delete_all_items_in_path(self, 
                                       folder_path: StrPath,
                                      ) -> None:
        """
        """
        folder_path = Path(folder_path).absolute()
        items = [folder_path.joinpath(q) for q in folder_path.iterdir()]
        self.check_delete_files_folders(items)

    def check_delete_files_folders(self, 
                                   paths: List[Path],
                                  ) -> None:
        """
        """
        for path in paths:
            if path.is_file():
                path.unlink(missing_ok=True)
            elif path.is_dir():
                path.rmdir()

    def check_rename_path(self,
                          from_path: StrPath,
                          timestamp: Optional[str] = None,
                          verbose: Optional[IntBool] = True
                         ) -> Path:
        """
        """
        from_path = Path(from_path).absolute()
        if not from_path.exists():
            return from_path
        else:
            if timestamp is None:
                now = datetime.now().strftime("%Y_%m_%d_%H_%M_%S_%fZ")[:-3]
            else:
                now = timestamp
            filepath = from_path.parent
            filename = from_path.stem
            extension = from_path.suffix
            tmp = re.search(r'\d{4}-\d{2}-\d{2}-\d{2}-\d{2}-\d{2}', filename)
            if tmp is not None:
                match = tmp.group()
            else:
                match = ""
            if match != "":
                new_filename = filename.replace(match, now)
            else:
                new_filename = "old_"+now+"_"+filename
            to_path = filepath.joinpath(new_filename+extension)
            from_path.rename(to_path)
            print(header_string,"\nThe file\n\t",str(from_path),"\nalready exists and has been moved to\n\t",str(to_path),"\n",show=verbose)
            return to_path

    def copy_and_save_folder(self,
                             from_path: StrPath,
                             to_path: StrPath,
                             timestamp: Optional[str] = None,
                             verbose: Optional[IntBool] = True
                             ) -> None:
        """
        """
        from_path = Path(from_path).absolute()
        to_path = Path(to_path).absolute()
        if not from_path.exists():
            raise FileNotFoundError("The source folder does not exist")
        self.check_rename_path(to_path, timestamp=timestamp, verbose=verbose)
        shutil.copytree(from_path, to_path)

    def generate_dump_file_name(self, 
                                filepath: StrPath, 
                                timestamp: Optional[str] = None,
                               ) -> Path:
        """
        """
        if timestamp is None:
            timestamp = datetime.now().strftime("%Y_%m_%d_%H_%M_%S_%fZ")[:-3]
        filepath = Path(filepath).absolute()
        dump_filepath = Path(filepath.parent).joinpath("dump_"+filepath.stem+"_"+timestamp+filepath.suffix)
        return dump_filepath

    def get_target_file_overwrite(self,
                                  input_file: StrPath,
                                  timestamp: Optional[str] = None,
                                  overwrite: StrBool = False,
                                  verbose: Optional[IntBool] = None
                                 ) -> Path:
        """
        """
        verbose, verbose_sub = self.set_verbosity(verbose)
        input_file = Path(input_file).absolute()
        if type(overwrite) == bool:
            output_file = input_file
            if not overwrite:
                self.check_rename_path(output_file, verbose=verbose_sub)
        elif overwrite == "dump":
            if timestamp is None:
                timestamp = utils.generate_timestamp()
            output_file = self.generate_dump_file_name(input_file, timestamp=timestamp)
        else:
            raise Exception("Invalid 'overwrite' argument. The argument should be either bool or 'dump'.")
        return output_file

    def get_parent_path(self, 
                        this_path: StrPath, 
                        level: int, 
                       ) -> Path:
        """
        """
        this_path = Path(this_path).absolute()
        parent_path = this_path
        for i in range(level):
            parent_path = parent_path.parent
        return parent_path

    def print_save_info(self,
                        filename: StrPath,
                        time: str,
                        extension_string: Optional[str] = None,
                        overwrite: StrBool = False,
                        verbose: Optional[IntBool] = None
                       )  -> None:
        """
        """
        filename = Path(filename).absolute()
        if extension_string is None:
            extension = filename.suffix.replace(".","")
        else:
            extension = extension_string
        if type(overwrite) == bool:
            if overwrite:
                print(header_string, "\n",self.obj_name,extension,"file\n\t", str(filename),"\nupdated (or saved if it did not exist) in", time, "s.\n", show=verbose)
            else:
                print(header_string, "\n",self.obj_name,extension,"file\n\t", str(filename),"\nsaved in", time, "s.\n", show=verbose)
        elif overwrite == "dump":
            print(header_string, "\n",self.obj_name,extension,"file dump\n\t",str(filename), "\nsaved in", time, "s.\n", show=verbose)

    def replace_strings_in_file(self, 
                                filename: StrPath, 
                                old_strings: str, 
                                new_string: str
                               ) -> None:
        """
        """
        filename = Path(filename).absolute()
        # Safely read the input filename using 'with'
        with filename.open() as f:
            found_any = []
            s = f.read()
            for old_string in old_strings:
                if old_string not in s:
                    found_any.append(False)
                    #print('"{old_string}" not found in {filename}.'.format(**locals()))
                else:
                    found_any.append(True)
                    #print('"{old_string}" found in {filename}.'.format(**locals()))
            if not np.any(found_any):
                return
        # Safely write the changed content, if found in the file
        with filename.open('w') as f:
            #print('Changing "{old_string}" to "{new_string}" in {filename}'.format(**locals()))
            for old_string in old_strings:
                s = s.replace(old_string, new_string)
            f.write(s)

class ParsInput():
    obj_name: str
    allowed_objects: List[str] = ["LikMain", "SamplerMain", "DataMain", "DnnLikMain"]
    #allowed_types = Union["LikMain", "SamplerMain", "DataMain", "DnnLikMain"]
    def __init__(self,
                 pars_central: Optional[Array],
                 pars_pos_poi: Optional[ArrayInt],
                 pars_pos_nuis: Optional[ArrayInt],
                 pars_labels: Optional[ArrayStr],
                 pars_bounds: Optional[Array],
                 verbose: Optional[IntBool] = None
                ) -> None:
        """
        """
        self._pars_pos_poi = pars_pos_poi
        self._pars_pos_nuis = pars_pos_nuis
        self._pars_central = pars_central
        self._pars_labels = pars_labels
        self._pars_bounds = pars_bounds

class ParsManager(Verbosity):
    obj_name: str
    allowed_objects: List[str] = ["LikMain", "SamplerMain", "DataMain", "DnnLikMain"]
    #allowed_types = Union["LikMain", "SamplerMain", "DataMain", "DnnLikMain"]
    def __init__(self,
                 ndims: Optional[int],
                 pars_input: ParsInput,
                 verbose: Optional[IntBool] = None
                ) -> None:
        """
        """
        super().__init__(verbose)
        self._ndims = ndims
        self._pars_input = pars_input
        self._pars_pos_poi = self._pars_input._pars_pos_poi
        self._pars_pos_nuis = self._pars_input._pars_pos_nuis
        self._pars_central = self._pars_input._pars_central
        self._pars_labels = self._pars_input._pars_labels
        self._pars_bounds = self._pars_input._pars_bounds
        verbose, verbose_sub = self.set_verbosity(verbose)
        self.__check_define_pars(verbose=verbose)

    def __get_pars_labels_auto(self,
                               pars_pos_poi: ArrayInt,
                               pars_pos_nuis: ArrayInt
                              ) -> List[str]:
        """
        """
        pars_labels_auto: List[str] = []
        i_poi: int = 1
        i_nuis: int = 1
        for i in range(len(pars_pos_poi)+len(pars_pos_nuis)):
            if i in pars_pos_poi:
                pars_labels_auto.append(r"$\theta_{%d}$" % i_poi)
                i_poi = i_poi+1
            else:
                pars_labels_auto.append(r"$\nu_{%d}$" % i_nuis)
                i_nuis = i_nuis+1
        return pars_labels_auto

    def __check_define_pars(self,
                            verbose: Optional[IntBool] = None
                           ) -> None:
        """
        Private method used by the :meth:`Data.__init__ <DNNLikelihood.Data.__init__>` one
        to check the consistency of the attributes

            - :attr:`Data.pars_pos_nuis <DNNLikelihood.Data.pars_pos_nuis>`
            - :attr:`Data.pars_pos_poi <DNNLikelihood.Data.pars_pos_poi>`
            - :attr:`Data.pars_central <DNNLikelihood.Data.pars_pos_poi>`
            - :attr:`Data.pars_labels <DNNLikelihood.Data.pars_labels>`
            - :attr:`Data.pars_bounds <DNNLikelihood.Data.pars_bounds>` 

        and to convert the attributes

            - :attr:`Data.pars_pos_nuis <DNNLikelihood.Data.pars_pos_nuis>`
            - :attr:`Data.pars_pos_poi <DNNLikelihood.Data.pars_pos_poi>`
            - :attr:`Data.pars_central <DNNLikelihood.Data.pars_pos_poi>`
            - :attr:`Data.pars_bounds <DNNLikelihood.Data.pars_bounds>` 
        
        to |numpy_link| arrays.
        If no parameters positions are specified, all parameters are assumed to be parameters of interest.
        If only the position of the parameters of interest or of the nuisance parameters is specified,
        the other is automatically generated by matching dimensions.
        If labels are not provided then :attr:`Data.pars_labels <DNNLikelihood.Data.pars_labels>`
        is set to the value of :attr:`Data.pars_labels_auto <DNNLikelihood.Data.pars_labels_auto>`.
        If parameters bounds are not provided, they are set to ``(-np.inf,np.inf)``.
        A check is performed on the length of the four attributes and an Exception is raised if the length
        does not match :attr:`Data.ndims <DNNLikelihood.Data.ndims>`.

        - **Arguments**

            - **verbose**
            
                See :argument:`verbose <common_methods_arguments.verbose>`.
        """
        verbose, _ = self.set_verbosity(verbose)
        if self._pars_central is not None:
            self.pars_central = np.array(self._pars_central)
            self.ndims = len(self.pars_central)
        elif self._pars_central is None and self._ndims is not None:
            self.ndims = self._ndims
            self.pars_central = np.zeros(self.ndims)
            print(header_string,"\nNo central values for the parameters 'pars_central' has been specified. The number of dimensions \
                    have been automatically determined from 'logpdf' and the central values have been set to zero for all \
                    parameters. If they are known it is better to build the object providing parameters central values.\n", show=verbose)
        else:
            raise Exception("Impossible to determine the number of parameters/dimensions and the parameters central values. \
                    Please specify the input parameter 'pars_central'.")
        if self._pars_pos_nuis is not None and self._pars_pos_poi is not None:
            if len(self._pars_pos_poi)+len(self._pars_pos_nuis) == self.ndims:
                self.pars_pos_nuis = np.array(self._pars_pos_nuis)
                self.pars_pos_poi = np.array(self._pars_pos_poi)
            else:
                raise Exception("The number of parameters positions do not match the number of dimensions.")
        elif self._pars_pos_nuis is None and self._pars_pos_poi is None:
            print(header_string,"\nThe positions of the parameters of interest (pars_pos_poi) and of the nuisance parameters\
                (pars_pos_nuis) have not been specified. Assuming all parameters are parameters of interest.\n", show=verbose)
            self.pars_pos_nuis = np.array([])
            self.pars_pos_poi = np.arange(self.ndims)
        elif self._pars_pos_nuis is not None and self._pars_pos_poi is None:
            print(header_string,"\nOnly the positions of the nuisance parameters have been specified.\
                Assuming all other parameters are parameters of interest.\n", show=verbose)
            self.pars_pos_poi = np.setdiff1d(np.arange(self.ndims), np.array(self._pars_pos_nuis))
        elif self._pars_pos_nuis is None and self._pars_pos_poi is not None:
            print(header_string,"\nOnly the positions of the parameters of interest.\
                Assuming all other parameters are nuisance parameters.\n", show=verbose)
            self.pars_pos_nuis = np.setdiff1d(np.arange(self.ndims), np.array(self._pars_pos_poi))
        self.pars_labels_auto = self.__get_pars_labels_auto(self.pars_pos_poi, self.pars_pos_nuis)
        if self._pars_labels is None:
            self.pars_labels = self.pars_labels_auto
        elif len(self._pars_labels) != self.ndims:
            raise Exception("The number of parameters labels do not match the number of dimensions.")
        if self._pars_bounds is not None:
            self.pars_bounds = np.array(self._pars_bounds)
        else:
            self.pars_bounds = np.vstack([np.full(self.ndims, -np.inf), np.full(self.ndims, np.inf)]).T
        if len(self.pars_bounds) != self.ndims:
            raise Exception("The length of the parameters bounds array does not match the number of dimensions.")

    def __set_pars_labels(self, 
                          pars_labels: Union[str,list]
                         ) -> Union[str,list]:
        """
        """
        if pars_labels == "original":
            return self.pars_labels
        elif pars_labels == "auto":
            return self.pars_labels_auto
        else:
            return pars_labels


class Predictions():
    """
    """
    allowed_objects: List[str] = ["HistfactoryMain", "LikMain", "SamplerMain", "DataMain", "DnnLikMain"]
    #allowed_types = Union["HistfactoryMain", "LikMain", "SamplerMain", "DataMain", "DnnLikMain"]
    allowed_attrs: Dict[str,List[str]] = {"HistfactoryMain": [], 
                                          "LikMain": ["logpdf_max", "logdff_profiled_max"], 
                                          "SamplerMain": ["gelman_rubin"], 
                                          "DataMain": [], 
                                          "DNNLikMain": ["model_evaluation","bayesian_inference","frequentist_inference"]}

    def __init__(self,
                 obj: Any) -> None:
        """
        """
        self.obj = obj
        self.obj_name = obj.obj_name
        if self.obj_name not in self.allowed_objects:
            raise ArgumentError(None, message=str("Predictions are not supported for the object")+self.obj_name+str("."))
        self.verbose = obj.verbose
        self.init_predictions(verbose=self.verbose)

    def init_predictions(self,
                         verbose: Optional[IntBool] = None) -> None:
        """
        """
        verbose, _ = self.obj.set_verbosity(verbose)
        print(header_string,"\nInitialize predictions.\n",show=verbose)
        start = timer()
        objn = self.obj_name
        if objn == "Lik":
            self.logpdf_max: LogPredDict = {}
            self.logpdf_profiled_max: LogPredDict = {}
        elif objn == "Sampler":
            self.gelman_rubin: LogPredDict = {}
        elif objn == "DNNLik":
            self.model_evaluation: LogPredDict = {}
            self.bayesian_inference: LogPredDict = {}
            self.frequentist_inference: LogPredDict = {}
        end = timer()
        print(header_string,"\nPredictions initialized in", end-start, "s.\n",show=verbose)

    def reset_predictions(self,
                          verbose: Optional[IntBool] = None) -> None:
        """
        """
        verbose, _ = self.obj.set_verbosity(verbose)
        print(header_string,"\nResetting predictions.\n",show=verbose)
        start = timer()
        timestamp = utils.generate_timestamp()
        self.init_predictions(verbose==False)
        self.obj.log[timestamp] = {"action": "reset predictions"}
        end = timer()
        print(header_string,"\nPredictions reset in", end-start, "s.\n",show=verbose)
        
    def validate_predictions(self) -> None:
        """
        """
        for attr in self.allowed_attrs[self.obj_name]:
            try:
                getattr(self, attr)
            except:
                raise InvalidPredictions("Predictions for object",self.obj_name,"are not consistent.")

class Figures():
    """
    """
    allowed_objects: List[str] = ["HistfactoryMain", "LikMain", "SamplerMain", "DataMain", "DnnLikMain"]
    #allowed_types = Union["HistfactoryMain", "LikMain", "SamplerMain", "DataMain", "DnnLikMain"]
    
    def __init__(self,
                 obj: Any) -> None:
        self.obj = obj
        self.obj_name = obj.obj_name
        if self.obj_name not in self.allowed_objects:
            raise ArgumentError(None, message=str("Predictions are not supported for the object")+self.obj_name+str("."))
        self.verbose = obj.verbose
        self.figures: FigDict = {}
        
    def show_figures(self,
                     fig_list: List[StrPath],
                    ) -> None:
        """
        """
        figs = [str(f) for f in np.array(fig_list).flatten().tolist()]
        for fig in figs:
            try:
                os.startfile(r'%s'%fig)
                print(header_string,"\nFile\n\t", fig, "\nopened.\n")
            except:
                print(header_string,"\nFile\n\t", fig, "\nnot found.\n")

    def check_figures_list(self,
                           fig_list: List[Path],
                           output_figures_folder: Path
                          ) -> List[Path]:
        """
        """
        figs = [str(f) for f in np.array(fig_list).flatten().tolist()]
        new_fig_list: List[Path] = []
        for fig in figs:
            fig_path = output_figures_folder.joinpath(fig).absolute()
            if fig_path.exists():
                new_fig_list.append(fig_path)
        return new_fig_list

    def check_figures_dic(self,
                          output_figures_folder: Path
                         ) -> FigDict:
        """
        """
        new_fig_dic: FigDict = {}
        for k in self.figures.keys():
            new_fig_dic[k] = self.check_figures_list(self.figures[k],output_figures_folder)
            if new_fig_dic[k] == {}:
                del new_fig_dic[k]
        return new_fig_dic

    def check_delete_figures(self, 
                             file_manager: FileManager,
                             delete_figures: bool = False, 
                             verbose: Optional[IntBool] = None
                            ) -> None:
        """
        """
        verbose, _ = self.obj.set_verbosity(verbose)
        print(header_string,"\nResetting predictions.\n",show=verbose)
        try:
            file_manager.output_figures_folder
        except:
            print("The object does not have an associated figures folder.")
            return
        if delete_figures:
            file_manager.check_delete_all_files_in_path(file_manager.output_figures_folder)
            self.figures = {}
            print(header_string,"\nAll predictions and figures have been deleted and the 'predictions' attribute has been initialized.\n",show=verbose)
        else:
            self.figures = self.check_figures_dic(output_figures_folder=file_manager.output_figures_folder)
            print(header_string,"\nAll predictions have been deleted and the 'predictions' attribute has been initialized. No figure file has been deleted.\n",show=verbose)

    def reset_figures(self, 
                      log: LogPredDict,
                      file_manager: FileManager,
                      delete_figures: bool = False,
                      verbose: Optional[IntBool] = None
                     ) -> None:
        """
        """
        verbose, verbose_sub = self.obj.set_verbosity(verbose)
        print(header_string,"\nResetting figures.\n",show=verbose)
        start = timer()
        timestamp = utils.generate_timestamp()
        self.check_delete_figures(file_manager = file_manager,
                                  delete_figures = delete_figures, 
                                  verbose = verbose_sub)
        end = timer()
        log[timestamp] = {"action": "reset predictions"}
        print(header_string,"\nFigures reset in", end-start, "s.\n",show=verbose)

    def update_figures(self,
                       figure_file: StrPath,
                       file_manager: FileManager,
                       log: LogPredDict,
                       timestamp: Optional[str] = None,
                       overwrite: StrBool = False,
                       verbose: Optional[IntBool] = None
                       ) -> Path:
        """
        """
        verbose, verbose_sub = self.obj.set_verbosity(verbose)
        print(header_string,"\nChecking and updating figures dictionary,\n",show=verbose)
        figure_file = Path(figure_file).absolute()
        new_figure_file = figure_file
        if type(overwrite) == bool:
            if not overwrite:
                # search figure
                timestamp = None
                for k, v in self.figures.items():
                    if figure_file in v:
                        timestamp = k
                    old_figure_file = file_manager.check_rename_path(from_path = file_manager.output_figures_folder.joinpath(figure_file),
                                                                     timestamp = timestamp,
                                                                     verbose = verbose_sub)
                    if timestamp is not None:
                        self.figures[timestamp] = [Path(str(f).replace(str(figure_file),str(old_figure_file))) for f in v]
        elif overwrite == "dump":
            new_figure_file = file_manager.generate_dump_file_name(figure_file, timestamp=timestamp)
        if timestamp is None:
            timestamp = utils.generate_timestamp()
        log[timestamp] = {"action": "checked/updated figures dictionary",
                          "figure_file": figure_file,
                          "new_figure_file": new_figure_file}
        #self.save_log(overwrite=True, verbose=verbose_sub)
        return new_figure_file


class Inference(Verbosity):
    """
    """
    def __init__(self,
                 verbose: Optional[IntBool] = None) -> None:
        """
        """
        super().__init__(verbose)

    def CI_from_sigma(self, 
                      sigma: Union[Number,Array]
                     ) -> Union[float, Array]:
        """
        """
        return 2*stats.norm.cdf(sigma)-1

    def sigma_from_CI(self,
                      CI: Union[float, Array]
                     ) -> Union[float, Array]:
        """
        """
        CI = np.array(CI)
        return stats.norm.ppf(CI/2+1/2)

    def delta_chi2_from_CI(self,
                           CI: Union[float,Array], 
                           dof: Union[Number,Array] = 1
                          ) -> Union[float, Array]:
        """
        """
        CI = np.array(CI)
        dof = np.array(dof)
        return stats.chi2.ppf(CI, dof)

    def ks_w(self,
             data1: Array, 
             data2: Array, 
             wei1: Optional[Array] = None, 
             wei2: Optional[Array] = None
            ) -> List[Any]:
        """ 
        Weighted Kolmogorov-Smirnov test. Returns the KS statistics and the p-value (in the limit of large samples).
        """
        data1 = np.array(data1)
        data2 = np.array(data2)
        if wei1 is None:
            wei1 = np.ones(len(data1))
        if wei2 is None:
            wei2 = np.ones(len(data2))
        wei1 = np.array(wei1)
        wei2 = np.array(wei2)
        ix1 = np.argsort(data1)
        ix2 = np.argsort(data2)
        data1 = np.array(data1[ix1])
        data2 = np.array(data2[ix2])
        wei1 = np.array(wei1[ix1])
        wei2 = np.array(wei2[ix2])
        n1 = len(data1)
        n2 = len(data2)
        data = np.concatenate([data1, data2])
        cwei1 = np.hstack([0, np.cumsum(wei1)/sum(wei1)])
        cwei2 = np.hstack([0, np.cumsum(wei2)/sum(wei2)])
        cdf1we = cwei1[np.searchsorted(data1, data, side='right').tolist()]
        cdf2we = cwei2[np.searchsorted(data2, data, side='right').tolist()]
        d = np.max(np.abs(cdf1we - cdf2we))
        en = np.sqrt(n1 * n2 / (n1 + n2))
        prob = stats.distributions.kstwobign.sf(en * d)
        return [d, prob]

    def sort_consecutive(self,
                         data: Array, 
                         stepsize: Number = 1):
        """
        """
        return np.split(data, np.where(np.diff(data) != stepsize)[0]+1)

    def HPDI(self,
             data: Array, 
             intervals: Union[Number,Array] = 0.68,
             weights: Optional[Array] = None, 
             nbins: int = 25,
             print_hist: bool = False, 
             optimize_binning: bool = True
            ) -> Dict[Number,Dict[str,Any]]:
        """
        """
        data = np.array(data)
        intervals = np.sort(np.array([intervals]).flatten())
        if weights is None:
            weights = np.ones(len(data))
        weights = np.array(weights)
        counter = 0
        results = {}
        result_previous = []
        binwidth_previous = 0
        for interval in intervals:
            counts, bins = np.histogram(data, nbins, weights=weights, density=True)
            #counts, bins = hist
            nbins_val = len(counts)
            if print_hist:
                integral = counts.sum()
                plt.step(bins[:-1], counts/integral, where='post',color='green', label=r"train")
                plt.show()
            binwidth = bins[1]-bins[0]
            arr0 = np.transpose(np.concatenate(([counts*binwidth], [(bins+binwidth/2)[0:-1]])))
            arr0 = np.transpose(np.append(np.arange(nbins_val),np.transpose(arr0)).reshape((3, nbins_val)))
            arr = np.flip(arr0[arr0[:, 1].argsort()], axis=0)
            q = 0
            bin_labels: npt.NDArray = np.array([])
            for i in range(nbins_val):
                if q <= interval:
                    q = q + arr[i, 1]
                    bin_labels = np.append(bin_labels, arr[i, 0])
                else:
                    bin_labels = np.sort(bin_labels)
                    result = [[arr0[tuple([int(k[0]), 2])], arr0[tuple([int(k[-1]), 2])]] for k in self.sort_consecutive(bin_labels)]
                    result_previous = result
                    binwidth_previous = binwidth
                    if optimize_binning:
                        while (len(result) == 1 and nbins_val+nbins < np.sqrt(len(data))):
                            nbins_val = nbins_val+nbins
                            result_previous = result
                            binwidth_previous = binwidth
                            #nbins_val_previous = nbins_val
                            HPD_int_val = self.HPDI(data=data, 
                                                    intervals=interval, 
                                                    weights=weights, 
                                                    nbins=nbins_val, 
                                                    print_hist=False)
                            result = HPD_int_val[interval]["Intervals"]
                            binwidth = HPD_int_val[interval]["Bin width"]
                    break
            #results.append([interval, result_previous, nbins_val, binwidth_previous])
            results[interval] = {"Probability": interval, "Intervals": result_previous, "Number of bins": nbins_val, "Bin width": binwidth_previous}
            counter = counter + 1
        return results

    def HPDI_error(self,
                   HPDI
                  ) -> Dict[Number,Dict[str,Any]]:
        """
        """
        res: Dict[Number,Dict[str,Any]] = {}
        different_lengths = False
        for key_par, value_par in HPDI.items():
            dic: Dict[str,Any] = {}
            for sample in value_par['true'].keys():
                true = value_par['true'][sample]
                pred = value_par['pred'][sample]
                dic2: Dict[str,Any] = {}
                for CI in true.keys():
                    dic3 = {"Probability": true[CI]["Probability"]}
                    if len(true[CI]["Intervals"])==len(pred[CI]["Intervals"]):
                        dic3["Absolute error"] = (np.array(true[CI]["Intervals"])-np.array(pred[CI]["Intervals"])).tolist() # type: ignore
                        dic3["Relative error"] = ((np.array(true[CI]["Intervals"])-np.array(pred[CI]["Intervals"]))/(np.array(true[CI]["Intervals"]))).tolist() # type: ignore
                    else:
                        dic3["Absolute error"] = None
                        dic3["Relative error"] = None
                        different_lengths = True
                    dic2 = {**dic2, **{CI: dic3}}
                dic = {**dic, **{sample: dic2}}
            res = {**res, **{key_par: dic}}
            if different_lengths:
                print("For some probability values there are different numbers of intervals. In this case error is not computed and is set to None.")
        return res

    def HPD_quotas(self,
                   data: Array, 
                   intervals: Union[Number,Array] = 0.68,
                   weights: Optional[Array] = None, 
                   nbins: int = 25,
                   from_top: bool = True):
        """
        """
        data = np.array(data)
        intervals = np.sort(np.array([intervals]).flatten())
        counts, _, _ = np.histogram2d(data[:, 0], data[:, 1], bins=nbins, range=None, normed=None, weights=weights, density=None)
        #counts, binsX, binsY = np.histogram2d(data[:, 0], data[:, 1], bins=nbins, range=None, normed=None, weights=weights, density=None)
        integral = counts.sum()
        counts_sorted = np.flip(np.sort(utils.flatten_list(counts)))
        quotas = intervals
        q = 0
        j = 0
        for i in range(len(counts_sorted)):
            if q < intervals[j] and i < len(counts_sorted)-1:
                q = q + counts_sorted[i]/integral
            elif q >= intervals[j] and i < len(counts_sorted)-1:
                if from_top:
                    quotas[j] = 1-counts_sorted[i]/counts_sorted[0]
                else:
                    quotas[j] = counts_sorted[i]/counts_sorted[0]
                j = j + 1
            else:
                for k in range(j, len(intervals)):
                    quotas[k] = 0
                j = len(intervals)
            if j == len(intervals):
                return quotas

    def weighted_quantiles(self,
                           data: Array, 
                           quantiles: Union[Number,Array] = 0.68,
                           weights: Optional[Array] = None, 
                           data_sorted: bool = False, 
                           onesided: bool = False
                          ) -> npt.NDArray:
        """ Very close to numpy.percentile, but supports weights.
        NOTE: quantiles should be in [0, 1]!

            - param data numpy.array with data
            - param quantiles array-like with many quantiles needed
            - param weights array-like of the same length as `array`
            - param data_sorted bool, if True, then will avoid sorting of initial array
            - return numpy.array with computed quantiles.
        """
        data = np.array(data)
        quantiles = np.sort(np.array([quantiles]).flatten())
        if onesided:
            data = np.array(data[data > 0])
        else:
            data = np.array(data)
        if weights is None:
            weights = np.ones(len(data))
        weights = np.array(weights)
        assert np.all(quantiles >= 0) and np.all(quantiles <= 1), 'quantiles should be in [0, 1]'
        if not data_sorted:
            sorter = np.argsort(data)
            data = np.array(data[sorter])
            weights = np.array(weights[sorter])
        w_quantiles = np.cumsum(weights) - 0.5 * weights
        w_quantiles -= w_quantiles[0]
        w_quantiles /= w_quantiles[-1]
        result = np.transpose(np.concatenate((quantiles, np.interp(quantiles, w_quantiles, data))).reshape(2, len(quantiles))).tolist()
        return result

    def weighted_central_quantiles(self,
                                   data: Array, 
                                   quantiles: Union[Number,Array] = 0.68,
                                   weights: Optional[Array] = None, 
                                   onesided: bool = False
                                  ) -> list:
        """
        """
        data = np.array(data)
        quantiles = np.sort(np.array([quantiles]).flatten())
        if onesided:
            data = np.array(data[data > 0])
        else:
            data = np.array(data)
        return [[i, [self.weighted_quantiles(data, (1-i)/2, weights), self.weighted_quantiles(data, 0.5, weights), self.weighted_quantiles(data, 1-(1-i)/2, weights)]] for i in quantiles]

    def __compute_maximum_logpdf(self,
                                 logpdf: Callable, 
                                 ndims: Optional[int] = None, 
                                 pars_init: Optional[Array] = None, 
                                 pars_bounds: Optional[Array] = None,
                                 optimizer: Dict[str,Any] = {},
                                 minimization_options: Dict[str,Any] = {},
                                 verbose: IntBool = True
                                ) -> list:
        """
        """
        if verbose < 0:
            verbose_sub = 0
        else:
            verbose_sub = verbose
        def minus_logpdf(x): return -logpdf(x)
        if ndims is None and pars_init is not None:
            ndims = len(pars_init)
        elif ndims is not None and pars_init is None:
            pars_init = np.full(ndims, 0)
        elif ndims is None and pars_init is None:
            print("Please specify npars or pars_init or both", show = verbose)
        utils.check_set_dict_keys(optimizer, ["name",
                                              "args",
                                              "kwargs"],
                                             ["scipy", [], {"method": "Powell"}], verbose=verbose_sub)
        args = optimizer["args"]
        kwargs = optimizer["kwargs"]
        options = minimization_options
        if pars_bounds is None:
            ml = optimize.minimize(minus_logpdf, pars_init, *args, options=options, **kwargs)
        else:
            pars_bounds = np.array(pars_bounds)
            bounds = optimize.Bounds(pars_bounds[:, 0], pars_bounds[:, 1])
            ml = optimize.minimize(minus_logpdf, pars_init, *args, bounds=bounds, options=options, **kwargs)
        return [ml['x'], -ml['fun']]

    def __compute_profiled_maximum_logpdf(self,
                                          logpdf: Callable, 
                                          pars: Array,
                                          pars_val: Array,
                                          ndims: Optional[int] = None, 
                                          pars_init: Optional[Array] = None, 
                                          pars_bounds: Optional[Array] = None,
                                          optimizer: Dict[str,Any] = {},
                                          minimization_options: Dict[str,Any] = {},
                                          verbose: IntBool = True
                                         ) -> list:
        """
        """
        if verbose < 0:
            verbose_sub = 0
        else:
            verbose_sub = verbose
        pars = np.array(pars)
        pars_val = np.array(pars_val)
        # Add check that pars are within bounds
        if len(pars)!=len(pars_val):
            raise Exception("The input arguments 'pars' and 'pars_val' should have the same length.")
        pars_insert = pars - range(len(pars))
        if pars_init is None:
            if ndims is not None:
                pars_init = np.full(ndims, 0)
            else:
                raise Exception("At lease one of the two arguments 'ndims' and 'pars_init' needs to be specified.")
        else:
            if ndims is None:
                ndims = len(pars_init)
            else:
                if len(pars_init)!=ndims:
                    raise Exception("Parameters initialization has the wrong dimension. The dimensionality should be"+str(ndims)+".")
        pars_init_reduced = np.delete(pars_init, pars)
        utils.check_set_dict_keys(optimizer, ["name",
                                              "args",
                                              "kwargs"],
                                             ["scipy", [], {"method": "Powell"}], verbose=verbose_sub)
        args = optimizer["args"]
        kwargs = optimizer["kwargs"]
        options = minimization_options
        def minus_logpdf(x):
            return -logpdf(np.insert(x, pars_insert, pars_val))
        if pars_bounds is not None:
            pars_bounds=np.array(pars_bounds)
            if len(pars_bounds)!=len(pars_init):
                raise Exception("The input argument 'pars_bounds' should be either 'None' or have the same length of 'pars'.")
            if not ((np.all(pars_val >= pars_bounds[pars, 0]) and np.all(pars_val <= pars_bounds[pars, 1]))):
                print("Parameter values",pars_val,"lies outside parameters bounds",pars_bounds,".")
                return []
        if pars_bounds is None:
            ml = optimize.minimize(minus_logpdf, pars_init_reduced, *args, options=options, **kwargs)
        else:
            pars_bounds_reduced = np.delete(pars_bounds, pars,axis=0)
            pars_bounds_reduced = np.array(pars_bounds_reduced)
            bounds=optimize.Bounds(pars_bounds_reduced[:, 0], pars_bounds_reduced[:, 1])
            ml = optimize.minimize(minus_logpdf, pars_init_reduced, *args, bounds=bounds, options=options, **kwargs)
        return [np.insert(ml['x'], pars_insert, pars_val, axis=0), -ml['fun']]

    def __compute_maximum_sample(self,
                                 X: Array,
                                 Y: Array
                                ) -> list:
        """
        """
        X = np.array(X)
        Y = np.array(Y)
        y_max = np.amax(Y)
        pos_max = np.where(Y == y_max)[0][0]
        Y[pos_max] = -np.inf
        y_next_max = np.amax(Y)
        pos_next_max = np.where(Y == y_next_max)[0][0]
        x_max = X[pos_max]
        x_next_max = X[pos_next_max]
        return [x_max, y_max, np.abs(x_next_max-x_max), np.abs(y_next_max-y_max)]

    def __compute_profiled_maximum_sample(self,
                                          pars: Array,
                                          pars_val: Array,
                                          X: Array,
                                          Y: Array,
                                          binwidths: Union[str,Number,Array] = "auto"):
        """
        """
        pars = np.array(pars)
        pars_val = np.array(pars_val)
        X = np.array(X)
        Y = np.array(Y)
        if type(binwidths) == float or type(binwidths) == int:
            binwidths = np.full(len(pars), binwidths)
        elif type(binwidths) == list or type(binwidths) == np.ndarray:
            binwidths = np.array(binwidths)
            if len(binwidths) != len(pars):
                raise Exception("List of bin widths specifications has length different from the number of parameters.")
        elif type(binwidths) == str and binwidths != "auto":
            raise Exception("Invalid bin widths specification. Only allowed string is 'auto'.")
        if type(binwidths) != "auto":
            binwidths = np.array(binwidths)
            slicings_list: list = []
            for i in range(len(pars)):
                slicings_list.append([p > pars_val[i]-binwidths[i]/2 and p <
                                 pars_val[i]+binwidths[i]/2 for p in X[:, pars[i]]])
            slicing = np.array(np.prod(np.array(slicings_list), axis=0).astype(bool))
            npoints = np.count_nonzero(slicing)
            y_max = np.amax(Y[slicing])
            pos_max = np.where(Y == y_max)[0][0]
            Y[pos_max] = -np.inf
            y_next_max = np.amax(Y[slicing])
            pos_next_max = np.where(Y == y_next_max)[0][0]
            x_max = X[pos_max]
            x_next_max = X[pos_next_max]
        else:
            binwidths = np.full(len(pars), 0.001)
            npoints = 0
            slicing = np.array([])
            while npoints < 2:
                binwidths = binwidths+0.001
                slicings_list = []
                for i in range(len(pars)):
                    slicings_list.append([p > pars_val[i]-binwidths[i]/2 and p <
                                     pars_val[i]+binwidths[i]/2 for p in X[:, pars[i]]])
                slicing = np.array(np.prod(np.array(slicings_list), axis=0).astype(bool))
                npoints = np.count_nonzero(slicing)
            y_max = np.amax(Y[slicing])
            pos_max = np.where(Y == y_max)[0][0]
            Y[pos_max] = -np.inf
            y_next_max = np.amax(Y[slicing])
            pos_next_max = np.where(Y == y_next_max)[0][0]
            x_max = X[pos_max]
            x_next_max = X[pos_next_max]
        return [x_max, y_max, np.abs(x_next_max-x_max), np.abs(y_next_max-y_max), npoints]


class Plots(Verbosity):
    """
    """
    def __init__(self,
                 verbose: Optional[IntBool] = None) -> None:
        """
        """
        super().__init__(verbose)

    def savefig(self,
                figure_path: StrPath,
                **kwargs: Dict):
        """
        """
        if 'win32' in sys.platform or "win64" in sys.platform:
            plt.savefig("\\\\?\\" + str(figure_path), **kwargs)
        else:
            plt.savefig(figure_path, **kwargs)