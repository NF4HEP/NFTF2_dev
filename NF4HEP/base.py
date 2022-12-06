__all__ = ["InputFileNotFoundError",
           "InvalidInput",
           "InvalidPredictions",
           "Name",
           "FileManager",
           "PredictionsManager",
           "FiguresManager",
           "Inference",
           "Plotter"]

from abc import ABC, abstractmethod
from argparse import ArgumentError
import h5py # type: ignore
import json
import os
import re
import shutil
import codecs
import pandas as pd # type: ignore
import sys
import traceback
import numpy as np
from matplotlib import pyplot as plt # type:  ignore
from numpy import typing as npt
from scipy import stats, optimize # type: ignore

from typing import Union, List, Dict, Callable, Tuple, Optional, NewType, Type, Generic, Any, TypeVar, TYPE_CHECKING
from typing_extensions import TypeAlias
from NF4HEP.utils.custom_types import Array, ArrayInt, ArrayStr, DataType, StrPath, IntBool, StrBool, StrList, FigDict, LogPredDict, Number, DTypeStr, DTypeStrList, DictStr

from datetime import datetime
from pathlib import Path
from timeit import default_timer as timer
import deepdish as dd # type: ignore

from NF4HEP.utils.verbosity import print, Verbosity
from NF4HEP.utils import utils

#from NF4HEP.nf import NF
#from .inputs.data import DataMain
#from .inputs.distributions import Distributions

header_string_1 = "=============================="
header_string_2 = "------------------------------"

class InputFileNotFoundError(FileNotFoundError):
    pass


class InvalidInput(Exception):
    pass

class InvalidPredictions(Exception):
    pass


class ObjectManager():
    """
    Managed objects need to be imported in the package __init__.py
    Current managed objects are DataMain and NFMain
    """
    managed_object_name: str
    #allowed_managed_objects: List[str] = ["DataMain", "NFMain"]
    #allowed_managed_object_types: TypeAlias = Union["DataMain", "NFMain"] # type: ignore
    def __init__(self,
                 managed_object: Any
                ) -> None:
        """
        """
        # Import managed objects modules
        #from NF4HEP import DataMain, NFMain
        # Attributes type declatation
        self._ManagedObject: Any
        # Initialize object
        self.ManagedObject = managed_object
    
    @property
    def ManagedObject(self) -> Any:
        return self._ManagedObject

    @ManagedObject.setter
    def ManagedObject(self,
                      managed_object: Any,
                     ) -> None:
        object_name = self.__class__.__name__
        exec("from NF4HEP import " + self.managed_object_name)
        if isinstance(managed_object,eval(self.managed_object_name)):
            self._ManagedObject = managed_object
        else:
            raise TypeError(object_name+" object does not support object of type "+str(type(managed_object))+" as managed object.")


class Name:
    """
    Object created by the classes inheriting from :obj:`FileManager <NF4HEP.FileManager>`
    to store the object name.
    """
    def __init__(self,
                 managed_object_name: str,
                 name: str = ""
                ) -> None:
        # Attributes type declarations
        self._name_str: str
        self._managed_object_name: str
        # Initialize object
        self._managed_object_name = managed_object_name
        self.__check_define_name(name = name)

    @property
    def name_str(self) -> str:
        return self._name_str

    @property
    def managed_object_name(self) -> str:
        return self._managed_object_name

    def __check_define_name(self,
                            name: str
                           ) -> None:
        """
        """
        if name == "":
            timestamp = utils.generate_timestamp()
            self._name_str = self._managed_object_name.lower()+"_"+timestamp
        else:
            self._name_str = utils.check_add_suffix(name, "_"+self._managed_object_name.lower())  

    def __call__(self) -> str:
        """
        """
        return self._name_str


class FileManager(ABC,ObjectManager,Verbosity):
    """
    Abstract class to define I/O files and save/load objects :obj:`DataMain <NF4HEP.DataMain>` and :obj:`NF <NF4HEP.NF>`.
    """
    managed_object_name: str
    #managed_object_type: TypeAlias
    def __init__(self,
                 name: Optional[str] = None,
                 input_file: Optional[StrPath] = None,
                 output_folder: Optional[StrPath] = None,
                 verbose: Optional[IntBool] = None
                ) -> None:
        # Attributes type declarations
        self._input_file: Optional[Path]
        self._name: Name
        self._output_folder: Path
        self._ManagedObject: Any
        # Initialise parent ABC and Verbosity classes
        ABC.__init__(self)
        Verbosity.__init__(self, verbose)
        # Initialize object
        print(header_string_1, "\nInitializing FileManager object.\n", show = self.verbose)
        timestamp = utils.generate_timestamp()
        # Define self._input_file, self._input_folder, self._input_object_h5_file, self._input_log_file
        input_file = input_file if input_file is None else Path(input_file)
        self.input_file = input_file
        self.name = name if name is not None else ""
        self.output_folder = output_folder # type: ignore

    @property
    def name(self) -> str:
        """
        Property that returns the name as string
        """
        return self._name()

    @name.setter
    def name(self,
             name: str
            ) -> None:
        if self.input_object_h5_file is not None:
            name = self.__read_name_from_input_file()
        self._name = Name(managed_object_name = self.managed_object_name,
                          name = name)

    @property
    def input_file(self) -> Optional[Path]:
        return self._input_file

    @input_file.setter
    def input_file(self, 
                   input_file: Optional[StrPath] = None,
                  ) -> None:
        if input_file is None:
            self._input_file = input_file
        else:
            try:
                self._input_file = Path(input_file).absolute()
                if self._input_file.with_suffix(".h5").exists():
                    if self._input_file.with_suffix(".log").exists():
                        print(header_string_2, "\nInput folder set to\n\t",str(self._input_file.parent), ".\n", show = self._verbose)
                    else:
                        raise InputFileNotFoundError("The file",str(self._input_file.with_suffix(".log")),"has not been found.")
                else:
                    raise InputFileNotFoundError("The file",str(self._input_file.with_suffix(".h5")),"has not been found.")
            except:
                raise Exception("Could not set input files.")
    
    @property
    def input_folder(self) -> Optional[Path]:
        if self._input_file is not None:
            return self._input_file.parent
        else:
            return None

    @property
    def input_object_h5_file(self) -> Optional[Path]:
        if self._input_file is not None:
            return self._input_file.with_suffix(".h5")
        else:
            return None

    @property
    def input_log_file(self) -> Optional[Path]:
        if self._input_file is not None:
            return self._input_file.with_suffix(".log")
        else:
            return None

    @property
    def input_predictions_h5_file(self) -> Optional[Path]:
        if self._input_file is not None:
            return self._input_file.parent.joinpath(self._input_file.stem + "_predictions.h5")
        else:
            return None

    @property
    def output_folder(self) -> Path:
        return self._output_folder

    @output_folder.setter
    def output_folder(self,
                      output_folder: Optional[StrPath] = None,
                     ) -> None:
        timestamp = utils.generate_timestamp()
        if output_folder is not None:
            self._output_folder = Path(output_folder).absolute()
            if self.input_folder is not None:
                self.copy_and_save_folder(self.input_folder, self._output_folder, timestamp=timestamp, verbose=self._verbose_sub)
        else:
            if self.input_folder is not None:
                self._output_folder = Path(self.input_folder)
            else:
                self._output_folder = Path("").absolute()
        self.check_create_folder(self._output_folder)
        print(header_string_2,"\nOutput folder set to\n\t", self._output_folder,".\n", show = self._verbose)

    @property
    def output_object_h5_file(self) -> Path:
        return self._output_folder.joinpath(self.name).with_suffix(".h5")

    @property
    def output_object_json_file(self) -> Path:
        return self._output_folder.joinpath(self.name).with_suffix(".json")

    @property
    def output_log_file(self) -> Path:
        return self._output_folder.joinpath(self.name+".log")

    @property
    def output_figures_folder(self) -> Path:
        return self.check_create_folder(self._output_folder.joinpath("figures"))

    @property
    def output_figures_base_file_name(self) -> str:
        return self.name+"_figure"

    @property
    def output_figures_base_file_path(self) -> Path:
        return self.output_figures_folder.joinpath(self.output_figures_base_file_name)

    @property
    def output_predictions_json_file(self) -> Path:
        return self._output_folder.joinpath(self.name+"_predictions.json")

    @property
    def output_predictions_h5_file(self) -> Path:
        return self._output_folder.joinpath(self.name+"_predictions.h5")

    def __load_h5(self,
                  input_h5_file: StrPath,
                  verbose: Optional[IntBool] = None
                 ) -> List[Dict]:
        """
        """
        verbose, _ = self.get_verbosity(verbose)
        start = timer()
        input_h5_file = Path(input_h5_file)
        dictionary = dd.io.load(input_h5_file)
        log = {}
        log[utils.generate_timestamp()] = {"action": "loaded h5",
                                           "file name": input_h5_file.name}
        end = timer()
        self.print_load_info(filename=input_h5_file,
                             time= str(end-start),
                             verbose = verbose)
        return [log,dictionary]

    def __load_json(self,
                    input_json_file: StrPath,
                    verbose: Optional[IntBool] = None
                   ) -> List[Dict]:
        """
        """
        verbose, _ = self.get_verbosity(verbose)
        start = timer()
        input_json_file = Path(input_json_file)
        with input_json_file.open() as json_file:
            dictionary = json.load(json_file)
        log = {}
        log[utils.generate_timestamp()] = {"action": "loaded json",
                                           "file name": input_json_file.name}
        end = timer()
        self.print_load_info(filename=input_json_file,
                             time= str(end-start),
                             verbose = verbose)
        return [log,dictionary]

    def __read_name_from_input_file(self) -> str:
        if self.input_object_h5_file is not None:
            input_h5_file = Path(self.input_object_h5_file)
            name_str: str = str(dd.io.load(input_h5_file,"/_name"))
            #dictionary = dd.io.load(input_h5_file,"/_name")
            #name_str: str = dictionary["_name"]
            return name_str
        else:
            raise Exception("Could not determine object name from 'input_file'")

    def __save_dict_h5(self,
                       dict_to_save: dict,
                       output_file: StrPath,
                       timestamp: Optional[str] = None,
                       overwrite: StrBool = False,
                       verbose: Optional[IntBool] = None
                      ) -> LogPredDict:
        """
        """
        verbose, verbose_sub = self.get_verbosity(verbose)
        if timestamp is None:
            timestamp = utils.generate_timestamp()
        start = timer()
        output_h5_file = self.get_target_file_overwrite(input_file = Path(output_file),
                                                        timestamp = timestamp,
                                                        overwrite = overwrite,
                                                        verbose = verbose_sub)
        dd.io.save(output_h5_file, dict_to_save)
        log = {}
        log[utils.generate_timestamp()] = {"action": "saved h5",
                                           "file name": output_h5_file.name}
        end = timer()
        self.print_save_info(filename=output_h5_file,
                             time= str(end-start),
                             overwrite = overwrite,
                             verbose = verbose)
        return log

    def __save_dict_json(self,
                         dict_to_save: dict,
                         output_file: StrPath,
                         timestamp: Optional[str] = None,
                         overwrite: StrBool = False,
                         verbose: Optional[IntBool] = None
                        ) -> LogPredDict:
        """
        """
        verbose, verbose_sub = self.get_verbosity(verbose)
        if timestamp is None:
            timestamp = utils.generate_timestamp()
        start = timer()
        output_json_file = self.get_target_file_overwrite(input_file = Path(output_file),
                                                          timestamp = timestamp,
                                                          overwrite = overwrite,
                                                          verbose = verbose_sub)
        dict_to_save = utils.convert_types_dict(dict_to_save)
        with codecs.open(str(output_json_file), "w", encoding="utf-8") as f:
            json.dump(dict_to_save, f, separators=(",", ":"), indent=4)
        log = {}
        log[utils.generate_timestamp()] = {"action": "saved json",
                                           "file name": output_json_file.name}
        end = timer()
        self.print_save_info(filename = output_json_file,
                             time= str(end-start),
                             overwrite = overwrite,
                             verbose = verbose)
        return log

    def __save_dict_h5_json(self,
                            dict_to_save: dict,
                            output_file: StrPath,
                            timestamp: Optional[str] = None,
                            overwrite: StrBool = False,
                            verbose: Optional[IntBool] = None
                           ) -> LogPredDict:
        if timestamp is None:
            timestamp = utils.generate_timestamp()
        output_file = Path(output_file).stem
        output_json_file = self.output_folder.joinpath(output_file+".json")
        output_h5_file = self.output_folder.joinpath(output_file+".h5")
        log_json = self.__save_dict_json(dict_to_save = dict_to_save,
                                         output_file = output_json_file,
                                         timestamp = timestamp,
                                         overwrite = overwrite,
                                         verbose = verbose)
        log_h5 = self.__save_dict_h5(dict_to_save = dict_to_save,
                                     output_file = output_h5_file,
                                     timestamp = timestamp,
                                     overwrite = overwrite,
                                     verbose = verbose)
        return {**log_json, **log_h5}

    def __save_object_h5(self,
                         obj: Any,
                         output_file: StrPath,
                         timestamp: Optional[str] = None,
                         overwrite: StrBool = False,
                         verbose: Optional[IntBool] = None
                        ) -> LogPredDict:
        """
        """
        verbose, verbose_sub = self.get_verbosity(verbose)
        if timestamp is None:
            timestamp = utils.generate_timestamp()
        start = timer()
        output_h5_file = self.get_target_file_overwrite(input_file = Path(output_file),
                                                        timestamp = timestamp,
                                                        overwrite = overwrite,
                                                        verbose = verbose_sub)
        dict_to_save = obj.__dict__
        dd.io.save(output_h5_file, dict_to_save)
        log = {}
        log[utils.generate_timestamp()] = {"action": "saved h5",
                                           "file name": output_h5_file.name}
        end = timer()
        self.print_save_info(filename=output_h5_file,
                             time= str(end-start),
                             overwrite = overwrite,
                             verbose = verbose)
        return log

    def __save_object_json(self,
                         obj: Any,
                         output_file: StrPath,
                         timestamp: Optional[str] = None,
                         overwrite: StrBool = False,
                         verbose: Optional[IntBool] = None
                        ) -> LogPredDict:
        """
        """
        verbose, verbose_sub = self.get_verbosity(verbose)
        if timestamp is None:
            timestamp = utils.generate_timestamp()
        start = timer()
        output_json_file = self.get_target_file_overwrite(input_file = Path(output_file),
                                                          timestamp = timestamp,
                                                          overwrite = overwrite,
                                                          verbose = verbose_sub)
        dict_to_save = utils.convert_types_dict(obj.__dict__)
        with codecs.open(str(output_json_file), "w", encoding="utf-8") as f:
            json.dump(dict_to_save, f, separators=(",", ":"), indent=4)
        log = {}
        log[utils.generate_timestamp()] = {"action": "saved json",
                                           "file name": output_json_file.name}
        end = timer()
        self.print_save_info(filename=output_json_file,
                             time= str(end-start),
                             overwrite = overwrite,
                             verbose = verbose)
        return log

    def __save_object_h5_json(self,
                            obj: Any,
                            output_file: StrPath,
                            timestamp: Optional[str] = None,
                            overwrite: StrBool = False,
                            verbose: Optional[IntBool] = None
                           ) -> LogPredDict:
        if timestamp is None:
            timestamp = utils.generate_timestamp()
        output_file = Path(output_file).stem
        output_json_file = self.output_folder.joinpath(output_file+".json")
        output_h5_file = self.output_folder.joinpath(output_file+".h5")
        log_json = self.__save_object_json(obj = obj,
                                           output_file = output_json_file,
                                           timestamp = timestamp,
                                           overwrite = overwrite,
                                           verbose = verbose)
        log_h5 = self.__save_object_h5(obj = obj,
                                       output_file = output_h5_file,
                                       timestamp=timestamp,
                                       overwrite = overwrite,
                                       verbose = verbose)
        return {**log_json, **log_h5}
    
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
            print(header_string_2,"\nThe file\n\t",str(from_path),"\nalready exists and has been moved to\n\t",str(to_path),"\n", show = verbose)
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

    def get_target_file_overwrite(self,
                                  input_file: StrPath,
                                  timestamp: Optional[str] = None,
                                  overwrite: StrBool = False,
                                  verbose: Optional[IntBool] = None
                                 ) -> Path:
        """
        """
        verbose, _ = self.get_verbosity(verbose)
        input_file = Path(input_file).absolute()
        if type(overwrite) == bool:
            output_file = input_file
            if not overwrite:
                self.check_rename_path(output_file, verbose = verbose)
        elif overwrite == "dump":
            if timestamp is None:
                timestamp = utils.generate_timestamp()
            output_file = self.generate_dump_file_name(input_file, timestamp = timestamp)
        else:
            raise Exception("Invalid 'overwrite' argument. The argument should be either bool or 'dump'.")
        return output_file

    @abstractmethod
    def load(self, verbose: Optional[IntBool] = None) -> None:
        pass

    def load_log(self, verbose: Optional[IntBool] = None) -> None:
        verbose, verbose_sub = self.get_verbosity(verbose)
        if self.input_log_file is not None:
            [log, dictionary] = self.__load_json(input_json_file = self.input_log_file, # type: ignore
                                                 verbose = verbose
                                                )
            self.ManagedObject._log.update(dictionary)
            self.ManagedObject._log = {**self.ManagedObject._log, **log}
        else:
            raise Exception("Input file not defined.")

    @abstractmethod
    def load_object(self, verbose: Optional[IntBool] = None) -> None:
        pass

    def load_predictions(self, verbose: Optional[IntBool] = None) -> None:
        verbose, verbose_sub = self.get_verbosity(verbose)
        if self.input_predictions_h5_file is not None:
            [log, dictionary] = self.__load_h5(input_h5_file = self.input_predictions_h5_file,
                                               verbose = verbose
                                              )
            self.ManagedObject._log = {**self.ManagedObject._log, **log}
            self.ManagedObject.Predictions._predictions_dict.update(dictionary["Predictions"])
            self.ManagedObject.Figures._figures_dict.update(dictionary["Figures"])
        else:
            raise Exception("Input file not defined.")

    def print_save_info(self,
                        filename: StrPath,
                        time: str,
                        extension_string: Optional[str] = None,
                        overwrite: StrBool = False,
                        verbose: Optional[IntBool] = None
                       )  -> None:
        """
        """
        verbose, _ = self.get_verbosity(verbose)
        filename = Path(filename).absolute()
        if extension_string is None:
            extension = filename.suffix.replace(".","")
        else:
            extension = extension_string
        if type(overwrite) == bool:
            if overwrite:
                print(header_string_2, "\n",self.ManagedObject.__class__.__name__,extension,"output file\n\t", str(filename),"\nupdated (or saved if it did not exist) in", time, "s.\n", show = verbose)
            else:
                print(header_string_2, "\n",self.ManagedObject.__class__.__name__,extension,"output file\n\t", str(filename),"\nsaved in", time, "s.\n", show = verbose)
        elif overwrite == "dump":
            print(header_string_2, "\n",self.ManagedObject.__class__.__name__,extension,"output file dump\n\t",str(filename), "\nsaved in", time, "s.\n", show = verbose)

    def print_load_info(self,
                        filename: StrPath,
                        time: str,
                        extension_string: Optional[str] = None,
                        verbose: Optional[IntBool] = None
                       )  -> None:
        """
        """
        verbose, _ = self.get_verbosity(verbose)
        filename = Path(filename).absolute()
        if extension_string is None:
            extension = filename.suffix.replace(".","")
        else:
            extension = extension_string
        print(header_string_2, "\n",self.ManagedObject,extension,"file\n\t", str(filename),"\nloaded in", time, "s.\n", show = verbose)

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

    @abstractmethod
    def save(self,
             timestamp: Optional[str] = None,
             overwrite: StrBool = False,
             verbose: Optional[IntBool] = None
            ) -> None:
        pass

    def save_log(self,
                 timestamp: Optional[str] = None,
                 overwrite: StrBool = False,
                 verbose: Optional[IntBool] = None
                ) -> None:
        """
        """
        verbose, verbose_sub = self.get_verbosity(verbose)
        if timestamp is None:
            timestamp = utils.generate_timestamp()
        dict_to_save = self.ManagedObject._log
        self.__save_dict_json(dict_to_save = dict_to_save, # type: ignore
                              output_file = self.output_log_file,
                              timestamp = timestamp,
                              overwrite = overwrite,
                              verbose = verbose)

    @abstractmethod
    def save_object(self, 
                    timestamp: Optional[str] = None,
                    overwrite: StrBool = False,
                    verbose: Optional[IntBool] = None
                   ) -> None:
        pass
    
    def save_predictions(self, 
                         timestamp: Optional[str] = None,
                         overwrite: StrBool = False,
                         verbose: Optional[IntBool] = None
                        ) -> None:
        if timestamp is None:
            timestamp = utils.generate_timestamp()
        pred_dict: Dict[str, Any] = {"Predictions": self.ManagedObject.Predictions._predictions_dict}
        figs_dict: Dict[str, Any] = {"Figures": self.ManagedObject.Figures._figures_dict}
        dict_to_save = {**pred_dict, **figs_dict}
        log = self.__save_dict_h5_json(dict_to_save = dict_to_save,
                                       output_file = self.output_predictions_h5_file,
                                       overwrite = overwrite,
                                       verbose = verbose)
        self.ManagedObject._log = {**self.ManagedObject._log, **log}


class PredictionsManager(ABC,ObjectManager):
    """
    Abstract class to define make and store predictions of objects :obj:`DataMain <NF4HEP.DataMain>` and :obj:`NF <NF4HEP.NF>`.
    """
    managed_object_name: str
    def __init__(self,
                 managed_object: Any
                ) -> None:
        """
        """
        # Attributes type declatation
        self._ManagedObject: Any
        self._managed_object_name: str
        self._predictions_dict: LogPredDict
        # Initialize parent ABC and ManagedObject class (sets self._ManagedObject)
        ABC.__init__(self)
        ObjectManager.__init__(self, managed_object = managed_object)
        # Set verbosity
        verbose, _ = self._ManagedObject.get_verbosity(self._ManagedObject._verbose)
        # Initialize object
        print(header_string_1,"\nInitializing Predictions.\n", show = verbose)
        self._predictions_dict = {}

    @property
    def predictions_dict(self) -> LogPredDict:
        return self._predictions_dict

    @abstractmethod
    def init_predictions(self):
        pass

    @abstractmethod
    def reset_predictions(self):
        pass

    @abstractmethod
    def validate_predictions(self):
        pass


class FiguresManager(ObjectManager):
    """
    """
    managed_object_name: str
    def __init__(self,
                 managed_object: Any
                ) -> None:
        """
        """
        # Attributes type declatation
        self._ManagedObject: Any
        self._figures_dict: FigDict
        # Initialize parent ManagedObject class (sets self._ManagedObject)
        super().__init__(managed_object = managed_object)
        # Set verbosity
        verbose, _ = self._ManagedObject.get_verbosity(self._ManagedObject._verbose)
        # Initialize object
        print(header_string_1,"\nInitializing Figures.\n", show = verbose)
        self._figures_dict = {}

    @property
    def figures_dict(self) -> FigDict:
        return self._figures_dict

    def check_delete_figures(self, 
                             delete_figures: bool = False, 
                             verbose: Optional[IntBool] = None
                            ) -> None:
        """
        """
        verbose, _ = self._ManagedObject.get_verbosity(verbose)
        print(header_string_2,"\nResetting predictions.\n", show = verbose)
        try:
            self._ManagedObject.FileManager.output_figures_folder
        except:
            print(header_string_2,"\nThe object does not have an associated figures folder.\n")
            return
        if delete_figures:
            self._ManagedObject.FileManager.check_delete_all_files_in_path(self._ManagedObject.FileManager.output_figures_folder)
            self._figures_dict = {}
            print(header_string_2,"\nAll predictions and figures have been deleted and the 'predictions' attribute has been initialized.\n", show = verbose)
        else:
            self._figures_dict = self.check_figures_dic(output_figures_folder=self._ManagedObject.FileManager.output_figures_folder)
            print(header_string_2,"\nAll predictions have been deleted and the 'predictions' attribute has been initialized. No figure file has been deleted.\n", show = verbose)

    def check_figures_dic(self,
                          output_figures_folder: Path
                         ) -> FigDict:
        """
        """
        new_fig_dic: FigDict = {}
        for k in self.figures_dict.keys():
            new_fig_dic[k] = self.check_figures_list(self.figures_dict[k],output_figures_folder)
            if new_fig_dic[k] == {}:
                del new_fig_dic[k]
        return new_fig_dic

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

    def init_figures(self):
        pass

    def reset_figures(self, 
                      delete_figures: bool = False,
                      verbose: Optional[IntBool] = None
                     ) -> None:
        """
        """
        verbose, verbose_sub = self._ManagedObject.get_verbosity(verbose)
        print(header_string_2,"\nResetting figures.\n", show = verbose)
        start = timer()
        timestamp = utils.generate_timestamp()
        self.check_delete_figures(delete_figures = delete_figures, 
                                  verbose = verbose_sub)
        end = timer()
        self._ManagedObject._log[utils.generate_timestamp()] = {"action": "reset predictions"}
        print(header_string_2,"\nFigures reset in", end-start, "s.\n", show = verbose)

    def show_figures(self,
                     fig_list: List[StrPath],
                    ) -> None:
        """
        """
        figs = [str(f) for f in np.array(fig_list).flatten().tolist()]
        for fig in figs:
            try:
                os.startfile(r'%s'%fig)
                print(header_string_2,"\nFile\n\t", fig, "\nopened.\n")
            except:
                print(header_string_2,"\nFile\n\t", fig, "\nnot found.\n")

    def update_figures(self,
                       figure_file: StrPath,
                       timestamp: Optional[str] = None,
                       overwrite: StrBool = False,
                       verbose: Optional[IntBool] = None
                       ) -> Path:
        """
        """
        verbose, verbose_sub = self._ManagedObject.get_verbosity(verbose)
        print(header_string_2,"\nChecking and updating figures dictionary,\n", show = verbose)
        figure_file = Path(figure_file).absolute()
        new_figure_file = figure_file
        if type(overwrite) == bool:
            if not overwrite:
                # search figure
                timestamp = None
                for k, v in self.figures_dict.items():
                    if figure_file in v:
                        timestamp = k
                    old_figure_file = self._ManagedObject.FileManager.check_rename_path(from_path = self._ManagedObject.FileManager.output_figures_folder.joinpath(figure_file),
                                                                         timestamp = timestamp,
                                                                         verbose = verbose_sub)
                    if timestamp is not None:
                        self.figures_dict[timestamp] = [Path(str(f).replace(str(figure_file),str(old_figure_file))) for f in v]
        elif overwrite == "dump":
            new_figure_file = self._ManagedObject.FileManager.generate_dump_file_name(figure_file, timestamp=timestamp)
        if timestamp is None:
            timestamp = utils.generate_timestamp()
        self._ManagedObject._log[utils.generate_timestamp()] = {"action": "checked/updated figures dictionary",
                                                      "figure_file": figure_file,
                                                      "new_figure_file": new_figure_file}
        #self.save_log(overwrite=True, verbose=verbose_sub)
        return new_figure_file

    def validate_figures(self):
        pass


class Inference(ObjectManager):
    """
    """
    managed_object_name: str
    def __init__(self,
                 managed_object: Any
                ) -> None:
        """
        """
        # Attributes type declatation
        self._ManagedObject: Any
        # Initialize parent ManagedObject class (sets self._ManagedObject)
        super().__init__(managed_object = managed_object)
        # Set verbosity
        verbose, _ = self._ManagedObject.get_verbosity(self._ManagedObject._verbose)
        # Initialize object
        print(header_string_1,"\nInitializing Inference.\n", show = verbose)

    def CI_from_sigma(self, 
                      sigma: Union[Number,Array]
                     ) -> Union[float, Array]:
        """
        """
        return 2*stats.norm.cdf(sigma)-1

    def delta_chi2_from_CI(self,
                           CI: Union[float,Array], 
                           dof: Union[Number,Array] = 1
                          ) -> Union[float, Array]:
        """
        """
        CI = np.array(CI)
        dof = np.array(dof)
        return stats.chi2.ppf(CI, dof)

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
                print(header_string_2,"\nFor some probability values there are different numbers of intervals. In this case error is not computed and is set to None.\n")
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

    def is_pos_def(self,x):
        return np.all(np.linalg.eigvals(x) > 0)

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

    def sigma_from_CI(self,
                      CI: Union[float, Array]
                     ) -> Union[float, Array]:
        """
        """
        CI = np.array(CI)
        return stats.norm.ppf(CI/2+1/2)

    def sort_consecutive(self,
                         data: Array, 
                         stepsize: Number = 1):
        """
        """
        return np.split(data, np.where(np.diff(data) != stepsize)[0]+1)

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

#    def __compute_maximum_logpdf(self,
#                                 logpdf: Callable, 
#                                 ndims: Optional[int] = None, 
#                                 pars_init: Optional[Array] = None, 
#                                 pars_bounds: Optional[Array] = None,
#                                 optimizer: Dict[str,Any] = {},
#                                 minimization_options: Dict[str,Any] = {},
#                                 verbose: IntBool = True
#                                ) -> list:
#        """
#        """
#        if verbose < 0:
#            verbose_sub = 0
#        else:
#            verbose_sub = verbose
#        def minus_logpdf(x): return -logpdf(x)
#        if ndims is None and pars_init is not None:
#            ndims = len(pars_init)
#        elif ndims is not None and pars_init is None:
#            pars_init = np.full(ndims, 0)
#        elif ndims is None and pars_init is None:
#            print("Please specify npars or pars_init or both", show = verbose)
#        utils.check_set_dict_keys(optimizer, ["name",
#                                              "args",
#                                              "kwargs"],
#                                             ["scipy", [], {"method": "Powell"}], verbose=verbose_sub)
#        args = optimizer["args"]
#        kwargs = optimizer["kwargs"]
#        options = minimization_options
#        if pars_bounds is None:
#            ml = optimize.minimize(minus_logpdf, pars_init, *args, options=options, **kwargs)
#        else:
#            pars_bounds = np.array(pars_bounds)
#            bounds = optimize.Bounds(pars_bounds[:, 0], pars_bounds[:, 1])
#            ml = optimize.minimize(minus_logpdf, pars_init, *args, bounds=bounds, options=options, **kwargs)
#        return [ml['x'], -ml['fun']]
#
#    def __compute_profiled_maximum_logpdf(self,
#                                          logpdf: Callable, 
#                                          pars: Array,
#                                          pars_val: Array,
#                                          ndims: Optional[int] = None, 
#                                          pars_init: Optional[Array] = None, 
#                                          pars_bounds: Optional[Array] = None,
#                                          optimizer: Dict[str,Any] = {},
#                                          minimization_options: Dict[str,Any] = {},
#                                          verbose: IntBool = True
#                                         ) -> list:
#        """
#        """
#        if verbose < 0:
#            verbose_sub = 0
#        else:
#            verbose_sub = verbose
#        pars = np.array(pars)
#        pars_val = np.array(pars_val)
#        # Add check that pars are within bounds
#        if len(pars)!=len(pars_val):
#            raise Exception("The input arguments 'pars' and 'pars_val' should have the same length.")
#        pars_insert = pars - range(len(pars))
#        if pars_init is None:
#            if ndims is not None:
#                pars_init = np.full(ndims, 0)
#            else:
#                raise Exception("At lease one of the two arguments 'ndims' and 'pars_init' needs to be specified.")
#        else:
#            if ndims is None:
#                ndims = len(pars_init)
#            else:
#                if len(pars_init)!=ndims:
#                    raise Exception("Parameters initialization has the wrong dimension. The dimensionality should be"+str(ndims)+".")
#        pars_init_reduced = np.delete(pars_init, pars)
#        utils.check_set_dict_keys(optimizer, ["name",
#                                              "args",
#                                              "kwargs"],
#                                             ["scipy", [], {"method": "Powell"}], verbose=verbose_sub)
#        args = optimizer["args"]
#        kwargs = optimizer["kwargs"]
#        options = minimization_options
#        def minus_logpdf(x):
#            return -logpdf(np.insert(x, pars_insert, pars_val))
#        if pars_bounds is not None:
#            pars_bounds=np.array(pars_bounds)
#            if len(pars_bounds)!=len(pars_init):
#                raise Exception("The input argument 'pars_bounds' should be either 'None' or have the same length of 'pars'.")
#            if not ((np.all(pars_val >= pars_bounds[pars, 0]) and np.all(pars_val <= pars_bounds[pars, 1]))):
#                print("Parameter values",pars_val,"lies outside parameters bounds",pars_bounds,".")
#                return []
#        if pars_bounds is None:
#            ml = optimize.minimize(minus_logpdf, pars_init_reduced, *args, options=options, **kwargs)
#        else:
#            pars_bounds_reduced = np.delete(pars_bounds, pars,axis=0)
#            pars_bounds_reduced = np.array(pars_bounds_reduced)
#            bounds=optimize.Bounds(pars_bounds_reduced[:, 0], pars_bounds_reduced[:, 1])
#            ml = optimize.minimize(minus_logpdf, pars_init_reduced, *args, bounds=bounds, options=options, **kwargs)
#        return [np.insert(ml['x'], pars_insert, pars_val, axis=0), -ml['fun']]
#
#    def __compute_maximum_sample(self,
#                                 X: Array,
#                                 Y: Array
#                                ) -> list:
#        """
#        """
#        X = np.array(X)
#        Y = np.array(Y)
#        y_max = np.amax(Y)
#        pos_max = np.where(Y == y_max)[0][0]
#        Y[pos_max] = -np.inf
#        y_next_max = np.amax(Y)
#        pos_next_max = np.where(Y == y_next_max)[0][0]
#        x_max = X[pos_max]
#        x_next_max = X[pos_next_max]
#        return [x_max, y_max, np.abs(x_next_max-x_max), np.abs(y_next_max-y_max)]
#
#    def __compute_profiled_maximum_sample(self,
#                                          pars: Array,
#                                          pars_val: Array,
#                                          X: Array,
#                                          Y: Array,
#                                          binwidths: Union[str,Number,Array] = "auto"):
#        """
#        """
#        pars = np.array(pars)
#        pars_val = np.array(pars_val)
#        X = np.array(X)
#        Y = np.array(Y)
#        if type(binwidths) == float or type(binwidths) == int:
#            binwidths = np.full(len(pars), binwidths)
#        elif type(binwidths) == list or type(binwidths) == np.ndarray:
#            binwidths = np.array(binwidths)
#            if len(binwidths) != len(pars):
#                raise Exception("List of bin widths specifications has length different from the number of parameters.")
#        elif type(binwidths) == str and binwidths != "auto":
#            raise Exception("Invalid bin widths specification. Only allowed string is 'auto'.")
#        if type(binwidths) != "auto":
#            binwidths = np.array(binwidths)
#            slicings_list: list = []
#            for i in range(len(pars)):
#                slicings_list.append([p > pars_val[i]-binwidths[i]/2 and p <
#                                 pars_val[i]+binwidths[i]/2 for p in X[:, pars[i]]])
#            slicing = np.array(np.prod(np.array(slicings_list), axis=0).astype(bool))
#            npoints = np.count_nonzero(slicing)
#            y_max = np.amax(Y[slicing])
#            pos_max = np.where(Y == y_max)[0][0]
#            Y[pos_max] = -np.inf
#            y_next_max = np.amax(Y[slicing])
#            pos_next_max = np.where(Y == y_next_max)[0][0]
#            x_max = X[pos_max]
#            x_next_max = X[pos_next_max]
#        else:
#            binwidths = np.full(len(pars), 0.001)
#            npoints = 0
#            slicing = np.array([])
#            while npoints < 2:
#                binwidths = binwidths+0.001
#                slicings_list = []
#                for i in range(len(pars)):
#                    slicings_list.append([p > pars_val[i]-binwidths[i]/2 and p <
#                                     pars_val[i]+binwidths[i]/2 for p in X[:, pars[i]]])
#                slicing = np.array(np.prod(np.array(slicings_list), axis=0).astype(bool))
#                npoints = np.count_nonzero(slicing)
#            y_max = np.amax(Y[slicing])
#            pos_max = np.where(Y == y_max)[0][0]
#            Y[pos_max] = -np.inf
#            y_next_max = np.amax(Y[slicing])
#            pos_next_max = np.where(Y == y_next_max)[0][0]
#            x_max = X[pos_max]
#            x_next_max = X[pos_next_max]
#        return [x_max, y_max, np.abs(x_next_max-x_max), np.abs(y_next_max-y_max), npoints]


class Plotter(ObjectManager):
    """
    """
    managed_object_name: str
    def __init__(self,
                 managed_object: Any
                ) -> None:
        """
        """
        # Attributes type declatation
        self._ManagedObject: Any
        # Initialize parent ManagedObject class (sets self._ManagedObject)
        super().__init__(managed_object = managed_object)
        # Set verbosity
        verbose, _ = self._ManagedObject.get_verbosity(self._ManagedObject._verbose)
        # Initialize object
        print(header_string_1,"\nInitializing Plotter.\n", show = verbose)

    def plot_corr_matrix(self,
                         X: Array
                        ) -> None:
        df = pd.DataFrame(X)
        f = plt.figure(figsize=(18, 18))
        plt.matshow(df.corr(), fignum=f.number) # type: ignore
        cb = plt.colorbar()
        plt.grid(False)
        plt.show()
        plt.close()

    def savefig(self,
                figure_path: StrPath,
                **kwargs: Dict
               ) -> None:
        """
        """
        if 'win32' in sys.platform or "win64" in sys.platform:
            plt.savefig("\\\\?\\" + str(figure_path), **kwargs)
        else:
            plt.savefig(figure_path, **kwargs)