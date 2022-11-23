__all__ = ["DataFileManager",
           "DataParsManager",
           "DataPredictions",
           "DataInference",
           "DataPlots",
           "Data"]

import numpy as np

from matplotlib import pyplot as plt # type:  ignore
from numpy import typing as npt
from pathlib import Path
from typing import Union, List, Dict, Callable, Tuple, Optional, NewType, Type, Generic, Any, TypeVar, TYPE_CHECKING

from NF4HEP import Figures
from NF4HEP import FileManager
from NF4HEP import Inference
from NF4HEP import mplstyle_path
from NF4HEP import ParsManager
from NF4HEP import Plots
from NF4HEP import Predictions
from NF4HEP import print
from NF4HEP import utils
from NF4HEP import Verbosity

Array = Union[List, npt.NDArray[Any]]
ArrayInt = Union[List[int], npt.NDArray[np.int_]]
ArrayStr = Union[List[str], npt.NDArray[np.str_]]
StrPath = Union[str,Path]
IntBool = Union[int, bool]
StrBool = Union[str, bool]
LogPredDict = Dict[str,Dict[str,Any]]

header_string = "=============================="
footer_string = "------------------------------"

class DataFileManager(FileManager):
    obj_name = "Data"

    def __init__(self,
                 name: Union[str,None] = None,
                 input_file: Optional[StrPath] = None, 
                 output_folder: Optional[StrPath] = None, 
                 verbose: Union[int,bool,None] = None
                ) -> None:
        # Define self.input_file, self.output_folder
        super().__init__(name=name,
                         input_file=input_file,
                         output_folder=output_folder,
                         verbose=verbose)
        verbose, verbose_sub = self.set_verbosity(verbose)
        self.__define_predictions_files()
        
    def __define_predictions_files(self) -> None:
        self.output_figures_folder = self.check_create_folder(self.output_folder.joinpath("figures"))
        self.output_figures_base_file_name = self.name_str+"_figure"
        self.output_figures_base_file_path = self.output_figures_folder.joinpath(self.output_figures_base_file_name)
        self.output_predictions_json_file = self.output_folder.joinpath(self.name_str+"_predictions.json")

class DataParsManager(ParsManager):
    """
    """
    def __init__(self,
                 pars_central: Optional[Array],
                 pars_pos_poi: Optional[ArrayInt],
                 pars_pos_nuis: Optional[ArrayInt],
                 pars_labels: Optional[ArrayStr],
                 pars_bounds: Optional[Array],
                 verbose: Optional[IntBool] = None) -> None:
        super().__init__(pars_central = pars_central,
                         pars_pos_poi = pars_pos_poi,
                         pars_pos_nuis = pars_pos_nuis,
                         pars_labels = pars_labels,
                         pars_bounds = pars_bounds,
                         verbose = verbose)

class DataPredictions(Predictions):
    """
    """
    def __init__(self,
                 obj_name: str,
                 verbose = None) -> None:
        super().__init__(obj_name = obj_name,
                         verbose=verbose)

class DataInference(Inference):
    """
    """
    def __init__(self,
                 verbose = None) -> None:
        super().__init__(verbose=verbose)
        # Declaration of needed types for attributes (Attributes of Lik available through mixin)
        self.log: LogPredDict
        self.parameters: DataParsManager
        self.predictions: DataPredictions
        # Declaration of methods called from Lik
        self.save: Callable
        self.save_log: Callable


class DataPlots(Plots):
    """
    """
    def __init__(self,
                 verbose = None) -> None:
        super().__init__(verbose=verbose)
        # Declaration of needed types for attributes (Attributes of Lik available through mixin)
        self.log: LogPredDict
        self.file_manager: DataFileManager
        self.figures: Figures
        self.name_str: str
        self.parameters: DataParsManager
        self.predictions: DataPredictions
        # Declaration of methods called from Lik
        self.save: Callable
        self.save_log: Callable


class Data(Verbosity):
    """
    This class is a container for the the :mod:`Histfactory <histfactory>` object created from an ATLAS histfactory workspace. It allows one to import histfactory workspaces, 
    read parameters and logpdf using the |pyhf_link| package, create :class:`Lik <DNNLikelihood.Lik>` objects and save them for later use
    (see the :mod:`Likelihood <likelihood>` object documentation).
    """

    def __init__(self,
                 file_manager: DataFileManager,
                 parameters: DataParsManager,
                 verbose: IntBool = True
                ) -> None:
        """
        """
        # Declaration of needed types for attributes
        self.log: LogPredDict
        self.name: str
        # Initialization of parent class
        super().__init__(verbose)
        # Initialization of verbosity mode
        verbose, verbose_sub = self.set_verbosity(self.verbose)
        # Initialization of object
        timestamp = utils.generate_timestamp()
        print(header_string, "\nInitialize Histfactory object.\n", show=verbose)
        self.file_manager = file_manager
        self.parameters = parameters
        self.predictions = DataPredictions(self.file_manager.obj_name, verbose=verbose_sub)         # Predictions need to be saved and loaded in a robust way
        self.figures = Figures(verbose=verbose_sub)