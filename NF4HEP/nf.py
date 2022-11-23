__all__ = ["NFFileManager",
           "NFMain",
           "NFManager",
           "NFPredictionsManager",
           "NFInference",
           "NFPlotter"]

import numpy as np

from asyncio import base_subprocess
from numpy import typing as npt
from pathlib import Path
from timeit import default_timer as timer

from typing import Union, List, Dict, Callable, Tuple, Optional, NewType, Type, Generic, Any, TypeVar, TYPE_CHECKING
from typing_extensions import TypeAlias
from NF4HEP.utils.custom_types import Array, ArrayInt, ArrayStr, DataType, StrPath, IntBool, StrBool, StrList, FigDict, LogPredDict, Number, DTypeStr, DTypeStrList, DictStr
from NF4HEP.bijectors import arqspline
from NF4HEP.bijectors import crqspline
from NF4HEP.bijectors import maf
from NF4HEP.bijectors import realnvp
from NF4HEP.bijectors.arqspline import ARQSplineChain
from NF4HEP.bijectors.crqspline import CRQSplineChain
from NF4HEP.bijectors.maf import MAFChain
from NF4HEP.bijectors.realnvp import RealNVPChain, RealNVPBijector, RealNVPNetwork
from NF4HEP.utils import corner
from NF4HEP.utils import utils
from NF4HEP.utils.corner import extend_corner_range
from NF4HEP.utils.corner import get_1d_hist
from NF4HEP.utils.resources import Resources
from NF4HEP.utils.verbosity import print
from NF4HEP.utils.verbosity import Verbosity
from NF4HEP.utils import mplstyle_path
from NF4HEP.inputs.data import DataMain, DataFileManager
from NF4HEP.inputs.distributions import Distribution
from NF4HEP.base import Name, FileManager, PredictionsManager, FiguresManager, Inference, Plotter

header_string_1 = "=============================="
header_string_2 = "------------------------------"

class NFFileManager(FileManager):
    """
    """
    managed_object: str = "NF"
    def __init__(self,
                 name: Optional[str] = None,
                 input_file: Optional[StrPath] = None,
                 input_data_main_file: Optional[StrPath] = None,
                 output_folder: Optional[StrPath] = None, 
                 verbose: Optional[IntBool] = None
                ) -> None:
        # Attributes type declarations (from parent FileManager class)
        self._input_data_main_file: Optional[Path]
        self._input_file: Optional[Path]
        self._name: Name
        self._output_folder: Path
        self._ManagedObject: "NFMain"
        # Attributes type declarations
        #        
        # Define self.input_file, self.output_folder
        super().__init__(name=name,
                         input_file=input_file,
                         output_folder=output_folder,
                         verbose=verbose)
        # Set verbosity
        verbose, verbose_sub = self.get_verbosity(verbose)
        # Initialize object
        self.input_data_main_file = input_data_main_file if input_data_main_file is None else Path(input_data_main_file)

    @property
    def ManagedObject(self) -> "NFMain":
        return self._ManagedObject

    @ManagedObject.setter
    def ManagedObject(self,
                      managed_object: "NFMain"
                     ) -> None:
        try:
            self._ManagedObject
            raise Exception("The 'ManagedObject' attribute is automatically set when initialising the NFMain object and cannot be replaced.")
        except:
            self._ManagedObject = managed_object

    @property
    def input_data_main_file(self) -> Optional[Path]:
        return self._input_data_main_file

    @input_data_main_file.setter
    def input_data_main_file(self, 
                             input_data_main_file: Optional[Path] = None,
                            ) -> None:
        self._input_data_main_file = input_data_main_file

    @property
    def input_flow_h5_file(self) -> Optional[Path]:
        if self._input_file is not None:
            return self._input_file.parent.joinpath(self._input_file.stem + "_flow.h5")
        else:
            return None

    @property
    def input_history_json_file(self) -> Optional[Path]:
        if self._input_file is not None:
            return self._input_file.parent.joinpath(self._input_file.stem + "_history.json")
        else:
            return None

    @property
    def input_model_h5_file(self) -> Optional[Path]:
        if self._input_file is not None:
            return self._input_file.parent.joinpath(self._input_file.stem + "_model.h5")
        else:
            return None

    @property
    def input_model_json_file(self) -> Optional[Path]:
        if self._input_file is not None:
            return self._input_file.parent.joinpath(self._input_file.stem + "_model.json")
        else:
            return None

    @property
    def output_flow_h5_file(self) -> Path:
        return self._output_folder.joinpath(self.name+"_flow.h5")

    @property
    def output_flow_json_file(self) -> Path:
        return self._output_folder.joinpath(self.name+"_flow.json")
    
    @property
    def output_history_json_file(self) -> Path:
        return self._output_folder.joinpath(self.name+"_history.json")

    @property
    def output_model_graph_file(self) -> Path:
        return self._output_folder.joinpath(self.name+"_model_graph.pdf")

    @property
    def output_model_h5_file(self) -> Path:
        return self._output_folder.joinpath(self.name+"_model.h5")

    @property
    def output_model_json_file(self) -> Path:
        return self._output_folder.joinpath(self.name+"_model.json")

    def load(self, verbose: Optional[IntBool] = None) -> None:
        """
        """
        verbose, _ = self.get_verbosity(verbose)
        self.load_log(verbose = verbose)
        self.load_object(verbose = verbose)
        self.load_flow(verbose = verbose)
        self.load_model(verbose = verbose)
        self.load_history(verbose = verbose)
        self.load_predictions(verbose = verbose)

    def load_object(self, verbose: Optional[IntBool] = None) -> None:
        verbose, _ = self.get_verbosity(verbose)

    def load_flow(self, verbose: Optional[IntBool] = None) -> None:
        verbose, _ = self.get_verbosity(verbose)

    def load_history(self, verbose: Optional[IntBool] = None) -> None:
        verbose, _ = self.get_verbosity(verbose)

    def load_model(self, verbose: Optional[IntBool] = None) -> None:
        verbose, _ = self.get_verbosity(verbose)

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
        self.save_flow(timestamp = timestamp, overwrite = overwrite, verbose = verbose)
        self.save_model(timestamp = timestamp, overwrite = overwrite, verbose = verbose)
        self.save_history(timestamp = timestamp, overwrite = overwrite, verbose = verbose)
        self.save_predictions(timestamp = timestamp, overwrite = overwrite, verbose = verbose)
        self.save_log(timestamp = timestamp, overwrite = overwrite, verbose = verbose)

    def save_object(self, 
                    timestamp: Optional[str] = None, 
                    overwrite: StrBool = False, 
                    verbose: Optional[IntBool] = None
                   ) -> None:
        verbose, verbose_sub = self.get_verbosity(verbose)

    def save_flow(self, 
                  timestamp: Optional[str] = None, 
                  overwrite: StrBool = False, 
                  verbose: Optional[IntBool] = None
                 ) -> None:
        verbose, verbose_sub = self.get_verbosity(verbose)

    def save_history(self, 
                     timestamp: Optional[str] = None, 
                     overwrite: StrBool = False, 
                     verbose: Optional[IntBool] = None
                    ) -> None:
        verbose, verbose_sub = self.get_verbosity(verbose)

    def save_model(self, 
                   timestamp: Optional[str] = None, 
                   overwrite: StrBool = False, 
                   verbose: Optional[IntBool] = None
                  ) -> None:
        verbose, verbose_sub = self.get_verbosity(verbose)


class NFChain(Verbosity):
    allowed_chain_types: TypeAlias = Union[ARQSplineChain,CRQSplineChain,MAFChain,RealNVPChain]
    allowed_bijector_types: TypeAlias = Union[ARQSplineBijector,CRQSplineBijector,MAFBijector,RealNVPBijector]
    allowed_NN_types: TypeAlias = Union[ARQSplineNetwork,CRQSplineNetwork,MAFNetwork,RealNVPNetwork]
    """
    """
    def __init__(self,
                 flow_name: str,
                 chain: allowed_chain_types,
                 verbose: Optional[IntBool] = None
                ) -> None:
        """
        """
        # Declaration of needed types for attributes
        self._Chain: NFChain.allowed_chain_types
        self._name: str
        self._Bijector: NFChain.allowed_bijector_types
        self._nbijectors: int
        self._NN: NFChain.allowed_NN_types
        # Initialise parent Verbosity class
        super().__init__(verbose)
        # Set verbosity
        verbose, _ = self.get_verbosity(verbose)
        # Initialize object
        print(header_string_1, "\nInitializing Chain object.\n", show = verbose)
        self.Chain = chain

    @property
    def allowed_chains_names(self) -> List[str]:
        return ["ARQSplineFlow","CRQSplineFlow","MAFFlow","RealNVPFlow"]

    @property
    def Chain(self) -> allowed_chain_types:
        return self._Chain

    @Chain.setter
    def Chain(self,
              chain: allowed_chain_types,
             ) -> None:
        try:
            self._Chain
            raise Exception("The 'Chain' attribute is automatically set when initialising the NFChain object and cannot be replaced.")
        except:
            if chain.name in self.allowed_chains_names:
                self._Chain = chain
                self._chain_name = chain.name
            else:
                raise Exception("The chain",chain.name,"is not supported.")

    @property
    def chain_name(self) -> str:
        return self._chain_name

    @property
    def Bijector(self) -> allowed_bijector_types:
        return self._Chain.Bijector

    @property
    def NN(self) -> allowed_NN_types:
        return self.Bijector._NN

    @property
    def nbijectors(self) -> int:
        return self.Chain.nbijectors

    def __call__(self) -> allowed_bijector_types:
        return self.Chain


class NFMain(Verbosity):
    managed_object : str = "NFMain"
    allowed_chain_types: TypeAlias = Union[ARQSplineChain,CRQSplineChain,MAFChain,RealNVPChain]

    """
    """
    def __init__(self,
                 file_manager: NFFileManager,
                 data: DataMain,
                 base_distribution: Distribution,
                 nf_chain: NFChain,
                 seed: Optional[int] = None,
                 verbose: IntBool = True
                ) -> None:
        """
        """
        # Attributes type declatation
        self._log: LogPredDict
        self._name: str
        self._BaseDistribution: Distribution
        self._Data: DataMain
        self._Figures: NFFiguresManager
        self._FileManager: NFFileManager
        self._Chain: NFChain
        self._Inference: NFInference
        self._NFManager: NFManager
        self._Plotter: NFPlotter
        self._Predictions: NFPredictionsManager
        self._Trainer: NFTrainer
        # Initialize parent Verbosity class
        super().__init__(verbose)
        # Set verbosity
        verbose, verbose_sub = self.get_verbosity(verbose)
        # Initialize object
        timestamp = utils.generate_timestamp()
        print(header_string_1, "\nInitializing NFMain object.\n", show = verbose)
        self._log = {}
        print(header_string_2,"\nSetting FileManager.\n", show = verbose)
        self.FileManager = file_manager # also sets the NFMain managed object FileManager attribute
        self.FileManager.ManagedObject = self
        self.Predictions = NFPredictionsManager(nf_main = self)
        print(header_string_2,"\nSetting Predictions.\n", show = self.verbose)
        self.Figures = NFFiguresManager(nf_main = self)
        print(header_string_2,"\nSetting Figures.\n", show = verbose)
        if self.FileManager.input_file is None:
            self.seed = seed if seed is not None else 0
            print(header_string_2,"\nInitializing new NFMain object.\n", show = verbose)
            self.__init_data_main(data = data, verbose = verbose)
            if base_distribution is not None:
                self.BaseDistribution = base_distribution
            else:
                self.BaseDistribution = Distribution(ndims = self.Data.DataManager.ndims,
                                                     seed = seed,
                                                     dtype = self.Data.InputData.dtype_required,
                                                     default_dist = "Normal",
                                                     tf_dist = None,
                                                     verbose = verbose)
            if nf_chain is not None:
                self.Chain = nf_chain
            else:
                raise Exception("When no input file is specified the 'nf_chain' argument needs to be different from 'None'.")
            self.NFManager = NFManager(nf_main = self, verbose = verbose)
            self.Trainer = NFTrainer(nf_main = self, verbose = verbose)
        else:
            print(header_string_2,"\nLoading existing NFMain object.\n", show = verbose)
            for attr in [nf_chain,base_distribution,data,seed]:
                if attr is not None:
                    print(header_string_2,"\nWarning: an input file was specified and the argument '",attr,"' will be ignored. The related attribute will be set from the input file.\n", show = True)
        self.Inference = NFInference(nf_main = self)
        print(header_string_2,"\nSetting Inference.\n", show = verbose)
        self.Plotter = NFPlotter(nf_main = self)
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
               "_Figures",
               "_FileManager",
               "_Inference",
               "_NFManager",
               "_Plotter",
               "_Predictions",
               "_Trainer"
              ]
        return tmp

    @property
    def FileManager(self) -> NFFileManager:
        return self._FileManager

    @FileManager.setter
    def FileManager(self,
                    file_manager: NFFileManager,
                   ) -> None:
        try:
            self._FileManager
            raise Exception("The 'FileManager' attribute is automatically set when initialising the NFMain object and cannot be manually set.")
        except:
            self._FileManager = file_manager
            self._FileManager.ManagedObject = self

    @property
    def name(self) -> str:
        return self._FileManager.name

    @property
    def Predictions(self) -> "NFPredictionsManager":
        return self._Predictions

    @Predictions.setter
    def Predictions(self,
                    predictions: "NFPredictionsManager"
                   ) -> None:
        self._Predictions = predictions

    @property
    def Figures(self) -> "NFFiguresManager":
        return self._Figures

    @Figures.setter
    def Figures(self,
                figures: "NFFiguresManager"
               ) -> None:
        self._Figures = figures

    @property
    def Data(self) -> DataMain:
        return self._Data

    @Data.setter
    def Data(self,
             data_main: DataMain
            ) -> None:
        try:
            self._Data
            raise Exception("The 'Data' attribute is automatically set when initialising the NFMain object and cannot be manually set.")
        except:
            self._Data = data_main
        print(header_string_2,"\nSetting Data.\n", show = self.verbose)

    @property
    def BaseDistribution(self) -> Distribution:
        return self._BaseDistribution

    @BaseDistribution.setter
    def BaseDistribution(self,
             base_distribution: Distribution
            ) -> None:
        self._BaseDistribution = base_distribution
        print(header_string_2,"\nSetting BaseDistriburion.\n", show = self.verbose)

    @property
    def Chain(self) -> NFChain:
        return self._Chain

    @Chain.setter
    def Chain(self,
             nf_chain: NFChain
            ) -> None:
        try:
            self._Chain
            raise Exception("The 'Chain' attribute is automatically set when initialising the NFMain object and cannot be manually set.")
        except:
            self._Chain = nf_chain
        print(header_string_2,"\nSetting NF Chain.\n", show = self.verbose)

    @property
    def NFManager(self) -> "NFManager":
        return self._NFManager

    @NFManager.setter
    def NFManager(self,
                  nf_manager: "NFManager"
                 ) -> None:
        try:
            self._NFManager
            raise Exception("The 'NFManager' attribute is automatically set when initialising the NFMain object and cannot be manually set.")
        except:
            self._NFManager = nf_manager
        print(header_string_2,"\nSetting NFManager.\n", show = self.verbose)

    @property
    def Trainer(self) -> "NFTrainer":
        return self._Trainer

    @Trainer.setter
    def Trainer(self,
                nf_trainer: "NFTrainer"
               ) -> None:
        try:
            self._Trainer
            raise Exception("The 'Trainer' attribute is automatically set when initialising the NFMain object and cannot be manually set.")
        except:
            self._Trainer = nf_trainer
        print(header_string_2,"\nSetting Trainer.\n", show = self.verbose)

    @property
    def Inference(self) -> "NFInference":
        return self._Inference

    @Inference.setter
    def Inference(self,
                  inference: "NFInference"
                 ) -> None:
        try:
            self._Inference
            raise Exception("The 'Inference' attribute is automatically set when initialising the NFMain object and cannot be manually set.")
        except:
            self._Inference = inference
        print(header_string_2,"\nSetting Inference.\n", show = self.verbose)

    @property
    def Plotter(self) -> "NFPlotter":
        return self._Plotter

    @Plotter.setter
    def Plotter(self,
                plotter: "NFPlotter"
               ) -> None:
        try:
            self._Plotter
            raise Exception("The 'Inference' attribute is automatically set when initialising the NFMain object and cannot be manually set.")
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

    def __init_data_main(self,
                         data: Optional[DataMain],
                         verbose: Optional[IntBool] = None
                        ) -> None:
        verbose, verbose_sub = self.get_verbosity(verbose)
        if self.FileManager.input_data_main_file is None:
            if data is not None:
                self.Data = data
            else:
                raise Exception("When no input file is specified the 'data' argument needs to be different from 'None'.")
        else:
            if data is not None:
                print(header_string_2,"\nWarning: an input datamain file was specified and the argument '",data,"' will be ignored. The related attribute will be set from the input file.\n", show = True)
            DataFileManagerLoad = DataFileManager(name = None,
                                                  input_file = self.FileManager.input_data_main_file,
                                                  output_folder = None,
                                                  load_on_RAM = False, # this is ignored
                                                  verbose = verbose_sub)
            self.Data = DataMain(file_manager = DataFileManagerLoad,
                                 pars_manager = None,
                                 input_data = None,
                                 npoints = None,
                                 preprocessing = None,
                                 seed = None,
                                 verbose = verbose_sub)

class NFManager(Verbosity):
    """
    Manages nf object
    """
    def __init__(self,
                 nf_main: NFMain,
                 verbose: Optional[IntBool] = None
                ) -> None:
        """
        """
    pass


class NFTrainer(Verbosity):
    """
    Manages nf training
    """
    def __init__(self,
                 nf_main: NFMain,
                 verbose: Optional[IntBool] = None
                ) -> None:
        """
        """
    pass


class NFPredictionsManager(PredictionsManager):
    """
    """
    def __init__(self,
                 nf_main: NFMain
                ) -> None:
        super().__init__(obj = nf_main)

    def init_predictions(self):
        pass

    def reset_predictions(self):
        pass

    def validate_predictions(self):
        pass


class NFFiguresManager(FiguresManager):
    """
    """
    def __init__(self,
                 nf_main: NFMain
                ) -> None:
        super().__init__(obj = nf_main)


class NFInference(Inference):
    """
    """
    def __init__(self,
                 nf_main: NFMain
                ) -> None:
        super().__init__(obj = nf_main)


class NFPlotter(Plotter):
    """
    """
    def __init__(self,
                 nf_main: NFMain
                ) -> None:
        super().__init__(obj = nf_main)