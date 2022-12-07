__all__ = ["NFFileManager",
           "NFChain",
           "NFMain",
           "NFManager",
           "NFTrainer",
           "NFPredictionsManager",
           "NFInference",
           "NFPlotter"]

from datetime import datetime
import numpy as np
import tensorflow as tf # type: ignore
import tensorflow.compat.v1 as tf1 # type: ignore
from tensorflow.keras import Input # type: ignore
from tensorflow.keras import layers, initializers, regularizers, constraints, callbacks, optimizers, metrics, losses # type: ignore
from tensorflow.keras.models import Model # type: ignore
from tensorflow.keras.layers import Layer #type: ignore
import tensorflow_probability as tfp # type: ignore
tfd = tfp.distributions
tfb = tfp.bijectors

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
from NF4HEP.bijectors.arqspline import ARQSplineNetwork, ARQSplineBijector
from NF4HEP.bijectors.crqspline import CRQSplineNetwork, CRQSplineBijector
from NF4HEP.bijectors.maf import MAFNetwork, MAFBijector
from NF4HEP.bijectors.realnvp import RealNVPNetwork, RealNVPBijector
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
    managed_object_name: str = "NFMain"
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

    #@property
    #def ManagedObject(self) -> "NFMain":
    #    return self._ManagedObject
#
    #@ManagedObject.setter
    #def ManagedObject(self,
    #                  managed_object: "NFMain"
    #                 ) -> None:
    #    try:
    #        self._ManagedObject
    #        raise Exception("The 'ManagedObject' attribute is automatically set when initialising the NFMain object and cannot be replaced.")
    #    except:
    #        self._ManagedObject = managed_object

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


class NFChain(tfb.Chain, Verbosity): # type: ignore
    allowed_bijector_types: TypeAlias = Union[ARQSplineBijector,CRQSplineBijector,MAFBijector,RealNVPBijector]
    allowed_NN_types: TypeAlias = Union[ARQSplineNetwork,CRQSplineNetwork,MAFNetwork,RealNVPNetwork]
    """
    model_chain_inputs can be of the following form:
    .. code-block:: python

        model_chain_inputs = {"nbijectors": 2,
                              "batch_normalization": False}
    """
    def __init__(self,
                 bijector: allowed_bijector_types,
                 model_chain_inputs: Optional[Dict[str, Any]] = None,
                 verbose: Optional[IntBool] = None
                ) -> None:
        # Attributes type declarations
        self._batch_normalization: bool
        self._Bijector: NFChain.allowed_bijector_types
        self._model_chain_inputs: Dict[str, Any]
        self._nbijectors: int
        # Initialise parent Verbosity class
        Verbosity.__init__(self, verbose)
        # Set verbosity
        verbose, _ = self.get_verbosity(verbose)
        # Initialize object
        print(header_string_1, "\nInitializing NFChain object.\n", show = verbose)
        self.__set_model_chain_inputs(model_chain_inputs = model_chain_inputs, verbose = verbose)
        self._Bijector = bijector
        ndims = self._bijector._ndims
        permutation = tf.cast(np.concatenate((np.arange(int(ndims/2), ndims), np.arange(0, int(ndims/2)))), tf.int32)
        Permute = tfb.Permute(permutation=permutation)
        print("\n",tfb.BatchNormalization())
        print("\n",self._bijector)
        print("\n",Permute)
        self._bijectors = []
        for _ in range(self._num_bijectors):
            if self._batch_normalization:
                self._bijectors.append(tfb.BatchNormalization())
            self._bijectors.append(self._Bijector)
            self._bijectors.append(Permute)
        tfb.Chain.__init__(self, bijectors=list(reversed(self._bijectors[:-1])), name=self.name)

    @property
    def allowed_chains_names(self) -> List[str]:
        return ["ARQSplineChain","CRQSplineChain","MAFChain","RealNVPChain"]

    @property
    def NN(self) -> allowed_NN_types:
        return self._Bijector._NN

    @property
    def batch_normalization(self) -> bool:
        return self._batch_normalization

    @property
    def Bijector(self) -> Union["ARQSplineBijector", "CRQSplineBijector", "MAFBijector", "RealNVPBijector"]: # type: ignore
        return self._Bijector

    @property
    def model_chain_inputs(self) -> Dict[str, Any]:
        return self._model_chain_inputs

    @property
    def nbijectors(self) -> int:
        return self._nbijectors

    @property
    def ndims(self) -> int:
        return self._Bijector._ndims

    def __set_model_chain_inputs(self,
                                 model_chain_inputs: Optional[Dict[str, Any]] = None,
                                 verbose: Optional[IntBool] = None
                                ) -> None:
        if model_chain_inputs is None:
            model_chain_inputs = {}
        try:
            self._num_bijectors = model_chain_inputs["nbijectors"]
        except:
            print("WARNING: The 'model_chain_inputs' argument misses the mandatory 'nbijectors' item. The corresponding attribute will be set to a default of 2.")
        utils.check_set_dict_keys(dic = model_chain_inputs, 
                                  keys = ["nbijectors","batch_normalization"],
                                  vals = [2,False],
                                  verbose = verbose)
        self._batch_normalization = model_chain_inputs["batch_normalization"]
        self._model_chain_inputs = model_chain_inputs


class NFMain(Verbosity):
    managed_object : str = "NFMain"
    allowed_bijector_types: TypeAlias = Union[ARQSplineBijector,CRQSplineBijector,MAFBijector,RealNVPBijector]
    """
    """
    def __init__(self,
                 file_manager: NFFileManager,
                 data: DataMain,
                 bijector: allowed_bijector_types,
                 model_chain_inputs: Optional[Dict[str, Any]] = None,
                 base_distribution: Optional[Union[str,tfp.distributions.distribution.AutoCompositeTensorDistribution]] = None, # type: ignore
                 seed: Optional[int] = None,
                 verbose: IntBool = True
                ) -> None:
        """
        """
        # Attributes type declatation
        self._log: LogPredDict
        self._name: str
        self._BaseDistribution: tfp.distributions.distribution.AutoCompositeTensorDistribution
        self._BaseDistribution_string: str
        self._Data: DataMain
        self._Figures: NFFiguresManager
        self._FileManager: NFFileManager
        self._Bijector: NFMain.allowed_bijector_types
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
            self.BaseDistribution = base_distribution
            self.Bijector = bijector
            self.Chain = NFChain(bijector = bijector,
                                 model_chain_inputs = model_chain_inputs,
                                 verbose = verbose_sub)
            self.NFManager = NFManager(nf_main = self, verbose = verbose)
            self.Trainer = NFTrainer(nf_main = self, verbose = verbose)
        else:
            print(header_string_2,"\nLoading existing NFMain object.\n", show = verbose)
            for attr in [bijector,model_chain_inputs,base_distribution,data,seed]:
                if attr is not None:
                    print(header_string_2,"\nWarning: an input file was specified and the argument '",attr,"' will be ignored. The related attribute will be set from the input file.\n", show = True)
            self.FileManager.load(verbose = verbose)
        self.Inference = NFInference(nf_main = self)
        print(header_string_2,"\nSetting Inference.\n", show = verbose)
        self.Plotter = NFPlotter(nf_main = self)
        print(header_string_2,"\nSetting Plotter.\n", show = verbose)
        if self.FileManager.input_file is None:
            self.log = {timestamp: {"action": "object created from input arguments"}}
            #self.FileManager.save(timestamp = timestamp, overwrite = False, verbose = verbose_sub)
        else:
            self.log = {timestamp: {"action": "object reconstructed from loaded files"}}
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
    def log(self) -> LogPredDict:
        return self._log

    @log.setter
    def log(self,
            log_action: LogPredDict
           ) -> None:
        try: 
            self._log
        except:
            self._log = {}
        if isinstance(log_action,dict):
            self._log = {**self._log, **log_action}
        else:
            raise TypeError("Only log-type dictionaries can be added to the '_log' dictionary.")

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
    def BaseDistribution(self) -> Union[str,tfp.distributions.distribution.AutoCompositeTensorDistribution]:
        return self._BaseDistribution

    @BaseDistribution.setter
    def BaseDistribution(self,
                         base_distribution: Optional[Union[str,tfp.distributions.distribution.AutoCompositeTensorDistribution]] = None, # type: ignore
                        ) -> None:
        dist_str: str = str(base_distribution) if type(base_distribution) is str else ""
        dist_dic: dict = dict(base_distribution) if type(base_distribution) is dict else {}
        dist_obj: tfp.distributions.distribution.AutoCompositeTensorDistribution = base_distribution if (type(base_distribution) is not str and type(base_distribution) is not dict) else None
        self.base_distribution = None
        if type(base_distribution) is str:
            dist = None
            if "(" in dist_str:
                try:
                    eval("tfd." + dist_str.replace("tfd.", ""))
                    dist_string = "tfd." + dist_str.replace("tfd.", "")
                except:
                    eval(dist_str)
                    dist_string = dist_str
            else:
                try:
                    eval("tfd." + dist_str.replace("tfd.", "") +"()")
                    dist_string = "tfd." + dist_str.replace("tfd.", "") +"()"
                except:
                    eval(dist_str +"()")
                    dist_string = dist_str +"()"
        elif type(base_distribution) is dict:
            dist = None
            try:
                name = dist_dic["name"]
            except:
                raise Exception("The distribution ", str(dist_dic), " has unspecified name.")
            try:
                args = dist_dic["args"]
            except:
                args = []
            try:
                kwargs = dist_dic["kwargs"]
            except:
                kwargs = {}
            dist_string = utils.build_method_string_from_dict("tfd", name, args, kwargs)
        elif dist_obj is not None:
            try:
                eval(dist_obj)
                dist_string = None
                dist = dist_obj
            except:
                raise Exception("Could not set base distribution. The 'distribution' input argument does not have a valid format.")
        else:
            raise Exception("Could not set base distribution. The 'distribution' input argument does not have a valid format.")
        if dist_string is None and dist is not None:
            self._BaseDistribution_string = str(dist_obj)
            self._BaseDistribution = dist
        elif dist_string is not None and dist is None:
            try:
                dist = eval(dist_string)
                self._BaseDistribution_string = "tfd.Sample("+dist_string+",sample_shape=["+str(self.ndims)+"])"
                self._BaseDistribution = tfd.Sample(dist,sample_shape=[self.ndims])
                print("Base distribution set to:", self._BaseDistribution_string, ".\n", show = self._verbose)
            except Exception as e:
                print(e)
                raise Exception("Could not set base distribution", dist_string, "\n")
        else:
            raise Exception("Could not set base distribution", dist_string, "\n")
        timestamp = "datetime_"+datetime.now().strftime("%Y_%m_%d_%H_%M_%S_%fZ")[:-3]
        self.log = {utils.generate_timestamp(): {"action": "base distribution set",
                                                 "distribution": self._BaseDistribution_string}}

    @property
    def Bijector(self) -> allowed_bijector_types:
        return self._Bijector

    @Bijector.setter
    def Bijector(self,
                 bijector: Optional[allowed_bijector_types]
                ) -> None:
        try:
            self._Bijector
            raise Exception("The 'Bijector' attribute is automatically set when initialising the NFMain object and cannot be manually set.")
        except:
            if bijector is not None:
                self._Bijector = bijector
            else:
                raise Exception("When no input file is specified the 'bijector' argument needs to be different from 'None'.")
        print(header_string_2,"\nSetting NF Bijector.\n", show = self.verbose)

    @property
    def ndims(self):
        return self.Bijector._ndims

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
        super().__init__(managed_object = nf_main)

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
        super().__init__(managed_object = nf_main)


class NFInference(Inference):
    """
    """
    def __init__(self,
                 nf_main: NFMain
                ) -> None:
        super().__init__(managed_object = nf_main)


class NFPlotter(Plotter):
    """
    """
    def __init__(self,
                 nf_main: NFMain
                ) -> None:
        super().__init__(managed_object = nf_main)