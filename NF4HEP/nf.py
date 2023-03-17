__all__ = ["NFFileManager",
           "NFTrainer",
           "NFMain",
           "NFDistribution",
           "NFPredictionsManager",
           "NFFiguresManager",
           "NFInference",
           "NFPlotter"]

from datetime import datetime
import numpy as np
import weakref
import matplotlib.pyplot as plt #type: ignore
import tensorflow as tf # type: ignore
import tensorflow.compat.v1 as tf1 # type: ignore
from tensorflow.python.keras import backend as K # type: ignore
from tensorflow.python.keras import Input # type: ignore
from tensorflow.python.keras import layers, initializers, models, regularizers, constraints, callbacks, optimizers, metrics, losses # type: ignore
from tensorflow.keras.callbacks import Callback # type: ignore
from tensorflow.keras.metrics import Metric # type: ignore
from tensorflow.keras.optimizers import Optimizer # type: ignore
from tensorflow.python.keras.models import Model
from tensorflow.keras.losses import Loss # type: ignore
from tensorflow.python.keras.layers import Layer
import tensorflow_probability as tfp # type: ignore
tfd = tfp.distributions
tfb = tfp.bijectors

from asyncio import base_subprocess
from numpy import typing as npt # type: ignore
from pathlib import Path
from timeit import default_timer as timer

from typing import Union, List, Dict, Callable, Tuple, Optional, NewType, Type, Generic, Any, TypeVar, TYPE_CHECKING
from typing_extensions import TypeAlias
from NF4HEP.utils.custom_types import Array, ArrayInt, ArrayStr, DataType, StrPath, IntBool, StrBool, StrList, StrArray, FigDict, LogPredDict, Number, DTypeStr, DTypeStrList, DictStr
from NF4HEP.bijectors import arqspline
from NF4HEP.bijectors import crqspline
from NF4HEP.bijectors import maf
from NF4HEP.bijectors import realnvp
from NF4HEP.bijectors.arqspline import ARQSplineNetwork, ARQSplineBijector
from NF4HEP.bijectors.crqspline import CRQSplineNetwork, CRQSplineBijector
from NF4HEP.bijectors.maf import MAFNetwork, MAFBijector
from NF4HEP.bijectors.realnvp import RealNVPChain#, RealNVPNetwork, RealNVPBijector
from NF4HEP.utils import corner
from NF4HEP.utils import utils
from NF4HEP.utils import custom_losses
from NF4HEP.utils.corner import extend_corner_range
from NF4HEP.utils.corner import get_1d_hist
from NF4HEP.utils.resources import ResourcesManager
from NF4HEP.utils.verbosity import print
from NF4HEP.utils.verbosity import Verbosity
from NF4HEP.utils import mplstyle_path
from NF4HEP.inputs.data import DataMain, DataFileManager
from NF4HEP.base import ObjectManager, Name, FileManager, PredictionsManager, FiguresManager, Inference, Plotter

header_string_1 = "=============================="
header_string_2 = "------------------------------"
strategy_header_string = "xxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"

class NFFileManager(FileManager):
    """
    """
    managed_object_name: str = "NFMain"
    def __init__(self,
                 name: Optional[str] = None,
                 input_file: Optional[StrPath] = None,
                 input_data_main_file: Optional[StrPath] = None,
                 output_folder: Optional[StrPath] = None,
                 load_on_RAM: Optional[bool] = None,
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
        self.load_on_RAM = load_on_RAM if load_on_RAM is not None else False
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
    def input_model_tf_file(self) -> Optional[Path]:
        if self._input_file is not None:
            return self._input_file.parent.joinpath(self._input_file.stem + "_model.tf")
        else:
            return None

    @property
    def input_model_json_file(self) -> Optional[Path]:
        if self._input_file is not None:
            return self._input_file.parent.joinpath(self._input_file.stem + "_model.json")
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
    def output_checkpoints_folder(self) -> Path:
        return self._output_folder.joinpath("checkpoints")

    @property
    def output_checkpoints_files(self) -> Path:
        return self.output_checkpoints_folder.joinpath(self.name+"_checkpoint.{epoch:02d}-{val_loss:.2f}.h5")

    @property
    def output_figure_plot_losses_keras_file(self) -> Path:
        return self.output_figures_folder.joinpath(self.output_figures_base_file_name+"_plot_losses_keras.pdf")

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
    def output_model_tf_file(self) -> Path:
        return self._output_folder.joinpath(self.name+"_model.tf")

    @property
    def output_model_json_file(self) -> Path:
        return self._output_folder.joinpath(self.name+"_model.json")

    @property
    def output_tensorboard_log_dir(self) -> Path:
        return self.output_folder.joinpath("tensorboard_logs")

    def load(self, verbose: Optional[IntBool] = None) -> None:
        """
        """
        verbose, _ = self.get_verbosity(verbose)
        self.load_log(verbose = verbose)
        self.load_object(verbose = verbose) # here also load the base_distribution
        #self.load_model(verbose = verbose)
        #self.load_history(verbose = verbose)
        self.load_predictions(verbose = verbose)

    def load_object(self, verbose: Optional[IntBool] = None) -> None:
        verbose, verbose_sub = self.get_verbosity(verbose)
        if self.input_object_json_file is not None:
            [log, dictionary] = self._FileManager__load_json(input_json_file = self.input_object_json_file, # type: ignore
                                                             verbose = verbose
                                                            )
            # Load log
            self.ManagedObject.log = log
            # Load DataMain main object attributes
            dict_to_load_main = dictionary["Main"]
            self.ManagedObject.__dict__.update(dict_to_load_main)
        else:
            raise Exception("Input file not defined.")

    def load_history(self, verbose: Optional[IntBool] = None) -> None:
        verbose, _ = self.get_verbosity(verbose)
        if self.input_history_json_file is not None:
            input_file = self.input_history_json_file
        else:
            raise Exception("Input file not defined.")
        if input_file.exists():
            [log, history] = self._FileManager__load_json(input_json_file = input_file, # type: ignore
                                                          verbose = verbose
                                                         )
            # Load log
            self.ManagedObject.log = log
            self.ManagedObject.Trainer.history = history
        else:
            print("No history file is present. History not loaded.")

    def load_model(self, verbose: Optional[IntBool] = None) -> None:
        verbose, _ = self.get_verbosity(verbose)
        start = timer()
        custom_objects = custom_losses.losses()
        if self.input_model_tf_file is not None:
            input_file = self.input_model_tf_file
        else:
            raise Exception("Input file not defined.")
        if input_file.exists():
            try:
                self.ManagedObject.Trainer._NFModel = models.load_model(input_file, custom_objects = custom_losses.losses())
            except:
                print(header_string_2,"\nWARNING: TF model load failed. The model attribute will be initialized to None.\n",show = True)
                self.ManagedObject.Trainer._NFModel = None
                return
            if self.ManagedObject.Trainer._NFModel is not None:
                try:
                    self.ManagedObject.Trainer.NFModel.history = callbacks.History()
                    self.ManagedObject.Trainer.NFModel.history.model = self.ManagedObject.Trainer.NFModel
                    self.ManagedObject.Trainer.NFModel.history.history = self.ManagedObject.Trainer.history
                    self.ManagedObject.Trainer.NFModel.history.params = {"verbose": 1, "epochs": self.ManagedObject.Trainer.epochs_available}
                    self.ManagedObject.Trainer.NFModel.history.epoch = np.arange(self.ManagedObject.Trainer.epochs_available).tolist()
                except:
                    print(header_string_2,"\nWARNING: No training history file available.\n",show = True)
                    return
            end = timer()
            self.ManagedObject.log = {utils.generate_timestamp(): {"action": "loaded tf model",
                                                                   "files names": [input_file.name]}}
            print(header_string_2,"\nTF model\n",input_file,"\nloaded in", str(end-start), ".\n", show = verbose)
        else:
            print(header_string_2,"\nWARNING: No TF model file is present.. The model attribute will be initialized to None.\n",show = True)
            self.ManagedObject.Trainer._NFModel = None
            return

    def save(self,
             timestamp: Optional[str] = None,
             overwrite: StrBool = False,
             verbose: Optional[IntBool] = None
            ) -> None:
        """
        Need to save the following:
        {EXCLUDE -- '_verbose': int,
         EXCLUDE -- '_verbose_sub': int,
         EXCLUDE -- '_FileManager': NF4HEP.nf.NFFileManager,
         save_log -- '_log': {...},
         save_predictions -- '_Predictions': NF4HEP.nf.NFPredictionsManager,
         save_predictions -- '_Figures': NF4HEP.nf.NFFiguresManager,
         already saved -- '_Data': NF4HEP.inputs.data.DataMain,
         save_object -- '_seed': int,
         save_object -- '_model_bijector_inputs': {...},
         save_object -- '_model_define_inputs': {...},
         save_object -- '_model_chain_inputs': {...},
         save_object -- '_model_optimizer_inputs': str,
         save_object -- '_model_callbacks_inputs': list,
         save_object -- '_model_compile_inputs': {},
         save_object -- '_model_train_inputs': {},
         save_object -- '_base_distribution_inputs': NoneType, * could be a tf object
         save_object -- '_resources_inputs': {},
         EXCLUDE (reconstruct from inputs) -- '_Chain': NF4HEP.bijectors.realnvp.RealNVPChain,
         save_history / save_model -- '_Trainer': NF4HEP.nf.NFTrainer, after reconstruction load model and weigths
         EXCLUDE (set from inputs) -- '_BaseDistribution': NF4HEP.nf.NFDistribution,
         EXCLUDE (set from inputs) -- '_TrainableDistribution': NF4HEP.nf.NFDistribution,
         EXCLUDE -- '_Inference': NF4HEP.nf.NFInference,
         EXCLUDE -- '_Plotter': NF4HEP.nf.NFPlotter}
        """
        if timestamp is None:
            timestamp = utils.generate_timestamp()
        self.save_object(timestamp = timestamp, overwrite = overwrite, verbose = verbose) # here also save the base_distribution
        self.save_history(timestamp = timestamp, overwrite = overwrite, verbose = verbose)
        self.save_predictions(timestamp = timestamp, overwrite = overwrite, verbose = verbose)
        self.save_model(timestamp = timestamp, overwrite = overwrite, verbose = verbose)
        self.save_log(timestamp = timestamp, overwrite = overwrite, verbose = verbose)

    def save_object(self, 
                    timestamp: Optional[str] = None,
                    overwrite: StrBool = False,
                    verbose: Optional[IntBool] = None
                   ) -> None:
        if timestamp is None:
            timestamp = utils.generate_timestamp()
        dict_to_save_main = utils.dic_minus_keys(self.ManagedObject.__dict__,self.ManagedObject.excluded_attributes)
        dict_to_save = {"_name": self.name, "Main": dict_to_save_main}
        log = self._FileManager__save_dict_json(dict_to_save = dict_to_save, # type: ignore
                                                output_file = self.output_object_json_file,
                                                timestamp = timestamp,
                                                overwrite = overwrite,
                                                verbose = verbose)
        self.ManagedObject.log = log

    def save_model(self, 
                   timestamp: Optional[str] = None, 
                   overwrite: StrBool = False, 
                   verbose: Optional[IntBool] = None
                  ) -> None:
        verbose, verbose_sub = self.get_verbosity(verbose)
        start = timer()
        output_tf_file = self.get_target_file_overwrite(input_file = Path(self.output_model_tf_file),
                                                        timestamp = timestamp,
                                                        overwrite = overwrite,
                                                        verbose = verbose_sub)
        try:
           self.ManagedObject.Trainer.NFModel.save(output_tf_file)
        except:
            print(header_string_1,"\nModel not defined. No TF model file saved.\n")
            return
        end = timer()
        self.print_save_info(filename = output_tf_file,
                             time= str(end-start),
                             overwrite = overwrite,
                             verbose = verbose)
        self.ManagedObject.log = {utils.generate_timestamp(): {"action": "saved tf model",
                                                               "file name": output_tf_file.name}}

    def save_history(self, 
                     timestamp: Optional[str] = None, 
                     overwrite: StrBool = False, 
                     verbose: Optional[IntBool] = None
                    ) -> None:
        if timestamp is None:
            timestamp = utils.generate_timestamp()
        try:
            dict_to_save = self.ManagedObject.Trainer.NFModel.history.history
        except:
            print(header_string_1,"\nModel not defined. No history file saved.\n")
            return
        log = self._FileManager__save_dict_json(dict_to_save = dict_to_save, # type: ignore
                                                output_file = self.output_history_json_file,
                                                timestamp = timestamp,
                                                overwrite = overwrite,
                                                verbose = verbose)
        self.ManagedObject.log = log

    


#class NFChain(tfb.Chain, Verbosity): # type: ignore
#    allowed_bijectors: List = [ARQSplineBijector,CRQSplineBijector,MAFBijector,RealNVPBijector]
#    allowed_NN_types: TypeAlias = Union[ARQSplineNetwork,CRQSplineNetwork,MAFNetwork,RealNVPNetwork]
#    """
#    model_chain_inputs can be of the following form:
#    .. code-block:: python
#
#        model_chain_inputs = {"nbijectors": 2,
#                              "batch_normalization": False}
#    """
#    def __init__(self,
#                 bijector: Union[ARQSplineBijector,CRQSplineBijector,MAFBijector,RealNVPBijector],
#                 model_chain_inputs: Optional[Dict[str, Any]] = None,
#                 verbose: Optional[IntBool] = None
#                ) -> None:
#        # Attributes type declarations
#        self._batch_normalization: bool
#        self._Bijector: Union[ARQSplineBijector,CRQSplineBijector,MAFBijector,RealNVPBijector]
#        self._model_chain_inputs: Dict[str, Any]
#        self._nbijectors: int
#        # Initialise parent Verbosity class
#        Verbosity.__init__(self, verbose)
#        # Set verbosity
#        verbose, _ = self.get_verbosity(verbose)
#        # Initialize object
#        print(header_string_1, "\nInitializing NFChain object.\n", show = verbose)
#        self.__set_model_chain_inputs(model_chain_inputs = model_chain_inputs, verbose = verbose)
#        self._Bijector = bijector
#        name = self.Bijector.name.replace("Bijector","Chain")
#        ndims = self.Bijector._ndims
#        permutation = tf.cast(np.concatenate((np.arange(int(ndims/2), ndims), np.arange(0, int(ndims/2)))), tf.int32)
#        Permute = tfb.Permute(permutation=permutation)
#        #print("\n",tfb.BatchNormalization())
#        #print("\n",self.Bijector)
#        #print("\n",Permute)
#        self._bijectors = []
#        for _ in range(self.nbijectors):
#            if self._batch_normalization:
#                self._bijectors.append(tfb.BatchNormalization())
#            self._bijectors.append(self.Bijector)
#            self._bijectors.append(Permute)
#        tfb.Chain.__init__(self, bijectors=list(reversed(self._bijectors[:-1])), name=name)
#
#    @property
#    def allowed_chains_names(self) -> List[str]:
#        return ["ARQSplineChain","CRQSplineChain","MAFChain","RealNVPChain"]
#
#    @property
#    def NN(self) -> allowed_NN_types:
#        return self._Bijector._NN
#
#    @property
#    def batch_normalization(self) -> bool:
#        return self._batch_normalization
#
#    @property
#    def Bijector(self) -> Union[ARQSplineBijector,CRQSplineBijector,MAFBijector,RealNVPBijector]:
#        return self._Bijector
#
#    @Bijector.setter
#    def Bijector(self,
#                 bijector: Union[ARQSplineBijector,CRQSplineBijector,MAFBijector,RealNVPBijector]
#                ) -> None:
#        try:
#            self._Bijector
#            raise Exception("The 'Bijector' attribute is automatically set when initialising the NFMain object and cannot be manually set.")
#        except:
#            if isinstance(bijector,tuple(self.allowed_bijectors)) is not None:
#                print(header_string_2,"\nSetting NF Bijector.\n", show = self.verbose)
#                self._Bijector = bijector
#            else:
#                raise Exception("The 'Bijector' attribute should be a supported bijector object.")
#
#    @property
#    def model_chain_inputs(self) -> Dict[str, Any]:
#        return self._model_chain_inputs
#
#    @property
#    def nbijectors(self) -> int:
#        return self._nbijectors
#
#    @property
#    def ndims(self) -> int:
#        return self._Bijector._ndims
#
#    def __set_model_chain_inputs(self,
#                                 model_chain_inputs: Optional[Dict[str, Any]] = None,
#                                 verbose: Optional[IntBool] = None
#                                ) -> None:
#        if model_chain_inputs is None:
#            model_chain_inputs = {}
#        try:
#            self._nbijectors = model_chain_inputs["nbijectors"]
#        except:
#            print("WARNING: The 'model_chain_inputs' argument misses the mandatory 'nbijectors' item. The corresponding attribute will be set to a default of 2.")
#            self._nbijectors = 2
#        utils.check_set_dict_keys(dic = model_chain_inputs, 
#                                  keys = ["nbijectors","batch_normalization"],
#                                  vals = [2,False],
#                                  verbose = verbose)
#        self._batch_normalization = model_chain_inputs["batch_normalization"]
#        self._model_chain_inputs = model_chain_inputs


#class NFModel(Model):
#    """
#    """
#    def init(self) -> None:
#        super().__init__()


class NFMain(Verbosity):
    object_name : str = "NFMain"
    allowed_chains: List = [RealNVPChain]
    allowed_chains_types: TypeAlias = Union[RealNVPChain,RealNVPChain]
    """
    """
    def __init__(self,
                 file_manager: NFFileManager,
                 data: Optional[DataMain] = None,
                 base_distribution_inputs: Optional[Union[Dict[str,Any],str,tfp.distributions.distribution.Distribution]] = None,
                 model_define_inputs: Optional[Dict[str, Any]] = None,
                 model_bijector_inputs: Optional[Dict[str, Any]] = None,
                 model_chain_inputs: Optional[Dict[str, Any]] = None,
                 model_optimizer_inputs: Optional[Union[Dict[str,Any],str]] = None,
                 model_callbacks_inputs: Optional[List[Union[Dict[str,Any],str]]] = None,
                 model_compile_inputs: Optional[Dict[str,Any]] = None,
                 model_train_inputs: Optional[Dict[str,Any]] = None,
                 resources_inputs: Optional[Dict[str,Any]] = None,
                 seed: Optional[int] = None,
                 verbose: IntBool = True
                ) -> None:
        """
        """
        # Attributes type declatation (objects)
        self._BaseDistribution: NFDistribution
        self._Data: DataMain
        self._Figures: NFFiguresManager
        self._FileManager: NFFileManager
        self._Chain: NFMain.allowed_chains_types
        self._Inference: NFInference
        self._Plotter: NFPlotter
        self._Predictions: NFPredictionsManager
        self._TrainableDistribution: NFDistribution
        self._Trainer: NFTrainer
        # Attributes type declatation (others)
        self._base_distribution_inputs: Optional[Union[Dict[str,Any],str,tfp.distributions.distribution.Distribution]]
        self._log: LogPredDict
        self._model_define_inputs: Dict[str, Any]
        self._model_bijector_inputs: Dict[str, Any]
        self._model_chain_inputs: Dict[str, Any]
        self._model_optimizer_inputs: Union[Dict[str,Any],str]
        self._model_callbacks_inputs: List[Union[Dict[str,Any],str]]
        self._model_compile_inputs: Dict[str,Any]
        self._model_train_inputs: Dict[str,Any]
        self._name: str
        self._resources_inputs: Dict[str,Any]
        self._seed: int
        # Initialize parent Verbosity class
        super().__init__(verbose)
        # Set verbosity
        verbose, verbose_sub = self.get_verbosity(verbose)
        # Initialize object
        timestamp = utils.generate_timestamp()
        print(header_string_1, "\nInitializing NFMain object.\n", show = verbose)
        self.FileManager = file_manager # also sets the NFMain managed object FileManager attribute
        if self.FileManager.input_file is None:
            self.log = {timestamp: {"action": "NFMain object created from input arguments"}}
        else:
            self.log = {timestamp: {"action": "NFMain object reconstructed from loaded files"}}
        self.Predictions = NFPredictionsManager(nf_main = self)
        self.Figures = NFFiguresManager(nf_main = self)
        self.__set_data(data = data)
        if self.FileManager.input_file is None:
            print(header_string_2,"\nInitializing new NFMain object.\n", show = verbose)
            self.base_distribution_inputs = base_distribution_inputs
            self.model_define_inputs = model_define_inputs if model_define_inputs is not None else {}
            self.model_bijector_inputs = model_bijector_inputs if model_bijector_inputs is not None else {}
            self.model_chain_inputs = model_chain_inputs if model_chain_inputs is not None else {}
            self.model_optimizer_inputs = model_optimizer_inputs if model_optimizer_inputs is not None else {}
            self.model_callbacks_inputs = model_callbacks_inputs if model_callbacks_inputs is not None else []
            self.model_compile_inputs = model_compile_inputs if model_compile_inputs is not None else {}
            self.model_train_inputs = model_train_inputs if model_train_inputs is not None else {}
            self.resources_inputs = resources_inputs if resources_inputs is not None else {}
            self.seed = seed if seed is not None else 0
            chain_name = self.model_bijector_inputs["name"]+"Chain"
            self.Trainer = NFTrainer(nf_main = self)
            print(strategy_header_string,"\nExecuting within strategy context:",self.Trainer.Resources.strategy,".\n",show=verbose)
            self.Chain = self.Trainer.Resources.strategy_executor(eval(chain_name)(model_define_inputs = self.model_define_inputs,
                                                                                   model_bijector_inputs = self.model_bijector_inputs,
                                                                                   model_chain_inputs = self.model_chain_inputs,
                                                                                   verbose = verbose))
            self.BaseDistribution = self.Trainer.Resources.strategy_executor(NFDistribution(nf_main = self, distribution = self.base_distribution_inputs))
            self.TrainableDistribution = self.Trainer.Resources.strategy_executor(NFDistribution(nf_main = self, distribution = tfd.TransformedDistribution(self.BaseDistribution.distribution,self.Chain)))
            print("\nExecution in strategy context done.",show=verbose)
            print(strategy_header_string,"\n",show=verbose)
        else:
            print(header_string_2,"\nLoading existing NFMain object.\n", show = verbose)
            for k,v in {"data": data, 
                        "model_define_inputs": model_define_inputs, 
                        "model_bijector_inputs": model_bijector_inputs,
                        "model_chain_inputs": model_chain_inputs,
                        "model_optimizer_inputs": model_optimizer_inputs,
                        "model_callbacks_inputs": model_callbacks_inputs, 
                        "model_compile_inputs": model_compile_inputs,
                        "model_train_inputs": model_train_inputs,
                        "base_distribution_inputs": base_distribution_inputs,
                        "resources_inputs": resources_inputs,
                        "seed": seed}.items():
                if v is not None:
                    print(header_string_2,"\nWARNING: an input file was specified and the argument '",k,"' will be ignored. The related attribute will be set from the input file.\n", show = True)
            self.FileManager.load(verbose = verbose)
            chain_name = self.model_bijector_inputs["name"]+"Chain"
            self.Trainer = NFTrainer(nf_main = self)
            print(strategy_header_string,"\nExecuting within strategy context:",self.Trainer.Resources.strategy,".\n",show=verbose)
            self.Chain = self.Trainer.Resources.strategy_executor(eval(chain_name)(model_define_inputs = self.model_define_inputs,
                                                                                   model_bijector_inputs = self.model_bijector_inputs,
                                                                                   model_chain_inputs = self.model_chain_inputs,
                                                                                   verbose = verbose))
            self.BaseDistribution = self.Trainer.Resources.strategy_executor(NFDistribution(nf_main = self, distribution = self.base_distribution_inputs))
            self.TrainableDistribution = self.Trainer.Resources.strategy_executor(NFDistribution(nf_main = self, distribution = tfd.TransformedDistribution(self.BaseDistribution.distribution,self.Chain)))
            self.Trainer.Resources.strategy_executor(self.FileManager.load_history(verbose = verbose_sub))
            self.Trainer.Resources.strategy_executor(self.FileManager.load_model(verbose = verbose_sub))
            print("\nExecution in strategy context done.",show=verbose)
            print(strategy_header_string,"\n",show=verbose)
        self.Inference = NFInference(nf_main = self)
        print(header_string_2,"\nSetting Inference.\n", show = verbose)
        self.Plotter = NFPlotter(nf_main = self)
        print(header_string_2,"\nSetting Plotter.\n", show = verbose)
        if self.FileManager.input_file is None:
            self.log = {timestamp: {"action": "object created from input arguments"}}
            self.FileManager.save(timestamp = timestamp, overwrite = False, verbose = verbose_sub)
        else:
            self.log = {timestamp: {"action": "object reconstructed from loaded files"}}
            self.FileManager.save_log(timestamp = timestamp, overwrite = True, verbose = verbose_sub)
        #self.FileManager.save_log(timestamp = timestamp, overwrite = bool(self.FileManager.input_file), verbose = verbose_sub)
        
    @property
    def excluded_attributes(self) -> list:
        tmp = ["_log",
               "_verbose",
               "_verbose_sub",
               "_BaseDistribution",
               "_Chain",
               "_Data",
               "_Figures",
               "_FileManager",
               "_Inference",
               "_Plotter",
               "_Predictions",
               "_Trainer",
               "_TrainableDistribution",
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
    def model_define_inputs(self) -> Dict[str, Any]:
        return self._model_define_inputs
    
    @model_define_inputs.setter
    def model_define_inputs(self,
                              model_define_inputs: Dict[str, Any]
                             ) -> None:
        self._model_define_inputs = dict(model_define_inputs)
        self._model_define_inputs["ndims"] = self.Data.DataManager.ndims

    @property
    def model_bijector_inputs(self) -> Dict[str, Any]:
        return self._model_bijector_inputs
    
    @model_bijector_inputs.setter
    def model_bijector_inputs(self,
                              model_bijector_inputs: Dict[str, Any]
                             ) -> None:
        self._model_bijector_inputs = dict(model_bijector_inputs)
        try:
            self._model_bijector_inputs["name"]
        except:
            raise KeyError("The bijector 'name' key must be specified in the 'model_bijector_inputs' input dictionary.")

    @property
    def model_chain_inputs(self) -> Dict[str, Any]:
        return self._model_chain_inputs
    
    @model_chain_inputs.setter
    def model_chain_inputs(self,
                           model_chain_inputs: Dict[str, Any]
                          ) -> None:
        self._model_chain_inputs = dict(model_chain_inputs)

    @property
    def model_optimizer_inputs(self) -> Union[Dict[str,Any],str]:
        return self._model_optimizer_inputs
    
    @model_optimizer_inputs.setter
    def model_optimizer_inputs(self,
                               model_optimizer_inputs: Union[Dict[str,Any],str],
                              ) -> None:
        verbose, verbose_sub = self.get_verbosity(self.verbose)
        if isinstance(model_optimizer_inputs, str):
            self._model_optimizer_inputs = str(model_optimizer_inputs)
        elif isinstance(model_optimizer_inputs, dict):
            try:
                model_optimizer_inputs["name"]
                self._model_optimizer_inputs = dict(model_optimizer_inputs)
            except:
                print("WARNING: The optimizer ", str(self.model_optimizer_inputs), " has unspecified name. Optimizer will be set to the default 'optimizers.Adam()'.", show = True)
                self._model_optimizer_inputs = {"name": "Adam"}
        else:
            print("WARNING: The 'model_optimizer_inputs' argument is neither a string nor a dictionary. Optimizer will be set to the default 'optimizers.Adam()'.", show = True)
            self._model_optimizer_inputs = dict({"name": "Adam"})

    @property
    def model_callbacks_inputs(self) -> List[Union[Dict[str,Any],str]]:
        return self._model_callbacks_inputs
    
    @model_callbacks_inputs.setter
    def model_callbacks_inputs(self,
                               model_callbacks_inputs: List[Union[Dict[str,Any],str]],
                              ) -> None:
        verbose, verbose_sub = self.get_verbosity(self.verbose)
        self._model_callbacks_inputs = list(model_callbacks_inputs)

    @property
    def model_compile_inputs(self) -> Dict[str,Any]:
        return self._model_compile_inputs
    
    @model_compile_inputs.setter
    def model_compile_inputs(self,
                             model_compile_inputs: Dict[str,Any],
                            ) -> None:
        """
        """
        verbose, verbose_sub = self.get_verbosity(self.verbose)
        self._model_compile_inputs = dict(model_compile_inputs)
        utils.check_set_dict_keys(self._model_compile_inputs, 
                                  ["loss","metrics"],
                                  [None,[]], 
                                  verbose = verbose_sub)

    @property
    def model_train_inputs(self) -> Dict[str,Any]:
        return self._model_train_inputs
    
    @model_train_inputs.setter
    def model_train_inputs(self,
                           model_train_inputs: Dict[str,Any],
                          ) -> None:
        verbose, verbose_sub = self.get_verbosity(self.verbose)
        self._model_train_inputs = dict(model_train_inputs)
        try:
            self._model_train_inputs["epochs"]
        except:
            print("WARNING: the number of epochs input 'epochs' has not been specified. Setting a default of 20 epochs.", show = True)
            self._model_train_inputs["epochs"] = 20
        try:
            self._model_train_inputs["batch_size"]
        except:
            print("WARNING: the batch size input 'batch_size' has not been specified. Setting a default of 512.", show = True)
            self._model_train_inputs["batch_size"] = 512

    @property
    def base_distribution_inputs(self) -> Optional[Union[Dict[str,Any],str,tfp.distributions.distribution.Distribution]]:
        return self._base_distribution_inputs
    
    @base_distribution_inputs.setter
    def base_distribution_inputs(self,
                                 base_distribution_inputs: Optional[Union[Dict[str,Any],str,tfp.distributions.distribution.Distribution]]
                                ) -> None:
        verbose, verbose_sub = self.get_verbosity(self.verbose)
        if isinstance(base_distribution_inputs,str):
            self._base_distribution_inputs = str(base_distribution_inputs)
        elif isinstance(base_distribution_inputs,tfp.distributions.distribution.Distribution):
            self._base_distribution_inputs = base_distribution_inputs
        elif isinstance(base_distribution_inputs, dict):
            self._base_distribution_inputs = dict(base_distribution_inputs)
        else:
            self._base_distribution_inputs = None
    
    @property
    def resources_inputs(self) -> Dict[str,Any]:
        return self._resources_inputs

    @resources_inputs.setter
    def resources_inputs(self,
                         resources_inputs: Dict[str,Any],
                        ) -> None:
        self._resources_inputs =  dict(resources_inputs)

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
            if isinstance(file_manager, NFFileManager):
                print(header_string_2,"\nSetting FileManager.\n", show = self.verbose)
                self._FileManager = file_manager
                self.log = {utils.generate_timestamp(): {"action": "FileManager object set"}}
                self._FileManager.ManagedObject = self
            else:
                raise Exception("The 'FileManager' attribute should be a 'NFFileManager' object.")

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
        if isinstance(predictions,NFPredictionsManager):
            print(header_string_2,"\nSetting Predictions.\n", show = self.verbose)
            self._Predictions = predictions
            self.log = {utils.generate_timestamp(): {"action": "Predictions object set"}}
        else:
            print("WARNING: The 'Predictions' attribute should be a 'NFPredictionsManager' object. Nothing was set.")

    @property
    def Figures(self) -> "NFFiguresManager":
        return self._Figures

    @Figures.setter
    def Figures(self,
                figures: "NFFiguresManager"
               ) -> None:
        if isinstance(figures,NFFiguresManager):
            print(header_string_2,"\nSetting Figures.\n", show = self.verbose)
            self._Figures = figures
            self.log = {utils.generate_timestamp(): {"action": "Figures object set"}}
        else:
            print("WARNING: The 'Figures' attribute should be a 'NFFiguresManager' object. Nothing was set.")

    @property
    def Data(self) -> DataMain:
        return self._Data

    @Data.setter
    def Data(self,
             data: DataMain
            ) -> None:
        try:
            self._Data
            raise Exception("The 'Data' attribute is automatically set when initialising the NFMain object and cannot be manually set.")
        except:
            if isinstance(data,DataMain):
                print(header_string_2,"\nSetting Data.\n", show = self.verbose)
                self._Data = data
                self.log = {utils.generate_timestamp(): {"action": "Data object set"}}
            else:
                raise Exception("When no input file is specified the 'data' argument should be a 'DataMain' object.")
        
    def __set_data(self,
                   data: Optional[DataMain]
                  ) -> None:
        if self.FileManager.input_data_main_file is None:
            if isinstance(data,DataMain):
                print(header_string_2,"\nSetting Data from input argument.\n", show = self.verbose)
                self.Data = data
            else:
                raise Exception("When no input file is specified the 'data' argument should be a 'DataMain' object.")
        else:
            if data is not None:
                print(header_string_2,"\nWARNING: an input datamain file was specified and the 'data' argument will be ignored. The related attribute will be set from the input file.\n", show = True)
            print(header_string_2,"\nSetting Data from input file.\n", show = self.verbose)
            DataFileManagerLoad = DataFileManager(name = None,
                                                  input_file = self.FileManager.input_data_main_file,
                                                  output_folder = None,
                                                  load_on_RAM = self.FileManager.load_on_RAM, # this is ignored
                                                  verbose = self.verbose_sub)
            self.Data = DataMain(file_manager = DataFileManagerLoad,
                                 pars_manager = None,
                                 input_data = None,
                                 npoints = None,
                                 preprocessing = None,
                                 seed = None,
                                 verbose = self.verbose_sub)

    @property
    def ndims(self) -> int:
        return self.Chain.ndims

    @property
    def Chain(self) -> allowed_chains_types:
        return self._Chain

    @Chain.setter
    def Chain(self,
              nf_chain: allowed_chains_types
             ) -> None:
        try:
            self._Chain
            raise Exception("The 'Chain' attribute is automatically set when initialising the NFMain object and cannot be manually set.")
        except:
            if isinstance(nf_chain, tuple(self.allowed_chains)):
                print(header_string_2,"\nSetting NF Chain.\n", show = self.verbose)
                self._Chain = nf_chain
                self.log = {utils.generate_timestamp(): {"action": "Chain object set"}}
            else:
                raise Exception("The 'Chain' attribute should be a 'NFChain' object.")

    @property
    def BaseDistribution(self) -> "NFDistribution":
        return self._BaseDistribution

    @BaseDistribution.setter
    def BaseDistribution(self,
                         base_distribution: "NFDistribution"
                        ) -> None:
        try:
            self._BaseDistribution
            raise Exception("The 'BaseDistribution' attribute is automatically set when initialising the NFMain object and cannot be manually set.")
        except:
            if isinstance(base_distribution,NFDistribution):
                print(header_string_2,"\nSetting BaseDistribution.\n", show = self.verbose)
                self._BaseDistribution = base_distribution
                self.log = {utils.generate_timestamp(): {"action": "BaseDistribution object set"}}
            else:
                raise Exception("The 'BaseDistribution' attribute should be a 'NFDistribution' object.")

    @property
    def TrainableDistribution(self) -> "NFDistribution":
        return self._TrainableDistribution

    @TrainableDistribution.setter
    def TrainableDistribution(self,
                              trainable_distribution: "NFDistribution"
                             ) -> None:
        try:
            self._TrainableDistribution
            raise Exception("The 'TrainableDistribution' attribute is automatically set when initialising the NFMain object and cannot be manually set.")
        except:
            if isinstance(trainable_distribution,NFDistribution):
                print(header_string_2,"\nSetting TrainableDistribution.\n", show = self.verbose)
                self._TrainableDistribution = trainable_distribution
                self.log = {utils.generate_timestamp(): {"action": "TrainableDistribution object set"}}
            else:
                raise Exception("The 'TrainableDistribution' attribute should be a 'NFDistribution' object.")

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
            if isinstance(nf_trainer, NFTrainer):
                print(header_string_2,"\nSetting Trainer.\n", show = self.verbose)
                self._Trainer = nf_trainer
                self.log = {utils.generate_timestamp(): {"action": "Trainer object set"}}
                self._Trainer.ManagedObject = self
            else:
                raise Exception("The 'Trainer' attribute should be a 'NFTrainer' object.")
        
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
            if isinstance(inference,NFInference):
                print(header_string_2,"\nSetting Inference.\n", show = self.verbose)
                self._Inference = inference
                self.log = {utils.generate_timestamp(): {"action": "Inference object set"}}
            else:
                raise Exception("The 'Inference' attribute should be a 'NFInference' object.")
        
    @property
    def Plotter(self) -> "NFPlotter":
        return self._Plotter

    @Plotter.setter
    def Plotter(self,
                plotter: "NFPlotter"
               ) -> None:
        try:
            self._Plotter
            raise Exception("The 'Plotter' attribute is automatically set when initialising the NFMain object and cannot be manually set.")
        except:
            if isinstance(plotter,NFPlotter):
                print(header_string_2,"\nSetting Plotter.\n", show = self.verbose)
                self._Plotter = plotter
                self.log = {utils.generate_timestamp(): {"action": "Plotter object set"}}
            else:
                raise Exception("The 'Plotter' attribute should be a 'NFPlotter' object.")

    @property
    def seed(self) -> int:
        return self._seed

    @seed.setter
    def seed(self,
             seed: int
            ) -> None:
        print(header_string_2,"\nSetting seed to",str(seed),".\n", show = self.verbose)
        self._seed = seed
        utils.reset_random_seeds(self._seed)
        self.log = {utils.generate_timestamp(): {"action": "seed reset",
                                                 "value": seed}}


#class NFManager(ObjectManager):
#    """
#    Manages nf object
#    """
#    def __init__(self, nf_main: NFMain) -> None:
#        """
#        """
#        # Attributes type declatation
#        self._ManagedObject: Any
#        self._managed_object_name: str
#        # Initialize parent ManagedObject class (sets self._ManagedObject)
#        super().__init__(managed_object = nf_main)
#        # Set verbosity
#        verbose, verbose_sub = self.ManagedObject.get_verbosity(self.ManagedObject.verbose)
#        # Initialize object
#        print(header_string_1,"\nInitializing NF Manager.\n", show = verbose)


class NFTrainer(ObjectManager, Verbosity):
    """
    Manages nf training
    """
    managed_object_name: str = "NFMain"
    def __init__(self,
                 nf_main: NFMain,
                 verbose: Optional[IntBool] = None
                ) -> None:
        """
        """
        # Attributes type declatation
        self._batch_size: int
        self._callbacks: List[Callback]
        self._callbacks_strings: List[str]
        self._epochs_available: int
        self._epochs_required: int
        self._loss: Loss
        self._loss_string: str
        self._metrics: List[Metric]
        self._metrics_strings: List[str]
        self._optimizer: Optimizer
        self._optimizer_string: str
        self._ManagedObject: "NFMain"
        self._history: Dict[str,Any]
        self._managed_object_name: str
        self._model_params: Dict[str,Optional[int]]
        self._NFModel: Optional[Model]
        self._Resources: ResourcesManager
        #self._strategy: Optional[tf.distribute.Strategy]
        #self._training_device: Optional[str]
        self._training_time: float
        # Initialise parent ObjectManager class
        ObjectManager.__init__(self, managed_object = nf_main)
        # Initialise parent Verbosity class
        Verbosity.__init__(self, verbose)
        # Set verbosity
        verbose, verbose_sub = self.get_verbosity(verbose)
        # Initialize object
        print(header_string_1,"\nInitializing NF Trainer.\n", show = verbose)
        self._model_params = {"total": None, "trainable": None, "non_trainable": None}
        self._NFModel = None
        #self._strategy = None
        #self._training_device = None
        self._training_time = 0.
        self._history = {}
        self._batch_size = self.model_train_inputs["batch_size"]
        self._epochs_required = self.model_train_inputs["epochs"]
        self._epochs_available = 0
        self.Resources = ResourcesManager(resources_inputs = self.ManagedObject.resources_inputs, verbose = self.verbose)
        self.__set_tf_objects(verbose = verbose_sub)

    @property
    def NFModel(self) -> Model:
        return self._NFModel

    @property
    def Resources(self) -> ResourcesManager:
        if self._Resources is None:
            print("The 'Resources' attribute is set once the 'ManagedObject' attribute is set.")
        return self._Resources

    @Resources.setter
    def Resources(self,
                  resources_manager: ResourcesManager
                 ) -> None:
        if isinstance(resources_manager,ResourcesManager):
            print(header_string_2,"\nSetting Resources.\n", show = self.verbose)
            self._Resources = resources_manager
            #self.ManagedObject.log = {utils.generate_timestamp(): {"action": "Resources set"}}
        else:
            print("WARNING: The 'ResourcesManager' attribute should be a 'ResourcesManager' object. Nothing was set.")

    @property
    def model_optimizer_inputs(self) -> Union[Dict[str,Any],str]:
        return self.ManagedObject.model_optimizer_inputs

    @property
    def model_callbacks_inputs(self) -> List[Union[Dict[str,Any],str]]:
        return self.ManagedObject.model_callbacks_inputs

    @property
    def model_compile_inputs(self) -> Dict[str,Any]:
        return self.ManagedObject.model_compile_inputs

    @property
    def model_compile_kwargs(self) -> Dict[str,Any]:
        return utils.dic_minus_keys(self.model_compile_inputs, ["loss", "metrics"])

    @property
    def model_train_inputs(self) -> Dict[str,Any]:
        return self.ManagedObject.model_train_inputs
        
    @property
    def model_train_kwargs(self) -> Dict[str,Any]:
        return utils.dic_minus_keys(self.model_train_inputs, ["epochs", "batch_size"])

    @property
    def resources_inputs(self) -> Dict[str,Any]:
        return self.ManagedObject.resources_inputs

    @property
    def batch_size(self) -> int:
        return self._batch_size

    @property
    def callbacks(self) -> List[Callback]:
        return self._callbacks

    @property
    def callbacks_strings(self) -> List[str]:
        return self._callbacks_strings

    @property
    def epochs_required(self) -> int:
        return self._epochs_required
    
    @epochs_required.setter
    def epochs_required(self,
                        epochs_required: int
                       ) -> None:
        self._epochs_required = epochs_required

    @property
    def epochs_available(self) -> int:
        return self._epochs_available

    @property
    def loss(self) -> Loss:
        return self._loss
    
    @property
    def loss_string(self) -> str:
        return self._loss_string

    @property
    def metrics(self) -> List[Metric]:
        return self._metrics

    @property
    def metrics_strings(self) -> List[str]:
        return self._metrics_strings

    @property
    def model_params(self) -> Dict[str,Optional[int]]:
        if self._model_params["total"] is None:
            print("The 'model_params' attribute is set once the 'model_define' method is called.")
        return self._model_params

    @property
    def optimizer(self) -> Optimizer:
        return self._optimizer

    @property
    def optimizer_string(self) -> str:
        return self._optimizer_string
    
    #@property
    #def strategy(self) -> Optional[tf.distribute.Strategy]:
    #    return self._strategy

    #@property
    #def training_device(self) -> Optional[str]:
    #    if self._training_device is None:
    #        print("The 'training_device' attribute is set once the 'model_compile' method  is called.")
    #    return self._training_device
    
    @property
    def training_time(self) -> float:
        return self._training_time

    @property
    def history(self) -> Dict[str,Any]:
        return self._history

    @history.setter
    def history(self,
                history_dict: Dict[str,Any]
               ) -> None:
        if self._history == {}:
            print("No existing history. Setting new history.\n", show = self.verbose)
            self._history = history_dict
        else:
            print("Found existing history. Appending new history.\n", show = self.verbose)
            for k, v in self.history.items():
                self._history[k] = v + history_dict[k]

    def __set_optimizer(self, verbose: Optional[IntBool] = None) -> None:
        """
        Private method used by the 
        :meth:`NF.__set_tf_objects <NF4HEP.NF._NF__set_tf_objects>` one
        to set the |tf_keras_optimizers_link| object. It sets the
        :attr:`NF.optimizer_string <NF4HEP.NF.optimizer_string>`
        and :attr:`NF.optimizer <NF4HEP.NF.optimizer>` attributes.
        The former is set from the :attr:`NF._model_optimizer_inputs <NF4HEP.NF._NF_model_optimizer_inputs>` 
        dictionary.  The latter is set to ``None`` and then updated by the 
        :meth:`NF.model_compile <NF4HEP.NF.model_compile>` method during model compilation). 

        - **Arguments**

            - **verbose**

                See :argument:`verbose <common_methods_arguments.verbose>`.
        """
        verbose, verbose_sub = self.get_verbosity(verbose)
        def func_to_run():
            print(header_string_1, "\nSetting optimizer\n", show = verbose)
            self._optimizer = None
            if isinstance(self.model_optimizer_inputs, str):
                if "(" in self.model_optimizer_inputs:
                    optimizer_string = "optimizers." + \
                        self.model_optimizer_inputs.replace("optimizers.", "")
                else:
                    optimizer_string = "optimizers." + \
                        self.model_optimizer_inputs.replace(
                            "optimizers.", "") + "()"
            else: #elif isinstance(self.model_optimizer_inputs, dict):
                try:
                    name = self.model_optimizer_inputs["name"]
                except:
                    raise Exception("The optimizer ", str(self.model_optimizer_inputs), " has unspecified name.")
                try:
                    args = self.model_optimizer_inputs["args"]
                except:
                    args = []
                try:
                    kwargs = self.model_optimizer_inputs["kwargs"]
                except:
                    kwargs = {}
                optimizer_string = utils.build_method_string_from_dict("optimizers", name, args, kwargs)
            if optimizer_string is not None:
                try:
                    eval(optimizer_string)
                    self._optimizer_string = optimizer_string
                    self._optimizer = eval(self.optimizer_string)
                    print("Optimizer set to:", self.optimizer_string, ".\n", show = verbose)
                except Exception as e:
                    print(e)
                    raise Exception("Could not set optimizer", optimizer_string, ",\n")
            else:
                raise Exception("Could not set optimizer. The model_optimizer_inputs argument does not have a valid format (str or dict).")
            self.ManagedObject.log = {utils.generate_timestamp(): {"action": "optimizer set",
                                                                   "optimizer": self.optimizer_string}}
        print(strategy_header_string,"\nExecuting within strategy context:",self.Resources.strategy,".\n",show=verbose)        
        self.Resources.strategy_executor(func_to_run())
        print("\nExecution in strategy context done.",show=verbose)
        print(strategy_header_string,"\n",show=verbose)

    def __set_loss(self, verbose: Optional[IntBool] = None) -> None:
        """
        Private method used by the 
        :meth:`DnnLik.__set_tf_objects <DNNLikelihood.DnnLik._DnnLik__set_tf_objects>` one
        to set the loss object (it could be a |tf_keras_losses_link| object or a custom loss defined
        in the :mod:`Dnn_likelihood <dnn_likelihood>` module). It sets the
        :attr:`DnnLik.loss_string <DNNLikelihood.DnnLik.loss_string>`
        and :attr:`DnnLik.loss <DNNLikelihood.DnnLik.loss>` attributes. The former is set from the
        :attr:`DnnLik.__model_compile_inputs <DNNLikelihood.DnnLik._DnnLik__model_compile_inputs>` 
        dictionary, while the latter is set by evaluating the former.

        - **Arguments**

            - **verbose**
            
                See :argument:`verbose <common_methods_arguments.verbose>`.
        """
        verbose, verbose_sub = self.get_verbosity(verbose)
        def func_to_run():
            print(header_string_1, "\nSetting loss\n", show = verbose)
            self._loss = None
            loss_string = None
            ls = self.model_compile_inputs["loss"]
            if ls is None:
                loss_string = "custom_losses.minus_y_pred()"
            elif isinstance(ls, str):
                ls = ls.replace("losses.", "")
                try:
                    eval("losses." + ls)
                    loss_string = "losses." + ls
                except:
                    try:
                        eval("losses."+metrics.deserialize(ls).__name__)
                        loss_string = "losses." + \
                            metrics.deserialize(ls).__name__
                    except:
                        try:
                            eval("custom_losses."+ls+"()")
                            loss_string = "custom_losses."+ls+"()"
                        except:
                            try:
                               eval("custom_losses."+custom_losses.metric_name_unabbreviate(ls)+"()")
                               loss_string = "custom_losses." + custom_losses.metric_name_unabbreviate(ls)+"()"
                            except:
                                loss_string = None
            elif isinstance(ls,dict):
                try:
                    name = ls["name"]
                except:
                    raise Exception("The loss ", str(ls), " has unspecified name.")
                try:
                    args = ls["args"]
                except:
                    args = []
                try:
                    kwargs = ls["kwargs"]
                except:
                    kwargs = {}
                loss_string = utils.build_method_string_from_dict("losses", name, args, kwargs)
            else:
                loss_string = None
                print("WARNING: Invalid input for loss: ", str(ls), ". The loss will not be added to the model.", show = True)
            if loss_string is not None:
                try:
                    eval(loss_string)
                    self._loss_string = loss_string
                    self._loss = eval(self.loss_string)
                    if "self." in loss_string:
                        print("\tSet custom loss:", loss_string.replace("self.", ""), ".\n", show = verbose)
                    else:
                        print("\tSet loss:", loss_string, ".\n", show = verbose)
                except Exception as e:
                    loss_string = "custom_losses.minus_y_pred()"
                    self._loss_string = loss_string
                    self._loss = eval(self.loss_string)
                    print(e)
                    print("Could not set loss", str(ls), ".\nDefault loss (custom_losses.minus_y_pred) has been set.\n", show = verbose)
            else:
                loss_string = "custom_losses.minus_y_pred()"
                self._loss_string = loss_string
                self._loss = eval(self.loss_string)
                print("Could not set loss", str(ls), ".\nDefault loss (custom_losses.minus_y_pred) has been set.\n", show = verbose)
            self.ManagedObject.log = {utils.generate_timestamp(): {"action": "loss set",
                                                                   "loss": self.loss_string}}
        print(strategy_header_string,"\nExecuting within strategy context:",self.Resources.strategy,".\n",show=verbose)        
        self.Resources.strategy_executor(func_to_run())
        print("\nExecution in strategy context done.",show=verbose)
        print(strategy_header_string,"\n",show=verbose)

    def __set_metrics(self, verbose: Optional[IntBool] = None) -> None:
        """
        Private method used by the 
        :meth:`NF.__set_tf_objects <NF4HEP.NF._NF__set_tf_objects>` one
        to set the metrics objects (as for the loss, metrics could be |tf_keras_metrics_link|
        objects or a custom metrics defined in the :mod:`Dnn_likelihood <dnn_likelihood>` module). 
        It sets the :attr:`NF.metrics_strings <NF4HEP.NF.metrics_strings>`
        and :attr:`NF.metrics <NF4HEP.NF.metrics>` attributes. The former is set from the
        :attr:`NF.__model_compile_inputs <NF4HEP.NF._NF__model_compile_inputs>` 
        dictionary, while the latter is set by evaluating each item in the the former.

        - **Arguments**

            - **verbose**

                See :argument:`verbose <common_methods_arguments.verbose>`.
        """
        verbose, verbose_sub = self.get_verbosity(verbose)
        def func_to_run():
            print(header_string_1, "\nSetting metrics\n", show = verbose)
            self._metrics_strings = []
            self._metrics = []
            if self.model_compile_inputs["metrics"] is []:
                print("No metrics have been defined.\n", show = verbose)
            for met in self.model_compile_inputs["metrics"]:
                if isinstance(met, str):
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
                                eval("custom_losses."+met+"()")
                                metric_string = "custom_losses."+met+"()"
                            except:
                                try:
                                    eval("custom_losses."+custom_losses.metric_name_unabbreviate(met)+"()")
                                    metric_string = "custom_losses." + custom_losses.metric_name_unabbreviate(met)+"()"
                                except:
                                    metric_string = None
                elif isinstance(met,dict):
                    try:
                        name = met["name"]
                    except:
                        raise Exception("The metric ", str(met), " has unspecified name.")
                    try:
                        args = met["args"]
                    except:
                        args = []
                    try:
                        kwargs = met["kwargs"]
                    except:
                        kwargs = {}
                    metric_string = utils.build_method_string_from_dict("metrics", name, args, kwargs)
                else:
                    metric_string = None
                    print("WARNING: Invalid input for metric: ", str(met), ". The metric will not be added to the model.", show = True)
                if metric_string is not None:
                    try:
                        eval(metric_string)
                        self._metrics_strings.append(metric_string)
                        if "self." in metric_string:
                            print("\tAdded custom metric:", metric_string.replace("self.", ""), ".\n", show = verbose)
                        else:
                            print("\tAdded metric:", metric_string, ".\n", show = verbose)
                    except Exception as e:
                        print(e)
                        print("Could not add metric", metric_string, ".\n", show = verbose)
                else:
                    print("Could not add metric", str(met), ".\n", show = verbose)
            for metric_string in self.metrics_strings:
                self._metrics.append(eval(metric_string))
            self.ManagedObject.log = {utils.generate_timestamp(): {"action": "metrics set",
                                                                   "metrics": self.metrics_strings}}
        print(strategy_header_string,"\nExecuting within strategy context:",self.Resources.strategy,".\n",show=verbose)        
        self.Resources.strategy_executor(func_to_run())
        print("\nExecution in strategy context done.",show=verbose)
        print(strategy_header_string,"\n",show=verbose)

    def __set_callbacks(self, verbose: Optional[IntBool] = None) -> None:
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
        verbose, verbose_sub = self.get_verbosity(verbose)
        def func_to_run():
            print(header_string_1, "\nSetting callbacks\n", show = verbose)
            self._callbacks_strings = []
            self._callbacks = []
            for cb in self.model_callbacks_inputs:
                if isinstance(cb,str):
                    if cb == "ModelCheckpoint":
                        self.ManagedObject.FileManager.check_create_folder(self.ManagedObject.FileManager.output_checkpoints_folder)
                        cb_string = "callbacks.ModelCheckpoint(filepath=r'" + str(self.ManagedObject.FileManager.output_checkpoints_files)+"')"
                    elif cb == "TensorBoard":
                        self.ManagedObject.FileManager.check_create_folder(self.ManagedObject.FileManager.output_tensorboard_log_dir)
                        cb_string = "callbacks.TensorBoard(log_dir=r'" + str(self.ManagedObject.FileManager.output_tensorboard_log_dir)+"')"
                    elif cb == "PlotLossesKeras":
                        try:
                            from livelossplot import PlotLossesKerasTF as PlotLossesKeras # type: ignore
                            from livelossplot.outputs import MatplotlibPlot # type: ignore
                            self.ManagedObject.FileManager.check_create_folder(self.ManagedObject.FileManager.output_figures_folder)
                            cb_string = "PlotLossesKeras(outputs=[MatplotlibPlot(figpath = r'" + str(self.ManagedObject.FileManager.output_figure_plot_losses_keras_file)+"')])"
                        except:
                            print(header_string_2, "\nNo module named 'livelossplot's. Continuing without.\nIf you wish to plot the loss in real time please install 'livelossplot'.\n")
                            cb_string = None
                    else:
                        if "(" in cb:
                            cb_string = "callbacks."+cb.replace("callbacks.", "")
                        else:
                            cb_string = "callbacks." + \
                                cb.replace("callbacks.", "")+"()"
                elif isinstance(cb,dict):
                    try:
                        name = cb["name"]
                    except:
                        raise Exception("The layer ", str(cb), " has unspecified name.")
                    try:
                        args = cb["args"]
                    except:
                        args = []
                    try:
                        kwargs = cb["kwargs"]
                    except:
                        kwargs = {}
                    if name == "ModelCheckpoint":
                        self.ManagedObject.FileManager.check_create_folder(self.ManagedObject.FileManager.output_checkpoints_folder)
                        kwargs["filepath"] = self.ManagedObject.FileManager.output_checkpoints_files
                    elif name == "TensorBoard":
                        self.ManagedObject.FileManager.check_create_folder(self.ManagedObject.FileManager.output_tensorboard_log_dir)
                        #utils.check_create_folder(path.join(self.output_folder, "tensorboard_logs/fit"))
                        kwargs["log_dir"] = self.ManagedObject.FileManager.output_tensorboard_log_dir
                    elif name == "PlotLossesKeras":
                        try:
                            from livelossplot import PlotLossesKerasTF as PlotLossesKeras # type: ignore
                            from livelossplot.outputs import MatplotlibPlot # type: ignore
                            self.ManagedObject.FileManager.check_create_folder(self.ManagedObject.FileManager.output_figures_folder)
                            kwargs["outputs"] = "[MatplotlibPlot(figpath = r'" + self.ManagedObject.FileManager.output_figure_plot_losses_keras_file+"')]"
                        except:
                            print(header_string_2, "\nNo module named 'livelossplot's. Continuing without.\nIf you wish to plot the loss in real time please install 'livelossplot'.\n")
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
                    print("WARNING: Invalid input for callback: ", cb, ". The callback will not be added to the model.", show = True)
                if cb_string is not None:
                    try:
                        eval(cb_string)
                        self._callbacks_strings.append(cb_string)
                        print("\tAdded callback:", cb_string, ".\n", show = verbose)
                    except Exception as e:
                        print("Could not add callback", cb_string, ".\n", show = verbose)
                        print(e)
                else:
                    print("Could not set callback", cb_string, "\n", show = verbose)
            for cb_string in self.callbacks_strings:
                self._callbacks.append(eval(cb_string))
            self.ManagedObject.log = {utils.generate_timestamp(): {"action": "callbacks set",
                                                                   "callbacks": self.callbacks_strings}}
        print(strategy_header_string,"\nExecuting within strategy context:",self.Resources.strategy,".\n",show=verbose)        
        self.Resources.strategy_executor(func_to_run())
        print("\nExecution in strategy context done.",show=verbose)
        print(strategy_header_string,"\n",show=verbose)
        
    def __set_tf_objects(self, verbose: Optional[IntBool] = None) -> None:
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
        _, verbose_sub = self.get_verbosity(verbose)
        # this defines the string optimizer_string and object optimizer
        self.__set_optimizer(verbose = verbose_sub)
        # this defines the string loss_string and the object loss
        self.__set_loss(verbose=verbose_sub)
        # this defines the lists metrics_strings and metrics
        self.__set_metrics(verbose = verbose_sub)
        # this defines the lists callbacks_strings and callbacks
        self.__set_callbacks(verbose = verbose_sub)

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

    def model_define(self, verbose: Optional[IntBool] = None) -> None:
        """
        """
        verbose, verbose_sub = self.get_verbosity(verbose)
        def func_to_run():
            print(header_string_2,"\nDefining Keras model.\n", show = verbose)
            start = timer()
            x_ = Input(shape=(self.ManagedObject.ndims,), dtype=self.ManagedObject.Data.InputData.dtype_required)
            log_prob_ = self.ManagedObject.TrainableDistribution.distribution.log_prob(x_)
            self._NFModel = Model(x_, log_prob_)
            model_params = int(self.NFModel.count_params())
            model_trainable_params = int(np.sum([K.count_params(p) for p in self.NFModel.trainable_weights]))
            model_non_trainable_params = int(np.sum([K.count_params(p) for p in self.NFModel.non_trainable_weights]))
            self._model_params = {"total": model_params, "trainable": model_trainable_params, "non_trainable": model_non_trainable_params}
            summary_list = []
            self.NFModel.summary(print_fn=lambda x: summary_list.append(x.replace("\"","'")))
            end = timer()
            self.ManagedObject.log = {utils.generate_timestamp(): {"action": "Keras Model defined",
                                                                   "model summary": summary_list,
                                                                   "gpu mode": self.Resources.gpu_mode,
                                                                   "device id": self.Resources.training_device}}
            print("NFModel defined in", str(end-start),"s.\n", show = verbose)
        print(strategy_header_string,"\nExecuting within strategy context:",self.Resources.strategy,".\n",show=verbose)        
        self.Resources.strategy_executor(func_to_run())
        print("\nExecution in strategy context done.",show=verbose)
        print(strategy_header_string,"\n",show=verbose)

    def model_compile(self, verbose: Optional[IntBool] = None) -> None:
        """
        """
        verbose, verbose_sub = self.get_verbosity(verbose)
        def func_to_run():
            print(header_string_2,"\nCompiling Keras model\n", show = verbose)
            # Compile model
            start = timer()
            self.NFModel.compile(loss=self.loss, optimizer=self.optimizer, metrics=self.metrics, **self.model_compile_kwargs)
            end = timer()
            self.ManagedObject.log = {utils.generate_timestamp(): {"action": "Keras Model compiled",
                                                                   "gpu mode": self.Resources.gpu_mode,
                                                                   "device id": self.Resources.training_device}}
            print("NFModel",self.ManagedObject.name,"compiled in",str(end-start),"s.\n", show = verbose)
        print(strategy_header_string,"\nExecuting within strategy context:",self.Resources.strategy,".\n",show=verbose)        
        self.Resources.strategy_executor(func_to_run())
        print("\nExecution in strategy context done.",show=verbose)
        print(strategy_header_string,"\n",show=verbose)

    def model_build(self,
                    force: bool = False,
                    verbose: Optional[IntBool] = None
                   ) -> None:
        """
        """
        verbose, verbose_sub = self.get_verbosity(verbose)
        print(header_string_1,"\nDefining and compiling Keras model\n", show = verbose)
        try:
            self.NFModel
            create = False
            if self.NFModel._is_compiled:
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
            print("Model already built.", show = verbose)
            return
        if create:
            self.model_define(verbose=verbose_sub)
        if compile:
            self.model_compile(verbose=verbose_sub)
        #if self.Resources.strategy is None:
        #    if create:
        #        self.model_define(verbose=verbose_sub)
        #    if compile:
        #        self.model_compile(verbose=verbose_sub)
        #else:
        #    #self._strategy = tf.distribute.OneDeviceStrategy(device=device_id)
        #    print("Building NFModel", self.ManagedObject.name,"on device", self.Resources.training_device,".\n", show = verbose)
        #    with self.Resources.strategy.scope():
        #        # Rebuild Chain in current scope
        #        chain_name = self.ManagedObject._model_bijector_inputs["name"]+"Chain"
        #        self.ManagedObject._Chain = eval(chain_name)(model_define_inputs = self.ManagedObject._model_define_inputs,
        #                                                     model_bijector_inputs = self.ManagedObject._model_bijector_inputs,
        #                                                     model_chain_inputs = self.ManagedObject._model_chain_inputs,
        #                                                     verbose = verbose_sub)
        #        # Rebuild TrainableDistribution in current scope
        #        self.ManagedObject.BaseDistribution = NFDistribution(nf_main = self.ManagedObject, distribution = self.ManagedObject.base_distribution_inputs)
        #        self.ManagedObject.TrainableDistribution = NFDistribution(nf_main = self.ManagedObject, distribution = tfd.TransformedDistribution(self.ManagedObject.BaseDistribution.distribution,self.ManagedObject.Chain))
        #        if create:
        #            self.model_define(verbose=verbose_sub)
        #        if compile:
        #            self.model_compile(verbose=verbose_sub)
        #    self.ManagedObject.log = {utils.generate_timestamp(): {"action": "built tf model",
        #                                                           "gpu mode": self.Resources.gpu_mode,
        #                                                           "device id": self.Resources.training_device}}

    def model_train(self,
                    reset_seed: bool = True,
                    verbose: Optional[IntBool] = None) -> None:
        """
        """
        verbose, verbose_sub = self.get_verbosity(verbose)
        print(header_string_2,"\nTraining Keras model\n", show = verbose)
        # Reset random state
        if reset_seed:
            utils.reset_random_seeds(self.ManagedObject.seed)
        start = timer()
        data_manager = self.ManagedObject.Data.DataManager
        train_data = data_manager.TrainData
        val_data = data_manager.ValData
        dtype = train_data.dtype_required
        epochs_before_run = self.epochs_available
        epochs_to_run = self.__set_epochs_to_run()
        print("Required a total of",self.epochs_required,"epochs.",self.epochs_available,"epochs already available. Training for a maximum of",epochs_to_run,"epochs.\n", show = verbose)
        if epochs_to_run == 0:
            print("Please increase epochs_required to train for more epochs.\n", show = verbose)
        else:
            if np.shape(train_data.data_X) == (1,0):
                data_manager.generate_train_val_data(verbose=verbose_sub)
            X_train = data_manager.transform_data_X(data_X = train_data.data_X)
            X_val = data_manager.transform_data_X(data_X = val_data.data_X)
            Y_train = train_data.data_Y
            Y_val = val_data.data_Y
            if "PlotLossesKeras" in str(self.callbacks_strings):
                self.ManagedObject.Plotter.set_style()
            # Train model
            print("Start training of NFModel",self.ManagedObject.name, ".\n", show = verbose)
            #if self.weighted:
            #    history = self.model.fit(X_train, Y_train, sample_weight=self.W_train, initial_epoch=self.epochs_available, epochs=self.epochs_required, batch_size=self.batch_size, verbose=verbose_sub,
            #            validation_data=(X_val, Y_val), callbacks=self.callbacks, **self.model_train_kwargs)
            #else:
            history_obj = self.NFModel.fit(X_train, 
                                           Y_train, 
                                           initial_epoch = self.epochs_available, 
                                           epochs = self.epochs_required, 
                                           batch_size = self.batch_size, 
                                           verbose = verbose_sub,
                                           validation_data = (X_val, Y_val), 
                                           callbacks=self.callbacks, 
                                           **self.model_train_kwargs)
            end = timer()
            history: Dict[str,Any] = history_obj.history
            for k, v in history.items():
                history[k] = list(np.array(v, dtype=dtype))
            self.history = history
            self._epochs_available = len(self.history["loss"])
            epochs_current_run = self.epochs_available-epochs_before_run
            training_time_current_run = (end - start)
            self._training_time = self.training_time + training_time_current_run
            print("Updating model.history and model.epoch attribute.\n", show = verbose)
            self.NFModel.history.history = self.history
            self.NFModel.history.params["epochs"] = self.epochs_available
            self.NFModel.history.epoch = np.arange(self.epochs_available).tolist()
            if "PlotLossesKeras" in str(self.callbacks_strings):
                self.ManagedObject.Plotter.close_plot()
            self.ManagedObject.log = {utils.generate_timestamp(): {"action": "trained Keras Model",
                                                                   "epochs to run": epochs_to_run,
                                                                   "epochs run": epochs_current_run,
                                                                   "epochs total": self.epochs_available,
                                                                   "batch size": self.batch_size,
                                                                   "training time for current run": training_time_current_run,
                                                                   "total training time": self.training_time}}
            self.ManagedObject.FileManager.save(overwrite = True, verbose = verbose)
            print("NFModel", self.ManagedObject.name, "successfully trained for", epochs_current_run, "epochs in", training_time_current_run, "s (",training_time_current_run/epochs_current_run,"s/epoch).\n", show = verbose)


class NFDistribution(ObjectManager): # type: ignore
    """
    """
    managed_object_name: str = "NFMain"
    def __init__(self,
                 nf_main: NFMain,
                 distribution: Optional[Union[Dict[str,Any],str,tfp.distributions.distribution.Distribution]] = None
                ) -> None:
        # Attributes type declarations
        self._distribution: tfp.distributions.distribution.Distribution
        self._distribution_string: str
        self._trainable: bool
        # Initialise parent ObjectManager class
        super().__init__(managed_object = nf_main)
        # Initialize object
        if isinstance(distribution,tfp.distributions.TransformedDistribution):
            print(header_string_1, "\nInitializing NFDistribution object (TrainableDistribution).\n", show = self.ManagedObject.verbose)
        else:
            print(header_string_1, "\nInitializing NFDistribution object (BaseDistribution).\n", show = self.ManagedObject.verbose)
        self.distribution = distribution
    
    @property
    def distribution(self) -> Union[Dict[str,Any],str,tfp.distributions.distribution.Distribution]:
        return self._distribution

    @distribution.setter
    def distribution(self,
                     distribution: Optional[Union[Dict[str,Any],str,tfp.distributions.distribution.Distribution]] = None,
                    ) -> None:
        self._distribution = None
        self._trainable = False
        dist = None
        dist_string = None
        #print("The distribution input is",distribution)
        if isinstance(distribution,str):
            #print("Option 1")
            dist_str: str = str(distribution)
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
        elif isinstance(distribution,dict):
            #print("Option 2")
            dist_dic: dict = dict(distribution)
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
        elif isinstance(distribution, tfp.distributions.TransformedDistribution):
            #print("Option 3")
            dist_obj_trainable: tfp.distributions.TransformedDistribution = distribution
            try:
                #eval(dist_obj_trainable)
                dist_string = "trainable distribution"
                dist = dist_obj_trainable
                self._trainable = True
            except:
                raise Exception("Could not set distribution. The 'distribution' input argument does not have a valid format.")
        elif isinstance(distribution,tfp.distributions.distribution.Distribution):
            #print("Option 4")
            dist_obj: tfp.distributions.distribution.Distribution = distribution
            try:
                #eval(dist_obj)
                dist_string = "tfd."+str(dist_obj.__class__.__name__)+"(**"+str(dist_obj.__dict__['_parameters'])+")"
                dist = dist_obj
            except:
                raise Exception("Could not set distribution. The 'distribution' input argument does not have a valid format.")
        elif distribution is None:
            #print("Option 5")
            print("WARNING: No 'distribution' input argument has been provided. Proceeding with a default Normal distribution.")
            data_dtype = self.ManagedObject.Data.InputData.dtype_required
            dist_string = "tfd.Normal(loc=np.array(0,dtype='"+np.dtype(data_dtype).str+"'), scale=1, allow_nan_stats=False)"
        else:
            #print("Option 6")
            raise Exception("Could not set distribution. The 'distribution' input argument does not have a valid format.")
        if dist is not None:
            self._distribution_string = dist_string
            self._distribution = dist
            print("Distribution set to:", self._distribution_string, ".\n", show = self.ManagedObject.verbose)
        elif dist is None:
            try:
                dist = eval(dist_string)
                self._distribution_string = "tfd.Sample("+dist_string+",sample_shape=["+str(self.ManagedObject.ndims)+"])"
                print(self._distribution_string)
                self._distribution = tfd.Sample(dist,sample_shape=[self.ManagedObject.ndims])
                print("Base distribution set to:", self._distribution_string, ".\n", show = self.ManagedObject.verbose)
            except Exception as e:
                print(e)
                raise Exception("Could not set base distribution", dist_string, "\n")
        else:
            raise Exception("Could not set base distribution", dist_string, "\n")
        self.ManagedObject.log = {utils.generate_timestamp(): {"action": "distribution set",
                                                               "distribution": self._distribution_string}}

    @property
    def distribution_string(self) -> str:
        return self._distribution_string

    def describe_distributions(self,
                               distribution: tfp.distributions.distribution
                              ) -> None:
        """
        Describes a 'tfp.distributions' object.
        """
        print('\n'.join([str(d) for d in distribution]))


class NFPredictionsManager(PredictionsManager):
    """
    """
    managed_object_name: str = "NFMain"
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
    managed_object_name: str = "NFMain"
    def __init__(self,
                 nf_main: NFMain
                ) -> None:
        super().__init__(managed_object = nf_main)


class NFInference(Inference):
    """
    """
    managed_object_name: str = "NFMain"
    def __init__(self,
                 nf_main: NFMain
                ) -> None:
        super().__init__(managed_object = nf_main)


class NFPlotter(Plotter):
    """
    """
    managed_object_name: str = "NFMain"
    def __init__(self,
                 nf_main: NFMain
                ) -> None:
        super().__init__(managed_object = nf_main)

    def set_style(self) -> None:
        plt.style.use(mplstyle_path)

    def close_plot(self) -> None:
        plt.close()