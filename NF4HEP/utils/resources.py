import os
import numpy as np
import builtins
#from multiprocessing import cpu_count
import tensorflow as tf # type: ignore
from tensorflow.python.client import device_lib # type: ignore
import tensorflow as tf # type: ignore
import cpuinfo # type: ignore

from typing import Union, List, Dict, Callable, Tuple, Optional, NewType, Type, Generic, Any, TypeVar, TYPE_CHECKING
from typing_extensions import TypeAlias
from NF4HEP.utils import utils
from NF4HEP.utils.custom_types import Array, ArrayInt, ArrayStr, DataType, StrPath, IntBool, StrBool, StrList, StrArray, FigDict, LogPredDict, Number, DTypeStr, DTypeStrList, DictStr
from NF4HEP.utils.verbosity import print
from NF4HEP.utils.verbosity import Verbosity

header_string_1 = "=============================="
header_string_2 = "------------------------------"

#https://stackoverflow.com/questions/42322698/tensorflow-keras-multi-threaded-model-fitting?utm_medium=organic&utm_source=google_rich_qa&utm_campaign=google_rich_qa

#import subprocess
#def get_available_gpus():
#    if os.name == 'nt':
#        try:
#            os.environ['PATH'] += os.pathsep + r'C:\Program Files\NVIDIA Corporation\NVSMI'
#            available_gpus = (str(subprocess.check_output(["nvidia-smi", "-L"])).replace("\\n'","").replace("b'","").split("\\n"))
#        except:
#            print("nvidia-smi.exe not found it its system folder 'C:\\Program Files\\NVIDIA Corporation\\NVSMI'. Please modify the PATH accordingly.")
#            available_gpus = []
#    else:
#        available_gpus = (str(subprocess.check_output(["nvidia-smi", "-L"])).replace("\\n'","").replace("b'","").split("\\n"))
#    #available_gpus_current = K.tensorflow_backend._get_available_gpus()
#    #available_gpus_current = K.tensorflow_backend._get_available_gpus()
#    print(str(len(available_gpus))+" GPUs available in current environment")
#    if len(available_gpus) >0:
#        print(available_gpus)
#    return available_gpus
#def get_available_cpus():
#    local_device_protos = device_lib.list_local_devices()
#    return [x.name for x in local_device_protos if x.device_type == 'CPU']

class ResourcesManager(Verbosity):
    """
    Class inherited by all other classes to provide the
    :meth:`Verbosity.get_verbosity <DNNLikelihood.Verbosity.get_verbosity>` method.
    """
    def __init__(self,
                 resources_inputs: Optional[Dict[str,Any]] = None,
                 verbose: Optional[IntBool] = None
                ) -> None:
        # Attributes type declatation
        self._resources_inputs: Dict[str,Any]
        self._available_gpus: List
        self._available_cpus: List
        self._active_gpus: List
        self._gpu_mode: bool
        self._training_device = None
        self._strategy: Optional[tf.distribute.Strategy]
        # Initialise parent Verbosity class
        super().__init__(verbose)
        # Set verbosity
        verbose, verbose_sub = self.get_verbosity(verbose)
        # Initialize object
        print(header_string_1,"\nInitializing Resources Manager.\n", show = verbose)
        self.resources_inputs = resources_inputs if resources_inputs is not None else {}
        self.__set_resources(verbose = verbose)

    @property
    def resources_inputs(self) -> Dict[str,Any]:
        return self._resources_inputs
    
    @resources_inputs.setter
    def resources_inputs(self,
                         resources_inputs: Dict[str,Any]
                        ) -> None:
        self._resources_inputs = resources_inputs

    @property
    def available_gpus(self) -> List:
        return self._available_gpus

    @property
    def available_cpus(self) -> List:
        return self._available_cpus

    @property
    def active_gpus(self) -> List:
        return self._active_gpus

    @property
    def gpu_mode(self) -> bool:
        return self._gpu_mode
    
    @property
    def training_device(self) -> Optional[str]:
        return self._training_device
    
    @property
    def strategy(self) -> Optional[tf.distribute.Strategy]:
        return self._strategy
    
    def __set_resources(self,
                        verbose: Optional[IntBool] = None
                       ) -> None:
        """
        Private method used by the :meth:`DnnLik.__init__ <DNNLikelihood.DnnLik.__init__>` one to set resources.
        If :attr:`DnnLik.__resources_inputs <DNNLikelihood.DnnLik.__resources_inputs` is ``None``, it 
        calls the methods 
        :meth:`DnnLik.get_available_cpus <DNNLikelihood.DnnLik.get_available_cpus` and
        :meth:`DnnLik.set_gpus <DNNLikelihood.DnnLik.set_gpus` inherited from the
        :class:`Verbosity <DNNLikelihood.Verbosity>` class, otherwise it sets resources from input arguments.
        The latter method is needed, when the object is a member of an esemble, to pass available resources from the parent
        :class:`DnnLikEnsemble <DNNLikelihood.DnnLikEnsemble>` object.
        
        - **Arguments**

            - **verbose**
            
                See :argument:`verbose <common_methods_arguments.verbose>`.
        """
        verbose, verbose_sub = self.get_verbosity(verbose)
        self.check_tf_gpu(verbose=verbose)
        utils.check_set_dict_keys(self._resources_inputs, 
                                  ["active_gpus","target_gpu","strategy"],
                                  ["all","auto",None], 
                                  verbose = verbose_sub)
        
        self.get_available_cpus(verbose=verbose_sub)
        self.set_gpus(verbose=verbose_sub)
        self.set_strategy(verbose=verbose_sub)

    def check_tf_gpu(self, verbose: Optional[IntBool] = None) -> None:
        if not tf.test.gpu_device_name():
            print("To enable GPU support please install GPU version of TensorFlow", show = verbose)

    def get_available_gpus(self, verbose: Optional[IntBool] = None) -> None:
        verbose, _ = self.get_verbosity(verbose)
        local_device_protos = device_lib.list_local_devices()
        available_gpus = [[x.name, x.physical_device_desc] # type: ignore
                          for x in local_device_protos if x.device_type == 'GPU'] # type: ignore
        print(str(len(available_gpus))+" GPUs available", show = verbose)
        self._available_gpus = available_gpus    

    def get_available_cpus(self, verbose: Optional[IntBool] = None) -> None:
        verbose, _ = self.get_verbosity(verbose)
        local_device_protos = device_lib.list_local_devices()
        id = [x.name for x in local_device_protos if x.device_type == 'CPU'][0] # type: ignore
        local_device_protos = cpuinfo.get_cpu_info()
        try:
            brand = local_device_protos['brand']
        except:
            try:
                brand = local_device_protos['brand_raw']
            except:
                brand = ""
        cores_count = local_device_protos['count']
        available_cpus = [id, brand, cores_count]
        print(str(cores_count)+" CPU cores available", show = verbose)
        self._available_cpus = available_cpus

    def set_gpus(self,
                 verbose: Optional[IntBool] = None
                ) -> None:
        verbose, verbose_sub = self.get_verbosity(verbose)
        gpus_list = self.resources_inputs["active_gpus"]
        self.get_available_gpus(verbose=verbose_sub)
        if len(self.available_gpus) == 0:
            print('No available GPUs. Running with CPU support only.', show = verbose)
            self._active_gpus = self.available_gpus
        else:
            # Set gpus_list
            if isinstance(gpus_list,str):
                if gpus_list == "all":
                    gpus_list = list(range(len(self.available_gpus)))
                else:
                    raise Exception("gpu_list argument should be a list or None or the string 'all'.")
            elif isinstance(gpus_list,list):
                if len(list(gpus_list)) > len(self.available_gpus):
                    print('WARNING: Not all selected GPU are available. Available GPUs are:\n', self.available_gpus, ". Proceeding with all available GPUs.", show = True)
                    gpus_list = list(range(len(self.available_gpus)))
            elif gpus_list is None:
                gpus_list = []
            else:
                raise Exception("gpu_list argument should be a list or None or the string 'all'.")
            # Set active_gpus
            if len(gpus_list) == 0:
                print("No GPUs have been set. Running with CPU support only.", show = verbose)
                self._active_gpus = []
            elif len(gpus_list) == 1:
                print("1 GPU has been set:\n"+str(self.available_gpus[gpus_list[0]]), '.', show = verbose)
                self._active_gpus = [self.available_gpus[gpus_list[0]]]
            else:
                selected_gpus = np.array(self.available_gpus)[gpus_list].tolist()
                print(len(gpus_list), "GPUs have been set:\n", "\n".join([str(x) for x in selected_gpus]), '.', show = verbose)
                self._active_gpus = selected_gpus
        # Set gpu_mode
        if self.active_gpus == []:
            self._gpu_mode = False
        else:
            self._gpu_mode = True
            
    def set_strategy(self,
                     verbose: Optional[IntBool] = None
                    ) -> None:
        gpu = self.resources_inputs["target_gpu"]
        strategy = self.resources_inputs["strategy"]
        if strategy is None:
            self._strategy = None
        elif strategy == "OneDevice":
            if isinstance(gpu,str):
                if gpu == "auto":
                    gpu = 0
                else:
                    print("WARNING: invalid 'target_gpus' input argument. Proceeding with 'target_gpus=[0]'.")
                    gpu = 0
            if self.gpu_mode:
                if gpu > len(self.available_gpus):
                    print("gpu", gpu, "does not exist. Continuing on gpu '0'.\n", show = verbose)
                    gpu = 0
                self._training_device = self.available_gpus[gpu]
                if self.training_device:
                    device_id = self.training_device[0]
                else:
                    device_id = None
            else:
                if gpu != "auto":
                    print("GPU mode selected without any active GPU. Proceeding with CPU support.\n", show = verbose)
                self._training_device = self.available_cpus[0]
                if self.training_device:
                    device_id = self.training_device
                else:
                    device_id = None
            self._strategy = tf.distribute.OneDeviceStrategy(device=device_id)
        else:
            print("WARNING: TF2 strategies other than OneDeviceStrategy need to be manually implemented.")

    def set_gpus_env(self,
                     gpus_list: Optional[Union[List[int],str]] = "all", 
                     verbose: Optional[IntBool] = None
                    ) -> None:
        verbose, _ = self.get_verbosity(verbose)
        self.get_available_gpus(verbose=False)
        if len(self.available_gpus) == 0:
            print('No available GPUs. Running with CPU support only.', show = verbose)
            self._active_gpus = self.available_gpus
        if isinstance(gpus_list, str):
            if gpus_list == "all":
                gpus_list = list(range(len(self.available_gpus)))
            else:
                raise Exception("gpu_list argument should be a list or None or the string 'all'.")
        elif isinstance(gpus_list, list):
            if np.amax(np.array(gpus_list)) > len(self.available_gpus)-1:
                print('WARNING: Not all selected GPU are available. Available GPUs are:\n', self.available_gpus, ". Proceeding with all available GPUs.", show = True)
                gpus_list = list(range(len(self.available_gpus)))
        elif gpus_list is None:
            gpus_list = []
        else:
            raise Exception("gpu_list argument should be a list or None or the string 'all'.")
        # Set active_gpus
        if len(gpus_list) == 0:
            os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
            os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
            print("No GPUs have been set. Running with CPU support only.", show = verbose)
            self._active_gpus = []
        elif len(gpus_list) == 1:
            os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
            os.environ["CUDA_VISIBLE_DEVICES"] = str(gpus_list[0])
            print("1 GPU hase been set:\n"+str(self.available_gpus[gpus_list[0]]), '.', show = verbose)
            self._active_gpus = [self.available_gpus[gpus_list[0]]]
        else:
            os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
            os.environ["CUDA_VISIBLE_DEVICES"]=str(gpus_list).replace('[','').replace(']','')
            selected_gpus = np.array(self.available_gpus)[gpus_list].tolist()
            print(len(gpus_list), "GPUs have been set:\n", "\n".join([str(x) for x in selected_gpus]), '.', show = verbose)
            self._active_gpus = selected_gpus
