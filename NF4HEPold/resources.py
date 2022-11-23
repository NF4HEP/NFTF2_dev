import os
import numpy as np
import builtins
#from multiprocessing import cpu_count
from tensorflow.python.client import device_lib
import tensorflow as tf
import cpuinfo

from .show_prints import print, Verbosity

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

class Resources(Verbosity):
    """
    Class inherited by all other classes to provide the
    :meth:`Verbosity.set_verbosity <NF4HEP.Verbosity.set_verbosity>` method.
    """
    def check_tf_gpu(self,verbose=None):
        if not tf.test.gpu_device_name():
            print("To enable GPU support please install GPU version of TensorFlow", show=verbose)

    def get_available_gpus(self,verbose=None):
        verbose, _ = self.set_verbosity(verbose)
        local_device_protos = device_lib.list_local_devices()
        available_gpus = [[x.name, x.physical_device_desc]
                          for x in local_device_protos if x.device_type == 'GPU']
        print(str(len(available_gpus))+" GPUs available",show=verbose)
        self.available_gpus = available_gpus    

    def get_available_cpu(self,verbose=None):
        verbose, _ = self.set_verbosity(verbose)
        local_device_protos = device_lib.list_local_devices()
        id = [x.name for x in local_device_protos if x.device_type == 'CPU'][0]
        local_device_protos = cpuinfo.get_cpu_info()
        try:
            brand = local_device_protos['brand']
        except:
            try:
                brand = local_device_protos['brand_raw']
            except:
                brand = ""
        cores_count = local_device_protos['count']
        available_cpu = [id, brand, cores_count]
        print(str(cores_count)+" CPU cores available",show=verbose)
        self.available_cpu = available_cpu

    def set_gpus(self,gpus_list="all", verbose=None):
        verbose, verbose_sub = self.set_verbosity(verbose)
        self.get_available_gpus(verbose=verbose_sub)
        if len(self.available_gpus) == 0:
            print('No available GPUs. Running with CPU support only.', show=verbose)
            self.active_gpus = self.available_gpus
        else:
            # Set gpus_list
            if gpus_list == "all":
                gpus_list = list(range(len(self.available_gpus)))
            if len(gpus_list) > len(self.available_gpus):
                print('Not all selected GPU are available.', show=verbose)
                print('Available GPUs are:\n', self.available_gpus, ".",show=verbose)
                print('Proceeding with all available GPUs.', show=verbose)
                gpus_list = list(range(len(self.available_gpus)))
            # Set active_gpus
            if gpus_list == None or len(gpus_list) == 0:
                print("No GPUs have been set. Running with CPU support only.", show=verbose)
                self.active_gpus = []
            elif len(gpus_list) == 1:
                print("1 GPU has been set:\n"+str(self.available_gpus[gpus_list[0]]), '.',show=verbose)
                self.active_gpus = [self.available_gpus[gpus_list[0]]]
            else:
                selected_gpus = np.array(self.available_gpus)[gpus_list].tolist()
                print(len(gpus_list), "GPUs have been set:\n" + "\n".join([str(x) for x in selected_gpus]), '.', show=verbose)
                self.active_gpus = selected_gpus
        # Set gpu_mode
        if self.active_gpus != []:
            self.gpu_mode = True
        else:
            self.gpu_mode = False

    def set_gpus_env(self,gpus_list="all", verbose=None):
        verbose, _ = self.set_verbosity(verbose)
        self.get_available_gpus(verbose=False)
        if len(self.available_gpus) == 0:
            print('No available GPUs. Running with CPU support only.', show=verbose)
            self.active_gpus = self.available_gpus
        if gpus_list == None:
            os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
            os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
            print("No GPUs have been set. Running with CPU support only.", show=verbose)
            self.active_gpus = []
        elif gpus_list == "all":
            gpus_list = list(range(len(self.available_gpus)))
        else:
            if np.amax(np.array(gpus_list)) > len(self.available_gpus)-1:
                print('Not all selected GPU are available.', show=verbose)
                print('Available GPUs are:\n', self.available_gpus,".",show=verbose)
                print('Proceeding with all available GPUs.', show=verbose)
                gpus_list = list(range(len(self.available_gpus)))
        if len(gpus_list) > 1:
            os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
            os.environ["CUDA_VISIBLE_DEVICES"]=str(gpus_list).replace('[','').replace(']','')
            selected_gpus = np.array(self.available_gpus)[gpus_list].tolist()
            print(len(gpus_list), "GPUs have been set:\n" +
                  "\n".join([str(x) for x in selected_gpus]), '.', show=verbose)
            self.active_gpus = selected_gpus
        else:
            os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
            os.environ["CUDA_VISIBLE_DEVICES"] = str(gpus_list[0])
            print("1 GPU hase been set:\n"+str(self.available_gpus[gpus_list[0]]), '.',show=verbose)
            self.active_gpus = [self.available_gpus[gpus_list[0]]]
