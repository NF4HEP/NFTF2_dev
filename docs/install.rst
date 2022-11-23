Installation
============

Quick start
-----------

Requirements
------------
Several packages are required by the DNNLikelihood module. We provide a script to automatically set up the environment
and a script for environment check, which allows the user to tune its existing environment to run the DNNLikelihood 
package. We also list here all required packages with links to their documentation. The list is divided in general
requirements, i.e. packages that are needed by most of the objects and functions, and the various classes with their
specific requirements.

General requirements
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The following modules are required by most objects in the DNNLikelihood package.

    - Default Python modules

        - |builtins_link|
        
        - |datetime_link|

        - |os_link|

        - |pickle_link|

        - |sys_link|

        - |timeit_link|

    - External modules

        - |matplotlib_link|
        
        - |numpy_link|

The Histfactory object
^^^^^^^^^^^^^^^^^^^^^^

    - Default modules

        - |json_link|

    - External modules

        - (optional) |ipywidgets_link|

        - |json_patch_link|
        
        - |json_schema_link|
        
        - |pyhf_link|

        - |requests_link|

    - Internal modules

        - :obj:`NF4HEP.utils <utils>`

        - :class:`NF4HEP.Lik`

The Likelihood object
^^^^^^^^^^^^^^^^^^^^^

    - External modules

        - |deepdish_link|        

        - |cloudpickle_link|        
        
        - |ipywidgets_link| (optional)

        - |json_patch_link|
        
        - |json_schema_link|
        
        - |pyhf_link|

        - |requests_link|

    - Internal modules

        - :obj:`NF4HEP.utils <utils>`

        - :obj:`NF4HEP.inference <inference_module>`


The Sampler object
^^^^^^^^^^^^^^^^^^

    - Default modules

        - |copy_link|

        - |importlib_link|

        - |multiprocessing_link|

    - External modules

        - |psutil_link|

        - |scipy_link|
        
        - |emcee_link|

    - Internal modules

        - :obj:`NF4HEP.utils <utils>`

        - :class:`NF4HEP.Data`

The Data object
^^^^^^^^^^^^^^^

    - External modules

        - |h5py_link|

        - |sklearn_link|

    - Internal modules

        - :obj:`NF4HEP.utils <utils>`

The DNNLikelihood object
^^^^^^^^^^^^^^^^^^^^^^^^^

    - Default modules

        - ||

    - External modules

        - ||

    - Internal modules

        - :obj:`NF4HEP.inference <inference_module>`

        - :obj:`NF4HEP.set_resources <set_resources_module>`

        - :obj:`NF4HEP.utils <utils>`

        - :class:`NF4HEP.Data`

        - :class:`NF4HEP.Data`

The DnnLikEnsemble object
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^


Environment
-----------

.. include:: external_links.rst