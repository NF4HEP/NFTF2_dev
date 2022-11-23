.. _DnnLik_usage:

Usage
^^^^^

We give here a brief introduction to the use of the :class:`NF <NF4HEP.NF>` class. Refer to the 
full class documentation for more details. All examples will be referred to the toy likelihood introduced in 
:ref:`the Likelihood object Usage <likelihood_usage>` section of the documentation.

A :class:`NF <NF4HEP.NF>` object is defined through a series of dictionaries (or lists of dictionaries) describing data and 
DNN inputs, optimizer, compilation, callbacks, and training parameters.
An example of the dictionaries defining the :class:`NF <NF4HEP.NF>` configurations is given by 
(the example is designed to be representative and to show most of the input possibilities even though it does 
not represent a "meaningful" choice of model and hyperparameters).

.. code-block:: python

    model_data_inputs = {"npoints": [10000, 3000, 5000],
                     "scalerX": True,
                     "scalerY": True,
                     "weighted": False}
    model_define_inputs = model_define_inputs = {"hidden_layers": ["Dense(1000,activation='relu',kernel_initializer='glorot_uniform',kernel_regularizer=regularizers.L1L2(l1=0.001, l2=0.001))",
                                                                   {"name": "Dense",
                                                                    "args": [1000],#{"units": 1000},
                                                                    "kwargs": {"activation": "relu", 
                                                                               "use_bias": True,
                                                                               "kernel_initializer": "GlorotUniform(seed=None)",
                                                                               "bias_initializer": "zeros", 
                                                                               "kernel_regularizer": "L1L2(l1=0.001, l2=0.001)",
                                                                               "bias_regularizer": "L1(l1=0.0001)", 
                                                                               "activity_regularizer": "L1(l1=0.0001)", 
                                                                               "kernel_constraint": "MaxNorm(max_value=2, axis=0)",
                                                                               "bias_constraint": None}},
                                                                   "Dense(500)",
                                                                   {"name": "Activation",
                                                                    "args": ["relu"]},
                                                                   [100,"selu"],
                                                                   [100,"relu","glorot_uniform"],
                                                                   {"name": "Dense",
                                                                    "args": [1000],#{"units": 1000},
                                                                    "kwargs": {"use_bias": True,
                                                                               "kernel_initializer": {"name": "GlorotUniform",
                                                                                                      "args": [],
                                                                                                      "kwargs": {"seed": None}},
                                                                               "bias_initializer": "zeros", 
                                                                               "kernel_regularizer": {"name": "L1L2",
                                                                                                      "args": [],
                                                                                                      "kwargs": {"l1": 0.001,
                                                                                                                 "l2": 0.001}},
                                                                               "bias_regularizer": "L1(l1=0.0001)", 
                                                                               "activity_regularizer": "L1(l1=0.0001)", 
                                                                               "kernel_constraint": {"name": "MaxNorm",
                                                                                                      "args": [],
                                                                                                      "kwargs": {"max_value": 2,
                                                                                                                 "axis": 0}},
                                                                               "bias_constraint": "MaxNorm(max_value=2, axis=0)"}},
                                                                   "Activation('selu')",
                                                                   {"name": "BatchNormalization",
                                                                    "args": [],
                                                                    "kwargs": {"axis": -1, 
                                                                               "momentum": 0.99, 
                                                                               "epsilon": 0.001, 
                                                                               "center": True, 
                                                                               "scale": True,
                                                                               "beta_initializer": "zeros", 
                                                                               "gamma_initializer": "ones",
                                                                               "moving_mean_initializer": "zeros",
                                                                               "moving_variance_initializer": "ones", 
                                                                               "beta_regularizer": None,
                                                                               "gamma_regularizer": None, 
                                                                               "beta_constraint": None, 
                                                                               "gamma_constraint": None}},
                                                                   {"name": "Dropout",
                                                                    "args": [0.],
                                                                    "kwargs": {"noise_shape": None, 
                                                                               "seed": None}},
                                                                   "BatchNormalization",
                                                                   "AlphaDropout(0.1)",
                                                                   {"name": "AlphaDropout",
                                                                    "args": [0.], 
                                                                    "kwargs": {"noise_shape": None, 
                                                                                "seed": None}},
                                                                   {"name": "Dense",
                                                                    "args": [1000],
                                                                    "kwargs": {"activation": "relu", 
                                                                                "use_bias": True,
                                                                                "kernel_initializer": "glorot_uniform",
                                                                                "bias_initializer": "zeros", 
                                                                                "kernel_regularizer": None,
                                                                                "bias_regularizer": None, 
                                                                                "activity_regularizer": None, 
                                                                                "kernel_constraint": None,
                                                                                "bias_constraint": None}}], 
                                                  "act_func_out_layer": "linear",
                                                  "dropout_rate": 0,
                                                  "batch_norm": False}
    #model_optimizer_inputs = "optimizers.Adam(learning_rate = 0.001, beta_1 = 0.9, beta_2 = 0.999, amsgrad = False)"
    model_optimizer_inputs = {"name": "Adam",
    #                          "args": [],
    #                          "kwargs": {"learning_rate": 0.001,
    #                                     "beta_1": 0.9,
    #                                     "beta_2": 0.999,
    #                                     "amsgrad": False}}
    model_compile_inputs = {"loss": "mse", 
                            #"loss": {"name": "MeanSquaredError",
                            #         "args": [],
                            #         "kwargs": {"reduction": "losses.Reduction.SUM"}},
                            #"loss": "MeanSquaredError(reduction=losses.Reduction.SUM)",
                            "metrics": ["mean_squared_error",
                                        {"name": "MeanAbsoluteError",
                                         "args": [],
                                         "kwargs": {"dtype": "float32"}},
                                        "MeanSquaredLogarithmicError(dtype='float32')",
                                        "mape",
                                        #"mean_error",
                                        "me"]}
    model_callbacks_inputs = [{"name": "EarlyStopping",
                               "args": [],
                               "kwargs": {"monitor": "val_loss",
                                          "mode": "min",
                                          "patience": 100,
                                          "min_delta": 0.0001,
                                          "restore_best_weights": True}},
                              "TerminateOnNaN",
                              "PlotLossesKeras",
                              {"name": "TensorBoard",
                               "args": [],
                               "kwargs": {"log_dir": "automatically set by the framework",
                                          "histogram_freq": 0,
                                          "write_graph": True,
                                          "write_images": True,
                                          "update_freq": "epoch",
                                          "profile_batch": 2,
                                          "embeddings_freq": 0,
                                          "embeddings_metadata": None}},
                              {"name": "ReduceLROnPlateau",
                               "args": [],
                               "kwargs": {"monitor": "val_loss",
                                          "mode": "min",
                                          "factor": 0.2,
                                          "min_lr": 0.00008,
                                          "patience": 10,
                                          "min_delta": 0.0001}},
                              {"name": "ModelCheckpoint",
                               "args": [],
                               "kwargs": {"filepath": "automatically set by the framework",
                                          "monitor": "val_loss",
                                          "mode": "min",
                                          "save_best_only": True,
                                          "save_freq": "epoch"}}]
    model_train_inputs={"epochs": 50,
                        "batch_size": 512,
                        "shuffle": False,
                        "validation_freq": 1}

Each of the arguments

    - :argument:`model_data_inputs <NF.model_data_inputs>`
    - :argument:`model_define_inputs <NF.model_define_inputs>`
    - :argument:`model_optimizer_inputs <NF.model_optimizer_inputs>`
    - :argument:`model_compile_inputs <NF.model_compile_inputs>`
    - :argument:`model_callbacks_inputs <NF.model_callbacks_inputs>`
    - :argument:`model_train_inputs <NF.model_train_inputs>`
    
are input arguments of the :class:`NF <NF4HEP.NF>` class 
(see the corresponding documentation pages for a complete description)

The :argument:`model_data_inputs <NF.model_data_inputs>` dictionary contains information about the way data are processed
by the DNN. In particular it specifies the number of points for training, validation, and test, if ``X`` and/or ``Y`` data 
need to be scaled through a |standard_scalers_link| and if ``Y`` data need to be weighted. These parameters are passed to the
:class:`Data <NF4HEP.Data>` container class that is automatically initialized by the :class:`NF <NF4HEP.NF>`
one and that is used for data management.

The :argument:`model_define_inputs <NF.model_define_inputs>` dictionary contains the main hyperparameters needed to define 
the DNN. They are given by the scruture of hidden layers, each of which can be specified in a number of different ways,
the activation function in the output layer, and, optionally, two inputs that simplify the addition of dropout and batch normalization.
When hidden layers are specified through a list of dictionaries, such dictionaries follow the general scheme according to which an object
is specified by three items: ``"name"``, representing the object name (e.g. ``"Dense"``), ``"args", representing all positional arguments
(e.g. ``1000`` for the number of unit of a Dense layer), and ``"kwargs", a dictionary containing all keyword arguments (e.g. 
``{"activation": "relu", "use_bias": True}`` for a Dense layer).

The :argument:`model_optimizer_inputs <NF.model_optimizer_inputs>` is devoted to the definition of the optimizer.
It can be specified both as a string or a dictionary. In the latter case the same scheme used to input hidden layers in the
:argument:`model_define_inputs <NF.model_define_inputs>` dictionary is used. All |tf_keras_optimizers_link| are supported.

The :argument:`model_compile_inputs <NF.model_compile_inputs>` dictionary defines inputs for the |tf_keras_model_compile_link|
class. The dictionary contains two items, corresponding to the ``"loss"`` and ``"metrics"``. The former can be passed both as a string
or a dictionary. The latter is a list of strings and or dictionaries. In both case the input as dictionaries follows the aforementioned
scheme. Lossed and metrics strings may be passed in all formats supported by |tf_keras_link| (e.g. inputs such as ``"mse"``,
``"mean_squared_error"``, ``"MeanSquaredError"`` are all allowed).

The :argument:`model_compile_inputs <NF.model_compile_inputs>` list contains the list of |tf_keras_callbacks_link| to be used
during training. All |tf_keras_callbacks_link| are supported. Entries of the list could be strings or dictionaries, following the same
scheme as the other inputs.

The :argument:`model_train_inputs <NF.model_train_inputs>` dictionary is used to define parameters for the |tf_keras_model_fit_link| 
method. The ``"epochs"`` and ``"batch_size"`` items are mandatory, while any other additional parameter can be added and is directly 
passed to the |tf_keras_model_fit_link| method

The :class:`NF <NF4HEP.NF>` object can be instantiated in three different ways, depending on the value of the arguments
:argument:`data <NF.data>`, :argument:`input_data_file <NF.dainput_data_fileta>`, 
and :argument:`input_data_file <NF.input_file>`. The latter argument determines whether a new object is created from
input arguments (when is set to the default ``None``) or the object is imported from a previously saved one (when is set to the corresponding
path). When a new object is created, data can be passed both through an already defined :class:`Data <NF4HEP.Data>` object 
or through the path to a previously saved one. If the path is passed, then it is always used and, if also a locally defined 
:class:`Data <NF4HEP.Data>` object is passed, it gets ignored.

Starting from the data object we have saved in :ref:`the Data object Usage <data_usage>` section of the documentation, 
and using the arguments defined above, we can create a :class:`NF <NF4HEP.NF>` object as follows:

.. code-block:: python

    dnnlikelihood = NF4HEP.NF(name="DNNLikelihood_toy",
                                         data=None,
                                         input_data_file="toy/data/toy_data",
                                         load_on_RAM=False,
                                         seed=1,
                                         dtype=None,
                                         same_data=True,
                                         model_data_inputs=model_data_inputs,
                                         model_define_inputs=model_define_inputs,
                                         model_optimizer_inputs=model_optimizer_inputs,
                                         model_compile_inputs=model_compile_inputs,
                                         model_callbacks_inputs=model_callbacks_inputs,
                                         model_train_inputs=model_train_inputs,
                                         resources_inputs=None,
                                         output_folder="toy/dnnlikelihood",
                                         ensemble_name=None,
                                         input_file=None,
                                         verbose=2)

    >>> ============================== 
        No DnnLik input files and folders specified.

        ============================== 
        DnnLik output folder set to
        	 <abs_path_to_output_folder> .

        ============================== 
        This is a 'standalone' DNNLikelihood and does not belong to a DNNLikelihood_ensemble. The attributes 'ensemble_name' and 'ensemble_folder' have therefore been set to None.

        ============================== 
        No Likelihood object has been specified.

        88 CPU cores available
        4 GPUs available
        4 GPUs have been set:
        ['/device:GPU:0', 'device: 0, name: NVIDIA GeForce RTX 2080 Ti, pci bus id: 0000:18:00.0, compute capability: 7.5']
        ['/device:GPU:1', 'device: 1, name: NVIDIA GeForce RTX 2080 Ti, pci bus id: 0000:3b:00.0, compute capability: 7.5']
        ['/device:GPU:2', 'device: 2, name: NVIDIA GeForce RTX 2080 Ti, pci bus id: 0000:5e:00.0, compute capability: 7.5']
        ['/device:GPU:3', 'device: 3, name: NVIDIA GeForce RTX 2080 Ti, pci bus id: 0000:86:00.0, compute capability: 7.5'] .
        ============================== 
        Initialize Data object.

        ============================== 
        Data input folder set to
        	 <abs_path_to_data_input_folder> .

        ============================== 
        Data object loaded in 0.01520899999923131 .

        ============================== 
        Data output folder set to
        	 <abs_path_to_data_output_folder> .

        ============================== 
        Data log file
        	 <abs_path_to_data_output_folder>\toy_data.log 
        updated (or saved if it did not exist) in 0.002114199999596167 s.

        ============================== 
        Setting hidden layers

        Added Input layer:  layers.Input(shape=(11,))
        Added hidden layer:  layers.Dense(1000,activation='relu',kernel_initializer='glorot_uniform',kernel_regularizer=regularizers.L1L2(l1=0.001, l2=0.001))
        Added hidden layer:  layers.Dense(1000, activation='relu', use_bias=True, kernel_initializer=initializers.GlorotUniform(seed=None), bias_initializer='zeros', kernel_regularizer=regularizers.L1L2(l1=0.001, l2=0.001), bias_regularizer=regularizers.L1(l1=0.0001), activity_regularizer=regularizers.L1(l1=0.0001), kernel_constraint=constraints.MaxNorm(max_value=2, axis=0), bias_constraint=None)
        Added hidden layer:  layers.Dense(500)
        Added hidden layer:  layers.Activation('relu')
        Added hidden layer:  layers.Dense(100, activation='selu', kernel_initializer='lecun_normal')
        Added hidden layer:  layers.Dense(100, activation='relu')
        Added hidden layer:  layers.Dense(1000, use_bias=True, kernel_initializer=initializers.GlorotUniform(seed=None), bias_initializer='zeros', kernel_regularizer=regularizers.L1L2(l1=0.001, l2=0.001), bias_regularizer=regularizers.L1(l1=0.0001), activity_regularizer=regularizers.L1(l1=0.0001), kernel_constraint=constraints.MaxNorm(max_value=2, axis=0), bias_constraint=constraints.MaxNorm(max_value=2, axis=0))
        Added hidden layer:  layers.Activation('selu')
        Added hidden layer:  layers.BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True, beta_initializer='zeros', gamma_initializer='ones', moving_mean_initializer='zeros', moving_variance_initializer='ones', beta_regularizer=None, gamma_regularizer=None, beta_constraint=None, gamma_constraint=None)
        Added hidden layer:  layers.Dropout(0.1, noise_shape=None, seed=None)
        Added hidden layer:  layers.BatchNormalization()
        Added hidden layer:  layers.AlphaDropout(0.1, seed=2)
        Added hidden layer:  layers.AlphaDropout(0.1, noise_shape=None, seed=None)
        Added hidden layer:  layers.Dense(1000, activation='relu', use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None)
        Added Output layer:  layers.Dense(1, activation='linear') .

        ============================== 
        Setting optimizer

        Optimizer set to: optimizers.Adam(learning_rate = 0.001, beta_1 = 0.9, beta_2 = 0.999, amsgrad = False) 

        ============================== 
        Setting loss

        Loss set to: losses.mse .

        ============================== 
        Setting metrics

        	Added metric: metrics.mean_squared_error
        	Added metric: metrics.MeanAbsoluteError(dtype='float32')
        	Added metric: metrics.MeanSquaredLogarithmicError(dtype='float32')
        	Added metric: metrics.mape
        	Added custom metric: mean_error

        ============================== 
        Setting callbacks

        	Added callback: callbacks.EarlyStopping(monitor='val_loss', mode='min', patience=100, min_delta=0.0001, restore_best_weights=True)
        	Added callback: callbacks.TerminateOnNaN()
        	Added callback: PlotLossesKeras()
        	Added callback: callbacks.TensorBoard(log_dir=r'<abs_path_to_output_folder>\tensorboard_logs', histogram_freq=0, write_graph=True, write_images=True, update_freq='epoch', profile_batch=2, embeddings_freq=0, embeddings_metadata=None)
        	Added callback: callbacks.ReduceLROnPlateau(monitor='val_loss', mode='min', factor=0.2, min_lr=8e-05, patience=10, min_delta=0.0001)
        	Added callback: callbacks.ModelCheckpoint(filepath=r'<abs_path_to_output_folder>\checkpoints\DNNLikelihood_toy_checkpoint.{epoch:02d}-{val_loss:.2f}.h5', monitor='val_loss', mode='min', save_best_only=True, save_freq='epoch')

        ============================== 
        \nJson file\n\t
        	 <abs_path_to_output_folder>\DNNLikelihood_toy_summary.json 
        saved in 0.004206500000691449 s.

        ============================== 
        DnnLik log file
        	 <abs_path_to_output_folder>\DNNLikelihood_toy.log 
        saved in 0.0011275000006207847 s.


where we used the placeholders <abs_path_to_input_data_folder>, <abs_path_to_output_data_folder> and <abs_path_to_output_folder> to indicate the 
respective paths.

If the :argument:`name <NF.name>` argument is not passed, then one is automatically generated. 
The :argument:`load_on_RAM <NF.load_on_RAM>` argument is passed directly to the :class:`Data <NF4HEP.Data>` object (see the 
documentation of :argument:`Data.load_on_RAM <Data.load_on_RAM>`). The :argument:`seed <NF.seed>` argument is used to
initialize the |numpy_link| and |tf_link| random number generators. The :argument:`output_folder <NF.output_folder>` argument
is specified as usual, with a relative or absolute path, and if left unspecified, it is automatically set to the current working 
directory.

The :argument:`same_data <NF.same_data>` and :argument:`ensemble_name <NF.ensemble_name>` arguments 
should not be specified by the user (or left ``None``) and are automatically passed in case the 
:class:`NF <NF4HEP.NF>` object is part of a :mod:`DNNLikEnsemble <dnn_likelihood_ensemble>`.

When the object is created, resources are allocated through the hinerithed class :class:`Resources <NF4HEP.Resources>` (
see :ref:`Resources management <resources_module>` for details). The :argument:`datresources_inputsa <NF.resources_inputs>`
should not be specified by the user (or left ``None``) since it is only used for automatic resources allocation when the object is part
of a :mod:`DNNLikEnsemble <dnn_likelihood_ensemble>` object.

The :meth:`NF.__init__ <NF4HEP.NF.__init__>` method sets several attributes, which can be seen in the following output:

.. code-block:: python

    dnnlikelihood.__dict__

    >>> ['active_gpus', [['/device:GPU:0', 'device: 0, name: NVIDIA GeForce RTX 2080 Ti, pci bus id: 0000:18:00.0, compute capability: 7.5'], ['/device:GPU:1', 'device: 1, name: NVIDIA GeForce RTX 2080 Ti, pci bus id: 0000:3b:00.0, compute capability: 7.5'], ['/device:GPU:2', 'device: 2, name: NVIDIA GeForce RTX 2080 Ti, pci bus id: 0000:5e:00.0, compute capability: 7.5'], ['/device:GPU:3', 'device: 3, name: NVIDIA GeForce RTX 2080 Ti, pci bus id: 0000:86:00.0, compute capability: 7.5']]]
        ['act_func_out_layer', 'linear']
        ['available_cpu', ['/device:CPU:0', 'Intel(R) Xeon(R) Gold 6152 CPU @ 2.10GHz', 88]]
        ['available_gpus', [['/device:GPU:0', 'device: 0, name: NVIDIA GeForce RTX 2080 Ti, pci bus id: 0000:18:00.0, compute capability: 7.5'], ['/device:GPU:1', 'device: 1, name: NVIDIA GeForce RTX 2080 Ti, pci bus id: 0000:3b:00.0, compute capability: 7.5'], ['/device:GPU:2', 'device: 2, name: NVIDIA GeForce RTX 2080 Ti, pci bus id: 0000:5e:00.0, compute capability: 7.5'], ['/device:GPU:3', 'device: 3, name: NVIDIA GeForce RTX 2080 Ti, pci bus id: 0000:86:00.0, compute capability: 7.5']]]
        ['batch_norm', 'custom']
        ['batch_size', 512]
        ['callbacks', []]
        ['callbacks_strings', ["callbacks.EarlyStopping(monitor='val_loss', mode='min', patience=100, min_delta=0.0001, restore_best_weights=True)", 'callbacks.TerminateOnNaN()', 'PlotLossesKeras()', "callbacks.TensorBoard(log_dir=r'<abs_path_to_output_folder>\\tensorboard_logs', histogram_freq=0, write_graph=True, write_images=True, update_freq='epoch', profile_batch=2, embeddings_freq=0, embeddings_metadata=None)", "callbacks.ReduceLROnPlateau(monitor='val_loss', mode='min', factor=0.2, min_lr=8e-05, patience=10, min_delta=0.0001)", "callbacks.ModelCheckpoint(filepath=r'<abs_path_to_output_folder>\\checkpoints\\DNNLikelihood_toy_checkpoint.{epoch:02d}-{val_loss:.2f}.h5', monitor='val_loss', mode='min', save_best_only=True, save_freq='epoch')"]]
        ['data', <NF4HEP.data.Data object at 0x00000219FD3327F0>]
        ['dropout_rate', 0]
        ['dtype', 'float64']
        ['ensemble_folder', None]
        ['ensemble_name', None]
        ['epochs_available', 0]
        ['epochs_required', 50]
        ['gpu_mode', True]
        ['hidden_layers', ["Dense(1000,activation='relu',kernel_initializer='glorot_uniform',kernel_regularizer=regularizers.L1L2(l1=0.001, l2=0.001))", {'name': 'Dense', 'args': [1000], 'kwargs': {'activation': 'relu', 'use_bias': True, 'kernel_initializer': 'GlorotUniform(seed=None)', 'bias_initializer': 'zeros', 'kernel_regularizer': 'L1L2(l1=0.001, l2=0.001)', 'bias_regularizer': 'L1(l1=0.0001)', 'activity_regularizer': 'L1(l1=0.0001)', 'kernel_constraint': 'MaxNorm(max_value=2, axis=0)', 'bias_constraint': None}}, 'Dense(500)', {'name': 'Activation', 'args': ['relu']}, [100, 'selu'], [100, 'relu', 'glorot_uniform'], {'name': 'Dense', 'args': [1000], 'kwargs': {'use_bias': True, 'kernel_initializer': {'name': 'GlorotUniform', 'args': [], 'kwargs': {'seed': None}}, 'bias_initializer': 'zeros', 'kernel_regularizer': {'name': 'L1L2', 'args': [], 'kwargs': {'l1': 0.001, 'l2': 0.001}}, 'bias_regularizer': 'L1(l1=0.0001)', 'activity_regularizer': 'L1(l1=0.0001)', 'kernel_constraint': {'name': 'MaxNorm', 'args': [], 'kwargs': {'max_value': 2, 'axis': 0}}, 'bias_constraint': 'MaxNorm(max_value=2, axis=0)'}}, "Activation('selu')", {'name': 'BatchNormalization', 'args': [], 'kwargs': {'axis': -1, 'momentum': 0.99, 'epsilon': 0.001, 'center': True, 'scale': True, 'beta_initializer': 'zeros', 'gamma_initializer': 'ones', 'moving_mean_initializer': 'zeros', 'moving_variance_initializer': 'ones', 'beta_regularizer': None, 'gamma_regularizer': None, 'beta_constraint': None, 'gamma_constraint': None}}, {'name': 'Dropout', 'args': [0.1], 'kwargs': {'noise_shape': None, 'seed': None}}, 'BatchNormalization', 'AlphaDropout(0.1, seed=2)', {'name': 'AlphaDropout', 'args': [0.1], 'kwargs': {'noise_shape': None, 'seed': None}}, {'name': 'Dense', 'args': [1000], 'kwargs': {'activation': 'relu', 'use_bias': True, 'kernel_initializer': 'glorot_uniform', 'bias_initializer': 'zeros', 'kernel_regularizer': None, 'bias_regularizer': None, 'activity_regularizer': None, 'kernel_constraint': None, 'bias_constraint': None}}]]
        ['history', {}]
        ['idx_test', array([], dtype=int32)]
        ['idx_train', array([], dtype=int32)]
        ['idx_val', array([], dtype=int32)]
        ['input_data_file', '<abs_path_to_data_input_folder>\\toy_data']
        ['input_files_base_name', None]
        ['input_folder', None]
        ['input_history_json_file', None]
        ['input_idx_h5_file', None]
        ['input_likelihood_file', None]
        ['input_log_file', None]
        ['input_predictions_h5_file', None]
        ['input_scalers_pickle_file', None]
        ['input_file', None]
        ['input_tf_model_h5_file', None]
        ['layers', []]
        ['layers_string', ['layers.Input(shape=(11,))', "layers.Dense(1000,activation='relu',kernel_initializer='glorot_uniform',kernel_regularizer=regularizers.L1L2(l1=0.001, l2=0.001))", "layers.Dense(1000, activation='relu', use_bias=True, kernel_initializer=initializers.GlorotUniform(seed=None), bias_initializer='zeros', kernel_regularizer=regularizers.L1L2(l1=0.001, l2=0.001), bias_regularizer=regularizers.L1(l1=0.0001), activity_regularizer=regularizers.L1(l1=0.0001), kernel_constraint=constraints.MaxNorm(max_value=2, axis=0), bias_constraint=None)", 'layers.Dense(500)', "layers.Activation('relu')", "layers.Dense(100, activation='selu', kernel_initializer='lecun_normal')", "layers.Dense(100, activation='relu')", "layers.Dense(1000, use_bias=True, kernel_initializer=initializers.GlorotUniform(seed=None), bias_initializer='zeros', kernel_regularizer=regularizers.L1L2(l1=0.001, l2=0.001), bias_regularizer=regularizers.L1(l1=0.0001), activity_regularizer=regularizers.L1(l1=0.0001), kernel_constraint=constraints.MaxNorm(max_value=2, axis=0), bias_constraint=constraints.MaxNorm(max_value=2, axis=0))", "layers.Activation('selu')", "layers.BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True, beta_initializer='zeros', gamma_initializer='ones', moving_mean_initializer='zeros', moving_variance_initializer='ones', beta_regularizer=None, gamma_regularizer=None, beta_constraint=None, gamma_constraint=None)", 'layers.Dropout(0.1, noise_shape=None, seed=None)', 'layers.BatchNormalization()', 'layers.AlphaDropout(0.1, seed=2)', 'layers.AlphaDropout(0.1, noise_shape=None, seed=None)', "layers.Dense(1000, activation='relu', use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None)", "layers.Dense(1, activation='linear')"]]
        ['likelihood', None]
        ['load_on_RAM', False]
        ['log', {'datetime_2021_12_02_14_15_23_3239': {'action': 'created'}, 'datetime_2021_12_02_14_15_25_0959': {'action': 'optimizer set', 'optimizer': 'optimizers.Adam(learning_rate = 0.001, beta_1 = 0.9, beta_2 = 0.999, amsgrad = False)'}, 'datetime_2021_12_02_14_15_25_0969': {'action': 'loss set', 'loss': 'losses.mse'}, 'datetime_2021_12_02_14_15_27_9999': {'action': 'metrics set', 'metrics': ['metrics.mean_squared_error', "metrics.MeanAbsoluteError(dtype='float32')", "metrics.MeanSquaredLogarithmicError(dtype='float32')", 'metrics.mape', 'self.mean_error']}, 'datetime_2021_12_02_14_15_28_0059': {'action': 'saved object json', 'file name': 'DNNLikelihood_toy_summary.json'}}]
        ['loss', None]
        ['loss_string', 'losses.mse']
        ['metrics', []]
        ['metrics_string', ['metrics.mean_squared_error', "metrics.MeanAbsoluteError(dtype='float32')", "metrics.MeanSquaredLogarithmicError(dtype='float32')", 'metrics.mape', 'self.mean_error']]
        ['model', None]
        ['model_train_kwargs', {'shuffle': False, 'validation_freq': 1}]
        ['name', 'DNNLikelihood_toy']
        ['ndims', 11]
        ['npoints_available', 200000]
        ['npoints_test', 5000]
        ['npoints_test_available', 60000]
        ['npoints_train', 10000]
        ['npoints_train_val_available', 140000]
        ['npoints_val', 3000]
        ['optimizer', None]
        ['optimizer_string', 'optimizers.Adam(learning_rate = 0.001, beta_1 = 0.9, beta_2 = 0.999, amsgrad = False)']
        ['output_checkpoints_files', '<abs_path_to_output_folder>\\checkpoints\\DNNLikelihood_toy_checkpoint.{epoch:02d}-{val_loss:.2f}.h5']
        ['output_checkpoints_folder', '<abs_path_to_output_folder>\\checkpoints']
        ['output_figures_base_file_name', 'DNNLikelihood_toy_figure']
        ['output_figures_base_file_path', '<abs_path_to_output_folder>\\figures\\DNNLikelihood_toy_figure']
        ['output_figures_folder', '<abs_path_to_output_folder>\\figures']
        ['output_figure_plot_losses_keras_file', None]
        ['output_files_base_name', '<abs_path_to_output_folder>\\DNNLikelihood_toy']
        ['output_folder', '<abs_path_to_output_folder>']
        ['output_history_json_file', '<abs_path_to_output_folder>\\DNNLikelihood_toy_history.json']
        ['output_idx_h5_file', '<abs_path_to_output_folder>\\DNNLikelihood_toy_idx.h5']
        ['output_log_file', '<abs_path_to_output_folder>\\DNNLikelihood_toy.log']
        ['output_predictions_h5_file', '<abs_path_to_output_folder>\\DNNLikelihood_toy_predictions.h5']
        ['output_predictions_json_file', '<abs_path_to_output_folder>\\DNNLikelihood_toy_predictions.json']
        ['output_scalers_pickle_file', '<abs_path_to_output_folder>\\DNNLikelihood_toy_scalers.pickle']
        ['output_json_file', '<abs_path_to_output_folder>\\DNNLikelihood_toy_summary.json']
        ['output_tensorboard_log_dir', '<abs_path_to_output_folder>\\tensorboard_logs']
        ['output_tf_model_graph_pdf_file', '<abs_path_to_output_folder>\\DNNLikelihood_toy_model_graph.pdf']
        ['output_tf_model_h5_file', '<abs_path_to_output_folder>\\DNNLikelihood_toy_model.h5']
        ['output_tf_model_json_file', '<abs_path_to_output_folder>\\DNNLikelihood_toy_model.json']
        ['output_tf_model_onnx_file', '<abs_path_to_output_folder>\\DNNLikelihood_toy_model.onnx']
        ['pars_bounds', array([[ -5.,   5.],
               [-inf,  inf],
               [-inf,  inf],
               [-inf,  inf],
               [-inf,  inf],
               [-inf,  inf],
               [-inf,  inf],
               [-inf,  inf],
               [-inf,  inf],
               [-inf,  inf],
               [-inf,  inf]])]
        ['pars_central', array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])]
        ['pars_labels', ['$\\mu$', '$\\nu_{0}$', '$\\nu_{1}$', '$\\nu_{2}$', '$\\nu_{3}$', '$\\nu_{4}$', '$\\nu_{5}$', '$\\nu_{6}$', '$\\nu_{7}$', '$\\nu_{8}$', '$\\nu_{9}$']]
        ['pars_labels_auto', ['$\\theta_{1}$', '$\\nu_{1}$', '$\\nu_{2}$', '$\\nu_{3}$', '$\\nu_{4}$', '$\\nu_{5}$', '$\\nu_{6}$', '$\\nu_{7}$', '$\\nu_{8}$', '$\\nu_{9}$', '$\\nu_{10}$']]
        ['pars_pos_nuis', array([ 1,  2,  3,  4,  5,  6,  7,  8,  9, 10])]
        ['pars_pos_poi', array([0])]
        ['predictions', {'Model_evaluation': {}, 'Bayesian_inference': {}, 'Frequentist_inference': {}, 'Figures': {}}]
        ['same_data', True]
        ['scalerX', None]
        ['scalerX_bool', True]
        ['scalerY', None]
        ['scalerY_bool', True]
        ['script_file', '<abs_path_to_output_folder>\\DNNLikelihood_toy_script.py']
        ['seed', 1]
        ['standalone', True]
        ['verbose', 2]
        ['weighted', False]
        ['W_train', array([], dtype=float64)]
        ['X_test', array([], shape=(1, 0), dtype=float64)]
        ['X_train', array([], shape=(1, 0), dtype=float64)]
        ['X_val', array([], shape=(1, 0), dtype=float64)]
        ['Y_test', array([], dtype=float64)]
        ['Y_train', array([], dtype=float64)]
        ['Y_val', array([], dtype=float64)]
        ['_NF__model_callbacks_inputs', [{'name': 'EarlyStopping', 'args': [], 'kwargs': {'monitor': 'loss', 'mode': 'min', 'patience': 100, 'min_delta': 0.0001, 'restore_best_weights': True}}, 'TerminateOnNaN', 'PlotLossesKeras', {'name': 'TensorBoard', 'args': [], 'kwargs': {'log_dir': '<abs_path_to_output_folder>\\tensorboard_logs', 'histogram_freq': 0, 'write_graph': True, 'write_images': True, 'update_freq': 'epoch', 'profile_batch': 2, 'embeddings_freq': 0, 'embeddings_metadata': None}}, {'name': 'ReduceLROnPlateau', 'args': [], 'kwargs': {'monitor': 'loss', 'mode': 'min', 'factor': 0.2, 'min_lr': 8e-05, 'patience': 10, 'min_delta': 0.0001}}, {'name': 'ModelCheckpoint', 'args': [], 'kwargs': {'filepath': '<abs_path_to_output_folder>\\checkpoints\\DNNLikelihood_toy_checkpoint.{epoch:02d}-{val_loss:.2f}.h5', 'monitor': 'loss', 'mode': 'min', 'save_best_only': True, 'save_freq': 'epoch'}}]]
        ['_NF__model_compile_inputs', {'loss': 'mse', 'metrics': ['mean_squared_error', {'name': 'MeanAbsoluteError', 'args': [], 'kwargs': {'dtype': 'float32'}}, "MeanSquaredLogarithmicError(dtype='float32')", 'mape', 'me']}]
        ['_NF__model_data_inputs', {'npoints': [10000, 3000, 5000], 'scalerX': True, 'scalerY': True, 'weighted': False}]
        ['_NF__model_define_inputs', {'hidden_layers': ["Dense(1000,activation='relu',kernel_initializer='glorot_uniform',kernel_regularizer=regularizers.L1L2(l1=0.001, l2=0.001))", {'name': 'Dense', 'args': [1000], 'kwargs': {'activation': 'relu', 'use_bias': True, 'kernel_initializer': 'GlorotUniform(seed=None)', 'bias_initializer': 'zeros', 'kernel_regularizer': 'L1L2(l1=0.001, l2=0.001)', 'bias_regularizer': 'L1(l1=0.0001)', 'activity_regularizer': 'L1(l1=0.0001)', 'kernel_constraint': 'MaxNorm(max_value=2, axis=0)', 'bias_constraint': None}}, 'Dense(500)', {'name': 'Activation', 'args': ['relu']}, [100, 'selu'], [100, 'relu', 'glorot_uniform'], {'name': 'Dense', 'args': [1000], 'kwargs': {'use_bias': True, 'kernel_initializer': {'name': 'GlorotUniform', 'args': [], 'kwargs': {'seed': None}}, 'bias_initializer': 'zeros', 'kernel_regularizer': {'name': 'L1L2', 'args': [], 'kwargs': {'l1': 0.001, 'l2': 0.001}}, 'bias_regularizer': 'L1(l1=0.0001)', 'activity_regularizer': 'L1(l1=0.0001)', 'kernel_constraint': {'name': 'MaxNorm', 'args': [], 'kwargs': {'max_value': 2, 'axis': 0}}, 'bias_constraint': 'MaxNorm(max_value=2, axis=0)'}}, "Activation('selu')", {'name': 'BatchNormalization', 'args': [], 'kwargs': {'axis': -1, 'momentum': 0.99, 'epsilon': 0.001, 'center': True, 'scale': True, 'beta_initializer': 'zeros', 'gamma_initializer': 'ones', 'moving_mean_initializer': 'zeros', 'moving_variance_initializer': 'ones', 'beta_regularizer': None, 'gamma_regularizer': None, 'beta_constraint': None, 'gamma_constraint': None}}, {'name': 'Dropout', 'args': [0.1], 'kwargs': {'noise_shape': None, 'seed': None}}, 'BatchNormalization', 'AlphaDropout(0.1, seed=2)', {'name': 'AlphaDropout', 'args': [0.1], 'kwargs': {'noise_shape': None, 'seed': None}}, {'name': 'Dense', 'args': [1000], 'kwargs': {'activation': 'relu', 'use_bias': True, 'kernel_initializer': 'glorot_uniform', 'bias_initializer': 'zeros', 'kernel_regularizer': None, 'bias_regularizer': None, 'activity_regularizer': None, 'kernel_constraint': None, 'bias_constraint': None}}], 'act_func_out_layer': 'linear', 'dropout_rate': 0, 'batch_norm': False}]
        ['_NF__model_optimizer_inputs', 'optimizers.Adam(learning_rate = 0.001, beta_1 = 0.9, beta_2 = 0.999, amsgrad = False)']
        ['_NF__model_train_inputs', {'epochs': 50, 'batch_size': 512, 'shuffle': False, 'validation_freq': 1}]
        ['_NF__resources_inputs', None]

Notice that the :meth:`NF.__init__ <NF4HEP.NF.__init__>` method initializes
all attributes, leaving empty (with the correct type), all those arguments that will be populated during the object usage.
As it can be seen from the output of the object creation code, the object is saved upon creation. In case old files exist in the
specified path, they (the old ones) are renamed by appending a timestamp to them.

Once the :class:`NF <NF4HEP.NF>` object is created, the :meth:`NF.model_build <NF4HEP.NF.model_build>` 
method is used to create the |tf_keras_model_link| and compile it on the available resources as

.. code-block:: python

    dnnlikelihood.model_build(force=True,gpu="auto",verbose=2)

    >>> ============================== 
        Defining and compiling Keras model

        Building tf model for DNNLikelihood DNNLikelihood_toy on device ['/device:GPU:0', 'device: 0, name: NVIDIA GeForce RTX 2080 Ti, pci bus id: 0000:18:00.0, compute capability: 7.5'] .

        ============================== 
        Defining Keras model

        ============================== 
        DnnLik log file
        	 <abs_path_to_output_folder>\DNNLikelihood_toy.log 
        updated (or saved if it did not exist) in 0.001136699999733537 s.

        Model for DNNLikelihood DNNLikelihood_toy defined in 1.8786916000008205 s.

        Model: "functional_1"
        _________________________________________________________________
        Layer (type)                 Output Shape              Param #   
        =================================================================
        input_1 (InputLayer)         [(None, 11)]              0         
        _________________________________________________________________
        dense_28 (Dense)             (None, 1000)              12000     
        _________________________________________________________________
        dense_29 (Dense)             (None, 1000)              1001000   
        _________________________________________________________________
        dense_30 (Dense)             (None, 500)               500500    
        _________________________________________________________________
        activation_8 (Activation)    (None, 500)               0         
        _________________________________________________________________
        dense_31 (Dense)             (None, 100)               50100     
        _________________________________________________________________
        dense_32 (Dense)             (None, 100)               10100     
        _________________________________________________________________
        dense_33 (Dense)             (None, 1000)              101000    
        _________________________________________________________________
        activation_9 (Activation)    (None, 1000)              0         
        _________________________________________________________________
        batch_normalization_8 (Batch (None, 1000)              4000      
        _________________________________________________________________
        dropout_4 (Dropout)          (None, 1000)              0         
        _________________________________________________________________
        batch_normalization_9 (Batch (None, 1000)              4000      
        _________________________________________________________________
        alpha_dropout_8 (AlphaDropou (None, 1000)              0         
        _________________________________________________________________
        alpha_dropout_9 (AlphaDropou (None, 1000)              0         
        _________________________________________________________________
        dense_34 (Dense)             (None, 1000)              1001000   
        _________________________________________________________________
        dense_35 (Dense)             (None, 1)                 1001      
        =================================================================
        Total params: 2,684,701
        Trainable params: 2,680,701
        Non-trainable params: 4,000
        _________________________________________________________________
        ============================== 
        Compiling Keras model

        ============================== 
        DnnLik log file
        	 <abs_path_to_output_folder>\DNNLikelihood_toy.log 
        updated (or saved if it did not exist) in 0.0010114000006069546 s.

        Model for DNNLikelihood DNNLikelihood_toy compiled in 0.022523999999975786 s.

        ============================== 
        DnnLik log file
        	 <abs_path_to_output_folder>\DNNLikelihood_toy.log 
        updated (or saved if it did not exist) in 0.0020632999994631973 s.

This method uses the methods :meth:`NF.model_define <NF4HEP.NF.model_define>` and 
:meth:`NF.model_compile <NF4HEP.NF.model_compile>` and implements the |tf_distribute_onedevicestrategy_link| strategy to
place the compiled model on a specified (e.g. ``gpu=1``) or unspecified (i.e. ``gpu="auto"``) device. The ``force`` argument determines
if the model is re-generated and/or re-compiled even if already available, while the ``verbose=2`` ensures that all ``print`` commands
from each of the methods called within the method are active. The method sets, among others, the
:attr:`NF.model <NF4HEP.NF.model>` attribute, which is the principal object containing the |tf_keras_model_link|.

The DNNLikelihood is now ready for training with the :meth:`NF.model_train <NF4HEP.NF.model_train>` method:

.. code-block:: python

    dnnlikelihood.model_train(verbose=2)

    >>> <image>

                Loss
        	training         	 (min:    0.228, max:   33.063, cur:    0.228)
        	validation       	 (min:    0.836, max:   32.746, cur:    0.857)
        mean_absolute_error
        	training         	 (min:    0.195, max:    1.033, cur:    0.195)
        	validation       	 (min:    0.585, max:    1.996, cur:    0.800)
        mean_absolute_percentage_error
        	training         	 (min:  491.900, max: 7270.143, cur: 1459.488)
        	validation       	 (min:  131.199, max: 1814.549, cur:  696.519)
        mean_error
        	training         	 (min:    0.036, max:    0.163, cur:    0.036)
        	validation       	 (min:    0.050, max:    1.997, cur:    0.798)
        mean_squared_error
        	training         	 (min:    0.065, max:    1.955, cur:    0.065)
        	validation       	 (min:    0.576, max:    4.190, cur:    0.716)
        mean_squared_logarithmic_error
        	training         	 (min:    0.010, max:    0.194, cur:    0.010)
        	validation       	 (min:    0.049, max:    0.663, cur:    0.132)
        20/20 - 4s - loss: 0.2276 - mean_squared_error: 0.0650 - mean_absolute_error: 0.1951 - mean_squared_logarithmic_error: 0.0100 - mean_absolute_percentage_error: 1459.4880 - mean_error: 0.0361 - val_loss: 0.8567 - val_mean_squared_error: 0.7160 - val_mean_absolute_error: 0.7998 - val_mean_squared_logarithmic_error: 0.1316 - val_mean_absolute_percentage_error: 696.5186 - val_mean_error: 0.7978
        No existing history. Setting new history.

        Updating model.history and model.epoch attribute.

        ============================== 
        DnnLik log file
        	 <abs_path_to_output_folder>\DNNLikelihood_toy.log 
        updated (or saved if it did not exist) in 0.0015487000000007356 s.

        Model for DNNLikelihood DNNLikelihood_toy successfully trained for 50 epochs in 309.6447003 s ( 6.192894006 s/epoch).

where <image> shows the training in real time whose final result is:

.. image:: ../figs/plotlosseskeras.png
    :class: with-shadow
    :scale: 60

In this example we have used the ``PlotLossesKeras`` callback from |livelossplot_link| to plot the metrics in real time 
during training. Notice that this may be useful for quick checks, but it slows down training and is not optimal for long trainings.
For more robust logging the use of the |tf_keras_tensorboard_callback_link| callback is suggested.
Since no data were available yet when the method was called, training and validation data generation was done automatically. 
The user may want to generate data in advance, which can be done as follows:

.. code-block:: python

    dnnlikelihood.generate_train_data(verbose=2)
    dnnlikelihood.generate_test_data(verbose=2)

    >>> ============================== 
        Generating train/validation data

        ============================== 
        Data log file
        	 <abs_path_to_data_output_folder>\toy_data.log 
        updated (or saved if it did not exist) in 0.0032989999999983866 s.

        ============================== 
        Added 10000 (X_train, Y_train) samples and 3000 (X_val, Y_val) samples in 16.587983899999998 s.

        ============================== 
        Setting standard scalers

        ============================== 
        Data log file
        	 <abs_path_to_data_output_folder>\toy_data.log 
        updated (or saved if it did not exist) in 0.0027511999999987324 s.

        ============================== 
        Standard scalers defined in 0.0039535000000014975 s.

        ============================== 
        DnnLik log file
        	 <abs_path_to_output_folder>\DNNLikelihood_toy.log 
        updated (or saved if it did not exist) in 0.0020138000000002876 s.

        ============================== 
        Generating test data

        ============================== 
        Data log file
        	 <abs_path_to_data_output_folder>\toy_data.log 
        updated (or saved if it did not exist) in 0.0026599999999987745 s.

        ============================== 
        Added 5000 (X_test, Y_test) samples in 0.08260440000000102 s.

        ============================== 
        DnnLik log file
        	 <abs_path_to_output_folder>\DNNLikelihood_toy.log 
        updated (or saved if it did not exist) in 0.0012887000000034732 s.

Data generation is taken care by the :attr:`NF.data <NF4HEP.NF.data>` attribute, which is a 
:class:`Data <NF4HEP.Data>` object (see :mod:`the Data object <data>` documentation for more details on data management).

The :class:`NF <NF4HEP.NF>` object is saved automatically after performing operations with its methods. 
However, the user can also save it at any time with the commands:

.. code-block:: python

    dnnlikelihood.save(overwrite=True,verbose=2)

    >>> ============================== 
        Idx h5 file
        	 <abs_path_to_output_folder>\DNNLikelihood_toy_idx.h5 
        updated (or saved if it did not exist) in 0.004034199999978227 s.

        ============================== 
        Model json file
        	 <abs_path_to_output_folder>\DNNLikelihood_toy_model.json 
        updated (or saved if it did not exist) in 0.004750500000000102 s.

        tf executing eager_mode: True
        tf.keras model eager_mode: False
        ============================== 
        Mode h5 file
        	 <abs_path_to_output_folder>\DNNLikelihood_toy_model.h5 
        updated (or saved if it did not exist) in 1.9676557999999886 s.

        The ONNX operator number change on the optimization: 35 -> 25
        The maximum opset needed by this model is only 9.
        ============================== 
        Mode onnx file
        	 <abs_path_to_output_folder>\DNNLikelihood_toy_model.onnx 
        updated (or saved if it did not exist) in 9.157906899999944 s.

        ============================== 
        Mode history file
        	 <abs_path_to_output_folder>\DNNLikelihood_toy_history.json 
        updated (or saved if it did not exist) in 0.004860199999939141 s.

        ============================== 
        Predictions json file
        	 <abs_path_to_output_folder>\DNNLikelihood_toy_predictions.json 
        updated (or saved if it did not exist) in 0.0007312000000183616 s.

        ============================== 
        Predictions h5 file
        	 <abs_path_to_output_folder>\DNNLikelihood_toy_predictions.h5 
        updated (or saved if it did not exist) in 0.004424099999937425 s.

        ============================== 
        \nJson file\n\t
        	 <abs_path_to_output_folder>\DNNLikelihood_toy_summary.json 
        updated (or saved if it did not exist) in 0.004021100000045408 s.

        ============================== 
        Scalers pickle file
        	 <abs_path_to_output_folder>\DNNLikelihood_toy_scalers.pickle 
        updated (or saved if it did not exist) in 0.0013580999999476262 s.

Now that the DNNLikelihood is built and trained, one can proceed to evaluate its performances. There are several methods to compute
metrics and make plots, whose extensive guide is available in the :ref:`methods documentation <DnnLik_methods>`. The framework also
includes the :meth:`NF.model_compute_predictions <NF4HEP.NF.model_compute_predictions>` method,
devoted to a comprehensive evaluation of the model performances. An example of its usage is the following:

.. code-block:: python

    dnnlikelihood.model_compute_predictions(CI=NF4HEP.inference.CI_from_sigma([NF4HEP.inference.sigma_from_CI(0.5), 1, 2, 3]),
                                            pars=[0,2,4,5,8],
                                            batch_size=None,
                                            model_predictions={"Model_evaluation": True, 
                                                               "Bayesian_inference": True, 
                                                               "Frequentist_inference": True},
                                            plots={"plot_training_history": True,
                                                   "plot_pars_coverage": True,
                                                   "plot_lik_distribution": True,
                                                   "plot_corners_1samp": True,
                                                   "plot_corners_2samp": True,
                                                   "plot_tmu_sources_1d": True},
                                            model_predict_kwargs = {"x_boundaries": "original",
                                                                    "y_boundaries": False,
                                                                    "save_log": False},
                                            HPDI_kwargs = {"nbins": 25, 
                                                           "print_hist": False, 
                                                           "optimize_binning": True},
                                            frequentist_inference_options={"input_likelihood_file": None,
                                                                           "compute_maximum_likelihood_kwargs": {"pars_init": np.full(11,1).tolist(),
                                                                                                                 "verbose": True},
                                                                           "compute_profiled_maxima_likelihood_kwargs": {"pars": [0],
                                                                                                                         "pars_ranges": [[0,1,10]],
                                                                                                                         "pars_init": None,
                                                                                                                         "spacing": "grid",
                                                                                                                         "optimizer": {"method": "Powell",
                                                                                                                                       "options": {"maxiter": 10000,
                                                                                                                                                   "ftol": 0.0001}},
                                                                                                                         "progressbar": True,
                                                                                                                         "verbose": True},
                                                                           "compute_maximum_sample_kwargs": {"samples": ["train","val","test"],
                                                                                                             "save": False,
                                                                                                             "verbose": True},
                                                                           "compute_profiled_maxima_sample_kwargs": {"pars": [0],
                                                                                                                     "pars_ranges": [[0,1,10]],
                                                                                                                     "samples": ["train","val","test"],
                                                                                                                     "spacing": "grid",
                                                                                                                     "binwidths": "auto",
                                                                                                                     "x_boundaries": False,
                                                                                                                     "progressbar": True,
                                                                                                                     "save": False,
                                                                                                                     "verbose": True},
                                                                           "compute_maximum_model_kwargs": {"pars_init": None,
                                                                                                            "optimizer": {"name": "SGD",
                                                                                                                          "learning_rate": 1,
                                                                                                                          "options": {"maxiter": 10000,
                                                                                                                                     "run_length": 100,
                                                                                                                                     "tolerance": 0.0001}},
                                                                                                                          #"beta_1": 0.9,
                                                                                                                          #"beta_2": 0.999,
                                                                                                                          #"amsgrad": False},
                                                                                                            "x_boundaries": False, 
                                                                                                            "y_boundaries": False,
                                                                                                            "verbose": True},
                                                                           "compute_profiled_maxima_model_kwargs": {"pars": [0],
                                                                                                                    "pars_ranges": [[0,1,10]],
                                                                                                                    "pars_init": None,
                                                                                                                    "optimizer": {"name": "SGD",
                                                                                                                                  "learning_rate": 1,
                                                                                                                                  "options": {"maxiter": 10000,
                                                                                                                                             "run_length": 100,
                                                                                                                                             "tolerance": 0.0001}},
                                                                                                                    #"optimizer": {"name": "scipy",
                                                                                                                    #              "method": "Powell",
                                                                                                                    #              "options": {"maxiter": 10000,
                                                                                                                    #                          "ftol": 0.000001}},
                                                                                                                    "spacing": "grid",
                                                                                                                    "x_boundaries": False,
                                                                                                                    "y_boundaries": False,
                                                                                                                    "progressbar": True,
                                                                                                                    "save": False,
                                                                                                                    "verbose": True},
                                                                          },
                                            plot_training_history_kwargs = {"metrics": ["loss", "me"], 
                                                                            "yscale": "linear", 
                                                                            "show_plot": True,
                                                                            "overwrite": True,
                                                                            "verbose": True},
                                            plot_pars_coverage_kwargs = {"loglik": False, 
                                                                         "show_plot": True,
                                                                         "overwrite": True,
                                                                         "verbose": True},
                                            plot_lik_distribution_kwargs = {"loglik": False, 
                                                                            "show_plot": True,
                                                                            "overwrite": True,
                                                                            "verbose": True},
                                            plot_corners_1samp_kwargs={"ranges_extend": 1.3, 
                                                                       "max_points": 100000, 
                                                                       "nbins": 60, 
                                                                       "legend_loc": "upper right",
                                                                       "show_plot": True,
                                                                       "overwrite": True,
                                                                       "verbose": True},
                                            plot_corners_2samp_kwargs = {"ranges_extend": 1.3, 
                                                                         "max_points": 100000, 
                                                                         "nbins": 60, 
                                                                         "legend_loc": "upper right",
                                                                         "show_plot": True,
                                                                         "overwrite": True,
                                                                         "verbose": True},
                                            plot_tmu_sources_1d_kwargs = {"sources": ["likelihood","model","train","val","test"],
                                                                          "show_plot": True,
                                                                          "overwrite": True,
                                                                          "verbose": True},
                                            show_plots=True,
                                            overwrite=False,
                                            verbose=2)

    >>> ============================== 
        Computing predictions for DNNLikelihood model

        ============================== 
        Performing Model evaluation

        ============================== 
        Generating train/validation data

        ============================== 
        Loaded train data corresponding to existing indices.

        ============================== 
        Loaded val data corresponding to existing indices.

        ============================== 
        Loaded test data corresponding to existing indices.

        ============================== 
        Data log file
        	 C:\Users\Admin\Dropbox\Work\09_Resources\Git\GitHub\DNNLikelihood\DNNLikelihood_dev_tutorials\tutorials\toy\toy\data\toy_data.log 
        updated (or saved if it did not exist) in 0.0036706000000208405 s.

        ============================== 
        Added 0 (X_train, Y_train) samples and 0 (X_val, Y_val) samples in 18.42952059999999 s.

        ============================== 
        Setting standard scalers

        ============================== 
        Data log file
        	 C:\Users\Admin\Dropbox\Work\09_Resources\Git\GitHub\DNNLikelihood\DNNLikelihood_dev_tutorials\tutorials\toy\toy\data\toy_data.log 
        updated (or saved if it did not exist) in 0.003842600000012908 s.

        ============================== 
        Standard scalers defined in 0.004359499999992522 s.

        ============================== 
        DnnLik log file
        	 <abs_path_to_output_folder>\DNNLikelihood_toy.log 
        updated (or saved if it did not exist) in 0.008558500000020786 s.

        ============================== 
        Generating test data

        ============================== 
        Data log file
        	 C:\Users\Admin\Dropbox\Work\09_Resources\Git\GitHub\DNNLikelihood\DNNLikelihood_dev_tutorials\tutorials\toy\toy\data\toy_data.log 
        updated (or saved if it did not exist) in 0.004392400000000407 s.

        ============================== 
        Added 0 (X_test, Y_test) samples in 0.009593399999999974 s.

        ============================== 
        DnnLik log file
        	 <abs_path_to_output_folder>\DNNLikelihood_toy.log 
        updated (or saved if it did not exist) in 0.008789100000001326 s.

        ============================== 
        Idx h5 file
        	 <abs_path_to_output_folder>\DNNLikelihood_toy_idx.h5 
        updated (or saved if it did not exist) in 0.004162400000012667 s.

        The key 'Metrics_on_scaled_data' was not specified and has been set to the default value '{}'.
        The key 'Prediction_time' was not specified and has been set to the default value '{}'.
        The key 'Metrics_on_unscaled_data' was not specified and has been set to the default value '{}'.
        ============================== 
        Evaluating all metrics on (scaled) train/val/test using best models

        ============================== 
        Evaluating model

        Scaling data.

        20/20 - 0s - loss: 0.7256 - mean_squared_error: 0.5790 - mean_absolute_error: 0.6838 - mean_squared_logarithmic_error: 0.1195 - mean_absolute_percentage_error: 5443.5829 - mean_error: 0.6709
        ============================== 
        DnnLik log file
        	 <abs_path_to_output_folder>\DNNLikelihood_toy.log 
        updated (or saved if it did not exist) in 0.009148500000009108 s.

        ============================== 
        Evaluating model

        Scaling data.

        6/6 - 0s - loss: 0.7604 - mean_squared_error: 0.6138 - mean_absolute_error: 0.6933 - mean_squared_logarithmic_error: 0.1171 - mean_absolute_percentage_error: 765.2350 - mean_error: 0.6736
        ============================== 
        DnnLik log file
        	 <abs_path_to_output_folder>\DNNLikelihood_toy.log 
        updated (or saved if it did not exist) in 0.00920990000000188 s.

        ============================== 
        Evaluating model

        Scaling data.

        10/10 - 0s - loss: 0.7351 - mean_squared_error: 0.5885 - mean_absolute_error: 0.6769 - mean_squared_logarithmic_error: 0.1100 - mean_absolute_percentage_error: 494.2122 - mean_error: 0.6477
        ============================== 
        DnnLik log file
        	 <abs_path_to_output_folder>\DNNLikelihood_toy.log 
        updated (or saved if it did not exist) in 0.009924600000005057 s.

        ============================== 
        Predicting Y for train/val/test samples

        ============================== 
        Predicting with Keras model

        Scaling data.

        WARNING:tensorflow:Callbacks method `on_predict_batch_end` is slow compared to the batch time (batch time: 0.0010s vs `on_predict_batch_end` time: 0.0080s). Check your callbacks.
        20/20 - 0s
        ============================== 
        Predicting with Keras model

        Scaling data.

        WARNING:tensorflow:Callbacks method `on_predict_batch_end` is slow compared to the batch time (batch time: 0.0010s vs `on_predict_batch_end` time: 0.0065s). Check your callbacks.
        6/6 - 0s
        ============================== 
        Predicting with Keras model

        Scaling data.

        WARNING:tensorflow:Callbacks method `on_predict_batch_end` is slow compared to the batch time (batch time: 0.0010s vs `on_predict_batch_end` time: 0.0060s). Check your callbacks.
        10/10 - 0s
        ============================== 
        Evaluating all metrics on (un-scaled) trai[timestamp]n/val/test using best models

        ============================== 
        Prediction on (10000,3000,5000) (train,val,test) points done in 2.6130508999999904 s.

        ============================== 
        Computing Bayesian inference benchmarks

        ============================== 
        Computing exp(Y_true) and exp(Y_pred) for train/val/test samples

        ============================== 
        Computing weights (pred vs true) for reweighting of distributions

        ============================== 
        Computing HPDI (pred vs true) using reweighted distributions

        For some probability values there are different numbers of intervals. In this case error is not computed and is set to None.
        For some probability values there are different numbers of intervals. In this case error is not computed and is set to None.
        For some probability values there are different numbers of intervals. In this case error is not computed and is set to None.
        For some probability values there are different numbers of intervals. In this case error is not computed and is set to None.
        For some probability values there are different numbers of intervals. In this case error is not computed and is set to None.
        The key 'HPDI' was not specified and has been set to the default value '{}'.
        The key 'HPDI_error' was not specified and has been set to the default value '{}'.
        The key 'KS' was not specified and has been set to the default value '{}'.
        The key 'KS_medians' was not specified and has been set to the default value '{}'.
        ============================== 
        Computing KS test between one-dimensional distributions (pred vs true) using reweighted distributions

        ============================== 
        Bayesian inference benchmarks computed in 0.4348025999999834 s.

        ============================== 
        Computing Frequentist inference benchmarks based on 'frequentist_inference_options'

        ============================== 
        A reference Likelihood object is already available. frequentist_inference_options["input_likelihood_file"] input will be ignored.

        ============================== 
        Computing Frequentist inference benchmarks for reference Likelihood

        ============================== 
        Computing global maximum for reference Likelihood

        The key 'optimizer' was not specified and has been set to the default value '{}'.
        The key 'minimization_options' was not specified and has been set to the default value '{}'.
        The key 'timestamp' was not specified and has been set to the default value 'datetime_2021_12_06_16_36_49_9820'.
        The key 'save' was not specified and has been set to the default value 'False'.
        ============================== 
        Computing global maximum
        The key 'name' was not specified and has been set to the default value 'scipy'.
        The key 'args' was not specified and has been set to the default value '[]'.
        The key 'kwargs' was not specified and has been set to the default value '{}'.
        The key 'method' was not specified and has been set to the default value 'Powell'.
        Maximum logpdf computed in 0.35594520000000784 s.

        ============================== 
        Likelihood log file
        	 C:\Users\Admin\Dropbox\Work\09_Resources\Git\GitHub\DNNLikelihood\DNNLikelihood_dev_tutorials\tutorials\toy\toy\likelihood\toy_likelihood.log 
        updated (or saved if it did not exist) in 0.0036088000000233933 s.

        The key 'logpdf_max_likelihood' was not specified and has been set to the default value '{}'.
        The key 'datetime_2021_12_06_16_36_49_9820' was not specified and has been set to the default value '{}'.
        ============================== 
        Computing profiled maxima for reference Likelihood

        The key 'pars_bounds' was not specified and has been set to the default value 'None'.
        The key 'timestamp' was not specified and has been set to the default value 'datetime_2021_12_06_16_36_49_9820'.
        The key 'save' was not specified and has been set to the default value 'False'.
        ============================== 
        Computing profiled maxima
        The key 'datetime_2021_12_06_16_36_49_9820' was not specified and has been set to the default value '{}'.
        The key 'args' was not specified and has been set to the default value '[]'.
        The key 'kwargs' was not specified and has been set to the default value '{}'.
        Total number of points: 10 .
        Optimizing for parameters: [0]  - values: [0.0] .
        Optimizing for parameters: [0]  - values: [0.1111111111111111] .
        Optimizing for parameters: [0]  - values: [0.2222222222222222] .
        Optimizing for parameters: [0]  - values: [0.3333333333333333] .
        Optimizing for parameters: [0]  - values: [0.4444444444444444] .
        Optimizing for parameters: [0]  - values: [0.5555555555555556] .
        Optimizing for parameters: [0]  - values: [0.6666666666666666] .
        Optimizing for parameters: [0]  - values: [0.7777777777777777] .
        Optimizing for parameters: [0]  - values: [0.8888888888888888] .
        Optimizing for parameters: [0]  - values: [1.0] .
        Computing global maximum to estimate tmu test statistics.
        Maximum logpdf computed in 0.21160299999999665 s.

        ============================== 
        Likelihood log file
        	 C:\Users\Admin\Dropbox\Work\09_Resources\Git\GitHub\DNNLikelihood\DNNLikelihood_dev_tutorials\tutorials\toy\toy\likelihood\toy_likelihood.log 
        updated (or saved if it did not exist) in 0.0029160999999930937 s.

        ============================== 
        10 local maxima computed in 1.1987985000000094 s.
        Log-pdf values lie in the range [ -57.07828497740044 , -48.11154184674124 ].

        The key 'logpdf_profiled_max_likelihood' was not specified and has been set to the default value '{}'.
        The key 'datetime_2021_12_06_16_36_49_9820' was not specified and has been set to the default value '{}'.
        ============================== 
        Computing Frequentist inference benchmarks from samples

        ============================== 
        Computing global maximum from samples

        The key 'timestamp' was not specified and has been set to the default value 'datetime_2021_12_06_16_36_49_9820'.
        ============================== 
        Computing global maximum from samples

        The key 'logpdf_max_sample' was not specified and has been set to the default value '{}'.
        The key 'datetime_2021_12_06_16_36_49_9820' was not specified and has been set to the default value '{}'.
        Maximum of train sample computed in 0.0027992999999923995 s.

        Maximum of val sample computed in 0.0007636999999931504 s.

        Maximum of test sample computed in 0.0007075999999983651 s.

        ============================== 
        Computing profiled maxima from samples

        The key 'timestamp' was not specified and has been set to the default value 'datetime_2021_12_06_16_36_49_9820'.
        ============================== 
        Computing profiled maxima from samples

        The key 'logpdf_profiled_max_sample' was not specified and has been set to the default value '{}'.
        The key 'datetime_2021_12_06_16_36_49_9820' was not specified and has been set to the default value '{}'.
        The key 'train' was not specified and has been set to the default value '{}'.
        Total number of points: 10 .

        Optimizing for parameters: [0]  - values: [0.0] .
        Optimizing for parameters: [0]  - values: [0.1111111111111111] .
        Optimizing for parameters: [0]  - values: [0.2222222222222222] .
        Optimizing for parameters: [0]  - values: [0.3333333333333333] .
        Optimizing for parameters: [0]  - values: [0.4444444444444444] .
        Optimizing for parameters: [0]  - values: [0.5555555555555556] .
        Optimizing for parameters: [0]  - values: [0.6666666666666666] .
        Optimizing for parameters: [0]  - values: [0.7777777777777777] .
        Optimizing for parameters: [0]  - values: [0.8888888888888888] .
        Optimizing for parameters: [0]  - values: [1.0] .
        ============================== 
        Profiled maxima of train sample for 10 points computed in 20.895175999999992 s.

        The key 'val' was not specified and has been set to the default value '{}'.
        Total number of points: 10 .

        Optimizing for parameters: [0]  - values: [0.0] .
        Optimizing for parameters: [0]  - values: [0.1111111111111111] .
        Optimizing for parameters: [0]  - values: [0.2222222222222222] .
        Optimizing for parameters: [0]  - values: [0.3333333333333333] .
        Optimizing for parameters: [0]  - values: [0.4444444444444444] .
        Optimizing for parameters: [0]  - values: [0.5555555555555556] .
        Optimizing for parameters: [0]  - values: [0.6666666666666666] .
        Optimizing for parameters: [0]  - values: [0.7777777777777777] .
        Optimizing for parameters: [0]  - values: [0.8888888888888888] .
        Optimizing for parameters: [0]  - values: [1.0] .
        ============================== 
        Profiled maxima of val sample for 10 points computed in 9.414752099999987 s.

        The key 'test' was not specified and has been set to the default value '{}'.
        Total number of points: 10 .

        Optimizing for parameters: [0]  - values: [0.0] .
        Optimizing for parameters: [0]  - values: [0.1111111111111111] .
        Optimizing for parameters: [0]  - values: [0.2222222222222222] .
        Optimizing for parameters: [0]  - values: [0.3333333333333333] .
        Optimizing for parameters: [0]  - values: [0.4444444444444444] .
        Optimizing for parameters: [0]  - values: [0.5555555555555556] .
        Optimizing for parameters: [0]  - values: [0.6666666666666666] .
        Optimizing for parameters: [0]  - values: [0.7777777777777777] .
        Optimizing for parameters: [0]  - values: [0.8888888888888888] .
        Optimizing for parameters: [0]  - values: [1.0] .
        ============================== 
        Profiled maxima of test sample for 10 points computed in 11.582379799999984 s.

        ============================== 
        Computing Frequentist inference benchmarks from model

        ============================== 
        Computing global maximum from model

        The key 'timestamp' was not specified and has been set to the default value 'datetime_2021_12_06_16_36_49_9820'.
        The key 'save' was not specified and has been set to the default value 'False'.
        ============================== 
        Computing global maximum of Keras model

        The key 'logpdf_max_model' was not specified and has been set to the default value '{}'.
        ============================== 
        Optimizing with tensorflow

        Step: 0 Value: -52.750979833311696 -- Step: 100 Value: -49.23497556652939 -- % Variation 0.06665287124319885
        Step: 100 Value: -49.23497556652939 -- Step: 200 Value: -47.80273714378706 -- % Variation 0.02908985748976354
        Step: 200 Value: -47.80273714378706 -- Step: 300 Value: -49.63739609127204 -- % Variation -0.038379788629392876
        Optimizer learning rate reduced from 1 to 0.5 .

        Step: 300 Value: -49.63739609127204 -- Step: 400 Value: -47.11896891766499 -- % Variation 0.050736488452702504
        Step: 400 Value: -47.11896891766499 -- Step: 500 Value: -47.3104310603121 -- % Variation -0.004063377171551919
        Optimizer learning rate reduced from 0.5 to 0.25 .

        Step: 500 Value: -47.3104310603121 -- Step: 600 Value: -46.99135095258422 -- % Variation 0.006744392316381758
        Step: 600 Value: -46.99135095258422 -- Step: 700 Value: -46.788951647372606 -- % Variation 0.0043071608095677305
        Step: 700 Value: -46.788951647372606 -- Step: 800 Value: -46.802268199895074 -- % Variation -0.0002846089098731946
        Optimizer learning rate reduced from 0.25 to 0.125 .

        Step: 800 Value: -46.802268199895074 -- Step: 900 Value: -46.84494452263145 -- % Variation -0.0009118430447452572
        Optimizer learning rate reduced from 0.125 to 0.0625 .

        Step: 900 Value: -46.84494452263145 -- Step: 1000 Value: -46.6758648175949 -- % Variation 0.0036093479618674114
        Step: 1000 Value: -46.6758648175949 -- Step: 1100 Value: -46.67349168996511 -- % Variation 5.0842713660732376e-05
        Converged to tolerance 0.0001 in 3.7045769000000064 s.

        Best tolerance 5.0842713660732376e-05 reached in 3.7076693000000205 s.

        ============================== 
        Maximum of DNN computed in 3.7076693000000205 s.

        ============================== 
        Computing profiled maxima from model

        The key 'timestamp' was not specified and has been set to the default value 'datetime_2021_12_06_16_36_49_9820'.
        ============================== 
        Computing profiled maxima of Keras model

        Total number of points: 10 .

        The key 'logpdf_profiled_max_model' was not specified and has been set to the default value '{}'.
        The key 'datetime_2021_12_06_16_36_49_9820' was not specified and has been set to the default value '{}'.
        Optimizing for parameters: [0]  - values: [0.0] .
        Optimizing for parameters: [0]  - values: [0.1111111111111111] .
        Optimizing for parameters: [0]  - values: [0.2222222222222222] .
        Optimizing for parameters: [0]  - values: [0.3333333333333333] .
        Optimizing for parameters: [0]  - values: [0.4444444444444444] .
        Optimizing for parameters: [0]  - values: [0.5555555555555556] .
        Optimizing for parameters: [0]  - values: [0.6666666666666666] .
        Optimizing for parameters: [0]  - values: [0.7777777777777777] .
        Optimizing for parameters: [0]  - values: [0.8888888888888888] .
        Optimizing for parameters: [0]  - values: [1.0] .
        
        ============================== 
        Profiled maxima of DNN for 10 points computed in 614.4056245 s.

        ============================== 
        Frequentist inference benchmarks computed in 661.5934887999999 s.

        ============================== 
        The file
        	 <abs_path_to_output_folder>\DNNLikelihood_toy_predictions.json 
        already exists and has been moved to
        	 <abs_path_to_output_folder>\old_2021_12_06_16_48_13_1074_DNNLikelihood_toy_predictions.json 

        ============================== 
        Predictions json file
        	 <abs_path_to_output_folder>\DNNLikelihood_toy_predictions.json 
        saved in 0.0268781999999419 s.

        ============================== 
        The file
        	 <abs_path_to_output_folder>\DNNLikelihood_toy_predictions.h5 
        already exists and has been moved to
        	 <abs_path_to_output_folder>\old_2021_12_06_16_48_13_1354_DNNLikelihood_toy_predictions.h5 

        ============================== 
        Predictions h5 file
        	 <abs_path_to_output_folder>\DNNLikelihood_toy_predictions.h5 
        saved in 0.0869822000000795 s.

        ============================== 
        Making plots for model evaluation

        ============================== 
        Plotting training history

        Checking and updating figures dictionary

        <image1>

        ==============================
        Figure file
        	 DNNLikelihood_toy_figure_training_history_loss.pdf 
        created and saved in 11.569495599999982 s.

        Checking and updating figures dictionary

        <image2>

        ==============================
        Figure file
        	 DNNLikelihood_toy_figure_training_history_mean_error.pdf 
        created and saved in 2.2338313999999855 s.

        ============================== 
        Plotting parameters coverage

        ============================== 
        Predicting with Keras model

        Scaling data.

        2/2 - 0s
        ============================== 
        DnnLik log file
        	 <abs_path_to_output_folder>\DNNLikelihood_toy.log 
        updated (or saved if it did not exist) in 0.010870199999999386 s.

        Checking and updating figures dictionary

        <image3>

        ==============================
        Figure file
        	 DNNLikelihood_toy_figure_par_lik_coverage_0.pdf 
        created and saved in 3.837169200000062 s.

        ============================== 
        Predicting with Keras model

        Scaling data.

        2/2 - 0s
        ============================== 
        DnnLik log file
        	 <abs_path_to_output_folder>\DNNLikelihood_toy.log 
        updated (or saved if it did not exist) in 0.011883099999977276 s.

        Checking and updating figures dictionary

        <image4>

        ==============================
        Figure file
        	 DNNLikelihood_toy_figure_par_lik_coverage_2.pdf 
        created and saved in 0.8404969000000619 s.

        ============================== 
        Predicting with Keras model

        Scaling data.

        2/2 - 0s
        ============================== 
        DnnLik log file
        	 <abs_path_to_output_folder>\DNNLikelihood_toy.log 
        updated (or saved if it did not exist) in 0.00968399999999292 s.

        Checking and updating figures dictionary

        <image5>

        ==============================
        Figure file
        	 DNNLikelihood_toy_figure_par_lik_coverage_4.pdf 
        created and saved in 0.8441845000000967 s.

        ============================== 
        Predicting with Keras model

        Scaling data.

        2/2 - 0s
        ============================== 
        DnnLik log file
        	 <abs_path_to_output_folder>\DNNLikelihood_toy.log 
        updated (or saved if it did not exist) in 0.00970319999998992 s.

        Checking and updating figures dictionary

        <image6>

        ==============================
        Figure file
        	 DNNLikelihood_toy_figure_par_lik_coverage_5.pdf 
        created and saved in 0.7553292000000056 s.

        ============================== 
        Predicting with Keras model

        Scaling data.

        2/2 - 0s
        ============================== 
        DnnLik log file
        	 <abs_path_to_output_folder>\DNNLikelihood_toy.log 
        updated (or saved if it did not exist) in 0.009699800000021241 s.

        Checking and updating figures dictionary

        <image7>

        ==============================
        Figure file
        	 DNNLikelihood_toy_figure_par_lik_coverage_8.pdf 
        created and saved in 0.7887908000000152 s.

        ============================== 
        Plotting likelihood distribution

        Checking and updating figures dictionary
        ============================== 
        Predicting with Keras model

        Scaling data.

        10/10 - 0s
        ============================== 
        DnnLik log file
        	 <abs_path_to_output_folder>\DNNLikelihood_toy.log 
        updated (or saved if it did not exist) in 0.010957399999938389 s.

        Checking and updating figures dictionary

        <image8>

        ==============================
        Figure file
        	 DNNLikelihood_toy_figure_lik_distribution.pdf 
        created and saved in 0.8909048000000439 s.

        ============================== 
        Making corner plots for Bayesian inference

        ============================== 
        Plotting 2d posterior distributions for single sample

        Checking and updating figures dictionary

        <image9>

        ==============================
        Figure file
        	 DNNLikelihood_toy_figure_corner_pars_train_true.pdf 
        created and saved in 7.74165359999995 s.

        ============================== 
        Plotting 2d posterior distributions for single sample

        Checking and updating figures dictionary

        <image10>

        ==============================
        Figure file
        	 DNNLikelihood_toy_figure_corner_pars_train_pred.pdf 
        created and saved in 3.298143200000027 s.

        ============================== 
        Plotting 2d posterior distributions for single sample

        Checking and updating figures dictionary

        <image11>

        ==============================
        Figure file
        	 DNNLikelihood_toy_figure_corner_pars_val_true.pdf 
        created and saved in 3.204072300000007 s.

        ============================== 
        Plotting 2d posterior distributions for single sample

        Checking and updating figures dictionary

        <image12>

        ==============================
        Figure file
        	 DNNLikelihood_toy_figure_corner_pars_val_pred.pdf 
        created and saved in 3.083647199999973 s.

        ============================== 
        Plotting 2d posterior distributions for single sample

        Checking and updating figures dictionary

        <image13>

        ==============================
        Figure file
        	 DNNLikelihood_toy_figure_corner_pars_test_true.pdf 
        created and saved in 3.5541520000000446 s.

        ============================== 
        Plotting 2d posterior distributions for single sample

        Checking and updating figures dictionary

        <image14>

        ==============================
        Figure file
        	 DNNLikelihood_toy_figure_corner_pars_test_pred.pdf 
        created and saved in 3.499598399999968 s.

        ============================== 
        Plotting 2d posterior distributions for two samples comparison

        Checking and updating figures dictionary

        <image15>

        ==============================
        Figure file
        	 DNNLikelihood_toy_figure_corner_pars_train_true_vs_train_pred.pdf 
        created and saved in 4.314005199999997 s.

        ============================== 
        Plotting 2d posterior distributions for two samples comparison

        Checking and updating figures dictionary

        <image16>

        ==============================
        Figure file
        	 DNNLikelihood_toy_figure_corner_pars_test_true_vs_test_pred.pdf 
        created and saved in 4.048993200000041 s.

        ============================== 
        Plotting 2d posterior distributions for two samples comparison

        Checking and updating figures dictionary

        <image17>

        ==============================
        Figure file
        	 DNNLikelihood_toy_figure_corner_pars_train_true_vs_test_pred.pdf 
        created and saved in 3.716974300000061 s.

        ============================== 
        Making t_mu plot for Frequentist inference

        ============================== 
        Plotting t_mu test statistics

        t_mu information for source 'model' is not available. The curve will not be included in the plot.
        Checking and updating figures dictionary

        <image18>

        ==============================
        Figure file
        	 DNNLikelihood_toy_figure_tmu_0.pdf 
        created and saved in 0.49693549999994957 s.

        ============================== 
        All plots done in 58.75557929999991 s.

        ============================== 
        Idx h5 file
        	 <abs_path_to_output_folder>\DNNLikelihood_toy_idx.h5 
        saved in 0.007333099999982551 s.

        ============================== 
        Model json file
        	 <abs_path_to_output_folder>\DNNLikelihood_toy_model.json 
        saved in 0.00832470000000285 s.

        ============================== 
        Model h5 file
        	 <abs_path_to_output_folder>\DNNLikelihood_toy_model.h5
        tf executing eager_mode: True
        INFO:keras2onnx:tf executing eager_mode: True
        tf.keras model eager_mode: False
        INFO:keras2onnx:tf.keras model eager_mode: False

        saved in 0.16234810000003108 s.


        The ONNX operator number change on the optimization: 35 -> 25
        INFO:keras2onnx:The ONNX operator number change on the optimization: 35 -> 25
        WARNING:onnxmltools:The maximum opset needed by this model is only 9.
        ============================== 
        Model onnx file
        	 <abs_path_to_output_folder>\DNNLikelihood_toy_model.onnx 
        saved in 4.907806100000016 s.

        ============================== 
        Model history file
        	 <abs_path_to_output_folder>\DNNLikelihood_toy_history.json 
        saved in 0.008585899999957292 s. 

        ============================== 
        Predictions json file
        	 <abs_path_to_output_folder>\DNNLikelihood_toy_predictions.json 
        saved in 0.09338190000005397 s.
        
        ============================== 
        Predictions h5 file
        	 <abs_path_to_output_folder>\DNNLikelihood_toy_predictions.h5 
        saved in 0.10175609999998869 s.

        ============================== 
        \nJson file\n\t
        	 <abs_path_to_output_folder>\DNNLikelihood_toy_summary.json 
        saved in 0.007870599999932892 s.

        ============================== 
        Scalers pickle file
        	 <abs_path_to_output_folder>\DNNLikelihood_toy_scalers.pickle 
        saved in 0.004544399999986126 s.

        ============================== 
        Model graph pdf file
        	 <abs_path_to_output_folder>\DNNLikelihood_toy_model_graph.pdf 
        saved in 3.346568599999955 s.

        ============================== 
        DnnLik log file
        	 <abs_path_to_output_folder>\DNNLikelihood_toy.log 
        saved in 0.014863800000057381 s.

        ============================== 
        All predictions computed in 741.998475 s.

where <image1> to <image18> correspond to the following figures:

.. image:: ../figs/image1.png
    :class: with-shadow
    :scale: 54

.. image:: ../figs/image2.png
    :class: with-shadow
    :scale: 54

.. image:: ../figs/image3.png
    :class: with-shadow
    :scale: 54

.. image:: ../figs/image4.png
    :class: with-shadow
    :scale: 54

.. image:: ../figs/image5.png
    :class: with-shadow
    :scale: 54

.. image:: ../figs/image6.png
    :class: with-shadow
    :scale: 54

.. image:: ../figs/image7.png
    :class: with-shadow
    :scale: 54

.. image:: ../figs/image8.png
    :class: with-shadow
    :scale: 54

.. image:: ../figs/image9.png
    :class: with-shadow
    :scale: 54

.. image:: ../figs/image10.png
    :class: with-shadow
    :scale: 54

.. image:: ../figs/image11.png
    :class: with-shadow
    :scale: 54

.. image:: ../figs/image12.png
    :class: with-shadow
    :scale: 54

.. image:: ../figs/image13.png
    :class: with-shadow
    :scale: 54

.. image:: ../figs/image14.png
    :class: with-shadow
    :scale: 54
    
.. image:: ../figs/image15.png
    :class: with-shadow
    :scale: 54

.. image:: ../figs/image16.png
    :class: with-shadow
    :scale: 54

.. image:: ../figs/image17.png
    :class: with-shadow
    :scale: 54

.. image:: ../figs/image18.png
    :class: with-shadow
    :scale: 54


.. include:: ../external_links.rst