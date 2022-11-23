.. _NF_arguments:

Arguments
"""""""""

.. currentmodule:: NF

.. argument:: name

   See :argument:`name <common_classes_arguments.name>`.

.. argument:: data

   Input :mod:`Data <data>` object.
   Either this or the :argument:`input_data_file` input argument should be provided to initialize a 
   :class:`NF <NF4HEP.NF>` object and to set the
   :attr:`NF.data <NF4HEP.NF.data>` attribute.
      
      - **type**: :mod:`Data <data>` object or ``None``
      - **default**: ``None``

.. argument:: input_data_file

   File name (either relative to the code execution folder or absolute, with or without any of the
   .json or .h5 extensions) of a saved :mod:`Data <data>` object. 
   Either this or the :argument:`data` input argument should be provided to initialize a 
   :class:`NF <NF4HEP.NF>` objectand to set the
   :attr:`NF.data <NF4HEP.NF.data>` attribute.

      - **type**: ``str`` or ``None``
      - **default**: ``None``

.. argument:: load_on_RAM

   If ``True`` all data available in the dataset corresponding to the :argument:`input_data_file` input
   are loaded into the RAM for faster generation of train/validation/test data. If ``False`` the HDF5
   dataset file is open in read mode and train/validation/test data are generated on demand.
   It is used to build the :attr:`NF.load_on_RAM <NF4HEP.NF.load_on_RAM>` attribute.

      - **type**: ``bool``
      - **default**: ``False``

.. argument:: seed

   Seed of the random number generator. It is used to initialize the |numpy_link| and |tf_link| random state.
   It is used to ste the :attr:`NF.seed <NF4HEP.NF.seed>` attribute.

      - **type**: ``int`` or ``None``
      - **default**: ``None``

.. argument:: dtype

   It represents the data type into which train/val/test data are converted. 
   It is used to ste the :attr:`NF.dtype <NF4HEP.NF.dtype>` attribute.

      - **type**: ``str`` or ``None``
      - **default**: ``None``

.. argument:: same_data

   This option is passed to the :class:`NF <NF4HEP.NF>` object by the 
   :class:`DnnLikEnsemble <NF4HEP.NFEnsemble>` object to specify if the same
   train/val data are used for the different :class:`DnnLiks <NF4HEP.NF>` in the
   ensemble or not. In this way :class:`NF <NF4HEP.NF>` object knows how to
   deal with the corresponding :mod:`Data <data>` object to generate train/val data.
   It is used to ste the :attr:`NF.same_data <NF4HEP.NF.same_data>` attribute.

      - **type**: ``bool``
      - **default**: ``True``

.. argument:: model_data_inputs

   Dictionary specifying inputs related to data.
   It is used to set the private attribute
   :attr:`NF.__model_data_inputs <NF4HEP.NF._NF__model_data_inputs>`.

      - **type**: ``None`` or ``dict`` with the following structure:

         - *"npoints"* (value type: ``list``, shape: ``[n_train, n_val, n_test]``)
            List with number of train, validation, and test points. 
            The validation and test entries could also be passed in the form of fractions (<=1) of n_train. 
            These will be automatically converted in absolute numbers.
         - *"scalerX"* (value type: ``bool``)
            If ``True`` X input points are scaled with a |standard_scaler_link| fit to the train data.
            If not given it is automatically set to ``False``.
         - *"scalerY"* (value type: ``bool``)
            If ``True`` Y input points are scaled with a |standard_scaler_link| fit to the train data.
            If not given it is automatically set to ``False``.
         - *"weighted"* (value type: ``bool``)
            If ``True`` X input points are weighted (see the 
            :meth:`NF.compute_sample_weights <NF4HEP.NF.compute_sample_weights>` 
            method). If not given it is automatically set to ``False``.

      - **example**: 
      
         .. code-block:: python

            model_data_inputs = {"npoints": [n_train, n_val, n_test],
                                 "scalerX": True,
                                 "scalerY": True,
                                 "weighted": False}
      
      - **default**: ``None``

.. argument:: model_define_inputs

   Dictionary specifying inputs related to the |tf_keras_model_link| specifications, including the structure of hidden 
   |tf_keras_layers_link| with their |tf_keras_activations_link| and |tf_keras_initializers_link|, 
   |tf_keras_activations_link| in the output layer, |tf_keras_dropout_link| rate, and |tf_keras_batch_normalization_link|.
   It is used to set the private attribute
   :attr:`NF.__model_define_inputs <NF4HEP.NF._NF__model_define_inputs>`.
   The |tf_keras_activations_link| and the |tf_keras_initializers_link| in each layer are specified together with the number 
   of nodes in the layer (see the example below). All |tf_keras_link| specifications for the |tf_keras_activations_link| 
   and |tf_keras_initializers_link| are supported. In the case of |tf_keras_selu_link| activation the 
   |tf_keras_lecun_normal_link| initializer is automatically used and possible |tf_keras_dropout_link| layers are replaced 
   by |tf_keras_alpha_dropout_link| ones. In case of |tf_keras_activations_link| other than |tf_keras_selu_link|, 
   if no |tf_keras_initializers_link| is specified, then the default one is used.

      - **type**: ``None`` or ``dict`` with the following structure

         - *"hidden_layers"* (value type: ``list``, shape: ``[[int,str,str],...,[int,str,str]]``)
            List with hidden layers specifications. All hidden layers in the module are |tf_keras_layers_dense_link| layers.
            Each hidden layer is represented by a list with number of nodes (``int``),
            activation function (``str``), and (optionally) initializer (``str``).
         - *"act_func_out_layer"* (value type: ``str``)
            Activation function for the output layer. If not specified it is automatically set to ``"linear"``.
         - *"dropout_rate"* (value type: ``float``)
            If different than ``0``, then |tf_keras_dropout_link| layers with the given dropout_rate are added
            between each pair of hidden layers. If not specified it is automatically set to ``0``.
         - *"batch_norm"* (value type: ``bool``)
            If ``True``, then |tf_keras_batch_normalization_link| layers are added after the input layer and between 
            each pair of hidden layers. If not specified it is automatically set to ``False``.

      - **example**: 
      
         .. code-block:: python

            model_define_inputs = {"hidden_layers": [[300, "selu"],
                                                     [300, "relu"], 
                                                     [300, "selu", "lecun_normal"], 
                                                     [300, "relu", "glorot_uniform"]],
                                   "act_func_out_layer": "linear",
                                   "dropout_rate": 0,
                                   "batch_norm": True}

      - **default**: ``None``

.. argument:: model_optimizer_inputs

   String or dictionary specifying inputs related to the |tf_keras_optimizers_link| specifications.
   It is used to set the private attribute
   :attr:`NF.__model_optimizer_inputs <NF4HEP.NF._NF__model_optimizer_inputs>`.
   It could be specified either as a string with the name of the |tf_keras_optimizers_link_2| (all default parameters will 
   be used in this case) or as a dictionary with one item of the form ``"name": "optimizer_name"`` and all other items 
   specifying parameters related to that |tf_keras_optimizers_link_2| in the form ``"parameter": value`` (see the example below). 
   All unspecified parameters are set to their default value.

      - **type**: ``str`` or ``None`` or ``dict`` with the following structure:

         - *"name"* (value type: ``str``)
            Name of the |tf_keras_optimizers_link_2|.
         - others
            All input arguments accepted by the corresponding |tf_keras_optimizers_link|.
            For all unspecified inputs default values will be used.
         
      - **example**: 

         .. code-block:: python
 
            model_optimizer_inputs = {"name": "Adam",
                                      "lr": 0.001,
                                      "beta_1": 0.9,
                                      "beta_2": 0.999,
                                      "amsgrad": False}

      - **default**: ``None``
      
.. argument:: model_compile_inputs

   Dictionary specifying inputs related to the |tf_keras_model_compile_link| specifications, including |tf_keras_losses_link|
   and |tf_keras_metrics_link|.
   It is used to set the private attribute
   :attr:`NF.__model_compile_inputs <NF4HEP.NF._NF__model_compile_inputs>`.
   Both the short name and the true name of a metrix are allowed (for instance, both ``"mse"`` and ``"mean_squared_error"``
   are allowed).

      - **type**: ``None`` or ``dict`` with the following structure:

         - *"loss"* (value type: ``str``)
            Name of the loss (either abbreviated or extended).
            If not specified the user is warned that it will be automatically set to ``"mse"``.
         - *"metrics"* (value type: ``list``)
            List with the name (either abbreviated or extended) of the metrics to be monitored.
            If not specified it will be automatically set to ``["mse","mae","mape","msle"]``

      - **example**: 

         .. code-block:: python
 
            model_compile_inputs = {"loss": "mse", 
                                    "metrics": ["mean_squared_error", "mae", "msle"]}

      - **default**: ``None``

.. argument:: model_callbacks_inputs

   List of strings and/or dictionaries specifying inputs related to the |tf_keras_callbacks_link| specifications.
   It is used to set the private attribute
   :attr:`NF.__model_callbacks_inputs <NF4HEP.NF._NF__model_callbacks_inputs>`.
   Each item in the list could be specified either as a string with the name of the |tf_keras_callbacks_link_2| 
   (all default parameters will be used in this case) or as a dictionary with one item of the form 
   ``"name": "callback_name"`` and all other items specifying parameters related to that |tf_keras_callbacks_link_2|
   in the form ``"parameter": value`` (see the example below). All unspecified parameters are set to their default value.
   All |tf_keras_callbacks_link| are available, together with the ``"PlotLossesKeras"`` callback from the |livelossplot_link|
   package.
   Arguments specifying output files/folders for callbacks writing to files, that are ``"PlotLossesKeras"``, 
   |tf_keras_model_checkpoint_callback_link|, and |tensorboard_link|, are automatically set and the related inputs are ignored.

      - **type**: ``list`` or ``None``
      - **list items type**: ``str`` and/or ``dict`` with the following structure

         - *"name"* (value type: ``str``)
            Name of the |tf_keras_callbacks_link_2|.
         - *others*
            All input arguments accepted by the corresponding |tf_keras_callbacks_link|.
            For all unspecified inputs default values will be used.
         
      - **example**: 

         .. code-block:: python
 
            model_callbacks_inputs = [{"name": "EarlyStopping",
                                       "monitor": "loss",
                                       "mode": "min",
                                       "patience": 100,
                                       "min_delta": 0.0001,
                                       "restore_best_weights": True},
                                      "TerminateOnNaN",
                                      "PlotLossesKeras",
                                      {"name": "ReduceLROnPlateau",
                                       "monitor": "loss",
                                       "mode": "min",
                                       "factor": 0.2,
                                       "min_lr": 0.00008,
                                       "patience": 10,
                                       "min_delta": 0.0001},
                                      {"name": "ModelCheckpoint",
                                       "filepath": "this is ignored and automatically set",
                                       "monitor": "loss",
                                       "mode": "min",
                                       "save_best_only": True,
                                       "save_freq": "epoch"}]
      
      - **default**: ``None``

.. argument:: model_train_inputs

   Dictionary specifying inputs arguments for the |tf_keras_model_fit_link| method. 
   It must include at least the two items corresponding to the keys `"epochs"` and `"batch_size"`
   and could optionally include other arguments such as, for instance, `"shuffle"`, `"validation_freq"`, etc. 
   (see the documentation of the |tf_keras_model_fit_link| method for details).
   It is used to set the private attribute
   :attr:`NF.__model_train_inputs <NF4HEP.NF._NF__model_train_inputs>`.

      - **type**: ``None`` or ``dict`` with the following structure:

         - *"epochs"* (value type: ``int``)
            Number of epochs to run.
         - *"batch_size"* (value type: ``int``)
            Batch size to use in training, evaluation, and prediction.
         - *optional*
            Other keyword arguments to be passed to the |tf_keras_model_fit_link| method.
         
      - **example**: 

         .. code-block:: python
 
            model_train_inputs={"epochs": 50,
                                "batch_size": 512,
                                "shuffle": False,
                                "validation_freq": 2}

      - **default**: ``None``

.. argument:: resources_inputs

   Dictionary specifying available resources. This input argument is assumed to be specified automatically when the 
   :class:`NF <NF4HEP.NF>` object is created from within the 
   :class:`DnnLikEnsemble <NF4HEP.NFEnsemble>` one in order to sync resources between the 
   different :class:`DnnLiks <NF4HEP.NF>`
   (see the :mod:`DNNLikEnsemble <dnn_likelihood_ensemble>` object).

   - **type**: ``None`` or ``dict`` with the following structure:

         - *"available_gpus"* (value type: ``list``, shape: ``(n_gpus,2)``)
            List of lists with device ID and information of the available GPUs.
         - *"available_cpu"* (value type: ``list``, shape: ``(3,)``)
            List with device ID, specifications, number of cores of the available CPU.
         - *"active_gpus"* (value type: ``list``, shape: ``(n_active_gpus,2)``)
            List of lists with device ID and information of the subset of available GPUs that are active in the current environment.
         - *"gpu_mode"* (value type: ``bool``)
            ``True`` if the current machine has available GPUs, ``False`` otherwise.

   - **default**: ``None``

.. argument:: output_folder

   See :argument:`output_folder <common_classes_arguments.output_folder>`.

.. argument:: ensemble_name

   Name of the :class:`DnnLikEnsemble <NF4HEP.NFEnsemble>` object
   of which the :class:`NF <NF4HEP.NF>` object is a member. This input argument
   is automatically passed by the generating :class:`DnnLikEnsemble <NF4HEP.NFEnsemble>` object.
   If the :class:`NF <NF4HEP.NF>` object is not a member of an ensemble, this input should be
   ``None`` (default).
   It is used to set the :attr:`NF.ensemble_name <NF4HEP.NF.ensemble_name>`
   and :attr:`NF.ensemble_folder <NF4HEP.NF.ensemble_folder>` attributes.
         
         - **type**: ``str`` or ``None``
         - **default**: ``None``

.. argument:: input_file

   See :argument:`input_file <common_classes_arguments.input_file>`.

.. argument:: verbose

   See :argument:`verbose <common_classes_arguments.verbose>`.
   
.. include:: ../external_links.rst