Attributes
""""""""""

.. currentmodule:: NF

.. py:attribute:: NF._NF__model_callbacks_inputs

   Private attribute corresponding to the input argument :argument:`model_callbacks_inputs`. 
   If the object is initialized from input arguments it is set to the value of :argument:`model_callbacks_inputs`,
   otherwise it is set from the file corresponding to the 
   :attr:`NF.input_file <NF4HEP.NF.input_file>` attribute.

      - **type**: ``list``
      - **list items type**: ``str`` and/or ``dict`` (see :argument:`model_callbacks_inputs` for the dictionary structure).

.. py:attribute:: NF._NF__model_compile_inputs

   Private attribute corresponding to the input argument :argument:`model_compile_inputs`.
   If the object is initialized from input arguments it is set to the value of :argument:`model_compile_inputs` if present,
   otherwise it is constructed setting the loss to ``"mse"`` and the metrics to ``["mse","mae","mape","msle"]``. If the
   object is imported from files the attribute is set from the file corresponding to the 
   :attr:`NF.input_file <NF4HEP.NF.input_file>` attribute.

      - **type**: ``dict`` (see :argument:`model_compile_inputs` for the dictionary structure).

.. py:attribute:: NF._NF__model_data_inputs

   Private attribute corresponding to the input argument :argument:`model_data_inputs`.
   If the object is initialized from input arguments it is set to the value of :argument:`model_data_inputs`
   otherwise is set from the file corresponding to the 
   :attr:`NF.input_file <NF4HEP.NF.input_file>` attribute.
   If the number of validation and test data points are smaller than one, then they are treated as fractions of the
   corresponding number of training points and are replaced by the absolute numbers. If ``"scaleX"`, ``"scaleY"``,
   and/or ``"weighted"`` are not specified than they are set to ``False``.

      - **type**: ``dict`` (see :argument:`model_data_inputs` for the dictionary structure).

.. py:attribute:: NF._NF__model_define_inputs

   Private attribute corresponding to the input argument :argument:`model_define_inputs`.
   If the object is initialized from input arguments it is set to the value of :argument:`model_define_inputs`
   otherwise is set from the file corresponding to the 
   :attr:`NF.input_file <NF4HEP.NF.input_file>` attribute.
   If ``"act_func_out_layer"`, ``"dropout_rate"``,
   and/or ``"batch_norm"`` are not specified than they are set to ``"linear"``, ``0``, and ``False``, respectively.

      - **type**: ``dict`` (see :argument:`model_define_inputs` for the dictionary structure).

.. py:attribute:: NF._NF__model_optimizer_inputs

   Private attribute corresponding to the input argument :argument:`model_optimizer_inputs`. 
   If the object is initialized from input arguments it is set to the value of :argument:`model_optimizer_inputs`,
   otherwise it is set from the file corresponding to the 
   :attr:`NF.input_file <NF4HEP.NF.input_file>` attribute.
   For all arguments of the |tf_keras_optimizers_link_2| that are not specified the default value is used.

      - **type**: ``str`` or ``dict`` (see :argument:`model_optimizer_inputs` for the dictionary structure).

.. py:attribute:: NF._NF__model_train_inputs

   Private attribute corresponding to the input argument :argument:`model_train_inputs`. 
   If the object is initialized from input arguments it is set to the value of :argument:`model_train_inputs`,
   otherwise it is set from the file corresponding to the 
   :attr:`NF.input_file <NF4HEP.NF.input_file>` attribute.

      - **type**: ``dict`` (see :argument:`model_train_inputs` for the dictionary structure).

.. py:attribute:: NF._NF__resources_inputs

   Private attribute corresponding to the input argument :argument:`resources_inputs`. If the input argument is not given, 
   i.e. it is set to the default value ``None``, the dictionary is set by the 
   :meth:`NF.__set_resources <NF4HEP.NF._NF__set_resources>` private method.

      - **type**: ``dict`` (see :argument:`resources_inputs` for the dictionary structure).

.. py:attribute:: NF.act_func_out_layer

   String representing the activation function in the output layer. 
   It is set from the 
   :attr:`NF.__model_define_inputs <NF4HEP.NF._NF__model_define_inputs>`
   dictionary.

      - **type**: ``str``

.. py:attribute:: NF.active_gpus

   List specifying the GPUs active in the current environment.
   For each GPU the list contains two strings with device ID and information, respectively.
   It is either set from the the 
   :attr:`NF.__resources_inputs <NF4HEP.NF._NF__resources_inputs>`
   dictionary if available or initialized by the 
   :attr:`NF.__set_resources <NF4HEP.NF._NF__set_resources>` method.

      - **type**: ``list``
      - **shape**: ``(n_active_gpus,2)``

.. py:attribute:: NF.available_cpu

   List specifying the CPU resources.
   It contains three strings with device ID, specifications, number of cores, respectively.
   It is either set from the the 
   :attr:`NF.__resources_inputs <NF4HEP.NF._NF__resources_inputs>`
   dictionary if available or initialized by the 
   :attr:`NF.__set_resources <NF4HEP.NF._NF__set_resources>` method.

      - **type**: ``list``
      - **shape**: ``(3,)``

.. py:attribute:: NF.available_gpus

   List specifying the GPUs available on the current machine.
   For each GPU the list contains two strings with device ID and information, respectively.
   It is either set from the the 
   :attr:`NF.__resources_inputs <NF4HEP.NF._NF__resources_inputs>`
   dictionary if available or initialized by the 
   :attr:`NF.__set_resources <NF4HEP.NF._NF__set_resources>` method.

      - **type**: ``list``
      - **shape**: ``(n_available_gpus,2)``

.. py:attribute:: NF.batch_norm

   If ``True``, then |tf_keras_batch_normalization_link| layers are added after the input layer and between 
   each pair of hidden layers. It is set from the the 
   :attr:`NF.__model_define_inputs <NF4HEP.NF._NF__model_define_inputs>`
   dictionary (``False`` if not specified in the input argument :argument:`model_define_inputs`).

      - **type**: ``bool``

.. py:attribute:: NF.batch_size

   Specifies the default batch size used for training, evaluation, and prediction.
   It is set from the  
   :attr:`NF.__model_train_inputs <NF4HEP.NF._NF__model_train_inputs>`
   dictionary.

      - **type**: ``int``

.. py:attribute:: NF.callbacks

   List of |tf_keras_callbacks_link| objects.
   It is set by the :meth:`NF.__set_callbacks <NF4HEP.NF._NF__set_callbacks>` 
   method by evaluating
   the corresponding strings in the :attr:`NF.callbacks_strings <NF4HEP.NF.callbacks_strings>`
   attribute.

      - **type**: ``list`` of |tf_keras_callbacks_link| objects

.. py:attribute:: NF.callbacks_strings

   List of |tf_keras_callbacks_link| strings.
   It is set by the :meth:`NF.__set_callbacks <NF4HEP.NF._NF__set_callbacks>` 
   method by parsing the content of the 
   :attr:`NF.__model_callbacks_inputs <NF4HEP.NF._NF__model_callbacks_inputs>`
   dictionary.

      - **type**: ``list`` of ``str``

.. py:attribute:: NF.data

   The :mod:`Data <data>` object used by the 
   :class:`NF <NF4HEP.NF>` one for data management.
   It is set to the value of the input ärgument :argument:`data` if passed. Otherwise,
   the :meth:`NF.__set_data <NF4HEP.NF._NF__set_data>`
   sets it to a :mod:`Data <data>` object imported from the 
   :attr:`NF.input_data_file <NF4HEP.NF.input_data_file>`.

      - **type**: :mod:`Data <data>` object

.. py:attribute:: NF.data_max

   Dictionary containing the maximum of the DNNLikelihood computed with the 
   :meth:`NF.model_compute_max <NF4HEP.NF.model_compute_max>` method.
   It contains ``"X"`` and ``"Y"`` items corresponding to the X and Y values at
   the maximum of the :meth:`NF.model_predict_scalar <NF4HEP.NF.model_predict_scalar>`
   function.

      - **type**: ``dict`` with the following structure:

         - *"X"* (value type: ``numpy.ndarray``, shape: ``(ndims,)``)
            |Numpy_link| array with the values of parameters at the model maximum.
         - *"Y"* (value type: ``float``)
            Value of the model at its maximum

.. py:attribute:: NF.data_profiled_max

   To be written

.. py:attribute:: NF.dropout_rate

   Specifies the dropout rate for the |tf_keras_dropout_link| layers.
   It is set from the  
   :attr:`NF.__model_define_inputs <NF4HEP.NF._NF__model_define_inputs>`
   dictionary. If the value is ``0`` no |tf_keras_dropout_link| layers are added to the model.

      - **type**: ``float``

.. py:attribute:: NF.dtype

   Required data type for the generation of the train/validation/test datasets. The attribute is set by the
   :meth:`NF.__set_dtype <NF4HEP.NF._NF__set_dtype>` methods to
   the value of the :argument:`dtype` if it is not ``None`` and to ``"float64"`` otherwise.

      - **type**: ``str``

.. py:attribute:: NF.ensemble_folder

   If the :class:`NF <NF4HEP.NF>` object is a member of a 
   :class:`DnnLikEnsemble <NF4HEP.NFEnsemble>` object, then this attribute is set to the
   parent directory of :attr:`NF.output_folder <NF4HEP.NF.output_folder>`, where, by default,
   the :class:`DnnLikEnsemble <NF4HEP.NFEnsemble>` object is stored
   (and corresponding to the 
   :attr:`DnnLikEnsemble.ensemble_folder <NF4HEP.NFEnsemble.ensemble_folder>` attribute of the
   :class:`DnnLikEnsemble <NF4HEP.NFEnsemble>` object).
   The attribute is by the 
   :meth:`NF.__check_define_ensemble_folder <NF4HEP.NF._NF__check_define_ensemble_folder>`
   method based on the value of the 
   :attr:`NF.ensemble_name <NF4HEP.NF.ensemble_name>` attribute. If the latter is ``None``,
   then also :attr:`NF.output_folder <NF4HEP.NF.output_folder>` attribute is set to ``None``,
   the :class:`NF <NF4HEP.NF>` object is considered "standalone", and the 
   :attr:`NF.standalone <NF4HEP.NF.standalone>` is set to ``True``.

      - **type**: ``str`` or ``None``

.. py:attribute:: NF.ensemble_name

   Attribute corresponding to the input argument :argument:`ensemble_name`. 
   If the :class:`NF <NF4HEP.NF>` object is a member of a
   :class:`DnnLikEnsemble <NF4HEP.NFEnsemble>` object, then the latter automatically passes
   the input argument :argument:`ensemble_name` when it creates the :class:`NF <NF4HEP.NF>` object.
   The attribute is by the 
   :meth:`NF.__check_define_ensemble_folder <NF4HEP.NF._NF__check_define_ensemble_folder>`
   method. If the input argument is ``None``,
   then also the attribute is set to ``None``,
   the :class:`NF <NF4HEP.NF>` object is considered "standalone", and the 
   :attr:`NF.standalone <NF4HEP.NF.standalone>` is set to ``True``.

      - **type**: ``str`` or ``None``

.. py:attribute:: NF.epochs_available

   Number if epochs for which :attr:`NF.model <NF4HEP.NF.model>` has been trained. It is updated by
   the :meth:`NF.model_train <NF4HEP.NF.model_train>` method after each training.

      - **type**: ``int``

.. py:attribute:: NF.epochs_required

   Number of epochs required for the training. The first time the object is created it is set to the value of the
   "epochs" item in the :attr:`NF.__model_train_inputs <NF4HEP.NF._NF__model_train_inputs>`
   dictionary. It can be manually changed before calling the 
   :meth:`NF.model_train <NF4HEP.NF.model_train>` method to train for more epochs.
   Notice that each call to the 
   :meth:`NF.model_train <NF4HEP.NF.model_train>` method only trains for a number of epochs
   equal to the difference between :attr:`NF.epochs_required <NF4HEP.NF.epochs_required>`
   and :attr:`NF.epochs_available <NF4HEP.NF.epochs_available>` (set by the
   private method :meth:`NF.__set_epochs_to_run <NF4HEP.NF._NF__set_epochs_to_run>`).    

      - **type**: ``int``

.. py:attribute:: NF.fig_base_title

   Common title for generated figures generated by the 
   :meth:`NF.generate_fig_base_title <NF4HEP.NF.generate_fig_base_title>` method and
   including information on 

      - :attr:`NF.ndims <NF4HEP.NF.ndims>`
      - :attr:`NF.ndims <NF4HEP.NF.npoints_train>`
      - :attr:`NF.ndims <NF4HEP.NF.hidden_layers>`
      - :attr:`NF.ndims <NF4HEP.NF.loss_string>`

      - **type**: ``list`` of ``str`` 

.. py:attribute:: NF.figures_list

   List of absolute paths to the generated figures.

      - **type**: ``list`` of ``str`` 

.. py:attribute:: NF.pars_labels_auto

   Copy of the :attr:`NF.data.pars_labels_auto <NF4HEP.Data.pars_labels_auto>`
   attribute of the :mod:`Data <data>` object used for data management
   (see the doc of the :attr:`Data.pars_labels_auto <NF4HEP.Data.pars_labels_auto>` attribute).

      - **type**: ``list``
      - **shape**: ``(ndims,)``

.. py:attribute:: NF.gpu_mode

   Attribute set by the 
   :meth:`NF.__set_resources <NF4HEP.NF._NF__set_resources>` method and indicating
   whether the object has GPU support (``True``) or not (``False``). If the object is a mamber of a 
   :class:`DnnLikEnsemble <NF4HEP.NFEnsemble>` one, then the attribute is set from the
   :attr:`NF.__resources_inputs <NF4HEP.NF._NF__resources_inputs>` attribute.

      - **type**: ``bool`` 

.. py:attribute:: NF.hidden_layers

   Attribute corresponding to the "hidden_layer" item of the 
   :attr:`NF.__model_define_inputs <NF4HEP.NF._NF__model_define_inputs>`
   attribute.
   It is a list with hidden layers specifications. All hidden layers in the module are |tf_keras_layers_dense_link| layers.
   Each hidden layer is represented by a list with number of nodes (``int``),
   activation function (``str``), and (optionally) initializer (``str``)

      - **type**: ``list``
      - **shape**: ``(n_hidden_layers,3)``

.. py:attribute:: NF.history

   Attribute corresponding to the "history" dictionary of the ``History`` object returned by the |tf_keras_model_fit_link|
   method. It contains a list with the history of values of metrics for each training epoch. The contento of this 
   dictionary is saved into the file
   :attr:`NF.output_history_json_file <NF4HEP.NF.output_history_json_file>`.

      - **type**: ``dict`` with the following structure:

         - *"metric"* (value type: ``list``)
            List with values of metric for each epoch. This key is present for all metric with the corresponding 
            full name of the metric, such as, for instance, ``"mean_squared_error_best", etc. Metrics include ``"loss"``.
         - *"lr"* (value type: ``float``)
            List with values of the learning rate at each epoch.

.. py:attribute:: NF.idx_test

   |Numpy_link| array with the integer position (indices) of the test data points in the data arrays contained in
   :attr:`NF.data.data_X <NF4HEP.Data.data_X>` and
   :attr:`NF.data.data_Y <NF4HEP.Data.data_Y>`.
   The same indices are also stored in the "idx_test" item of the 
   :attr:`Data.data_dictionary <NF4HEP.Data.data_dictionary>` object 
   :attr:`NF.data <NF4HEP.NF.data>`.

      - **type**: ``numpy.ndarray``
      - **shape**: ``(n_points_test,)``

.. py:attribute:: NF.idx_train

   |Numpy_link| array with the integer position (indices) of the training data points in the data arrays contained in
   :attr:`NF.data.data_X <NF4HEP.Data.data_X>` and
   :attr:`NF.data.data_Y <NF4HEP.Data.data_Y>`.
   The same indices are also stored in the "idx_train" item of the 
   :attr:`Data.data_dictionary <NF4HEP.Data.data_dictionary>` object 
   :attr:`NF.data <NF4HEP.NF.data>`.

      - **type**: ``numpy.ndarray``
      - **shape**: ``(n_points_train,)``

.. py:attribute:: NF.idx_val

   |Numpy_link| array with the integer position (indices) of the validation data points in the data arrays contained in
   :attr:`NF.data.data_X <NF4HEP.Data.data.data_X>` and
   :attr:`NF.data.data_Y <NF4HEP.Data.data.data_Y>`.
   The same indices are also stored in the "idx_val" item of the 
   :attr:`NF.data.data_dictionary <NF4HEP.Data.data_dictionary>` attribute.

      - **type**: ``numpy.ndarray``
      - **shape**: ``(n_points_val,)``

.. py:attribute:: NF.input_data_file

   Absolute path corresponding to the input argument :argument:`input_data_file`. If the latter is not passed,
   :attr:`NF.input_data_file <NF4HEP.NF.input_data_file>` is set to
   :class:`NF.data.input_file <NF4HEP.Data.input_file>`.
           
        - **type**: ``str``

.. py:attribute:: NF.input_file

   See :attr:`input_file <common_classes_attributes.input_file>`.

.. py:attribute:: NF.input_folder

   See :attr:`input_folder <common_classes_attributes.input_folder>`.

.. py:attribute:: NF.input_h5_file

   See :attr:`input_h5_file <common_classes_attributes.input_h5_file>`.

.. py:attribute:: NF.input_history_json_file

   Absolute path to the .json file containing saved :attr:`NF.history <NF4HEP.NF.history>`
   attribute (see the :meth:`NF.save_history_json <NF4HEP.NF.save_history_json>`
   method for details).
   It is automatically generated from the attribute 
   :attr:`NF.input_files_base_name <NF4HEP.NF.input_files_base_name>`.
   When the latter is ``None``, the attribute is set to ``None``.
          
      - **type**: ``str`` or ``None``

.. py:attribute:: NF.input_idx_h5_file

   Absolute path to the .h5 file containing saved data indices (see the 
   :meth:`NF.save_data_indices <NF4HEP.NF.save_data_indices>`
   method for details).
   It is automatically generated from the attribute 
   :attr:`NF.input_files_base_name <NF4HEP.NF.input_files_base_name>`.
   When the latter is ``None``, the attribute is set to ``None``.
          
      - **type**: ``str`` or ``None``

.. py:attribute:: NF.input_log_file

   See :attr:`input_log_file <common_classes_attributes.input_log_file>`.

.. py:attribute:: NF.input_predictions.h5_file

   Absolute path to the .json file containing saved 
   :attr:`NF.predictions <NF4HEP.NF.predictions>` (see the 
   :meth:`NF.save_data_indices <NF4HEP.NF.save_data_indices>`
   method for details).
   It is automatically generated from the attribute 
   :attr:`NF.input_files_base_name <NF4HEP.NF.input_files_base_name>`.
   When the latter is ``None``, the attribute is set to ``None``.
          
      - **type**: ``str`` or ``None``

.. py:attribute:: NF.input_scalers_pickle_file

   Absolute path to the .pickle file containing saved data standard scalers (see the 
   :meth:`NF.save_scalers <NF4HEP.NF.save_scalers>`
   method for details).
   It is automatically generated from the attribute 
   :attr:`NF.input_files_base_name <NF4HEP.NF.input_files_base_name>`.
   When the latter is ``None``, the attribute is set to ``None``.
          
      - **type**: ``str`` or ``None``

.. py:attribute:: NF.input_file

   Absolute path to the .json file containing saved :class:`NF <NF4HEP.NF>` object 
   (see the :meth:`NF.save_json <NF4HEP.NF.save_json>`
   method for details).
   It is automatically generated from the attribute 
   :attr:`NF.input_files_base_name <NF4HEP.NF.input_files_base_name>`.
   When the latter is ``None``, the attribute is set to ``None``.
          
      - **type**: ``str`` or ``None``

.. py:attribute:: NF.input_tf_model_h5_file

   Absolute path to the .h5 file containing saved |tf_keras_model_link| corresponding to the 
   :attr:`NF.model <NF4HEP.NF.model>` 
   (see the :meth:`NF.save_json <NF4HEP.NF.save_json>`
   method for details).
   It is automatically generated from the attribute 
   :attr:`NF.input_files_base_name <NF4HEP.NF.input_files_base_name>`.
   When the latter is ``None``, the attribute is set to ``None``.
          
      - **type**: ``str`` or ``None``
 
.. py:attribute:: NF.load_on_RAM

   Attribute corresponding to the input argument :argument:`load_on_RAM`.
   If ``True`` all data available in the HDF5 dataset 
   :attr:`NF.input_data_file <NF4HEP.NF.input_data_file>`
   are loaded into the RAM for faster generation of train/validation/test data. 
   If ``False`` the HDF5 file is open in read mode and train/validation/test data are generated on demand.
         
      - **type**: ``str`` or ``None``

.. py:attribute:: NF.log

   See :attr:`log <common_classes_attributes.log>`.
                     
.. py:attribute:: NF.loss

   |tf_keras_losses_link| object.
   It is set by the :meth:`NF.__set_loss <NF4HEP.NF._NF__set_loss>` method by evaluating
   the corresponding strings in the :attr:`NF.loss_string <NF4HEP.NF.loss_string>`
   attribute.

      - **type**: |tf_keras_losses_link| object

.. py:attribute:: NF.loss_string

   String representing a |tf_keras_losses_link| object.
   It is set by the :meth:`NF.__set_loss <NF4HEP.NF._NF__set_loss>` method by parsing
   the content of the 
   :attr:`NF.__model_compile_inputs <NF4HEP.NF._NF__model_compile_inputs>`
   dictionary.

      - **type**: ``str``

.. py:attribute:: NF.metrics

   List of |tf_keras_metrics_link| objects.
   It is set by the :meth:`NF.__set_metrics <NF4HEP.NF._NF__set_metrics>` 
   method by evaluating
   the corresponding strings in the :attr:`NF.metrics_string <NF4HEP.NF.metrics_string>`
   attribute.

      - **type**: ``list`` of |tf_keras_metrics_link| objects

.. py:attribute:: NF.metrics_string

   List of |tf_keras_metrics_link| strings.
   It is set by the :meth:`NF.__set_metrics <NF4HEP.NF._NF__set_metrics>` 
   method by parsing the content of the 
   :attr:`NF.__model_compile_inputs <NF4HEP.NF._NF__model_compile_inputs>`
   dictionary.

      - **type**: ``list`` of ``str``

.. py:attribute:: NF.model

   |tf_keras_model_link| object representing the NF4HEP.

      - **type**: |tf_keras_model_link| object

.. py:attribute:: NF.model_max

   Dictionary containing the maximum of the DNNLikelihood computed with the 
   :meth:`NF.model_compute_max <NF4HEP.NF.model_compute_max>` method.
   It contains x and y values at
   the maximum of the :meth:`NF.model <NF4HEP.NF.model>`
   function.
   It is initialized to an empty dictionary ``{}`` when the 
   :class:`DNNLik <NF4HEP.DNNLik>` object is created.

      - **type**: ``dict`` with the following structure:

         - *"x"* (value type: ``numpy.ndarray``, shape: ``(ndims,)``)
            |Numpy_link| array with the values of parameters at the model maximum.
         - *"y"* (value type: ``float``)
            Value of the model at its maximum

.. py:attribute:: NF.model_train_kwargs

   Dictionary corresponding to the additional keyword arguments to be passed to the 
   |tf_keras_model_fit_link| method. It is constructed from the input dictionary
   :argument:`model_train_inputs` by removing the two items corresponding to 
   `"epochs"` and `"batch_size"`, which are saved into dedicated attributes.

.. py:attribute:: NF.model_profiled_max

   To be written

.. py:attribute:: NF.name

   See :attr:`name <common_classes_attributes.name>`.

.. py:attribute:: NF.ndims

   See :attr:`ndims <common_classes_attributes.ndims>`.

.. py:attribute:: NF.npoints_available

   Total number of points available in the class:`Data <NF4HEP.Data>` object
   :attr:`NF.data <NF4HEP.NF.data>`, corresponding to the
   :attr:`Data.npoints <NF4HEP.Data.npoints>` attribute.

      - **type**: ``int``

.. py:attribute:: NF.npoints_test

   Number of test points for the current model. It is set by the  
   :meth:`NF.__init__ <NF4HEP.NF.__init__>` method by parsing
   the item ``"npoints"`` of the 
   :attr:`NF.__model_data_inputs <NF4HEP.NF._NF__model_data_inputs>`
   dictionary.

      - **type**: ``int``

.. py:attribute:: NF.npoints_test_available

   Total number of test points available in the class:`Data <NF4HEP.Data>` object
   :attr:`NF.data <NF4HEP.NF.data>`, computed as the rounded product of
   one minus :attr:`Data.test_fraction <NF4HEP.Data.test_fraction>` and
   :attr:`NF.npoints_available <NF4HEP.NF.npoints_available>`.

      - **type**: ``int``

.. py:attribute:: NF.npoints_train

   Number of training points for the current model. It is set by the  
   :meth:`NF.__init__ <NF4HEP.NF.__init__>` method by parsing
   the item ``"npoints"`` of the 
   :attr:`NF.__model_data_inputs <NF4HEP.NF._NF__model_data_inputs>`
   dictionary.

      - **type**: ``int``

.. py:attribute:: NF.npoints_train_val_available

   Total number of train plus validation points available in the class:`Data <NF4HEP.Data>` object
   :attr:`NF.data <NF4HEP.NF.data>`, computed as the rounded product of
   :attr:`Data.test_fraction <NF4HEP.Data.test_fraction>` and
   :attr:`NF.npoints_available <NF4HEP.NF.npoints_available>`.

         - **type**: ``int``

.. py:attribute:: NF.npoints_val

   Number of validation points for the current model. It is set by the  
   :meth:`NF.__init__ <NF4HEP.NF.__init__>` method by parsing
   the item ``"npoints"`` of the 
   :attr:`NF.__model_data_inputs <NF4HEP.NF._NF__model_data_inputs>`
   dictionary.

      - **type**: ``int``

.. py:attribute:: NF.optimizer

   |tf_keras_optimizers_link| object.
   It is set by the :meth:`NF.__set_optimizer <NF4HEP.NF._NF__set_optimizer>` 
   method by evaluating the corresponding strings in the 
   :attr:`NF.optimizer_string <NF4HEP.NF.optimizer_string>`
   attribute.

      - **type**: |tf_keras_optimizers_link| object

.. py:attribute:: NF.optimizer_string

   String representing a |tf_keras_optimizers_link| object.
   It is set by the :meth:`NF.__set_optimizer <NF4HEP.NF._NF__set_optimizer>`
   method by parsing the content of the 
   :attr:`NF.__model_optimizer_inputs <NF4HEP.NF._NF__model_optimizer_inputs>`
   dictionary.

      - **type**: ``str``

.. py:attribute:: NF.output_checkpoints_files

   When the callback |tf_keras_model_checkpoint_callback_link| is present, this attribute represents the absolute path 
   to the .h5 files containing model checkpoints.
   It is initialized to ``None`` by the 
   :meth:`NF.__check_define_output_files <NF4HEP.NF._NF__check_define_output_files>`
   method and automatically generated when needed by the
   :meth:`NF.__set_callbacks <NF4HEP.NF._NF__set_callbacks>` method
   from the attribute 
   :attr:`NF.output_files_base_name <NF4HEP.NF.output_files_base_name>`.
          
      - **type**: ``str`` or ``None``

.. py:attribute:: NF.output_checkpoints_folder

   When the callback |tf_keras_model_checkpoint_callback_link| is present, this attribute represents the absolute path 
   to the folder where the .h5 files containing model checkpoints are saved.
   It is initialized to ``None`` by the 
   :meth:`NF.__check_define_output_files <NF4HEP.NF._NF__check_define_output_files>`
   method and automatically generated when needed by the
   :meth:`NF.__set_callbacks <NF4HEP.NF._NF__set_callbacks>` method
   from the attribute 
   :attr:`NF.output_folder <NF4HEP.NF.output_folder>`.
          
      - **type**: ``str`` or ``None``

.. py:attribute:: NF.output_figure_plot_losses_keras_file

   When the callback |livelossplot_link| is present, this attribute represents the absolute path 
   to the folder where the corresponding figures are saved.
   It is initialized to ``None`` by the 
   :meth:`NF.__check_define_output_files <NF4HEP.NF._NF__check_define_output_files>`
   method and automatically generated when needed by the
   :meth:`NF.__set_callbacks <NF4HEP.NF._NF__set_callbacks>` method
   from the attribute 
   :attr:`NF.output_figures_base_file <NF4HEP.NF.output_figures_base_file>`.
          
      - **type**: ``str`` or ``None``

.. py:attribute:: NF.output_figures_base_file_name

   See :attr:`output_figures_base_file_name <common_classes_attributes.output_figures_base_file_name>`.

.. py:attribute:: NF.output_figures_base_file_path

   See :attr:`output_figures_base_file_path <common_classes_attributes.output_figures_base_file_path>`.

.. py:attribute:: NF.output_figures_folder

   See :attr:`output_figures_folder <common_classes_attributes.output_figures_folder>`.

.. py:attribute:: NF.output_files_base_name

   Base name with absolute path of output files. It is 
   automatically generated from the
   :attr:`NF.output_folder <NF4HEP.NF.output_folder>` and
   :attr:`NF.name <NF4HEP.NF.name>` attributes.

      - **type**: ``str``

.. py:attribute:: NF.output_folder

   See :attr:`output_folder <common_classes_attributes.output_folder>`.

.. py:attribute:: NF.output_h5_file

   See :attr:`output_h5_file <common_classes_attributes.output_h5_file>`.

.. py:attribute:: NF.output_history_json_file

   Absolute path to the .json file where the :attr:`NF.history <NF4HEP.NF.history>` 
   attribute is saved (see the :meth:`NF.save_history_json <NF4HEP.NF.save_history_json>`
   method for details).
   It is automatically generated from the 
   :attr:`NF.output_files_base_name <NF4HEP.NF.output_files_base_name>` attribute.
         
      - **type**: ``str`` 

.. py:attribute:: NF.output_idx_h5_file

   Absolute path to the .h5 file where the data indices are saved (see the 
   :meth:`NF.save_data_indices <NF4HEP.NF.save_data_indices>`
   method for details).
   It is automatically generated from the 
   :attr:`NF.output_files_base_name <NF4HEP.NF.output_files_base_name>` attribute.
         
      - **type**: ``str`` 

.. py:attribute:: NF.output_json_file

   See :attr:`output_json_file <common_classes_attributes.output_json_file>`.

.. py:attribute:: NF.output_log_file

   See :attr:`output_log_file <common_classes_attributes.output_log_file>`.

.. py:attribute:: NF.output_predictions_h5_file

   Absolute path to the .json file where the :attr:`NF.predictions <NF4HEP.NF.predictions>` 
   attribute is saved (see the :meth:`NF.save_predictions_h5 <NF4HEP.NF.save_predictions_h5>`
   method for details).
   It is automatically generated from the 
   :attr:`NF.output_files_base_name <NF4HEP.NF.output_files_base_name>` attribute.
         
      - **type**: ``str`` 

.. py:attribute:: NF.output_predictions_json_file

   See :attr:`output_predictions_json_file <common_classes_attributes.output_predictions_json_file>`.

.. py:attribute:: NF.output_scalers_pickle_file

   Absolute path to the .pickle file where the data standard scalers are saved (see the 
   :meth:`NF.save_scalers <NF4HEP.NF.save_scalers>`
   method for details).
   It is automatically generated from the 
   :attr:`NF.output_files_base_name <NF4HEP.NF.output_files_base_name>` attribute.
         
      - **type**: ``str`` 

.. py:attribute:: NF.output_json_file

   Absolute path to the .json file where where part of the :class:`NF <NF4HEP.NF>` 
   object is saved (see the 
   :meth:`NF.save_json <NF4HEP.NF.save_json>`
   method for details).
   It is automatically generated from the 
   :attr:`NF.output_files_base_name <NF4HEP.NF.output_files_base_name>` attribute.
         
      - **type**: ``str`` 

.. py:attribute:: NF.output_tensorboard_log_dir

   When the callback |tensorboard_link| is present, this attribute represents the absolute path 
   to the folder where log files are saved.
   It is automatically generated from the 
   :attr:`NF.output_folder <NF4HEP.NF.output_folder>` attribute.
         
      - **type**: ``str`` or ``None``

.. py:attribute:: NF.output_tf_model_graph_pdf_file

   Absolute path to the .pdf file where the model graph is saved (see the 
   :meth:`NF.save_model_graph_pdf <NF4HEP.NF.save_model_graph_pdf>`
   method for details).
   It is automatically generated from the 
   :attr:`NF.output_files_base_name <NF4HEP.NF.output_files_base_name>` attribute.
         
      - **type**: ``str``

.. py:attribute:: NF.output_tf_model_h5_file

   Absolute path to the .h5 file where where the :attr:`NF.model <NF4HEP.NF.model>` 
   object is saved (see the 
   :meth:`NF.save_model_h5 <NF4HEP.NF.save_model_h5>`
   method for details).
   It is automatically generated from the 
   :attr:`NF.output_files_base_name <NF4HEP.NF.output_files_base_name>` attribute.
         
      - **type**: ``str`` 

.. py:attribute:: NF.output_tf_model_json_file

   Absolute path to the .json file where where the :attr:`NF.model <NF4HEP.NF.model>` 
   object is saved (see the 
   :meth:`NF.save_model_json <NF4HEP.NF.save_model_json>`
   method for details).
   It is automatically generated from the 
   :attr:`NF.output_files_base_name <NF4HEP.NF.output_files_base_name>` attribute.
         
      - **type**: ``str`` 

.. py:attribute:: NF.output_tf_model_onnx_file

   Absolute path to the .onnx file where where the :attr:`NF.model <NF4HEP.NF.model>` 
   object is saved in |onnx_link| format (see the 
   :meth:`NF.save_model_onnx <NF4HEP.NF.save_model_onnx>`
   method for details).
   It is automatically generated from the 
   :attr:`NF.output_files_base_name <NF4HEP.NF.output_files_base_name>` attribute.
         
      - **type**: ``str`` 

.. py:attribute:: NF.pars_bounds

   See :attr:`pars_bounds <common_classes_attributes.pars_bounds>`.

.. py:attribute:: NF.pars_bounds_train

   Bounds on the parameters computed from their minimum and maximum values in the training dataset.
   This attribute could be used to constrain the prediction of the DNNLikelihood within a region
   where ``X`` training data were available. See the :meth:`NF.model_predict <NF4HEP.NF.model_predict>`
   method for details.

      - **type**: ``numpy.ndarray``
      - **shape**: ``(ndims,2)``

.. py:attribute:: NF.pars_central   

   See :attr:`pars_central <common_classes_attributes.pars_central>`.

.. py:attribute:: NF.pars_labels

   See :attr:`pars_labels <common_classes_attributes.pars_labels>`.

.. py:attribute:: NF.pars_labels_auto   

   See :attr:`pars_labels_auto <common_classes_attributes.pars_labels_auto>`.

.. py:attribute:: NF.pars_pos_nuis

   See :attr:`pars_pos_nuis <common_classes_attributes.pars_pos_nuis>`.

.. py:attribute:: NF.pars_pos_poi

   See :attr:`pars_pos_poi <common_classes_attributes.pars_pos_poi>`.

.. py:attribute:: NF.pred_bounds_train

   Maximum and minimum value of ``Y`` in training data.
   This attribute could be used to constrain the prediction of the DNNLikelihood within a region
   where ``Y`` training data were available. See the :meth:`NF.model_predict <NF4HEP.NF.model_predict>`
   method for details.

      - **type**: ``numpy.ndarray``
      - **shape**: ``(2,)``

.. py:attribute:: NF.predictions

   Nested dictionary containing all predictions generated by the 
   :meth:`NF.model_compute_predictions <NF4HEP.NF.model_compute_predictions>`
   method. The contento of this dictionary is saved into the file
   :attr:`NF.output_predictions_h5_file <NF4HEP.NF.output_predictions_h5_file>`.

      - **type**: ``dict`` with the following structure:
.. 
..          - *"HPDI"* (value type: ``dict``)
..             Highest posterior density intervals (HPDI) for the parameters. This dictionary has the following structure:
.. 
..             - *"par"* (value type: ``dict``)
..                Dictionary corresponding to parameter par. This dictionary has the following structure:
.. 
..                - *"pred"* (value type: ``dict``)
..                   Dictionary corresponding to the prediction computed reweighting distributions using the DNN prediction. 
..                   Has the same structure as the item "true".
..                - *"true"* (value type: ``dict``)
..                   Dictionary corresponding to the prediction computed using the original data. 
..                   This dictionary has the following structure:
.. 
..                   - *"test"* (value type: ``dict``)
..                      Dictionary corresponding to the prediction computed from test data.
..                      Has the same structure as the items "train" and "val".
..                   - *"train"* (value type: ``dict``)
..                      Dictionary corresponding to the prediction computed from test data. 
..                      Has the same structure as the items "test" and "val".
..                   - *"val"* (value type: ``dict``)
..                      Dictionary corresponding to the prediction computed from test data. 
..                      This dictionary has the following structure:
.. 
..                      - *"Probability"* (value type: ``dict``)
..                         Dictionary with HPDI corresponding to the interval ``interval``. Default intervals for the
..                         :meth:`NF.model_compute_predictions <NF4HEP.NF.model_compute_predictions>`
..                         method are ``"0.5"``, ``"0.6826894921370859"``, ``"0.9544997361036416"``, and ``"0.9973002039367398"``.
..                         The latter three correspond to 1,2,3 standard deviations for a 1D normal distribution (obtained through the
..                         :func:`inference.CI_from_sigma <NF4HEP.inference.CI_from_sigma>` function).
..                         This dictionary has the following structure:
.. 
..                         - *"Bin width"* (value type: ``float``)
..                            Width of the binning used to compute the HPDI (see the 
..                            :func:`inference.HPDI <NF4HEP.inference.HPDI>`
..                            function for details). It gives an estimate of the uncertainty due to the algorithm for computing
..                            HPDI.
..                         - *"Intervals"* (value type: ``list`` of ``list``)
..                            List of intervals of parameter values for the given probability.
..                         - *"Number of bins"* (value type: ``int``)
..                            Number of bins used for the computation of HPDI (see the 
..                            :func:`inference.HPDI <NF4HEP.inference.HPDI>`
..                            function for details).
..                         - *"Probability"* (value type: ``float``)
..                            Probability interval.
.. 
..          - *"HPDI_error"* (value type: ``dict``)
..             Errors on the HPDI quantified as differences (absolute and relative) between the predictions of HPDI computed
..             with input data and with DNN reweight. This dictionary has the following structure:
.. 
..             - *"par"* (value type: ``dict``)
..                Dictionary corresponding to parameter par. This dictionary has the following structure:
.. 
..                - *"test"* (value type: ``dict``)
..                   Dictionary corresponding to the prediction computed from test data.
..                   Has the same structure as the items "train" and "val".
..                - *"train"* (value type: ``dict``)
..                   Dictionary corresponding to the prediction computed from test data. 
..                   Has the same structure as the items "test" and "val".
..                - *"val"* (value type: ``dict``)
..                   Dictionary corresponding to the prediction computed from test data. 
..                   This dictionary has the following structure:
.. 
..                   - *"Probability"* (value type: ``dict``)
..                      Dictionary with HPDI corresponding to the interval ``interval``. Default intervals for the
..                      :meth:`NF.model_compute_predictions <NF4HEP.NF.model_compute_predictions>`
..                      method are ``"0.5"``, ``"0.6826894921370859"``, ``"0.9544997361036416"``, and ``"0.9973002039367398"``.
..                      The latter three correspond to 1,2,3 standard deviations for a 1D normal distribution (obtained through the
..                      :func:`inference.CI_from_sigma <NF4HEP.inference.CI_from_sigma>` function).
..                      This dictionary has the following structure:
.. 
..                      - *"Absolute error"* (value type: ``list`` of ``list``)
..                         List of absolute errors (computed as ``true-pred``) for each boundary of the intervals (see the 
..                         :func:`inference.HPDI_error <NF4HEP.inference.HPDI_error>`
..                         function for details)
..                      - *"Probability"* (value type: ``float``)
..                         Probability interval.
..                      - *"Relative error"* (value type: ``list`` of ``list``)
..                         List of relative errors (computed as ``(true-pred)/true``) for each boundary of the intervals (see the 
..                         :func:`inference.HPDI_error <NF4HEP.inference.HPDI_error>`
..                         function for details)
.. 
..          - *"KS"* (value type: ``dict``)
..             Weighted Kolmogorov-Smirnov test statistics and p-values for two sample distributions for each parameter
..             (see the :func:`inference.ks_w <NF4HEP.inference.ks_w>` function for details).
..             This dictionary has the following structure:
.. 
..             - *"Test vs pred on train"* (value type: ``list``, shape: ``(ndims,2)``)
..                List of KS test statistics and p-value for each parameter for two sample KS test between input test data and
..                DNN reweighted prediction on train data.
..             - *"Test vs pred on val"* (value type: ``list``, shape: ``(ndims,2)``)
..                List of KS test statistics and p-value for each parameter for two sample KS test between input test data and
..                DNN reweighted prediction on validation data.
..             - *"Train vs pred on train"* (value type: ``list``, shape: ``(ndims,2)``)
..                List of KS test statistics and p-value for each parameter for two sample KS test between input train data and
..                DNN reweighted prediction on train data.
..             - *"Val vs pred on test"* (value type: ``list``, shape: ``(ndims,2)``)
..                List of KS test statistics and p-value for each parameter for two sample KS test between input validation data and
..                DNN reweighted prediction on test data.
.. 
..          - *"KS medians"* (value type: ``dict``)
..             Median over parameters of the KS p-values for two sample distributions.
..             This dictionary has the following structure:
.. 
..             - *"Test vs pred on train"* (value type: ``float``)
..                Median over parameters of the p-values for two sample KS test between input test data and
..                DNN reweighted prediction on train data.
..             - *"Test vs pred on val"* (value type: ``float``)
..                Median over parameters of the p-values for two sample KS test between input test data and
..                DNN reweighted prediction on validation data.
..             - *"Train vs pred on train"* (value type: ``float``)
..                Median over parameters of the p-values for two sample KS test between input train data and
..                DNN reweighted prediction on train data.
..             - *"Val vs pred on test"* (value type: ``float``)
..                Median over parameters of the p-values for two sample KS test between input validation data and
..                DNN reweighted prediction on test data.
.. 
..          - *"Metrics on scaled data"* (value type: ``dict``)
..             Value of all metrics (loss and metrics) evaluated on scaled data for train/val/test data.
..             This dictionary has the following structure:
.. 
..             - *"metric_best"* (value type: ``float``)
..                Value of each metric evaluated on train data.
..                This key is present for all metric with the corresponding full name of the metric, such as,
..                for instance, ``"mean_squared_error_best", etc. Metrics include ``"loss"``.
..             - *"test_metric_best"* (value type: ``float``)
..                Value of each metric evaluated on test data.
..                This key is present for all metric with the corresponding full name of the metric, such as,
..                for instance, ``"test_mean_squared_error_best", etc. Metrics include ``"loss"``.
..             - *"val_metric_best"* (value type: ``float``)
..                Value of each metric evaluated on validation data.
..                This key is present for all metric with the corresponding full name of the metric, such as,
..                for instance, ``"val_mean_squared_error_best", etc. Metrics include ``"loss"``.
..       
..          - *"Metrics on unscaled data"* (value type: ``dict``)
..             Value of all metrics (loss and metrics) evaluated on original data for train/val/test data.
..             This dictionary has the same structure of the previous one with all keys having the suffix "_unscaled".
.. 
..          - *"Prediction time"* (value type: ``float``)
..             Averege prediction time per point in second with batch size equal to 
..             :attr:`NF.batch_size <NF4HEP.NF.batch_size>`. 

      - **schematic example**:

         .. code-block:: python

            {'HPDI': {'par_1': {'pred': {'test': {'prob_1': {'Bin width': float,
                                                             'Intervals': list,
                                                             'Number of bins': int,
                                                             'Probability': float
                                                            },
                                                  ...,
                                                  {'prob_n': ...
                                                  },
                                         'train': ...,
                                         'val': ...
                                        },
                                'true': ...},
                      ...,
                      'par_n': ...
                     },
             'HPDI_error': {'par_1': {'test': {'prob_1': {'Absolute error': list,
                                                          'Probability': float,
                                                          'Relative error': list
                                                         },
                                               ...,
                                               {'prob_n': ...
                                               },
                                      'train': ...,
                                      'val': '...
                                     },
                            ...,
                            'par_n': ...
                           },
             'KS': {'Test vs pred on train': list,
                    'Test vs pred on val': list,
                    'Train vs pred on train': list,
                    'Val vs pred on test': list},
             'KS medians': {'Test vs pred on train': float,
                            'Test vs pred on val': float,
                            'Train vs pred on train': float,
                            'Val vs pred on test': float},
             'Metrics on scaled data': {'loss_best': float,
                                        'mean_absolute_error_best': float,
                                        ...
                                       },
             'Metrics on unscaled data': {'loss_best_unscaled': float,
                                          'mean_absolute_error_best_unscaled': float,
                                          ...
                                         },
             'Prediction time': float}

.. py:attribute:: NF.same_data

   Attribute corresponding to the input argument :argument:`same_data`. 
   If ``True``, all the :class:`NF <NF4HEP.NF>` objects part of a  
   :class:`DnnLikEnsemble <NF4HEP.NFEnsemble>` one will share the same data.
   If false data will be generated differently for each :class:`NF <NF4HEP.NF>` object.
   When the :attr:`NF.standalone <NF4HEP.NF.standalone>` is ``True``, then this attribute
   is ``True`` by default and is not used.

      - **type**: ``bool``

.. py:attribute:: NF.scalerX

   |standard_scaler_link| for X data points fit to the training data 
   :attr:`NF.X_train <NF4HEP.NF.X_train>`. Standard scalers are always defined 
   together with training data. Whenever the 
   :attr:`NF.scalerX_bool <NF4HEP.NF.scalerX_bool>` attribute is ``False``
   the scaler is just a representation of the identity matrix.

      - **type**: |standard_scaler_link| object

.. py:attribute:: NF.scalerX_bool

   If ``True`` then the |standard_scaler_link| for the X data points is fit to the training data
   :attr:`NF.X_train <NF4HEP.NF.X_train>`, otherwise it is set to the identity matrix.
   It is set by the 
   :meth:`NF.__set_model_hyperparameters <NF4HEP.NF._NF__set_model_hyperparameters>` 
   method by parsing the item ``"scalerX"`` of the 
   :attr:`NF.__model_data_inputs <NF4HEP.NF._NF__model_data_inputs>`
   dictionary.

      - **type**: ``bool``

.. py:attribute:: NF.scalerY

   |standard_scaler_link| for Y data points fit to the training data 
   :attr:`NF.Y_train <NF4HEP.NF.X_train>`. Standard scalers are always defined 
   together with training data. Whenever the 
   :attr:`NF.scalerY_bool <NF4HEP.NF.scalerY_bool>` attribute is ``False``
   the scaler is just a representation of the identity matrix.

      - **type**: |standard_scaler_link| object

.. py:attribute:: NF.scalerY_bool

   If ``True`` then the |standard_scaler_link| for the Y data points is fit to the training data
   :attr:`NF.Y_train <NF4HEP.NF.Y_train>`, otherwise it is set to the identity matrix.
   It is set by the 
   :meth:`NF.__set_model_hyperparameters <NF4HEP.NF._NF__set_model_hyperparameters>` 
   method by parsing the item ``"scalerY"`` of the 
   :attr:`NF.__model_data_inputs <NF4HEP.NF._NF__model_data_inputs>`
   dictionary.

      - **type**: ``bool``

.. py:attribute:: NF.seed

   Attribute corresponding to the input argument :argument:`seed`. If the latter is ``None`` then the attribute value is set
   to ``1``. It is used by the :meth:`NF.__set_seed <NF4HEP.NF._NF__set_seed>`
   method to initialize the |numpy_link| and |tf_link| random state.

      - **type**: ``int``

.. py:attribute:: NF.standalone

   If ``True`` the :class:`NF <NF4HEP.NF>` object is not part of a
   :class:`DnnLikEnsemble <NF4HEP.NFEnsemble>` one.
   It is set by the 
   :meth:`NF.__check_define_ensemble_folder <NF4HEP.NF._NF__check_define_ensemble_folder>`
   method based on the value of the :attr:`NF.ensemble_name <NF4HEP.NF.ensemble_name>`
   attribute. If the latter is ``None`` then the attribute is set to ``True``.

      - **type**: ``bool``

.. py:attribute:: NF.verbose

   See :attr:`verbose <common_classes_attributes.verbose>`.

.. py:attribute:: NF.W_train

   |Numpy_link| array with the weights of the training points computed by the 
   :meth:`NF.compute_sample_weights <NF4HEP.NF.compute_sample_weights>` method.
   If :attr:`NF.weighted <NF4HEP.NF.weighted>`
   is ``True`` then the weights are computed automatically when generating training data by calling the
   :meth:`NF.compute_sample_weights <NF4HEP.NF.compute_sample_weights>` method
   with default parameters. Alternatively, the latter method can be called manually with custom arguments.

      - **type**: ``numpy.ndarray``
      - **shape**: ``(npoints_train,)``

.. py:attribute:: NF.weighted

   If ``True`` then, when generating training data, weights for the X data points are computed automatically 
   (with default parameters) by the 
   :meth:`NF.compute_sample_weights <NF4HEP.NF.compute_sample_weights>` method.
   It is set by the 
   :meth:`NF.__set_model_hyperparameters <NF4HEP.NF._NF__set_model_hyperparameters>` 
   method by parsing the item ``"weighted"`` of the 
   :attr:`NF.__model_data_inputs <NF4HEP.NF._NF__model_data_inputs>`
   dictionary.
   If ``True`` the :meth:`NF.model_train <NF4HEP.NF.model_train>` method calls the
   |tf_keras_model_fit_link| method with argument ``sample_weight`` equal to 
   :attr:`NF.W_train <NF4HEP.NF.W_train>`.

      - **type**: ``bool``

.. py:attribute:: NF.X_test

   |Numpy_link| array with X test data points generated by the 
   :meth:`NF.generate_test_data <NF4HEP.NF.generate_test_data>` method.

      - **type**: ``numpy.ndarray``
      - **shape**: ``(npoints_test,ndim)`` 

.. py:attribute:: NF.X_train

   |Numpy_link| array with X train data points generated by the 
   :meth:`NF.generate_train_data <NF4HEP.NF.generate_train_data>` method.

      - **type**: ``numpy.ndarray``
      - **shape**: ``(npoints_train,ndim)`` 

.. py:attribute:: NF.X_val

   |Numpy_link| array with X validation data points generated by the 
   :meth:`NF.generate_train_data <NF4HEP.NF.generate_train_data>` method.

      - **type**: ``numpy.ndarray``
      - **shape**: ``(npoints_val,ndim)`` 

.. py:attribute:: NF.Y_test

   |Numpy_link| array with Y test data points generated by the 
   :meth:`NF.generate_test_data <NF4HEP.NF.generate_test_data>` method.

      - **type**: ``numpy.ndarray``
      - **shape**: ``(npoints_test,)`` 

.. py:attribute:: NF.Y_train

   |Numpy_link| array with Y train data points generated by the 
   :meth:`NF.generate_train_data <NF4HEP.NF.generate_train_data>` method.

      - **type**: ``numpy.ndarray``
      - **shape**: ``(npoints_train,)`` 

.. py:attribute:: NF.Y_val

   |Numpy_link| array with Y validation data points generated by the 
   :meth:`NF.generate_train_data <NF4HEP.NF.generate_train_data>` method.

      - **type**: ``numpy.ndarray``
      - **shape**: ``(npoints_val,)`` 

.. include:: ../external_links.rst