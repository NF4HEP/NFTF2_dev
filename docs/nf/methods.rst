.. _DnnLik_methods:

Methods
"""""""

.. currentmodule:: DNNLikelihood

.. automethod:: NF.__init__

.. automethod:: NF._NF__set_resources

.. automethod:: NF._NF__check_define_input_files

.. automethod:: NF._NF__check_define_output_files

.. automethod:: NF._NF__check_define_name

.. automethod:: NF._NF__check_npoints

.. automethod:: NF._NF__check_define_model_data_inputs

.. automethod:: NF._NF__check_define_model_define_inputs

.. automethod:: NF._NF__check_define_model_compile_inputs

.. automethod:: NF._NF__check_define_model_train_inputs

.. automethod:: NF._NF__check_define_ensemble_folder

.. automethod:: NF._NF__set_seed

.. automethod:: NF._NF__set_dtype

.. automethod:: NF._NF__set_data

.. automethod:: NF._NF__set_pars_info

.. automethod:: NF._NF__set_model_hyperparameters

.. automethod:: NF._NF__set_tf_objects

.. automethod:: NF._NF__load_json_and_log

.. automethod:: NF._NF__load_history

.. automethod:: NF._NF__load_model

.. automethod:: NF._NF__load_scalers

.. automethod:: NF._NF__load_data_indices

.. automethod:: NF._NF__load_predictions

.. automethod:: NF._NF__set_optimizer

.. automethod:: NF._NF__set_loss

.. automethod:: NF._NF__set_metrics

.. automethod:: NF._NF__set_callbacks

.. automethod:: NF._NF__set_epochs_to_run

.. automethod:: NF._NF__set_pars_labels

.. automethod:: NF.compute_sample_weights

.. automethod:: NF.define_rotation

.. automethod:: NF.define_scalers

.. automethod:: NF.generate_train_data

.. automethod:: NF.generate_test_data

.. automethod:: NF.model_define

.. automethod:: NF.model_compile

.. automethod:: NF.model_build

.. automethod:: NF.model_train

.. automethod:: NF.model_predict

.. automethod:: NF.model_predict_scalar

.. automethod:: NF.compute_maximum_model

.. automethod:: NF.compute_profiled_maximum_model

.. automethod:: NF.model_evaluate

.. automethod:: NF.generate_fig_base_title

.. automethod:: NF.update_figures

.. automethod:: NF.plot_training_history

.. automethod:: NF.plot_pars_coverage

.. automethod:: NF.plot_lik_distribution

.. automethod:: NF.plot_corners_1samp

.. automethod:: NF.plot_corners_2samp

.. automethod:: NF.model_compute_predictions

.. automethod:: NF.save_log

.. automethod:: NF.save_data_indices

.. automethod:: NF.save_model_json

.. automethod:: NF.save_model_h5

.. automethod:: NF.save_model_onnx

.. automethod:: NF.save_history_json

.. automethod:: NF.save_json

.. automethod:: NF.generate_summary_text

.. automethod:: NF.save_predictions_h5

.. automethod:: NF.save_scalers

.. automethod:: NF.save_model_graph_pdf

.. automethod:: NF.save

.. automethod:: NF.show_figures

.. py:method:: NF4HEP.NF.get_available_gpus

   Method inherited from the :class:`Resources <NF4HEP.Resources>` object.
   See the documentation of :meth:`Resources.get_available_gpus <NF4HEP.Resources.get_available_gpus>`.

.. py:method:: NF4HEP.NF.get_available_cpu

   Method inherited from the :class:`Resources <NF4HEP.Resources>` object.
   See the documentation of :meth:`Resources.get_available_cpu <NF4HEP.Resources.get_available_cpu>`.

.. py:method:: NF4HEP.NF.set_gpus

   Method inherited from the :class:`Resources <NF4HEP.Resources>` object.
   See the documentation of :meth:`Resources.set_gpus <NF4HEP.Resources.set_gpus>`.
   
.. py:method:: NF4HEP.NF.set_gpus_env

   Method inherited from the :class:`Resources <NF4HEP.Resources>` object.
   See the documentation of :meth:`Resources.set_gpus_env <NF4HEP.Resources.set_gpus_env>`.

.. py:method:: NF4HEP.NF.set_verbosity

   Method inherited from the :class:`Verbosity <NF4HEP.Verbosity>` object.
   See the documentation of :meth:`Verbosity.set_verbosity <NF4HEP.Verbosity.set_verbosity>`.

.. include:: ../external_links.rst