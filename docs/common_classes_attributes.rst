.. _common_classes_attributes:

.. currentmodule:: common_classes_attributes

Common classes attributes
-------------------------

.. py:attribute:: input_file
     
   Absolute path corresponding to the input argument 
   :argument:`input_file <common_classes_arguments.input_file>`.
   Whenever this attribute is not ``None`` the corresponding object
   is reconstructed from input files. See the specific objects ``__init__`` methods for details.

      - **type**: ``str`` or ``None``

.. py:attribute:: input_folder

   Absolute path corresponding to the folder containing the
   :attr:`input_file <common_classes_attributes.input_file>` file.
   It is automatically generated from the corresponding
   :attr:`input_file <common_classes_attributes.input_file>`.
   When the latter attribute is ``None``, then it is set to ``None``.

      - **type**: ``str`` or ``None``

.. py:attribute:: input_h5_file

   Absolute path to the .h5 file containing a saved object (see the dedicated save 
   method of the different objects for details).
   It is automatically generated from the
   :attr:`input_file <common_classes_attributes.input_file>` attribute.
   When the latter attribute is ``None``, then it is set to ``None``.
         
      - **type**: ``str`` or ``None``

.. py:attribute:: input_log_file

   Absolute path to the .log file containing a saved object log (see the dedicated save 
   methods of the different objects for details).
   It is automatically generated from the
   :attr:`input_file <common_classes_attributes.input_file>` attribute.
   When the latter attribute is ``None``, then it is set to ``None``.
         
      - **type**: ``str`` or ``None``

.. py:attribute:: log

   Dictionary containing a log of the calls to the object methods (creation, calculations, I/O, etc.). 
   The dictionary has datetime strings as keys and actions as values. 
   Actions are dictionaries, containing details of the methods calls.
         
      - **type**: ``dict``
      - **keys**: ``datetime.now().strftime("%Y-%m-%d-%H-%M-%S.%fZ")[:-3]``
      - **values**: ``dict`` with the following structure:

         - *"action"* (value type: ``str``)
            Short description of the action.
         - other items representing details of the corresponding action

.. py:attribute:: name

   Attribute corresponding to the input argument :argument:`name` (when present) or automatically
   generated from the object ``__init__``  method. The attribute is set by the ``__check_define_name`` private method of each class.
   In general, the :attr:`name <common_classes_attributes.name>` attribute is generated with the following
   scheme:

   +---------------------------------------+------------------------------------------------------------------+----------------------------------------------------------+
   | Object                                | Input argument :argument:`name <common_classes_arguments.name>`  | Attribute :attr:`name <common_classes_attributes.name>`  |
   +=======================================+==================================================================+==========================================================+
   | :mod:`Histfactory <histfactory>`      | ``None``                                                         | "model_"+<timestamp>+"_histfactory"                      |
   |                                       +------------------------------------------------------------------+----------------------------------------------------------+
   |                                       | <str>                                                            | <str>+"_histfactory"                                     |
   +---------------------------------------+------------------------------------------------------------------+----------------------------------------------------------+
   | :mod:`Likelihood <likelihood>`        | ``None``                                                         | "model_"+<timestamp>+"_likelihood"                       |
   |                                       +------------------------------------------------------------------+----------------------------------------------------------+
   |                                       | <str>                                                            | <str>+"_likelihood"                                      |
   +---------------------------------------+------------------------------------------------------------------+----------------------------------------------------------+
   | :mod:`Sampler <sampler>`              | not available                                                    | same name as input :class:`Lik <NF4HEP.Lik>`      |
   |                                       |                                                                  |                                                          |
   |                                       |                                                                  | with "_likelihood" suffix replaced by "_sampler"         |
   +---------------------------------------+------------------------------------------------------------------+----------------------------------------------------------+
   | :mod:`Data <data>`                    | ``None``                                                         | "model_"+<timestamp>+"_data"                             |
   |                                       +------------------------------------------------------------------+----------------------------------------------------------+
   |                                       | <str>                                                            | <str>+"_data"                                            |
   +---------------------------------------+------------------------------------------------------------------+----------------------------------------------------------+
   | :mod:`DNNLikelihood <dnn_likelihood>` | ``None``                                                         | "model_"+<timestamp>+"_dnnlikelihood"                    |
   |                                       +------------------------------------------------------------------+----------------------------------------------------------+
   |                                       | <str>                                                            | <str>+"_dnnlikelihood"                                   |
   +---------------------------------------+------------------------------------------------------------------+----------------------------------------------------------+

   where <timestamp> is the string ``datetime.now().strftime("%Y-%m-%d-%H-%M-%S.%fZ")[:-3]`` 
   and suffixes are appended preventing duplication.
   
      - **type**: ``str``


.. py:attribute:: output_folder

   Absolute path corresponding to the input argument
   :argument:`output_folder <common_classes_arguments.output_folder>`. 
   If the latter is ``None``, then it is set to
   the code execution folder. This is the folder where the object is saved
   (see the ``save`` method of the different classes for details).
   If the folder does not exist it is created
   by the :func:`utils.check_create_folder <NF4HEP.utils.check_create_folder>`
   function.

      - **type**: ``str``

.. py:attribute:: output_h5_file

   Absolute path to the .h5 file where the
   object is saved (see the ``save_h5`` method of the different classes for details).
   It is automatically generated from the attribute
   :attr:`output_folder <common_classes_attributes.output_folder>`.
         
      - **type**: ``str`` 

.. py:attribute:: output_json_file

   Absolute path to the .json file where the
   object is saved (see the ``save_json`` method of the different classes for details).
   It is automatically generated from the attribute
   :attr:`output_folder <common_classes_attributes.output_folder>`.
         
      - **type**: ``str`` 

.. py:attribute:: output_log_file

   Absolute path to the .log file where the object log stored in the 
   :attr:`log <common_classes_attributes.log>` attribute
   is saved (see the ``save_log`` method of the different classes for details).
   It is automatically generated from the attribute
   :attr:`output_folder <common_classes_attributes.output_folder>`.
         
      - **type**: ``str`` 

.. py:attribute:: output_predictions_json_file

   Absolute path to the .json file where the 
   ``predictions`` dictionary, containing predictions for the corresponding object is saved.
   It is automatically generated from the
   :attr:`output_folder <common_classes_attributes.output_folder> and
   :attr:`name <common_classes_attributes.name> attributes.

   Notice that the file is only saved to provide human readable predictions. Indeed, the
   ``predictions`` dictionary is saved and restored together with the other attributes,
   in the :attr:`output_h5_file <common_classes_attributes.output_h5_file> file.

      - **type**: ``str``

.. py:attribute:: verbose

   Attribute corresponding to the input argument 
   :argument:`verbose <common_classes_arguments.verbose>`.
   It represents verbosity mode of the 
    ``__init__`` method and the default verbosity mode of all class methods 
   that accept a ``verbose`` argument.
   See :ref:`Verbosity mode <verbosity_mode>` for more details.
 
      - **type**: ``bool`` or ``int``

.. py:attribute:: pars_bounds   

   |Numpy_link| array containing the parameters bounds.
   For the 
      
      - :mod:`Likelihood <likelihood>`
      - :mod:`Data <data>`

   objects it is set from the corresponding input argument 
   :argument:`pars_bounds <common_classes_arguments.pars_bounds>`. If the input argument is ``None``
   then bounds for all parameters are set to ``[-np.inf,np.inf]``.
   For the objects

        - :mod:`Sampler <sampler>`
        - :mod:`DNNLikelihood <dnn_likelihood>`

   it is automatically set from the corresponding attribute of the :mod:`Likelihood <likelihood>`
   and :mod:`Data <data>` objects that are inherited by these classes respectively.

   It is used by various methods to ensure/check that parameters lie within specified bounds.

      - **type**: ``numpy.ndarray``
      - **shape**: ``(ndims,2)``

.. py:attribute:: pars_central   

   |Numpy_link| array containing central values of the parameters.
   For the 
      
      - :mod:`Likelihood <likelihood>`
      - :mod:`Data <data>`

   objects it is set from the corresponding input argument 
   :argument:`pars_central <common_classes_arguments.pars_central>`. 
   If the input argument is ``None``
   then central values for all parameters are set to ``0``.
   For the objects

        - :mod:`Sampler <sampler>`
        - :mod:`DNNLikelihood <dnn_likelihood>`

   it is automatically set from the corresponding attribute of the :mod:`Likelihood <likelihood>`
   and :mod:`Data <data>` objects that are inherited by these classes respectively.
   
   It is used as initialization by several methods to perform MCMC and optimization.
       
      - **type**: ``numpy.ndarray``
      - **shape**: ``(ndims,)``

.. py:attribute:: pars_labels   

   List containing parameters names as strings.
   For the 
      
      - :mod:`Likelihood <likelihood>`
      - :mod:`Data <data>`

   objects it is set from the corresponding input argument 
   :argument:`pars_central <common_classes_arguments.pars_central>`. 
   If the input argument is ``None`` then
   it is set equal to the automatically
   generated :attr:`pars_labels_auto <common_classes_attributes.pars_labels_auto>`
   For the objects

        - :mod:`Sampler <sampler>`
        - :mod:`DNNLikelihood <dnn_likelihood>`

   it is automatically set from the corresponding attribute of the :mod:`Likelihood <likelihood>`
   and :mod:`Data <data>` objects that are inherited by these classes respectively.

   Parameters labels are always parsed as "raw" strings (like, for instance, ``r"%s"%pars_labels[0]``) 
   and can contain latex expressions that are properly compiled when making plots.
   
      - **type**: ``list``
      - **shape**: ``(ndims,)``

.. py:attribute:: pars_labels_auto   

   List containing parameters names automatically generated by the function
   :func:`utils.define_pars_labels_auto <NF4HEP.utils.define_pars_labels_auto>`.
   All parameters of interest are named ``r"$\theta_{i}$"`` with ``i`` ranging between
   one to the number of parameters of interest and all nuisance parameters are named
   ``r"$\nu_{j}$"`` with ``j`` ranging between one to the number of nuisance parameters.
   
   Parameters labels are always parsed as "raw" strings (like, for instance, ``r"%s"%pars_labels_auto[0]``) 
   and can contain latex expressions that are properly compiled when making plots.

      - **type**: ``list``
      - **shape**: ``(ndims,)``

.. py:attribute:: pars_pos_nuis   

   |Numpy_link| array containing the positions in the parameters list of the nuisance parameters.
   For the 
      
      - :mod:`Likelihood <likelihood>`
      - :mod:`Data <data>`

   objects it is set from the corresponding input argument 
   :argument:`pars_pos_nuis <common_classes_arguments.pars_pos_nuis>`. 
   For the objects

        - :mod:`Sampler <sampler>`
        - :mod:`DNNLikelihood <dnn_likelihood>`

   it is automatically set from the corresponding attribute of the :mod:`Likelihood <likelihood>`
   and :mod:`Data <data>` objects that are inherited by these classes respectively.

      - **type**: ``list`` or ```numpy.ndarray``
      - **shape**: ``(n_nuis,)``

.. py:attribute:: pars_pos_poi   

   |Numpy_link| array containing the positions in the parameters list of the parameters of interest.
   For the 
      
      - :mod:`Likelihood <likelihood>`
      - :mod:`Data <data>`

   objects it is set from the corresponding input argument 
   :argument:`pars_pos_poi <common_classes_arguments.pars_pos_poi>`. 
   For the objects

        - :mod:`Sampler <sampler>`
        - :mod:`DNNLikelihood <dnn_likelihood>`

   it is automatically set from the corresponding attribute of the :mod:`Likelihood <likelihood>`
   and :mod:`Data <data>` objects that are inherited by these classes respectively.

      - **type**: ``list`` or ```numpy.ndarray``
      - **shape**: ``(n_nuis,)``

.. py:attribute:: ndims

   Number of parameters on which the likelihood (or logpdf) depends, i.e. number of dimensions
   of the input vector X. It is automatically generated as explained in the following table:

   +---------------------------------------+-------------------------------------------------------------------------------------------+----------------------------------------------------------+
   | Object                                | Method                                                                                    | Details                                                  |
   +=======================================+===========================================================================================+==========================================================+
   | :mod:`Likelihood <likelihood>`        | :meth:`Lik.__check_define_ndims <NF4HEP.Lik._Lik__check_define_ndims>`             | The output of the logpdf  function is evaluated by       |
   |                                       |                                                                                           | changing the dimension of the input vector,              |
   |                                       |                                                                                           | starting from one, until no error is raised.             |
   +---------------------------------------+-------------------------------------------------------------------------------------------+----------------------------------------------------------+
   | :mod:`Sampler <sampler>`              | :meth:`Sampler.__init_likelihood <NF4HEP.Sampler._Sampler__init_likelihood>`       | It is set equal to the corresponding attribute           |
   |                                       |                                                                                           | of the :class:`Lik <NF4HEP.Lik>` class.           |
   |                                       |                                                                                           |                                                          |
   +---------------------------------------+-------------------------------------------------------------------------------------------+----------------------------------------------------------+
   | :mod:`Data <data>`                    | :meth:`Data.__init__ <NF4HEP.Data.__init__>`                                       | It is set equal to the length of the first entry of the  |
   |                                       |                                                                                           | :attr:`Data.data_X <NF4HEP.Data.data_X>`          |
   |                                       |                                                                                           | attribute.                                               |
   +---------------------------------------+-------------------------------------------------------------------------------------------+----------------------------------------------------------+
   | :mod:`DNNLikelihood <dnn_likelihood>` | :meth:`NF.__set_data <NF4HEP.NF._DNNLik__set_data>`                        | It is set equal to the corresponding attribute           |
   |                                       |                                                                                           | of the :class:`Data <NF4HEP.Data>` class.         |
   |                                       |                                                                                           |                                                          |
   +---------------------------------------+-------------------------------------------------------------------------------------------+----------------------------------------------------------+
       
      - **type**: ``int`` 


.. py:attribute:: output_figures_base_file_name

   Base name of the files to which figures are saved.
   It is automatically generated by appending the suffix "_figure" to the corresponding 
   :attr:`name <common_classes_attributes.name>` attribute.

      - **type**: ``str``

.. py:attribute:: output_figures_base_file_path

   Base name of the files to which figures are saved, including absolute path. 
   It is automatically generated from the corresponding 
   :attr:`output_figures_base_file_name <common_classes_attributes.output_figures_base_file_name>` and 
   :attr:`output_figures_folder <common_classes_attributes.output_figures_folder>` attributes.

     - **type**: ``str`` 

.. py:attribute:: output_figures_folder

   Absolute path to the folder where figures are saved. It is 
   automatically set to a subfolder, named "figures", of the folder corresponding to the
   :attr:`output_folder <common_classes_attributes.output_folder>` attribute.

      - **type**: ``str`` 
