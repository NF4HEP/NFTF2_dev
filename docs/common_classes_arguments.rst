.. _common_classes_arguments:

.. currentmodule:: common_classes_arguments

Common classes arguments
------------------------

.. argument:: name
     
    Name of the object.
    It is used to build the corresponding attribute in the objects:

        - :mod:`Histfactory <histfactory>`
        - :mod:`Likelihood <likelihood>`
        - :mod:`Data <data>`
        - :mod:`DNNLikelihood <dnn_likelihood>`
     
        - **type**: ``str`` or ``None``
        - **default**: ``None``

.. argument:: input_file

    File name (either relative to the code execution folder or absolute, with or without extension) 
    of a saved object. 
    It is used to set the corresponding attribute with the following scheme:

    +---------------------------------------+----------------------------------+
    | Object                                | File name scheme                 |
    +=======================================+==================================+
    | :mod:`Histfactory <histfactory>`      | <object_name>_histfactory.h5     |
    +---------------------------------------+----------------------------------+
    | :mod:`Likelihood <likelihood>`        | <object_name>_likelihood.h5      |
    +---------------------------------------+----------------------------------+
    | :mod:`Sampler <sampler>`              | <object_name>_sampler.h5         |
    +---------------------------------------+----------------------------------+
    | :mod:`Data <data>`                    | <object_name>_data.h5            |
    +---------------------------------------+----------------------------------+
    | :mod:`DNNLikelihood <dnn_likelihood>` | <object_name>_dnnlikelihood.json |
    +---------------------------------------+----------------------------------+

    where the object name is stored in the corresponding attribute.

.. argument:: output_folder
     
    Path (either relative to the code execution folder or absolute) where output files are saved.
    It is used to set the corresponding attribute in the objects:

        - :mod:`Histfactory <histfactory>`
        - :mod:`Likelihood <likelihood>`
        - :mod:`Data <data>`
        - :mod:`Sampler <sampler>`
        - :mod:`DNNLikelihood <dnn_likelihood>`

    In the case of the :class:`Sampler <NF4HEP.Sampler>` object the argument is also used 
    when importing an existing object. In this case if it is ``None`` (default), then the input files 
    are searched for in the same directory of the 
    :attr:`likelihood_script_file <NF4HEP.Sampler.likelihood_script_file>` file,
    otherwise, input files are searched for in the 
    :argument:`output_folder <common_classes_arguments.output_folder>` folder.

        - **type**: ``str`` or ``None``
        - **default**: ``None``

.. argument:: verbose

    Argument used to set the verbosity mode of the 
    ``__init__`` method and the default verbosity mode of all class methods 
    that accept a ``verbose`` argument.
    See :ref:`Verbosity mode <verbosity_mode>` for more details.

        - **type**: ``bool`` or ``int``
        - **default**: ``True``

.. argument:: pars_central   

    List or |numpy_link| array containing central values of the parameters.
    It is used to build the corresponding attribute
    in the objects:

        - :mod:`Likelihood <likelihood>`
        - :mod:`Data <data>`
        
        - **type**: ``list`` or ```numpy.ndarray``
        - **shape**: ``(ndims,)``
        - **default**: ``None`` 

.. argument:: pars_pos_poi   

    List or |numpy_link| array containing the positions in the parameters list of the
    parameters of interest.
    It is used to build the corresponding attribute
    in the objects:

        - :mod:`Likelihood <likelihood>`
        - :mod:`Data <data>`

        - **type**: ``list`` or ```numpy.ndarray``
        - **shape**: ``(n_poi,)``
        - **default**: ``None`` 

.. argument:: pars_pos_nuis   

    List or |numpy_link| array containing the positions in the parameters list of the
    nuisance parameters.
    It is used to build the corresponding attribute
    in the objects:

        - :mod:`Likelihood <likelihood>`
        - :mod:`Data <data>`

        - **type**: ``list`` or ``numpy.ndarray``
        - **shape**: ``(n_nuis,)``
        - **default**: ``None`` 

.. argument:: pars_labels   

    List containing the parameters names as strings.
    Parameters labels are always parsed as "raw" strings (like, for instance, ``r"%s"%pars_labels[0]``) 
    and can contain latex expressions that are properly compiled when making plots.
    It is used to build the corresponding attribute
    in the objects:

        - :mod:`Likelihood <likelihood>`
        - :mod:`Data <data>`

        - **type**: ``list``
        - **shape**: ``(ndims,)``
        - **default**: ``None`` 

.. argument:: pars_bounds   

    List or |numpy_link| array containing containing bounds for the parameters.
    It is used to build the corresponding attribute
    in the objects:

        - :mod:`Likelihood <likelihood>`
        - :mod:`Data <data>`

        - **type**: ``numpy.ndarray`` or ``None``
        - **shape**: ``(ndims,2)``
        - **default**: ``None`` 

.. include:: ../external_links.rst