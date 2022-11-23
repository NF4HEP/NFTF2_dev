.. _data_usage:

Usage
^^^^^

We give here a brief introduction to the use of the :class:`Data <NF4HEP.Data>` class. Refer to the 
full class documentation for more details. The :mod:`Data <data>` object is not supposed to be 
used directly by the user, but it is instead automatically used by the :class:`NF <NF4HEP.NF>` and 
:class:`DnnLikEnsemble <NF4HEP.NFEnsemble>` objects to manage training/validation/test data.
For more details refer to the :mod:`Data <data>` object documentation and to the documentation of the :mod:`DNNLikelihood <dnn_likelihood>` and 
the :mod:`DnnLikEnsemble <dnn_likelihood_ensemble>` objects. Examples will be referred to the toy likelihood introduced in the
:mod:`Likelihood <likelihood>` object :ref:`Usage <likelihood_usage>` section of the documentation.

The :mod:`Data <data>` object can be created from a :class:`Sampler <NF4HEP.Sampler>` object or from
input arguments. An example of the first method is given in the :mod:`Sampler <sampler>` object :ref:`Usage <sampler_usage>` section of the documentation
and consists of the following code:

.. code-block:: python

    import DNNLikelihood

    sampler = NF4HEP.Sampler(new_sampler=False, input_file=<my_output_folder>/toy_sampler)

    data = sampler.get_data_object(nsamples=200000, burnin=5000, thin=10, dtype="float64", test_fraction=0)

This gives a :mod:`Data <data>` object with 200K points which gets automatically stored in the files

    - <my_output_folder>/toy_data.log
    - <my_output_folder>/toy_data.h5
    - <my_output_folder>/toy_data_samples.h5 
    
To give an example of the creation of the object from input arguments we first define data arrays (for simplicity we use here
100K of the 200K points available in ``data``, but the user can of course use any other data) and parameters central values,
and then create the object as follows:

.. code-block:: python

    data_X = data.data_X[:100000]
    data_Y = data.data_Y[:100000]
    dtype = ["float64","float32"]
    pars_central = data.pars_central

    data_new = NF4HEP.Data(name = "new_data",
                                  data_X = data_X,
                                  data_Y = data_Y,
                                  dtype = ["float64","float32"],
                                  pars_central = pars_central,
                                  pars_pos_poi = [5],
                                  pars_pos_nuis = None,
                                  pars_labels = None,
                                  pars_bounds = None,
                                  test_fraction = 0.2,
                                  output_folder = <my_output_folder>,
                                  input_file = None,
                                  verbose = True)

This created the new object ``data_new`` and saves it to the files

    - <my_output_folder>/new_data.log
    - <my_output_folder>/new_data.h5
    - <my_output_folder>/new_data_samples.h5

See the documentation of the :meth:`Data.save <NF4HEP.Data.save>`, :meth:`Data.save_log <NF4HEP.Data.save_log>`,
:meth:`Data.save_object_h5 <NF4HEP.Data.save_object_h5>`, and :meth:`Data.save_samples_h5 <NF4HEP.Data.save_samples_h5>` 
methods for details on how the object is saved.

If the :argument:`name <Data.name>` argument is not passed, then one is automatically generated. In the example above we passed a list for the
:argument:`dtype <Data.dtype>` argument, where the first entry represents the data type of the stored data, while the second is the required data
type of the train/validation/test data created from the :mod:`Data <data>` object (see below). Concerning the arguments
related to parameters, if neither :argument:`pars_pos_poi <Data.pars_pos_poi>` nor :argument:`pars_pos_nuis <Data.pars_pos_nuis>` are specified, 
then all parameters are assumed to be parameters of interest, while if only one is specified, the other is automatically set. 
The arguments :argument:`pars_labels <Data.pars_labels>` and :argument:`pars_bounds <Data.pars_bounds>` can be optionally speficied, 
otherwise they are automatically set (see the full documentation for details).
The :argument:`output_folder <Data.output_folder>` is also optional and, if not specified, will automatically be set to the absolute 
path to the code execution folder.

The fraction of train/validation vs test sets is specified through the :argument:`test_fraction <Data.test_fraction>` attribute, which, 
in our example, has been set to ``0.2``. If :argument:`test_fraction <Data.test_fraction>` is not specified (default is ``None``), 
it is automatically set to ``0`` and no test data will be available.
The fraction of train/validation vs test sets can be changed at any time by changing the 
:attr:`Data.test_fraction <NF4HEP.Data.test_fraction>` attribute and calling the 
:meth:`Data.define_test_fraction <NF4HEP.Data.define_test_fraction>` method as follows:

.. code-block:: python

    data.test_fraction = 0.3
    data.define_test_fraction()

This updates the attributes :attr:`Data.train_range <NF4HEP.Data.train_range>` and 
:attr:`Data.test_range <NF4HEP.Data.test_range>`, used for train/validation/test data generation.

The :mod:`Data <data>` object can also be initialized importing it from saved files. 
In this case only the :argument:`input_file <Data.input_file>` argument needs to be specified,
while all other arguments are ignored. One could also optionally specify a new :argument:`output_folder <Data.output_folder>`. 
In case this is not specified, the 
:attr:`Lik.output_folder <NF4HEP.Lik.output_folder>` attribute from the imported object is used and the object is
saved by updating existing files. If a new :argument:`output_folder <Data.output_folder>` is specified, then the whole object is saved to the new location.
One could decide, when importing the object, to load all data stored in the 
:attr:`Data.input_h5_file <NF4HEP.Data.input_h5_file>` into the RAM for faster generation of train/validation/test data
by setting to ``True`` the argument :argument:`load_on_RAM <Data.load_on_RAM>`.
One could also specify the option :argument:`dtype <Data.dtype>`. This will affect neither saved data, nor the 
:attr:`Data.dtype_stored <NF4HEP.Data.dtype_stored>`, which is read from input files, but will change the value of the
:attr:`Data.dtype_required <NF4HEP.Data.dtype_required>` that will affect the dtype of the generated 
train/validation/test data. If :argument:`dtype <Data.dtype>` is not specified, then the attribute
:attr:`Data.dtype_required <NF4HEP.Data.dtype_required>` is set equal to the 
:attr:`Data.dtype_stored <NF4HEP.Data.dtype_stored>` attribute. Finally, also the argument 
:argument:`test_fraction <Data.test_fraction>` can be specified, which will update the corresponding attribute available from 
input files. An example of code to create the object from input files is given by:

.. code-block:: python
    
   import DNNLikelihood

   data = NF4HEP.Data(input_file="<my_output_folder>/new_data",
                             dtype = "float32",
                             load_on_RAM = True
                            )

When the object is imported, the :attr:`Data.log <NF4HEP.Data.log>` 
attribute is updated and saved in the corresponding file :attr:`Data.output_log_file <NF4HEP.Data.output_log_file>`.

Data stored in the object can be accessed from the :attr:`Data.data_X <NF4HEP.Data.data_X>` 
and :attr:`Data.data_Y <NF4HEP.Data.data_Y>` attributes as follows:

.. code-block:: python

    data.data_X

    >>> array([[-0.27304407, -0.46952336,  0.02573462, ..., -0.50863902, -1.12434653,  0.03133135],
               ...,
               [-0.1232975 ,  0.44791999, -0.21759863, ..., -1.04695986, -1.5426192 , -0.21464772]])

    data.data_Y

    >>> array([-58.52285607, -51.57067745, -52.23295711, ..., -52.55021968, -53.44788316, -54.33490431])

Upon creation of the :mod:`Data <data>` object, the :attr:`Data.data_dictionary <NF4HEP.Data.data_dictionary>`
dictionary is initialized as an empty dictionary containing empty |numpy_link| arrays with ``dtype`` equal to the value of
:attr:`Data.dtype_required <DNNLikeliohood.Data.dtype_required>` (for data) and to ``int`` for indices:

.. code-block:: python

    data.data_dictionary

    >>> {'X_train': array([], shape=(1, 0), dtype=float32),
         'Y_train': array([], dtype=float32),
         'X_val': array([], shape=(1, 0), dtype=float32),
         'Y_val': array([], dtype=float32),
         'X_test': array([], shape=(1, 0), dtype=float32),
         'Y_test': array([], dtype=float32),
         'idx_train': array([], dtype=int32),
         'idx_val': array([], dtype=int32),
         'idx_test': array([], dtype=int32)}

Such dictionary is updated any time data are generated or updated. For instance, we could generate 10000 train, 5000 validation,
and 5000 test points as follows:

.. code-block:: python

    data.generate_train_data(npoints_train=10000, npoints_val=5000, seed=1)
    data.generate_test_data(npoints_test=5000)

    data.data_dictionary

    >>> {'X_train': array([[-2.8694883e-01,  4.3361101e-01, -4.6597528e-01, ..., -9.9797267e-01, -9.8734242e-01, -9.5092475e-01],
                            ...,
                           [-4.1338271e-01, -2.6408151e-01,  4.8945403e-01, ..., -7.6410377e-01, -1.5141882e+00, -4.5764816e-01]], dtype=float32),
         'Y_train': array([-52.884396, -55.569386, -54.26178 , ..., -52.6744  , -57.04926 , -55.466824], dtype=float32),
         'X_val':   array([[ 0.19161688, -0.45214146, -0.10997196, ..., -0.86869633, -1.1265236 ,  0.6219092 ],
                            ...,
                           [-0.46064648,  0.5269843 ,  0.19721822, ..., -0.47361913, -0.27832463, -0.06421166]], dtype=float32),
         'Y_val':   array([-53.06865 , -53.42976 , -51.243603, ..., -55.466824, -54.91352 , -50.538002], dtype=float32),
         'X_test':  array([[-0.75995547,  0.49505967,  0.58410156, ..., -1.2593005 , -0.32470152,  0.8180257 ],
                            ...,
                           [-0.79108167,  0.7511644 ,  0.51859653, ..., -0.10589346, 1.1247568 ,  0.6004382 ]], dtype=float32),
         'Y_test':  array([-52.339706, -53.3896  , -55.370182, ..., -53.374393, -53.940056, -53.93778 ], dtype=float32),
         'idx_train': array([   10,    12,    19, ..., 69991, 69993, 69996]),
         'idx_val': array([   14,    67,    74, ..., 69946, 69987, 69998]),
         'idx_test': array([70000, 70001, 70002, ..., 74997, 74998, 74999])}

Where we have also print the new value of the :attr:`Data.data_dictionary <NF4HEP.Data.data_dictionary>` attribute.
In the generation of train/validation data one can profit of the ``seed`` argument to make the process reproducible. Indeed this
initializes the |numpy_link| random state to ``seed`` every time before selecting and splitting data. Concerning test data, 
these are always generated (deterministically) by taking the first ``npoints_test`` points in 
:attr:`Data.test_range <NF4HEP.Data.test_range>` range of data.

If scaling of ``X`` or ``Y`` data is necessary to train the neural network, one can define |standard_scalers_link| with the 
:meth:`Data.define_scalers <NF4HEP.Data.define_scalers>` method. For instance, scalers can be defined from the training 
data generated above with the following code:

.. code-block:: python

    data.define_scalers(data.data_dictionary["X_train"], data.data_dictionary["Y_train"], scalerX_bool=True, scalerY_bool=True)

    >>> [StandardScaler(copy=True, with_mean=True, with_std=True),
         StandardScaler(copy=True, with_mean=True, with_std=True)]

One could also define weights for data points, that allow to train on weighted data. This is done using the 
:meth:`Data.compute_sample_weights <NF4HEP.Data.compute_sample_weights>` method. This method bins data and weights them
according to the inverse frequency of the bins (and also allows for an arbitrary exponent). For instance we can compute weights 
making an histogram with ``50`` bins and weights equal to the inverse square root of frequencies as follows:

.. code-block:: python

    data.compute_sample_weights(data.data_dictionary["Y_train"], nbins=50, power=0.5)

    >>> array([0.70231453, 0.89124505, 0.72738372, ..., 0.71332963, 1.16068553, 0.89124505])

Notice that the :meth:`Data.define_scalers <NF4HEP.Data.define_scalers>` and 
:meth:`Data.compute_sample_weights <NF4HEP.Data.compute_sample_weights>` methods return a result and do not set any 
attribute (they are used by the :mod:`DNNLikelihood <dnn_likelihood>` object to set its corresponding attributes).

Data in the :attr:`Data.data_dictionary <NF4HEP.Data.data_dictionary>` as well as scalers and sample weights
are not stored when saving the :mod:`Data <data>` object, and will only be saved as part of trained models and 
models ensemble by the :class:`NF <NF4HEP.NF>` and 
:class:`DnnLikEnsemble <NF4HEP.NFEnsemble>` objects.
For more details refer to the full class documentation and to the documentation of the :mod:`DNNLikelihood <dnn_likelihood>` and 
the :mod:`DNNLikEnsemble <dnn_likelihood_ensemble>` objects.

The :mod:`Data <data>` object offers some methods useful to make plots that can be used to inspect the data. 
**Bla bla bla: add 1d histograms of X and Y data analogous to the methods in the Sampler class.**

Finally, the :meth:`plot_corners_1samp <NF4HEP.Data.plot_corners_1samp>` methos allows one to make a corner plot 
of 2D histograms of the data. As an example, **bla bla bla.** 

.. image:: ../figs/toy_data_plot_corners_1samp.png
    :class: with-shadow
    :scale: 48

Even though the files corresponding to the saved object are usually kept sync with the object state, manual change of some attributes
does not update them. Nevertheless, the full object can be saved at any time through

.. code-block:: python 

    data.save(overwrite=True)

.. include:: ../external_links.rst