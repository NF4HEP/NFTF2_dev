from tensorflow_probability.python.internal import tensorshape_util
from tensorflow_probability.python.internal import prefer_static as ps
from tensorflow_probability.python.bijectors import RationalQuadraticSpline
from . import inference, utils  # , custom_losses
from .show_prints import Verbosity, print

import numpy as np
import six
from timeit import default_timer as timer
import tensorflow as tf
import tensorflow.compat.v1 as tf1
from tensorflow.keras.layers import Layer
import tensorflow_probability as tfp
tfd = tfp.distributions
tfb = tfp.bijectors


header_string = "=============================="
footer_string = "------------------------------"

__all__ = ['MAFNetwork_default',
           'MAFNetwork_custom',
           'MAFBijector_default',
           'MAFBijector_custom',
           'MAFFlow']


class MAFNetwork_default(tfb.AutoregressiveNetwork, Verbosity):
    """
    Callable object that returns the MAF network from tfb.AutoregressiveNetwork
    """

    def __init__(self,
                 model_define_inputs,
                 verbose=True):
        self.verbose = verbose
        verbose, verbose_sub = self.set_verbosity(verbose)
        start = timer()
        self._model_define_inputs = model_define_inputs
        self.__set_inputs(verbose=verbose_sub)
        super(MAFNetwork_default, self).__init__(**utils.dic_minus_keys(self.model_define_inputs,["kwargs"]), **self.model_define_inputs["kwargs"])
        end = timer()
        print(footer_string, "\nMAF default neural network architecture (tfb.AutoregressiveNetwork) set in ", str(
            end-start), "s.\n", show=verbose)

    def __set_inputs(self, verbose=None):
        """

        """
        verbose, verbose_sub = self.set_verbosity(verbose)
        try:
            self._model_define_inputs["params"]
        except:
            raise Exception(
                "model_define_inputs dictionary should contain at least a key 'params'.")
        utils.check_set_dict_keys(self._model_define_inputs, ["event_shape",
                                                              "conditional",
                                                              "conditional_event_shape",
                                                              "conditional_input_layers",
                                                              "hidden_units",
                                                              "input_order",
                                                              "hidden_degrees",
                                                              "activation",
                                                              "use_bias",
                                                              "kernel_initializer",
                                                              "bias_initializer",
                                                              "kernel_regularizer",
                                                              "bias_regularizer",
                                                              "kernel_constraint",
                                                              "bias_constraint",
                                                              "validate_args",
                                                              "kwargs"],
                                                              [None,
                                                               False,
                                                               None,
                                                               "all_layers",
                                                               [64, 64, 64],
                                                               "left-to-right",
                                                               "equal",
                                                               None,
                                                               True,
                                                               "glorot_uniform",
                                                               "zeros",
                                                               None,
                                                               None,
                                                               None,
                                                               None,
                                                               False,
                                                               {}],
                                                              verbose=verbose_sub)

    def get_config(self):
        config = super().get_config()
        config.update({
            "model_define_inputs": self._model_define_inputs,
            "verbose": self.verbose,
        })
        return config

    @property
    def model_define_inputs(self):
        return self._model_define_inputs

class MAFNetwork_custom(Layer, Verbosity):
    """
    Defines a custom MAF network
    """

    def __init__(self,
                 model_define_inputs,
                 verbose=True):
        self.verbose = verbose
        verbose, verbose_sub = self.set_verbosity(verbose)
        start = timer()
        self._model_define_inputs = model_define_inputs
        self.__set_inputs()
        super(MAFNetwork_custom, self).__init__(**self.model_define_inputs["kwargs"]) ##passing additional keyword erguments to Layer

        if self._event_shape is not None:
            self._event_size = self._event_shape[-1]
            self._event_ndims = len(self._event_shape)

            if self._event_ndims != 1:
                raise ValueError('Parameter `event_shape` must describe a rank-1 '
                                 'shape. `event_shape: {!r}`'.format(self._event_shape))

        if self._conditional:
            if self._event_shape is None:
                raise ValueError('`event_shape` must be provided when '
                                 '`conditional` is True')
            if self._conditional_event_shape is None:
                raise ValueError('`conditional_event_shape` must be provided when '
                                 '`conditional` is True')
            self._conditional_size = self._conditional_event_shape[-1]
            self._conditional_ndims = len(self._conditional_event_shape)
            if self._conditional_ndims != 1:
                raise ValueError('Parameter `conditional_event_shape` must describe a '
                                 'rank-1 shape')
            if not ((self._conditional_layers == 'first_layer') or
                    (self._conditional_layers == 'all_layers')):
                raise ValueError('`conditional_input_layers` must be '
                                 '"first_layers" or "all_layers"')
        else:
            if self._conditional_event_shape is not None:
                raise ValueError('`conditional_event_shape` passed but `conditional` '
                                 'is set to False.')

        # To be built in `build`.
        self._input_order = None
        self._masks = None
        self._network = None
        end = timer()
        print(footer_string, "\nMAF custom neural network architecture set in ", str(
            end-start), "s.\n", show=verbose)

    def __set_inputs(self, verbose=None):
        """

        """
        verbose, verbose_sub = self.set_verbosity(verbose)
        try:
            self._model_define_inputs["params"]
        except:
            raise Exception(
                "model_define_inputs dictionary should contain at least a key 'params'.")
        utils.check_set_dict_keys(self._model_define_inputs, ["event_shape",
                                                              "conditional",
                                                              "conditional_event_shape",
                                                              "conditional_input_layers",
                                                              "hidden_units",
                                                              "input_order",
                                                              "hidden_degrees",
                                                              "activation",
                                                              "use_bias",
                                                              "kernel_initializer",
                                                              "bias_initializer",
                                                              "kernel_regularizer",
                                                              "bias_regularizer",
                                                              "kernel_constraint",
                                                              "bias_constraint",
                                                              "validate_args",
                                                              "kwargs"],
                                                              [None,
                                                               False,
                                                               None,
                                                               "all_layers",
                                                               [64, 64, 64],
                                                               "left-to-right",
                                                               "equal",
                                                               None,
                                                               True,
                                                               "glorot_uniform",
                                                               "zeros",
                                                               None,
                                                               None,
                                                               None,
                                                               None,
                                                               False,
                                                               {}],
                                                              verbose=verbose_sub)
        self._params = self.model_define_inputs["params"]
        event_shape = self.model_define_inputs["event_shape"]
        self._event_shape = _list(event_shape) if event_shape is not None else None
        self._conditional = self.model_define_inputs["conditional"]
        conditional_event_shape = self.model_define_inputs["conditional_event_shape"]
        self._conditional_event_shape = (
            _list(conditional_event_shape) if conditional_event_shape is not None else None)
        self._conditional_layers = self.model_define_inputs["conditional_input_layers"]
        hidden_units = self.model_define_inputs["hidden_units"]
        self._hidden_units = hidden_units if hidden_units is not None else []
        self._input_order_param = self.model_define_inputs["input_order"]
        self._hidden_degrees = self.model_define_inputs["hidden_degrees"]
        self._activation = self.model_define_inputs["activation"]

        self._use_bias = self.model_define_inputs["use_bias"]
        if "ListWrapper" not in str(type(self._use_bias)):
            self._use_bias = [self._use_bias for _ in range(len(self._hidden_units)+1)]

        self._kernel_initializer = self.model_define_inputs["kernel_initializer"]
        if "ListWrapper" not in str(type(self._kernel_initializer)):
            self._kernel_initializer = [self._kernel_initializer for _ in range(len(self._hidden_units)+1)]

        self._bias_initializer = self.model_define_inputs["bias_initializer"]
        if "ListWrapper" not in str(type(self._bias_initializer)):
            self._bias_initializer = [self._bias_initializer for _ in range(len(self._hidden_units)+1)]

        self._kernel_regularizer = self.model_define_inputs["kernel_regularizer"]
        if "ListWrapper" not in str(type(self._kernel_regularizer)):
            self._kernel_regularizer = [self._kernel_regularizer for _ in range(len(self._hidden_units)+1)]

        self._bias_regularizer = self.model_define_inputs["bias_regularizer"]
        if "ListWrapper" not in str(type(self._bias_regularizer)):
            self._bias_regularizer = [self._bias_regularizer for _ in range(len(self._hidden_units)+1)]

        kernel_constraint = self.model_define_inputs["kernel_constraint"]
        if "ListWrapper" not in str(type(kernel_constraint)):
            self._kernel_constraint = [tf.keras.constraints.get(kernel_constraint) for _ in range(len(self._hidden_units)+1)]
        else:
            self._kernel_constraint = [tf.keras.constraints.get(x) for x in kernel_constraint]

        self._bias_constraint = self.model_define_inputs["bias_constraint"]
        if "ListWrapper" not in str(type(self._bias_constraint)):
            self._bias_constraint = [self._bias_constraint for _ in range(len(self._hidden_units)+1)]
            
        self._validate_args = self.model_define_inputs["validate_args"]
        self._kwargs = self.model_define_inputs["kwargs"]

    def get_config(self):
        config = super().get_config()
        config.update({
            "model_define_inputs": self._model_define_inputs,
            "verbose": self.verbose,
        })
        return config

    def build(self, input_shape):
        """See tfkl.Layer.build."""
        if self._event_shape is None:
            # `event_shape` wasn't specied at __init__, so infer from `input_shape`.
            self._event_shape = [tf.compat.dimension_value(input_shape[-1])]
            self._event_size = self._event_shape[-1]
            self._event_ndims = len(self._event_shape)
            # Should we throw if input_shape has rank > 2?

        if input_shape[-1] != self._event_shape[-1]:
            raise ValueError('Invalid final dimension of `input_shape`. '
                             'Expected `{!r}`, but got `{!r}`'.format(
                                 self._event_shape[-1], input_shape[-1]))

        # Construct the masks.
        self._input_order = _create_input_order(
            self._event_size,
            self._input_order_param,
        )

        self._masks = _make_dense_autoregressive_masks(
            params=self._params,
            event_size=self._event_size,
            hidden_units=self._hidden_units,
            input_order=self._input_order,
            hidden_degrees=self._hidden_degrees,
        )

        outputs = [tf.keras.Input((self._event_size,), dtype=self.dtype)]
        inputs = outputs[0]
        if self._conditional:
            conditional_input = tf.keras.Input((self._conditional_size,),
                                               dtype=self.dtype)
            inputs = [inputs, conditional_input]

        # Input-to-hidden, hidden-to-hidden, and hidden-to-output layers:
        #  [..., self._event_size] -> [..., self._hidden_units[0]].
        #  [..., self._hidden_units[k-1]] -> [..., self._hidden_units[k]].
        #  [..., self._hidden_units[-1]] -> [..., event_size * self._params].
        layer_output_sizes = self._hidden_units + [self._event_size * self._params]
        for k in range(len(self._masks)):#list(np.array(self._input_order)-1):
            #print("Inputs:\n",
            #      "use_bias=",self._use_bias[k],"\n",
            #      "kernel_initializer=",self._kernel_initializer[k],"\n",
            #      "bias_initializer=",self._bias_initializer[k],"\n",
            #      "kernel_regularizer=",self._kernel_regularizer[k],"\n",
            #      "bias_regularizer=",self._bias_regularizer[k],"\n",
            #      "kernel_constraint=",self._kernel_constraint[k],"\n",
            #      "bias_constraint=",self._bias_constraint[k],"\n")
            autoregressive_output = tf.keras.layers.Dense(units=layer_output_sizes[k],
                                                          activation=None,
                                                          use_bias=self._use_bias[k],
                                                          kernel_initializer=_make_masked_initializer(
                                                              self._masks[k], self._kernel_initializer[k]),
                                                          bias_initializer=self._bias_initializer[k],
                                                          kernel_regularizer=self._kernel_regularizer[k],
                                                          bias_regularizer=self._bias_regularizer[k],
                                                          kernel_constraint=_make_masked_constraint(
                                                              self._masks[k], self._kernel_constraint[k]),
                                                          bias_constraint=self._bias_constraint[k],
                                                          dtype=self.dtype)(outputs[-1])
            if (self._conditional and
                ((self._conditional_layers == 'all_layers') or
                 ((self._conditional_layers == 'first_layer') and (k == 0)))):
                conditional_output = tf.keras.layers.Dense(units=layer_output_sizes[k],
                                                           activation=None,
                                                           use_bias=False,
                                                           kernel_initializer=self._kernel_initializer[k],
                                                           bias_initializer=None,
                                                           kernel_regularizer=self._kernel_regularizer[k],
                                                           bias_regularizer=None,
                                                           kernel_constraint=self._kernel_constraint[k],
                                                           bias_constraint=None,
                                                           dtype=self.dtype)(conditional_input)
                outputs.append(tf.keras.layers.Add()([autoregressive_output,
                                                      conditional_output]))
            else:
                outputs.append(autoregressive_output)
            if k + 1 < len(self._masks):
                outputs.append(tf.keras.layers.Activation(
                    self._activation)(outputs[-1]))
        self._network = tf.keras.models.Model(inputs=inputs,
                                              outputs=outputs[-1])
        # Allow network to be called with inputs of shapes that don't match
        # the specs of the network's input layers.
        self._network.input_spec = None
        # Record that the layer has been built.
        super().build(input_shape)

    def call(self, x, conditional_input=None):
        """Transforms the inputs and returns the outputs.
        Suppose `x` has shape `batch_shape + event_shape` and `conditional_input`
        has shape `conditional_batch_shape + conditional_event_shape`. Then, the
        output shape is:
        `broadcast(batch_shape, conditional_batch_shape) + event_shape + [params]`.
        Also see `tfkl.Layer.call` for some generic discussion about Layer calling.
        Args:
          x: A `Tensor`. Primary input to the layer.
          conditional_input: A `Tensor. Conditional input to the layer. This is
            required iff the layer is conditional.
        Returns:
          y: A `Tensor`. The output of the layer. Note that the leading dimensions
             follow broadcasting rules described above.
        """
        with tf.name_scope(self.name or 'AutoregressiveNetwork_call'):
            x = tf.convert_to_tensor(x, dtype=self.dtype, name='x')
            # TODO(b/67594795): Better support for dynamic shapes.
            input_shape = ps.shape(x)
            if tensorshape_util.rank(x.shape) == 1:
                x = x[tf.newaxis, ...]
            if self._conditional:
                if conditional_input is None:
                    raise ValueError('`conditional_input` must be passed as a named '
                                     'argument')
                conditional_input = tf.convert_to_tensor(
                    conditional_input, dtype=self.dtype, name='conditional_input')
                conditional_batch_shape = ps.shape(conditional_input)[:-1]
                if tensorshape_util.rank(conditional_input.shape) == 1:
                    conditional_input = conditional_input[tf.newaxis, ...]
                x = [x, conditional_input]
                output_shape = ps.concat(
                    [ps.broadcast_shape(conditional_batch_shape,
                                        input_shape[:-1]),
                     input_shape[-1:]], axis=0)
            else:
                output_shape = input_shape
            return tf.reshape(self._network(x),
                              tf.concat([output_shape, [self._params]], axis=0))

    def compute_output_shape(self, input_shape):
        """See tfkl.Layer.compute_output_shape."""
        return input_shape + (self._params,)

    @property
    def event_shape(self):
        return self._event_shape

    @property
    def params(self):
        return self._params

    @property
    def model_define_inputs(self):
        return self._model_define_inputs


class MAFBijector_default(tfb.MaskedAutoregressiveFlow, Verbosity):
    """
    Callable object that returns the MAF bijector from tfb.MaskedAutoregressiveFlow
    """

    def __init__(self,
                 model_define_inputs,
                 verbose=True):
        self.verbose = verbose
        verbose, verbose_sub = self.set_verbosity(verbose)
        start = timer()
        self._model_define_inputs = model_define_inputs
        self.__set_inputs(verbose=verbose_sub)
        if self.model_define_inputs["default_NN"]:
            NN = MAFNetwork_default(utils.dic_minus_keys(self.model_define_inputs,["default_NN"]))
        else:
            NN = MAFNetwork_custom(utils.dic_minus_keys(self.model_define_inputs,["default_NN"]))
        super(MAFBijector_default, self).__init__(shift_and_log_scale_fn=NN)
        end = timer()
        print(footer_string, "\nMAF default bijector (tfb.MaskedAutoregressiveFlow) set in ", str(
            end-start), "s.\n", show=verbose)

    def __set_inputs(self, verbose=None):
        """

        """
        verbose, verbose_sub = self.set_verbosity(verbose)
        try:
            self._model_define_inputs["params"]
        except:
            raise Exception("model_define_inputs dictionary should contain at least a key 'params'.")
        utils.check_set_dict_keys(self._model_define_inputs, ["default_NN"], [True], verbose=verbose_sub)

    def get_config(self):
        config = super().get_config()
        config.update({
            "model_define_inputs": self._model_define_inputs,
            "verbose": self.verbose,
        })
        return config

    @property
    def model_define_inputs(self):
        return self._model_define_inputs


class MAFBijector_custom(tfb.Bijector, Verbosity):
    """
    Defines a custom MAF (spline) bijector
    """

    def __init__(self,
                 model_define_inputs,
                 model_chain_inputs,
                 verbose=True):
        self.verbose = verbose
        verbose, verbose_sub = self.set_verbosity(verbose)
        start = timer()
        self._model_define_inputs = model_define_inputs
        self._model_chain_inputs = model_chain_inputs
        self.__set_inputs(verbose=verbose_sub)
        if self.model_define_inputs["default_NN"]:
            NN = MAFNetwork_default(utils.dic_minus_keys(self.model_define_inputs,["default_NN"]))
        else:
            NN = MAFNetwork_custom(utils.dic_minus_keys(self.model_define_inputs,["default_NN"]))
        self._shift_and_log_scale_fn = NN

        def _bijector_fn(x, **condition_kwargs):

            def reshape(params):
                # print(params)
                factor = tf.cast(2*abs(self.range_min), dtype=tf.float32)
                bin_widths = params[:, :, :self.spline_knots]
                #bin_widths=tf.reshape(bin_widths, (x.shape[0],self.tran_ndims,spline_knots), name=None)
                bin_widths = tf.math.softmax(bin_widths)
                bin_widths = tf.math.scalar_mul(factor, bin_widths)
                # print(bin_widths)

                bin_heights = params[:, :, self.spline_knots:self.spline_knots*2]
                #bin_heights=tf.reshape(bin_heights, (x.shape[0],self.tran_ndims,spline_knots), name=None)
                bin_heights = tf.math.softmax(bin_heights)
                bin_heights = tf.math.scalar_mul(factor, bin_heights)

                knot_slopes = params[:, :, self.spline_knots*2:]
                knot_slopes = tf.math.softplus(knot_slopes)
                #knot_slopes=tf.reshape(knot_slopes, (x.shape[0],self.tran_ndims,spline_knots-1), name=None)
                return bin_widths, bin_heights, knot_slopes

            params = self._shift_and_log_scale_fn(x, **condition_kwargs)
            bin_widths, bin_heights, knot_slopes = reshape(params)

            return RationalQuadraticSpline(bin_widths=bin_widths, bin_heights=bin_heights, knot_slopes=knot_slopes, range_min=self.range_min, validate_args=False)

        self._bijector_fn = _bijector_fn

        if self._validate_args:
            self._bijector_fn = _validate_bijector_fn(self._bijector_fn)

        self._parameters['bijector_fn'] = self._bijector_fn,
        self._parameters['shift_and_log_scale_fn'] = self._shift_and_log_scale_fn

        super().__init__(forward_min_event_ndims=self._event_ndims,
                         is_constant_jacobian=self._is_constant_jacobian,
                         validate_args=self._validate_args,
                         parameters=self._parameters,
                         name=self._name)
        end = timer()
        print(footer_string, "\nMAF custom bijector set in ",
              str(end-start), "s.\n", show=verbose)

    def get_config(self):
        config = super().get_config()
        config.update({
            "model_define_inputs": self._model_define_inputs,
            "model_chain_inputs": self._model_chain_inputs,
            "verbose": self.verbose,
        })
        return config

    def __set_inputs(self, verbose=None):
        """

        """
        verbose, verbose_sub = self.set_verbosity(verbose)
        try:
            self._model_define_inputs["params"]
        except:
            raise Exception("model_define_inputs dictionary should contain at least a key 'params'.")
        utils.check_set_dict_keys(self._model_define_inputs, ["default_NN"], [True], verbose=verbose_sub)
        utils.check_set_dict_keys(self.model_chain_inputs, ["is_constant_jacobian",
                                                           "validate_args",
                                                           "unroll_loop",
                                                           "event_ndims",
                                                           "name",
                                                           "spline_knots",
                                                           "range_min"],
                                                          [False,
                                                           False,
                                                           False,
                                                           1,
                                                           "custom_maf_bijector",
                                                           2,
                                                           -1])
        self._is_constant_jacobian = self.model_chain_inputs["is_constant_jacobian"]
        self._validate_args = self.model_chain_inputs["validate_args"]
        self._unroll_loop = self.model_chain_inputs["unroll_loop"]
        self._event_ndims = self.model_chain_inputs["event_ndims"]
        self._name = self.model_chain_inputs["name"]
        self._spline_knots = self.model_chain_inputs["spline_knots"]
        self._range_min = self.model_chain_inputs["range_min"]
        self._parameters = {'is_constant_jacobian': self._is_constant_jacobian,
                            'validate_args': self._validate_args,
                            'unroll_loop': self._unroll_loop,
                            'event_ndims': self._event_ndims,
                            'name': self._name}
    
    @classmethod
    def _parameter_properties(cls, dtype):
        return dict()

    def _forward(self, x, **kwargs):
        static_event_size = tensorshape_util.num_elements(
            tensorshape_util.with_rank_at_least(
                x.shape, self._event_ndims)[-self._event_ndims:])

        if self._unroll_loop:
            if not static_event_size:
                raise ValueError(
                    'The final {} dimensions of `x` must be known at graph '
                    'construction time if `unroll_loop=True`. `x.shape: {!r}`'.format(
                        self._event_ndims, x.shape))
            y = tf.zeros_like(x, name='y0')

            for _ in range(static_event_size):
                y = self._bijector_fn(y, **kwargs).forward(x)
            return y

        event_size = ps.reduce_prod(ps.shape(x)[-self._event_ndims:])
        y0 = tf.zeros_like(x, name='y0')
        # call the template once to ensure creation
        if not tf.executing_eagerly():
            _ = self._bijector_fn(y0, **kwargs).forward(y0)
        bijector = self._bijector_fn(y0, **kwargs)

        def _loop_body(y0):
            """While-loop body for autoregression calculation."""
            # Set caching device to avoid re-getting the tf.Variable for every while
            # loop iteration.
            with tf1.variable_scope(tf1.get_variable_scope()) as vs:
                if vs.caching_device is None and not tf.executing_eagerly():
                    vs.set_caching_device(lambda op: op.device)
                bijector = self._bijector_fn(y0, **kwargs)

            y = bijector.forward(x)
            return (y,)
        (y,) = tf.while_loop(
            cond=lambda _: True,
            body=_loop_body,
            loop_vars=(y0,),
            maximum_iterations=event_size)
        return y

    def _inverse(self, y, **kwargs):
        bijector = self._bijector_fn(y, **kwargs)
        return bijector.inverse(y)

    def _inverse_log_det_jacobian(self, y, **kwargs):
        return self._bijector_fn(y, **kwargs).inverse_log_det_jacobian(
            y, event_ndims=self._event_ndims)

    @property
    def model_define_inputs(self):
        return self._model_define_inputs

    @property
    def model_chain_inputs(self):
        return self._model_chain_inputs

    @property
    def spline_knots(self):
        return self._spline_knots

    @property
    def range_min(self):
        return self._range_min

class MAFFlow(tfb.Chain, Verbosity):
    """
    Chain object returning the full flow
    """

    def __init__(self,
                 model_define_inputs,
                 model_chain_inputs,
                 verbose=True):
        self.verbose = verbose
        verbose, verbose_sub = self.set_verbosity(verbose)
        start = timer()
        print(header_string, "\nInitializing MAF Flow.\n", show=verbose)
        self._model_define_inputs = model_define_inputs
        self._model_chain_inputs = model_chain_inputs
        self.__set_inputs(verbose=verbose_sub)
        default_NN = self.model_define_inputs["default_NN"]
        default_bijector = self.model_chain_inputs["default_bijector"]
        batch_norm = self.model_define_inputs.pop("batch_norm")
        ndims = self.model_chain_inputs["ndims"]
        if default_bijector:
            bijector = MAFBijector_default(self.model_define_inputs)
        else:
            self.model_define_inputs["params"] = 3*self.model_chain_inputs["spline_knots"]-1
            bijector = MAFBijector_custom(self.model_define_inputs, self.model_chain_inputs)
        permutation = tf.cast(np.concatenate((np.arange(int(ndims/2), ndims), np.arange(0, int(ndims/2)))), tf.int32)
        Permute = tfb.Permute(permutation=permutation)
        bijectors = []
        for _ in range(self.model_chain_inputs["num_bijectors"]):
            if batch_norm:
                bijectors.append(tfb.BatchNormalization())
            bijectors.append(bijector)
            bijectors.append(Permute)
        if default_bijector and default_NN:
            name = 'default_maf_chain'
        else:
            name = 'custom_maf_chain'
        super(MAFFlow, self).__init__(bijectors=list(reversed(bijectors[:-1])), name=name)
        end = timer()
        print(footer_string, "\nMAF Flow initialized in",
              str(end-start), "s.\n", show=verbose)
    
    def __set_inputs(self, verbose=None):
        """

        """
        verbose, verbose_sub = self.set_verbosity(verbose)
        try:
            self._model_define_inputs["params"]
        except:
            raise Exception("model_define_inputs dictionary should contain at least a key 'params'.")
        utils.check_set_dict_keys(self._model_define_inputs, ["default_NN"], [True], verbose=verbose_sub)
        utils.check_set_dict_keys(self._model_chain_inputs, ["default_bijector"], [True], verbose=verbose_sub)

    def get_config(self):
        config = super().get_config()
        config.update({
            "model_define_inputs": self._model_define_inputs,
            "model_chain_inputs": self._model_chain_inputs,
            "verbose": self.verbose,
        })
        return config

    @property
    def model_define_inputs(self):
        return self._model_define_inputs

    @property
    def model_chain_inputs(self):
        return self._model_chain_inputs


def _make_dense_autoregressive_masks(
    params,
    event_size,
    hidden_units,
    input_order='left-to-right',
    hidden_degrees='equal',
    seed=None,
):
    """Creates masks for use in dense MADE [Germain et al. (2015)][1] networks.
    See the documentation for `AutoregressiveNetwork` for the theory and
    application of MADE networks. This function lets you construct your own dense
    MADE networks by applying the returned masks to each dense layer. E.g. a
    consider an autoregressive network that takes `event_size`-dimensional vectors
    and produces `params`-parameters per input, with `num_hidden` hidden layers,
    with `hidden_size` hidden units each.
    ```python
    def random_made(x):
      masks = tfb._make_dense_autoregressive_masks(
          params=params,
          event_size=event_size,
          hidden_units=[hidden_size] * num_hidden)
      output_sizes = [hidden_size] * num_hidden
      input_size = event_size
      for (mask, output_size) in zip(masks, output_sizes):
        mask = tf.cast(mask, tf.float32)
        x = tf.matmul(x, tf.random.normal([input_size, output_size]) * mask)
        x = tf.nn.relu(x)
        input_size = output_size
      x = tf.matmul(
          x,
          tf.random.normal([input_size, params * event_size]) * masks[-1])
      x = tf.reshape(x, [-1, event_size, params])
      return x
    y = random_made(tf.zeros([1, event_size]))
    assert [1, event_size, params] == y.shape
    ```
    Each mask is a Numpy boolean array. All masks have the shape `[input_size,
    output_size]`. For example, if we `hidden_units` is a list of two integers,
    the mask shapes will be: `[event_size, hidden_units[0]], [hidden_units[0],
    hidden_units[1]], [hidden_units[1], params * event_size]`.
    You can extend this example with trainable parameters and constraints if
    necessary.
    Args:
      params: Python integer specifying the number of parameters to output
        per input.
      event_size: Python integer specifying the shape of the input to this layer.
      hidden_units: Python `list`-like of non-negative integers, specifying
        the number of units in each hidden layer.
      input_order: Order of degrees to the input units: 'random', 'left-to-right',
        'right-to-left', or an array of an explicit order. For example,
        'left-to-right' builds an autoregressive model
        p(x) = p(x1) p(x2 | x1) ... p(xD | x<D).
      hidden_degrees: Method for assigning degrees to the hidden units:
        'equal', 'random'. If 'equal', hidden units in each layer are allocated
        equally (up to a remainder term) to each degree. Default: 'equal'.
      seed: If not `None`, seed to use for 'random' `input_order` and
        `hidden_degrees`.
    Returns:
      masks: A list of masks that should be applied the dense matrices of
        individual densely connected layers in the MADE network. Each mask is a
        Numpy boolean array.
    # References
    [1]: Mathieu Germain, Karol Gregor, Iain Murray, and Hugo Larochelle. MADE:
         Masked Autoencoder for Distribution Estimation. In _International
         Conference on Machine Learning_, 2015. https://arxiv.org/abs/1502.03509
    """
    if seed is None:
        input_order_seed = None
        degrees_seed = None
    else:
        input_order_seed, degrees_seed = np.random.RandomState(seed).randint(
            2**31, size=2)
    input_order = _create_input_order(
        event_size, input_order, seed=input_order_seed)
    masks = _create_masks(_create_degrees(
        input_size=event_size,
        hidden_units=hidden_units,
        input_order=input_order,
        hidden_degrees=hidden_degrees,
        seed=degrees_seed))
    # In the final layer, we will produce `params` outputs for each of the
    # `event_size` inputs.  But `masks[-1]` has shape `[hidden_units[-1],
    # event_size]`.  Thus, we need to expand the mask to `[hidden_units[-1],
    # event_size * params]` such that all units for the same input are masked
    # identically.  In particular, we tile the mask so the j-th element of
    # `tf.unstack(output, axis=-1)` is a tensor of the j-th parameter/unit for
    # each input.
    #
    # NOTE: Other orderings of the output could be faster -- should benchmark.
    masks[-1] = np.reshape(
        np.tile(masks[-1][..., tf.newaxis], [1, 1, params]),
        [masks[-1].shape[0], event_size * params])
    return masks


def _list(xs):
    """Convert the given argument to a list."""
    try:
        return list(xs)
    except TypeError:
        return [xs]


def _create_input_order(input_size, input_order='left-to-right', seed=None):
    """Returns a degree vectors for the input."""
    if isinstance(input_order, six.string_types):
        if input_order == 'left-to-right':
            return np.arange(start=1, stop=input_size + 1)
        elif input_order == 'right-to-left':
            return np.arange(start=input_size, stop=0, step=-1)
        elif input_order == 'random':
            ret = np.arange(start=1, stop=input_size + 1)
            if seed is None:
                rng = np.random
            else:
                rng = np.random.RandomState(seed)
            rng.shuffle(ret)
            return ret
    elif np.all(np.sort(np.array(input_order)) == np.arange(1, input_size + 1)):
        return np.array(input_order)

    raise ValueError('Invalid input order: "{}".'.format(input_order))


def _create_degrees(input_size,
                    hidden_units=None,
                    input_order='left-to-right',
                    hidden_degrees='equal',
                    seed=None):
    """Returns a list of degree vectors, one for each input and hidden layer.
    A unit with degree d can only receive input from units with degree < d. Output
    units always have the same degree as their associated input unit.
    Args:
      input_size: Number of inputs.
      hidden_units: list with the number of hidden units per layer. It does not
        include the output layer. Each hidden unit size must be at least the size
        of length (otherwise autoregressivity is not possible).
      input_order: Order of degrees to the input units: 'random', 'left-to-right',
        'right-to-left', or an array of an explicit order. For example,
        'left-to-right' builds an autoregressive model
        p(x) = p(x1) p(x2 | x1) ... p(xD | x<D).
      hidden_degrees: Method for assigning degrees to the hidden units:
        'equal', 'random'.  If 'equal', hidden units in each layer are allocated
        equally (up to a remainder term) to each degree.  Default: 'equal'.
      seed: If not `None`, use as a seed for the 'random' hidden_degrees.
    Raises:
      ValueError: invalid input order.
      ValueError: invalid hidden degrees.
    """
    input_order = _create_input_order(input_size, input_order)
    degrees = [input_order]

    if hidden_units is None:
        hidden_units = []

    for units in hidden_units:
        if isinstance(hidden_degrees, six.string_types):
            if hidden_degrees == 'random':
                if seed is None:
                    rng = np.random
                else:
                    rng = np.random.RandomState(seed)
                # samples from: [low, high)
                degrees.append(
                    rng.randint(low=min(np.min(degrees[-1]), input_size - 1),
                                high=input_size,
                                size=units))
            elif hidden_degrees == 'equal':
                min_degree = min(np.min(degrees[-1]), input_size - 1)
                degrees.append(np.maximum(
                    min_degree,
                    # Evenly divide the range `[1, input_size - 1]` in to `units + 1`
                    # segments, and pick the boundaries between the segments as degrees.
                    np.ceil(np.arange(1, units + 1)
                            * (input_size - 1) / float(units + 1)).astype(np.int32)))
        else:
            raise ValueError(
                'Invalid hidden order: "{}".'.format(hidden_degrees))

    return degrees


def _create_masks(degrees):
    """Returns a list of binary mask matrices enforcing autoregressivity."""
    return [
        # Create input->hidden and hidden->hidden masks.
        inp[:, np.newaxis] <= out
        for inp, out in zip(degrees[:-1], degrees[1:])
    ] + [
        # Create hidden->output mask.
        degrees[-1][:, np.newaxis] < degrees[0]
    ]


def _make_masked_initializer(mask, initializer):
    """Returns a masked version of the given initializer."""
    initializer = tf.keras.initializers.get(initializer)

    def masked_initializer(shape, dtype=None, partition_info=None):
        # If no `partition_info` is given, then don't pass it to `initializer`, as
        # `initializer` may be a `tf.initializers.Initializer` (which don't accept a
        # `partition_info` argument).
        if partition_info is None:
            x = initializer(shape, dtype)
        else:
            x = initializer(shape, dtype, partition_info)
        return tf.cast(mask, x.dtype) * x
    return masked_initializer


def _make_masked_constraint(mask, constraint=None):
    constraint = tf.keras.constraints.get(constraint)

    def masked_constraint(x):
        x = tf.convert_to_tensor(x, dtype_hint=tf.float32, name='x')
        if constraint is not None:
            x = constraint(x)
        return tf.cast(mask, x.dtype) * x
    return masked_constraint


def _validate_bijector_fn(bijector_fn):
    """Validates the output of `bijector_fn`."""

    def _wrapper(x, **condition_kwargs):
        """A wrapper that validates `bijector_fn`."""
        bijector = bijector_fn(x, **condition_kwargs)
        if bijector.forward_min_event_ndims != bijector.inverse_min_event_ndims:
            # Current code won't really work with this, but in principle we could
            # implement this.
            raise ValueError(
                'Bijectors which alter `event_ndims` are not supported.')
        if bijector.forward_min_event_ndims > 0:
            # Mustn't break auto-regressivity,
            raise ValueError(
                'Bijectors with `forward_min_event_ndims` > 0 are not supported.')
        return bijector

    return _wrapper
