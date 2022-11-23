from . import inference, utils  # , custom_losses
from .verbosity import Verbosity, print
__all__ = ['RealNVPNetwork',
           'RealNVPBijector',
           'RealNVPFlow']

import numpy as np
from timeit import default_timer as timer
import tensorflow as tf
from tensorflow.keras import layers, Input
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Layer
import tensorflow_probability as tfp
tfd = tfp.distributions
tfb = tfp.bijectors


header_string = "=============================="
footer_string = "------------------------------"


class RealNVPNetwork(Layer, Verbosity):
    # """
    # """
    # def define_network(self, verbose = None):
    #    verbose, verbose_sub = self.set_verbosity(verbose)
    #    self.NN = tfb.AutoregressiveNetwork(**self.model_define_inputs
    """
    Neural Network Architecture for calcualting s and t for Real-NVP
    """

    def __init__(self,
                 model_define_inputs,
                 verbose=True):
        self.verbose = verbose
        verbose, verbose_sub = self.set_verbosity(verbose)
        self.model_define_inputs = model_define_inputs
        self.__set_inputs()
        self.__check_set_ndims()
        self.__set_layers()
        super().__init__()

    def __set_inputs(self, verbose=None):

        self.hidden_layers = self.model_define_inputs["hidden_layers"]
        self.dropout_rate = self.model_define_inputs["dropout_rate"]
        self.batch_norm = self.model_define_inputs["batch_norm"]

    def __check_set_ndims(self, verbose=None):
        self.ndims = self.model_define_inputs["ndims"]
        self.rem_dims = self.model_define_inputs["rem_dims"]
        if self.rem_dims < 1 or self.rem_dims > self.ndims-1:
            raise Exception("rem_dims must be 1 < rem_dims < ndims-1")
        self.tran_ndims = self.ndims-self.rem_dims

    def __set_layers(self, verbose=None):
        """
        Method that defines strings representing the |tf_keras_layers_link| that are stored in the 
        :attr:`NF.layers_string <NF4HEP.NF.layers_string>` attribute.
        These are defined from the attributes

            - :attr:`NF.hidden_layers <NF4HEP.NF.hidden_layers>`
            - :attr:`NF.batch_norm <NF4HEP.NF.batch_norm>`
            - :attr:`NF.dropout_rate <NF4HEP.NF.dropout_rate>`

        If |tf_keras_batch_normalization_link| layers are specified in the 
        :attr:`NF.hidden_layers <NF4HEP.NF.hidden_layers>` attribute, then the 
        :attr:`NF.batch_norm <NF4HEP.NF.batch_norm>` attribute is ignored. Otherwise,
        if :attr:`NF.batch_norm <NF4HEP.NF.batch_norm>` is ``True``, then a 
        |tf_keras_batch_normalization_link| layer is added after the input layer and before
        each |tf_keras_dense_link| layer.

        If |tf_keras_dropout_link| layers are specified in the 
        :attr:`NF.hidden_layers <NF4HEP.NF.hidden_layers>` attribute, then the 
        :attr:`NF.dropout_rate <NF4HEP.NF.dropout_rate>` attribute is ignored. Otherwise,
        if :attr:`NF.dropout_rate <NF4HEP.NF.dropout_rate>` is larger than ``0``, then
        a |tf_keras_dropout_link| layer is added after each |tf_keras_dense_link| layer 
        (but the output layer).

        The method also sets the three attributes:

            - :attr:`NF.layers <NF4HEP.NF.layers>` (set to an empty list ``[]``, filled by the 
                :meth:`NF.model_define <NF4HEP.NF.model_define>` method)
            - :attr:`NF.model_params <NF4HEP.NF.model_params>`
            - :attr:`NF.model_trainable_params <NF4HEP.NF.model_trainable_params>`
            - :attr:`NF.model_non_trainable_params <NF4HEP.NF.model_non_trainable_params>`

        - **Arguments**

            - **verbose**

                See :argument:`verbose <common_methods_arguments.verbose>`.

        - **Produces file**

            - :attr:`NF.output_log_file <NF4HEP.NF.output_log_file>`
        """
        #verbose, verbose_sub = self.set_verbosity(verbose)
        #print(header_string,"\nSetting hidden layers\n",show=verbose)
        self.layers_string = []
        self.layers = []
        #layer_string = "layers.Input((self.rem_dims,))"
        # self.layers_string.append(layer_string)
        #print("Added Input layer: ", layer_string, show=verbose)
        i = 0
        if "dropout" in str(self.hidden_layers).lower():
            insert_dropout = False
            self.dropout_rate = "custom"
        elif "dropout" not in str(self.hidden_layers).lower() and self.dropout_rate != 0:
            insert_dropout = True
        else:
            insert_dropout = False
        if "batchnormalization" in str(self.hidden_layers).lower():
            self.batch_norm = "custom"
        for layer in self.hidden_layers:
            if type(layer) == str:
                if "(" in layer:
                    layer_string = "layers."+layer
                else:
                    layer_string = "layers."+layer+"()"
            elif type(layer) == dict:
                try:
                    name = layer["name"]
                except:
                    raise Exception("The layer ", str(layer),
                                    " has unspecified name.")
                try:
                    args = layer["args"]
                except:
                    args = []
                try:
                    kwargs = layer["kwargs"]
                except:
                    kwargs = {}
                layer_string = utils.build_method_string_from_dict(
                    "layers", name, args, kwargs)
            elif type(layer) == list:
                units = layer[0]
                activation = layer[1]
                try:
                    initializer = layer[2]
                except:
                    initializer = None
                if activation == "selu":
                    layer_string = "layers.Dense(" + str(
                        units) + ", activation='" + activation + "', kernel_initializer='lecun_normal')"
                elif activation != "selu" and initializer != None:
                    layer_string = "layers.Dense(" + str(units) + \
                        ", activation='" + activation + "')"
                else:
                    layer_string = "layers.Dense(" + str(units)+", activation='" + \
                        activation + "', kernel_initializer='" + initializer + "')"
            else:
                layer_string = None
                print("Invalid input for layer: ", layer,
                      ". The layer will not be added to the model.", show=verbose)
            if self.batch_norm == True and "dense" in layer_string.lower():
                self.layers_string.append("layers.BatchNormalization()")
                print("Added hidden layer: layers.BatchNormalization()", show=verbose)
                i = i + 1
            if layer_string is not None:
                try:
                    eval(layer_string)
                    self.layers_string.append(layer_string)
                    print("Added hidden layer: ", layer_string, show=verbose)
                except Exception as e:
                    print(e)
                    print("Could not add layer",
                          layer_string, "\n", show=verbose)
                i = i + 1
            if insert_dropout:
                try:
                    act = eval(layer_string+".activation")
                    if "selu" in str(act).lower():
                        layer_string = "layers.AlphaDropout(" + \
                            str(self.dropout_rate)+")"
                        self.layers_string.append(layer_string)
                        print("Added hidden layer: ",
                              layer_string, show=verbose)
                        i = i + 1
                    elif "linear" not in str(act):
                        layer_string = "layers.Dropout(" + \
                            str(self.dropout_rate)+")"
                        self.layers_string.append(layer_string)
                        print("Added hidden layer: ",
                              layer_string, show=verbose)
                        i = i + 1
                except:
                    layer_string = "layers.AlphaDropout(" + \
                        str(self.dropout_rate)+")"
                    self.layers_string.append(layer_string)
                    print("Added hidden layer: ", layer_string, show=verbose)
                    i = i + 1
        if self.batch_norm == True and "dense" in layer_string.lower():
            self.layers_string.append("layers.BatchNormalization()")
            print("Added hidden layer: layers.BatchNormalization()", show=verbose)
        t_layer_string = "layers.Dense(self.tran_ndims, name='t')"
        log_s_layer_string = "layers.Dense(self.tran_ndims, activation='tanh', name='log_s')"
        self.layers_string.append(t_layer_string)
        self.layers_string.append(log_s_layer_string)
        for layer_string in self.layers_string:
            try:
                print("Building layer:", layer_string)
                self.layers.append(eval(layer_string))
            except:
                print("Failed to evaluate:", layer_string)

    def call(self, x):
        """
        """
        # Define and return Model
        y = x
        for layer in self.layers[:-2]:
            y = layer(y)
        t = self.layers[-2](y)
        log_s = self.layers[-1](y)
        return t, log_s


class RealNVPBijector(tfb.Bijector, Verbosity):
    """
    Implementation of a Real-NVP for Denisty Estimation. L. Dinh “Density estimation using Real NVP,” 2016.

    """

    def __init__(self,
                 model_define_inputs,
                 model_flow_inputs,
                 verbose=True):
        self.verbose = verbose
        verbose, verbose_sub = self.set_verbosity(verbose)
        super().__init__(validate_args=model_flow_inputs["validate_args"],
                         forward_min_event_ndims=model_flow_inputs["forward_min_event_ndims"])
        NN = RealNVPNetwork(model_define_inputs)
        self.tran_ndims = NN.tran_ndims
        x = Input((model_define_inputs["rem_dims"],))
        t, log_s = NN(x)
        self.NN = Model(x, [t, log_s])

    def _bijector_fn(self, x):
        t, log_s = self.NN(x)
        #print('this is t')
        # print(t)
        return tfb.Shift(shift=t)(tfb.Scale(log_scale=log_s))
        # return tfb.affine_scalar.AffineScalar(shift=t, log_scale=log_s)

    def _forward(self, x):
        #x_a, x_b = tf.split(x, 2, axis=-1)
        x_a = x[:, :self.tran_ndims]
        x_b = x[:, self.tran_ndims:]
        # print('x_a')
        # print(x_a)
        y_b = x_b
        y_a = self._bijector_fn(x_b).forward(x_a)
        y = tf.concat([y_a, y_b], axis=-1)
        return y

    def _inverse(self, y):
        #y_a, y_b = tf.split(y, 2, axis=-1)
        y_a = y[:, :self.tran_ndims]
        y_b = y[:, self.tran_ndims:]
        x_b = y_b
        x_a = self._bijector_fn(y_b).inverse(y_a)
        x = tf.concat([x_a, x_b], axis=-1)
        return x

    def _forward_log_det_jacobian(self, x):
        #x_a, x_b = tf.split(x, 2, axis=-1)
        x_a = x[:, :self.tran_ndims]
        x_b = x[:, self.tran_ndims:]
        return self._bijector_fn(x_b).forward_log_det_jacobian(x_a, event_ndims=1)

    def _inverse_log_det_jacobian(self, y):
        #y_a, y_b = tf.split(y, 2, axis=-1)
        y_a = y[:, :self.tran_ndims]
        y_b = y[:, self.tran_ndims:]
        return self._bijector_fn(y_b).inverse_log_det_jacobian(y_a, event_ndims=1)


class RealNVPFlow(tfb.Chain, Verbosity):
    """
    
    """

    def __init__(self,
                 model_define_inputs,
                 model_flow_inputs,
                 verbose=True):
        self.verbose = verbose
        verbose, verbose_sub = self.set_verbosity(verbose)
        batch_norm = model_define_inputs["batch_norm"]
        ndims = model_define_inputs["ndims"]
        bijector = RealNVPBijector(model_define_inputs, model_flow_inputs)
        permutation = tf.cast(np.concatenate(
            (np.arange(int(ndims/2), ndims), np.arange(0, int(ndims/2)))), tf.int32)
        Permute = tfb.Permute(permutation=permutation)
        bijectors = []
        for _ in range(model_flow_inputs["num_bijectors"]):
            if batch_norm:
                bijectors.append(tfb.BatchNormalization())
            bijectors.append(bijector)
            bijectors.append(Permute)
        super(RealNVPFlow, self).__init__(bijectors=list(
            reversed(bijectors[:-1])), name='realnvp_chain')
