__all__ = ['CSplineNetwork','CSplineBijector','CSplineFlow']

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions
tfb= tfp.bijectors

from tensorflow.python.keras import layers, Input
from tensorflow.keras.layers import Layer, Dense
from tensorflow.keras.models import Model

from .rqspline import RQSplineBijector

from . import inference, utils#, custom_losses
from .show_prints import Verbosity, print


class CSplineNetwork(Verbosity):
    """
    """
    def define_network(self, verbose = None):
        verbose, verbose_sub = self.set_verbosity(verbose)
        self.NN = tfb.AutoregressiveNetwork(**self.model_define_inputs)

    """
    Neural Network Architecture for calcualting s and t for Real-NVP
  
    """
    def __init__(self, 
                 tran_dims, 
                 spline_knots,
                 range_min, 
                 n_hidden=[5,5],
                 activation="relu",
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros', 
                 kernel_regularizer=None,
                 bias_regularizer=None, 
                 activity_regularizer=None, 
                 kernel_constraint=None,
                 bias_constraint=None):
        super(CSplineNetwork, self).__init__()
        
        
        #print(n_hidden)
        self.tran_dims=tran_dims
        self.range_min=range_min
        layer_list = []
        for i, hidden in enumerate(n_hidden):
            layer_list.append(Dense(hidden, activation=activation,use_bias=use_bias,
        kernel_initializer=kernel_initializer,
        bias_initializer=bias_initializer, kernel_regularizer=kernel_regularizer,
        bias_regularizer=bias_regularizer, activity_regularizer=activity_regularizer, kernel_constraint=kernel_constraint,
        bias_constraint=bias_constraint))
        self.layer_list = layer_list
       # self.log_s_layer = Dense(input_shape, activation="tanh", name='log_s')
       # self.t_layer = Dense(input_shape, name='t')
        
        bin_widths=[]
        bin_heights=[]
        knot_slopes=[]
      
        for _ in range(tran_dims):
            bin_widths.append(Dense(spline_knots,activation="softmax"))
            bin_heights.append(Dense(spline_knots,activation="softmax"))
    
        for _ in range(tran_dims):
            knot_slopes.append(Dense(spline_knots-1,activation="softplus"))
            
        
        '''
        bin_widths=Dense(self.tran_dims*spline_knots,activation="softmax")
        bin_heights=Dense(self.tran_dims*spline_knots,activation="softmax")
        knot_slopes=Dense(self.tran_dims*(spline_knots-1),activation="softplus")
        '''
        
        
        
        self.bin_widths=bin_widths
        self.bin_heights=bin_heights
        self.knot_slopes=knot_slopes
        
       
        

    def call(self, x):
      
        y = x
      
        for layer in self.layer_list:
            y = layer(y)
            
        
        bin_widths=[]
        bin_heights=[]
        knot_slopes=[]
        for j in range(self.tran_dims):
            bin_widths.append(self.bin_widths[j](y))
            #print('self bin)widths')
            #print(self.bin_widths[j](y))
            bin_heights.append(self.bin_heights[j](y))
    
        for j in range(self.tran_dims):
            knot_slopes.append(self.knot_slopes[j](y))
        '''
        #print('ageqrgkwegbkehjb kehbgqkeruhbgke')
        #print(y)
        
        #print('hhjhjss')
        bin_widths=tf.keras.layers.Reshape((self.tran_dims,spline_knots))(bin_widths)
        #print('heeeeeyy')
        bin_heights=tf.keras.layers.Reshape((self.tran_dims,spline_knots))(bin_heights)
        #print('crrroooor')
        knot_slopes=tf.keras.layers.Reshape((self.tran_dims,spline_knots-1))(knot_slopes)
        #print('ggooog')
        
        bin_widths = Lambda(lambda x: x *(10+abs(self.range_min)))(bin_widths)
        bin_heights = Lambda(lambda x: x *(10+abs(self.range_min)))(bin_heights)
        knot_slopes = Lambda(lambda x: x +10e-5)(knot_slopes)
        '''
        '''
        
        bin_widths=self.bin_widths(y)
        bin_heights=self.bin_heights(y)
        knot_slopes=self.knot_slopes(y)
        '''
        
        return bin_widths,bin_heights,knot_slopes

class CSplineBijector(CSplineNetwork):
    """
    """
    def define_bijector(self, verbose = None):
        verbose, verbose_sub = self.set_verbosity(verbose)
        self.Bijector = tfb.MaskedAutoregressiveFlow(shift_and_log_scale_fn=self.NN)

    """
    """
    """
    Implementation of a Cspline

    """

    def __init__(self, ndims, rem_dims, spline_knots, range_min, n_hidden=[64, 64, 64], activation='relu', use_bias=True,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros', kernel_regularizer=None,
                 bias_regularizer=None, activity_regularizer=None, kernel_constraint=None,
                 bias_constraint=None, forward_min_event_ndims=1, validate_args: bool = False):
        super(CSplineBijector, self).__init__(
            validate_args=validate_args, forward_min_event_ndims=forward_min_event_ndims
        )

        if rem_dims < 1 or rem_dims > ndims-1:
            print('ERROR: rem_dims must be 1<rem_dims<ndims-1')
            exit()

        self.range_min = range_min
        self.tran_ndims = ndims-rem_dims

        self.spline_knots = spline_knots
        #input_shape = input_shape // 2
        nn_layer = CSplineNetwork(self.tran_ndims, self.spline_knots, self.range_min, n_hidden, activation, use_bias,
                      kernel_initializer,
                      bias_initializer, kernel_regularizer,
                      bias_regularizer, activity_regularizer, kernel_constraint,
                      bias_constraint)
        x = tf.keras.Input((rem_dims,))
        bin_widths, bin_heights, knot_slopes = nn_layer(x)

        self.nn = Model(x, [bin_widths, bin_heights, knot_slopes])

    #@tf.function

    def _bijector_fn(self, x):

        def reshape():

            [bin_widths, bin_heights, knot_slopes] = self.nn(x)

            #print('hello')

            #print(tf.shape(x)[0])
            #print(output[0])

            #output=tf.reshape(output, (x.shape[0],self.tran_ndims,3*spline_knots-1), name=None)

            #bin_widths=output[:,:,:spline_knots]
            #bin_widths=tf.reshape(output[0], (x.shape[0],self.tran_ndims,self.spline_knots), name=None)
            bin_widths = tf.reshape(bin_widths, (tf.shape(
                x)[0], self.tran_ndims, self.spline_knots), name=None)
            bin_widths = tf.math.scalar_mul(tf.constant(
                2*abs(self.range_min), dtype=tf.float32), bin_widths)
            #print('bin_widths')
            #print(bin_widths)

            #bin_heights=tf.reshape(output[1], (x.shape[0],self.tran_ndims,self.spline_knots), name=None)
            bin_heights = tf.reshape(bin_heights, (tf.shape(
                x)[0], self.tran_ndims, self.spline_knots), name=None)
            bin_heights = tf.math.scalar_mul(tf.constant(
                2*abs(self.range_min), dtype=tf.float32), bin_heights)
            #print('bin_heights')
            #print(bin_heights)

            #knot_slopes=tf.reshape(output[2], (x.shape[0],self.tran_ndims,self.spline_knots-1), name=None)+tf.constant(1e-5,dtype=tf.float32)
            knot_slopes = tf.reshape(knot_slopes, (tf.shape(
                x)[0], self.tran_ndims, self.spline_knots-1), name=None)
            #print('knot_slopes')
            #print(knot_slopes)
            #knot_slopes=tf.math.scalar_mul(2*abs(self.range_min),knot_slopes)

            return bin_widths, bin_heights, knot_slopes

        bin_widths, bin_heights, knot_slopes = reshape()
        #print('hey')
        #RQS=tf.cast(RQS,dtype=tf.float32)
        return RQSplineBijector(
            bin_widths=bin_widths, bin_heights=bin_heights, knot_slopes=knot_slopes, range_min=self.range_min, validate_args=False)

    def _forward(self, x):
        #x_a, x_b = tf.split(x, 2, axis=-1)

        x_a = x[:, :self.tran_ndims]
        x_b = x[:, self.tran_ndims:]
        #print(x_b)
        y_b = x_b
        #print('gggggghhhggg')
        y_a = self._bijector_fn(x_b).forward(x_a)
        #print('did i get here?')
        #print('y_a')
        #print(y_a)
        #print(y_a)
        y = tf.concat([y_a, y_b], axis=-1)

        return y

    def _inverse(self, y):
        #y_a, y_b = tf.split(y, 2, axis=-1)
        #print('niverse')
        y_a = y[:, :self.tran_ndims]
        y_b = y[:, self.tran_ndims:]
        x_b = y_b
        x_a = self._bijector_fn(y_b).inverse(y_a)
        x = tf.concat([x_a, x_b], axis=-1)
        #print('hello')
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

class CSplineFlow(CSplineBijector):
    def CsplineN(ndims,
                 rem_dims,
                 spline_knots,
                 n_bijectors,
                 range_min,
                 n_hidden=[128,128,128],
                 activation='relu',
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros', 
                 kernel_regularizer=None,
                 bias_regularizer=None, 
                 activity_regularizer=None, 
                 kernel_constraint=None,
                 bias_constraint=None):
        bijectors=[]
        permutation=tf.cast(np.concatenate((np.arange(int(ndims/2),ndims),np.arange(0,int(ndims/2)))), tf.int32)
        #bijectors.append(tfb.BatchNormalization())
        for i in range(n_bijectors):
            #bijectors.append(tfb.BatchNormalization())
            bijectors.append(CSplineBijector(ndims, rem_dims, spline_knots, range_min, n_hidden, activation, use_bias,
        kernel_initializer,
        bias_initializer, kernel_regularizer,
        bias_regularizer, activity_regularizer, kernel_constraint,
        bias_constraint))
            bijectors.append(tfp.bijectors.Permute(permutation))
        bijector = tfb.Chain(bijectors=list(reversed(bijectors[:-1])))
        return bijector

class CSpline(CSplineFlow):
    """
    """
    def __init__(self,
                 model_define_inputs = None,
                 model_chain_inputs = None,
                 verbose = True):
        self.verbose = verbose
        verbose, verbose_sub = self.set_verbosity(verbose)
        self.__model_define_inputs = model_define_inputs
        self.__model_chain_inputs = model_chain_inputs
        # setting inputs
        self.__set_inputs(verbose = verbose_sub)
        # define network through the __define_network method of MAFNetwork
        self.define_network()
        # ora ho self.NN -> define bijector through the __define_bijector method of MAFBijector
        self.define_bijector()
        # ora ho self.Bijector -> define flow through the __define_flow method of MAFFlow
        self.define_flow()

    def __set_inputs(self, verbose=None):
        self.model_define_inputs = utils.dic_minus_keys(self.__model_define_inputs, ["batch_norm"])
        self.batch_norm = self.__model_define_inputs["batch_norm"]
        self.ndims = self.__model_chain_inputs["ndims"]
        self.num_bijectors = self.__model_chain_inputs["num_bijectors"]