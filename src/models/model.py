from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import re

from constants import *



    
    



class Net(object):
  """Base Net class 
  """
  def __init__(self, model_param,specific_param):

    pass
    #pretrained variable collection
    #self.pretrained_collection = []
    #trainable variable collection
    #self.trainable_collection = []
  def str_to_bool(self,s):

      s = s.strip()
      print(s)
      if s == 'True':
           return True
      elif s == 'False':
           return False
      else:
           raise ValueError # evil ValueError that doesn't tell you what the wrong value wa    
  


  def choose_initializer(self,init_def,init_deviation=STDDEV):
    if init_def == 'Normal':
      #print('Normal Init')
      initializer = tf.random_normal_initializer(stddev=init_deviation)
    elif init_def == 'Uniform':
      #print('Uniform Init')
      initializer = tf.random_uniform_initializer(-1*init_deviation,init_deviation)
    elif init_def == 'Xavier':  
      #print('Xavier Init')
      initializer = tf.contrib.layers.xavier_initializer()
    else:
      print('NO INITIALIZER SPECIFIED!! Check parameters. Using Normal distribution')
      initializer = tf.random_normal_initializer(stddev=init_deviation)
      
      
      
    return initializer
  
  def choose_regulizer(self,reg_def,reg):
    if reg_def == 'L2':
      #print('L2 reg')
      regulizer = tf.contrib.layers.l2_regularizer(reg)
    elif reg_def == 'L1':
      #print('L1 reg')
      regulizer = tf.contrib.layers.l1_regularizer(reg)
    else:
      #print('No reg')
      regulizer = None
    
    return regulizer
    
  def log10(self,x):
    numerator = tf.log(x)
    denominator = tf.log(tf.constant(10, dtype=numerator.dtype))
    return numerator / denominator    





  def flatten_conv2d(self,x):
    '''
    Flattens a 2d conv layer...
    '''

    (batch_size,x_size,y_size,chan) = x.shape
    size = x_size*y_size*chan

    flat = tf.reshape(x,[-1,size])
    return flat


  ##################################################
  '''
  A convolutional layer!!!!

  x is expected to have shape (batch_size, x,y, in chan) (3 for rgb for example))
  the output will have shape (batch_size ,x,y, out chan) or filters...

  '''
  ##################################################
  def conv2d(self,x,
          filter_x=3,filter_y=3,
          out_chan =10,
          stride = 1,is_training = True,flatten = False,
          init_deviation = STDDEV,reg = REG, reg_def = None,
          do_batch_norm = False,
          padding = 'SAME'):

          
    in_chan = x.shape[3]
    
    #define variables
    W = tf.get_variable("Wconv", shape=[filter_x, filter_y, in_chan, out_chan],
            initializer=tf.random_normal_initializer(stddev=init_deviation),
            #regularizer = tf.contrib.layers.l2_regularizer(reg)  
            regularizer = self.choose_regulizer(reg_def,reg) 
          )
    b = tf.get_variable("bconv", shape = [out_chan],initializer=tf.constant_initializer(0.0))

    # apply conv. layer
    y = tf.nn.conv2d(x, W, strides=[1,stride,stride,1], padding=padding) + b

    # apply a ReLU filter
    y = tf.nn.leaky_relu(y)    
    
    #spacial normalization    
    if do_batch_norm == True:
      y = tf.contrib.layers.batch_norm(
        y,
        data_format='NHWC',  # Matching the "cnn" tensor 
        center=True,
        scale=True,
        is_training=is_training)

      
    if (flatten == True):
      y = self.flatten_conv2d(y)

    return y

    
  def linear(self,x,output_dim,is_training,
          reg_def = None, 
          do_batch_norm = False, init_deviation = STDDEV,
          reg = REG,
          do_drop_out = False, drop_out_rate = DROP_OUT_RATE,
          residual = False
          ):
    """
    affine transformation Wx+b
    assumes x.shape = (batch_size, num_features)
    """
    # define trainable weights
    w=tf.get_variable("w", [x.get_shape()[1], output_dim],
          initializer=tf.random_normal_initializer(stddev=init_deviation),
          # this line is causing a warning... how to supress/fix???
          regularizer = self.choose_regulizer(reg_def,reg) 
        ) 
        
        
    b=tf.get_variable("b", [output_dim], initializer=tf.constant_initializer(0.0))
        
    
    if do_drop_out == True:
      x = tf.layers.dropout(x,rate=drop_out_rate,training= is_training)
  
        
    # apply linear layer.
    y = tf.matmul(x,w)+b
    
    # batch norm
    if do_batch_norm == True:
      y = tf.layers.batch_normalization(y,axis = 1, training = is_training,trainable = True) 
    
    # activation function
    # in a word embedding we do not need the non-linearity.
    y = tf.nn.leaky_relu(y)
    #y = tf.nn.relu(y)
    
    # do a residual layer ONLY IF SAME SIZE
    if (residual == True) and (x.get_shape()[1] == output_dim):
      y = x + y
    
    
    return y
  

  def _variable_on_cpu(self, name, shape, initializer, pretrain=True, train=True):
    """Helper to create a Variable stored on CPU memory.

    Args:
      name: name of the Variable
      shape: list of ints
      initializer: initializer of Variable

    Returns:
      Variable Tensor
    """
    with tf.device('/cpu:0'):
      var = tf.get_variable(name, shape, initializer=initializer, dtype=tf.float32)
      if pretrain:
        self.pretrained_collection.append(var)
      if train:
        self.trainable_collection.append(var)
    return var 


  def _variable_with_weight_decay(self, name, shape, stddev, wd, pretrain=True, train=True):
    """Helper to create an initialized Variable with weight decay.

    Note that the Variable is initialized with truncated normal distribution
    A weight decay is added only if one is specified.

    Args:
      name: name of the variable 
      shape: list of ints
      stddev: standard devision of a truncated Gaussian
      wd: add L2Loss weight decay multiplied by this float. If None, weight 
      decay is not added for this Variable.

   Returns:
      Variable Tensor 
    """
    var = self._variable_on_cpu(name, shape,
      tf.truncated_normal_initializer(stddev=stddev, dtype=tf.float32), pretrain, train)
    if wd is not None:
      weight_decay = tf.multiply(tf.nn.l2_loss(var), wd, name='weight_loss')
      tf.add_to_collection('losses', weight_decay)
    return var 


  def max_pool(self, input, kernel_size, stride):
    """max_pool layer

    Args:
      input: 4-D tensor [batch_zie, height, width, depth]
      kernel_size: [k_height, k_width]
      stride: int32
    Return:
      output: 4-D tensor [batch_size, height/stride, width/stride, depth]
    """
    return tf.nn.max_pool(input, ksize=[1, kernel_size[0], kernel_size[1], 1], strides=[1, stride, stride, 1],
                  padding='SAME')

  def local(self, scope, input, in_dimension, out_dimension, leaky=True, pretrain=True, train=True):
    """Fully connection layer

    Args:
      scope: variable_scope name
      input: [batch_size, ???]
      out_dimension: int32
    Return:
      output: 2-D tensor [batch_size, out_dimension]
    """
    with tf.variable_scope(scope) as scope:
      reshape = tf.reshape(input, [tf.shape(input)[0], -1])

      weights = self._variable_with_weight_decay('weights', shape=[in_dimension, out_dimension],
                          stddev=0.04, wd=self.weight_decay, pretrain=pretrain, train=train)
      biases = self._variable_on_cpu('biases', [out_dimension], tf.constant_initializer(0.0), pretrain, train)
      local = tf.matmul(reshape, weights) + biases

      if leaky:
        local = self.leaky_relu(local)
      else:
        local = tf.identity(local, name=scope.name)

    return local

  def leaky_relu(self, x, alpha=0.1, dtype=tf.float32):
    """leaky relu 
    if x > 0:
      return x
    else:
      return alpha * x
    Args:
      x : Tensor
      alpha: float
    Return:
      y : Tensor
    """
    x = tf.cast(x, dtype=dtype)
    bool_mask = (x > 0)
    mask = tf.cast(bool_mask, dtype=dtype)
    return 1.0 * mask * x + alpha * (1 - mask) * x

    
    
  def rmsle(self, y_true, y_pred):
    return tf.sqrt(tf.reduce_mean(tf.pow(tf.log1p(y_true) - tf.log1p(y_pred), 2)))  
    