from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import math

from models.model import Net
from constants import *



class Model(Net):
    def __init__(self,model_param, model_def):
        """
        common params: a params dict
        model_params   : a params dict
        """

        self.name = model_param['model_type']

        # Define all the model variables....
        self.batch_norm = model_def['batch_norm']
        self.drop_out =  model_def['drop_out']
        self.reg_type = model_def['reg_type']
        self.reg_strength = model_def['reg_strength']
        self.layers = model_def['layers']
        
        self.drop_out_rate = model_def['drop_out_rate']
        self.residual = model_def['residual']
        


    def build(self, features, categories):
        # build graph!
        self.inference(features,categories)
        self.loss()
        print('build model complete!')
        

    def inference(self, features,categories):
        
        # Define our input data placeholders
        # we have both scalar quantities and categorical data 
        # we turn the categorical data into one-hot representation
        # and concatenate all the features together.
        self.x = tf.placeholder(tf.float32,[None,features])

        
        
        
        
        # create a categorical input and transform into one hot representation.
        self.y = tf.placeholder(tf.int64,[None])
        self.y_onehot = tf.one_hot(self.y,categories)
        
 
        # grab batch size  -> this is not needed in this model.
        # but is often useful to have on hand.
        self.batch_size = tf.shape(self.x)[0]


        
        # this propogates whether the model is training or being
        # used for inference for methods like dropout and batchnorm.
        self.is_training = tf.placeholder(tf.bool, name='is_training')


        # define the fc model....
        h = self.x
        for i in range(len(self.layers) - 1):
            var_scope = ('hidden%d' % i)
            with tf.variable_scope(var_scope):
                h = self.linear(
                    h,
                    output_dim=self.layers[i],
                    is_training=self.is_training,
                    reg_def = self.reg_type,
                    do_batch_norm=self.batch_norm,
                    init_deviation = 1.0 / math.sqrt(float(h.shape[1].value)),
                    reg=self.reg_strength,
                    do_drop_out = self.drop_out,
                    drop_out_rate = self.drop_out_rate,
                    residual = self.residual
                )
  
        # now compute scores from the last hidden layer.
        self.scores = self.linear(
            h,
            output_dim=categories,
            is_training=self.is_training,
            init_deviation = 1.0 / math.sqrt(float(categories))
        )





    def loss(self):

    
      # we will use soft-max loss
      self.scores = tf.nn.softmax(self.scores)

    
      # get the cross entropy loss
      L_class = -tf.reduce_mean(self.y_onehot * tf.log(self.scores + EPS),1)
      L_class = tf.reduce_mean(L_class)
      
      # get predictions
      self.correct_prediction = tf.equal(tf.argmax(self.scores, 1), self.y)
      # %% And now we can look at the mean of our network's correct guesses
      self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, "float"))
    
      
      # grab our regularization loss
      L_reg = sum( tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))

      self.loss = L_class + L_reg


      # collect logging variables
      tf.summary.scalar("Reg_Loss", L_reg)
      tf.summary.scalar("Class_Loss", L_class)
      tf.summary.scalar("Loss", self.loss)
      tf.summary.scalar("accuracy", self.accuracy)



