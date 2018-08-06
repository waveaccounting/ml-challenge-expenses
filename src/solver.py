'''
xxx
yyy
This solver unit is to simplify the model 
code and to reduce repeated code...

it is designed to take in a model and data and train the thing.
'''

import os
import numpy as np
import math
import io
import tensorflow as tf

# utils imports
import utils as utils
from constants import *

class Solver(object):
  """
   This creates a solver architecture to
   train a word2vec model.
   
   It has two primary functions:
    train -> train a given model with a given dataset
    run   -> extract a list of learned word embeddings from 
      the model.
      
      
    The solver takes target_word data, context data as 
    word pairings. These should be provided as numpy arrays.
  """
  
  def __init__(self,model):
    self.m = model


  def restore_model_weights(self,saver,sess,root_dir,model_dest):
    '''
    This routine checks to see if there are available weights to load. 
    Given the models destination. If not the weights will be initialized 
    randomly.
    '''
    fname = './%s/checkpoint' % (root_dir)
    if os.path.exists(root_dir):
      print('Trying to load BEST model...')
      try :
        best_model_dest = '%s_best' % model_dest
        saver.restore(sess,best_model_dest)
        print('BEST Model Loaded.')
      except :
        print('Trying to load finished model...')
        try :
          saver.restore(sess,model_dest)
          print('Model Loaded.')
        except tf.errors.NotFoundError:
          print('Could not find finished model. Trying to load last checkpoint...')
          try :
            saver.restore(sess,tf.train.latest_checkpoint(root_dir))  
            print('Last checkpoint loaded.')
          except :
            print('Initializing weights randomly.')
            tf.global_variables_initializer().run()  

    else:  
      print('No previous model found. Initializing weights randomly...')
      tf.global_variables_initializer().run()  


  
      
  def optimize(self,learning_rate,var_list = None):    
    ## OPTIMIZER ## 
    
    model = self.m
    with tf.name_scope('Optimize'):
      # the adam optimizer is much more stable and optimizes quicker than
      # basic gradient descent. (*Ask me for details*)
      #optimizer=tf.train.AdamOptimizer(learning_rate, beta1=0.9)
      optimizer=tf.train.GradientDescentOptimizer(learning_rate)
      
      update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
      with tf.control_dependencies(update_ops):
        # This is generally for fine-tuning,
        # Instead of optimizing all variables you can 
        # choose a set to train.
        if (var_list == None):
          grads=optimizer.compute_gradients(model.loss)
        else:
          grads=optimizer.compute_gradients(model.loss,var_list = var_list)
          
        # we cap the gradient sizes to make sure we do not
        # get outlier large grads from computational small number errors
        with tf.name_scope('Clip_Grads'):
          for i,(g,v) in enumerate(grads):
              if g is not None:
                  grads[i]=(tf.clip_by_norm(g,5),v) # clip gradients
                  
          self.train_op = optimizer.apply_gradients(grads)
          
    return         

  
  def train(self,X,y,param,X_val = [], y_val = []):
    '''
    This routine takes in the database connection to the 
    training data target_words and 
    context pairing and trains the word embedding, using the 
    model architecture fed into the solver object.
    
    The model variables will be initialized as specified, unless
    a previous model has been saved to the models destination directory.
    
    All logs and model data will be saved and loaded from
     './model.name/id/' as defined in the parameters.
     
     
    All hyper parameters can be adjusted in paramters.yml. 
    
    '''
  
    # define params locally for readability
    model = self.m     
    learning_rate = param['learning_rate']
    epoch = param['epoch']
    batch_size = param['batch_size']
    id = param['id']
    N = X.shape[0]

    # set up all the directories to send the model and logs...
    root_dir, model_dest,log_dir = utils.generate_directories(model.name,id)
    best_dest = '%s_best' % (model_dest)

   
  
    # get training op.. set learning rate...
    self.optimize(learning_rate)


    # set the training feed.
    fetches=[]
    fetches.extend([self.train_op])
    
    # set up the validation fetches...
    val_fetches=[]
    val_fetches.extend([
                          model.loss
                          ])
  
    

    # determine the counter values.
    iterations_per_epoch = N // batch_size
    if (N % batch_size) > 0:
      iterations_per_epoch+=1  


    num_iterations = epoch * iterations_per_epoch

    print('***********************************************')
    print('Begining training of %s model. ID: %s' % (model.name,id))    
    print('Training Points: %d' % (N))
    print('Batch size = %d' % (batch_size))
    
    print('Epoch: %d  iters/epoch: %d' % (epoch,iterations_per_epoch))
    print('Total Iterations: %d' % (num_iterations))
    print('model will be saved as: %s' % model_dest)
    print('logs will be stored in: %s' % log_dir)

    
    best_val = 9999999
    best_val_epoch = 0
    #with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as sess:
    with tf.Session() as sess:

      # setup a saver object to save model...
      # create model saver
      saver = tf.train.Saver()
      
    
      # initialize variables -> need to add code to load if loading trained model.
      self.restore_model_weights(saver,sess,root_dir,model_dest)
  
      # create session writers
      writer = tf.summary.FileWriter(os.path.join(log_dir,'train'),sess.graph) # for 1.0
      val_writer = tf.summary.FileWriter(os.path.join(log_dir,'val'),sess.graph)
      #test_writer = tf.summary.FileWriter(os.path.join(test_dir , model.model_name)) # for 1.0
      merged = tf.summary.merge_all()
      
      
      
      print('Begin Training')
      print('***********************************************')
      tot_itts = 0
      for e in range(epoch):
        # create a mask for the data
        mask = np.arange(N)
        np.random.shuffle(mask)
        

          
        for i in range(iterations_per_epoch):
            
            # print out tracking information to make sure everything is running correctly...
            if ( (i + iterations_per_epoch*e ) % param['read_out'] == 0):
              print('%d)  %d of %d for epoch %d' % (tot_itts ,i, iterations_per_epoch,e))
              
            # Grab the batch data... (handle modified batches...)
            if batch_size*(i + 1) > N:
              X_b = X[mask[batch_size*i:]]
              y_b = y[mask[batch_size*i:]]            
            else :
              X_b = X[mask[batch_size*i:batch_size*(i +1)]]
              y_b= y[mask[batch_size*i:batch_size*(i +1)]]
            

            feed_dict={model.x:X_b,
                       model.y:y_b,
                       model.is_training:True}      
            

            
            
            # do training on batch, return the summary and any results...
            [summary,_] = sess.run([merged,fetches],feed_dict)

            
            # write summary to writer
            writer.add_summary(summary, tot_itts)
            tot_itts+=1
          
        # epoch done, check word similarities....
        # checkpoint the model while training... 
        if (e % param['model_check_point'] == 0):
          saver.save(sess,model_dest, global_step=e+1)
          
        if (e % param['validation_check_point'] == 0): 
          # record accuracy with validation data
          # add a best model routine.
          if not X_val == []:
            feed_dict={model.x:X_val,
             model.y:y_val,
             model.is_training:False}
            

            [summary,info]=sess.run([merged,val_fetches],feed_dict) 


            [val_loss] = info
            
            # check for best model... if so, save.
            if (val_loss < best_val):
              print('best model saved. %1.4f improved on %1.4f' % (val_loss,best_val))
              best_val = val_loss
              best_val_epoch = e
              saver.save(sess,best_dest)
              
            
            val_writer.add_summary(summary, tot_itts)
          
          
          
        print('%d of %d epoch complete.' % (1+e,epoch))
        
        
  
      ## TRAINING FINISHED ##
      # saves variables learned during training
      saver.save(sess,model_dest)
      
      # make sure the log writer closes and sess is done.
      writer.close()
      val_writer.close()
      sess.close()  


    print('***********************************************')
    print('Done training')
    print('model saved to: %s' % model_dest)
    print('logs stored in: %s' % log_dir)
    print('***********************************************')
    return