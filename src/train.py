"""
Author: David McNamee
E-mail: krebbet@gmail.com

This is the main training script to train a DNN model on 
a dataset with the form given in the question set. 


For details please see README.md
"""

# bring in imports
import yaml
import numpy as np
import pandas as pd
import os

# local imports 
from solver import Solver
import utils 
from constants import *


def main():

  # read in the parameters defining the model.
  parameters = yaml.load(open(PARAM_FILE_DIRECTORY))

  # read in all the parameters defining the model and 
  # how it will be trained.
  solver_param = parameters['solver_param']        
  model_param = parameters['model_param']
  model_specific_params = parameters[model_param['model_type']]
  print("Training data using %s model" % (model_param['model_type']))

  print('Loading Data set...')  
  data = utils.load_data()

  print('Drawing model...')
  # initialize the model
  model = utils.create_model(model_param,model_specific_params)
  
  # build it with feature, label sizes...
  model.build(data['features'],data['categories'])
  print('Model drawn.')


  # Initialize the solver object.
  solver = Solver(model)
  
  # train model....
  solver.train(data['X_train'],data['y_train'],solver_param,data['X_val'],data['y_val'])
  
  
  print('done!')



if __name__ == '__main__':
    main()
    
    
    
    