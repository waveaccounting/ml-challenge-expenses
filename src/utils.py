
import yaml

from datetime import datetime
from sklearn.preprocessing import MinMaxScaler

import os
import numpy as np
import pandas as pd
import importlib

from constants import *





def data_to_np(data):
  '''
  purpose: convert training data into an
    ingestible form (a numpy array of 
    scalar values and one-hot encodings)
  '''


  id_oh = to_one_hot(data['employee id'].values)  
  role_oh = to_one_hot(data['role'].values)
  
  data_np = np.concatenate([
                  id_oh,
                  role_oh,
                  data['date'].values.reshape(-1,1),
                  data['tax amount'].values.reshape(-1,1),
                  data['pre-tax amount'].values.reshape(-1,1)
                  ],axis = -1)
                  
  return data_np

  
  
def to_one_hot(targets):
    cat = np.max(targets) + 1
    res = np.eye(cat)[np.array(targets).reshape(-1)]
    return res.reshape(list(targets.shape)+[cat])  



def load_data():
  '''
  purpose: load the data a process it into a form 
    that is ingestible as an input into a DNN model.
    
    The return will be a dictionary of numpy arrays that can be 
    directly fed into the model.
    
  we will:
    - process the input data into either scalar magnetudes or categorical data.
    - Scale all scalar magnetudes to a range of [0,1]
    - turn all categorical data into a one hot encoding
  '''


  train = pd.read_csv(training_data_file) 
  val = pd.read_csv(validation_data_file) 
  employee = pd.read_csv(employee_file) 
  


  train, val,categories = preprocess(train,val,employee)
  
  X_train= train.drop(columns = 'category')
  X_val = val.drop(columns = 'category')
  
  # we turn our categorical data into one hot format.
  X_train = data_to_np(train)
  X_val = data_to_np(val)
  
  # We finally output or categorical data as a numpy array
  y_train = train['category'].values
  y_val = val['category'].values
  
      
  return {
          'X_train':X_train,
          'y_train':y_train,
          'X_val':X_val,
          'y_val':y_val,
          'features': X_train.shape[1],
          'categories':categories
          }
          
          

    
def create_model(model_param,model_specific_params):
    import_string = ('models.%s' % model_param['model_type'])
    print(import_string)
    model_def = importlib.import_module(import_string)
    return model_def.Model(model_param,model_specific_params)

    
    

 
def dates_to_int(data):
    '''
    purpose: take in an array of string dates with the 
        format dd/mm/yyyy and turn them into inputable data.
    '''

    sec_dates = []
    for item in data['date']:
        sec_dates.append(datetime.strptime(item, '%m/%d/%Y'))

    data['date'] = sec_dates    
    return data
 

def scale_column(train,val,col):
    '''
    purpose: scale the training data to a 
        range of [0,1]. Scale the validation
        data accordingly.
    '''
    
    scaler = MinMaxScaler()
    
    train[col] = scaler.fit_transform(np.reshape(train[col].values,(-1,1)))
    val[col] = scaler.transform(np.reshape(val[col].values,(-1,1))) 
    

    return train,val


def transform_employees(data,employee):
    '''
    purpose: take employee id and extract useful data
        for model.
    '''
    # the only extra useful information is the role so add that into the data.
    dict, reverse = category_to_dict(employee['role'])
    roles = np.zeros(len(data['employee id']),dtype = int)

    for i,id in enumerate(data['employee id']):
        roles[i] = dict[employee['role'][id-1]]
    data['role'] = roles    
    

    return data

def preprocess(train,val,employee):
    '''
    purpose: Take in a training set and validation set
        and the employee records and transform the data
        into something ingestable for a model.
        
        - Categorical data is turned into an integer label
        - Spurious data is dropped
        - Employee data is added to train and val set.
        - Dates are transformed into a number
        - All numerical values are scaled to a range of [0,1]
    '''

    
    # first we will turn the prediction labels into a numerical 
    # representation
    dict, reverse = category_to_dict(train['category'])
    train['category'] = category_to_vec(train['category'],dict)
    val['category'] = category_to_vec(val['category'],dict)
    categories = len(dict)
    
    # we turn the tax type into a numerical type as well
    dict, reverse = category_to_dict(train['tax name'])
    train['tax name'] = category_to_vec(train['tax name'],dict)
    val['tax name'] = category_to_vec(val['tax name'],dict)
    
    # for the purposes of this model we will drop the descriptions
    # with more data I have some better suggestions, but all these
    # labels are basically unique and won't provide much to a model.
    # note: NLP methods can be used to turn this into very useful information
    train = train.drop(columns= ['expense description'])
    val = val.drop(columns= ['expense description'])
    
    # employee information
    train = transform_employees(train,employee)
    val = transform_employees(val,employee)
    
    train = dates_to_int(train)
    val = dates_to_int(val)

    # Now we scale all our columns to a linear range of (0,1).
    # This will just initial training time and allow 
    # easier use of fancy methods like batch norm.
    # NOTE: The validation and training data must be 
    # scaled identically.
    
    train, val = scale_column(train,val,'date')
    train, val = scale_column(train,val,'pre-tax amount')
    train, val = scale_column(train,val,'tax amount')

    print('categories',categories)
    return train, val, categories
    
def category_to_dict(cat):
    '''
    purpose: take an arbitrary set of categories 
        and creates a dictionary and reverse lookup 
        of the categories
        
    cat: a list of categories for a set of data
    
    '''
    dict = {}
    reverse = {}
    for i,item in enumerate(cat.unique()):
        dict[item] = i
        reverse[i] = item
        
    return dict,reverse
    
    
    
def category_to_vec(cat,dict):
    vec = np.zeros(len(cat),dtype = int)
    
    for i,item in enumerate(cat):
        vec[i] = dict[item] 
    
    
    return vec     

def generate_directories(name,id):
    model_dest = os.path.join(name, id,'model')
    log_dir = os.path.join(name, id, 'logs')
    root_dir = os.path.join(name, id)
    return root_dir, model_dest,log_dir
    
        
    
      