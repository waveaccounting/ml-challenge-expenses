"""
Author: David McNamee
E-mail: krebbet@gmail.com

This script will read in the training data set 
and extract the personal expenses and write them 
out to 'personal_data_file'.

A task like this can not be completed with very much 
percision without more information and more data. I 
have discussed this and possible solutions in README.md, but for the 
purposes of demonstration I have included this simple
solution. 

The 'expense description' column is searched for predefined
tagged words that identify personal expenses. This list could 
be formed by searching through properties of a larger data set. 
But here I have just added a few words for demonstration.
"""

import pandas as pd
import os


from constants import *

# constants for the script.
tags = ['personal','family','coffee']
personal_data_file = 'personal.csv'

def main():

  # load the data
  train = pd.read_csv(training_data_file) 
  
  # init variables
  personal = []
  
  # search through each expense description to see if 
  # any of the predefined tags are present in the string.
  for i,str in enumerate(train['expense description']):
    tag_found = False
    for tag in tags:
      if tag in str.lower():
        tag_found = True
        
    # if any of the tags are present add to list of personal expenses entries.
    if tag_found:
      personal.append(i)
  
  # subset the personal data and write it out to a csv file.
  personal_data = train.iloc[personal]
  personal_data.to_csv(personal_data_file)
  
  
  print('personal expenses have been written to %s' % personal_data_file)



if __name__ == '__main__':
    main()
    
    
    
    
    
    
    


                  
                  
                  
                  
                  
                  
                  
                  
                  
                  
                  
                  
                  
                  
                  
                  
                  