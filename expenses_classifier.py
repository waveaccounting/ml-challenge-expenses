#!/usr/bin/env python

import tensorflow as tf
import numpy as np

from file_parser import *

# converting text into vectors
#################################################
def check_expense(ID, sub_string, string):   
   if sub_string.lower() in string.lower():
      return ID
   else:
      return 0

def create_data_attributes(training_list):
   training_list_mod = []
   new_record = []

# the conversion into vectors is supposed to spread apart the values for business and personal expenses

   for record in training_list:
#      month,day,year = record[0].split("/")
#      day_value = int(month)*30 + int(day) + (int(year)-2010)*365
      new_record = []
#      new_record.append(day_value)
      category_value = {
         "Travel": 2,
         "Meals and Entertainment": -2,
         "Computer - Hardware": 5,
         "Computer - Software": 5,
         "Office Supplies": 10}

      new_record.append(category_value.get(record[1], "0")) # when none of the categories is a match
           
      expense_description  = 0
      expense_description += check_expense(2,   "taxi",         record[3])
      expense_description += check_expense(10,  "team",         record[3])
      expense_description += check_expense(5,   "laptop",       record[3])
      expense_description += check_expense(-10, "coffee",       record[3])
      expense_description += check_expense(3,   "flight",       record[3])
      expense_description += check_expense(0,   "airplane",     record[3])
      expense_description += check_expense(3,   "dinner",       record[3])
      expense_description += check_expense(10,  "client",       record[3])

      new_record.append(expense_description)

      price_criteria = float(record[4])/10-10;
      if (price_criteria > 10): price_criteria = 10

      new_record.append(price_criteria)

      training_list_mod.append(new_record)      
      training_list_ = tf.stack(axis=0, values=training_list_mod)

   return training_list_
#################################################
# Main program

seed = 700

training_list,   training_header   = parse_csv_data("training_data_example.csv")
#training_list,   training_header   = parse_csv_data("validation_data_example.csv")

training_samples                   = create_data_attributes(training_list) # create attributes from the data set

#training_samples = tf.constant([[float(1), float(1)], [float(10), float(10)], [float(2), float(1)], [float(11), float(11)]])


init = tf.global_variables_initializer()

# get two random points, which will serve as centroids
n_samples = tf.shape(training_samples)[0]                             # gets the number of samples
random_indexes = tf.random_shuffle(tf.range(0, n_samples), seed=seed) # reshuffles the array of indexes
centroid_indexes = tf.slice(random_indexes, [0], [2])                 # takes the first 2 indexes from the reshuffled array
initial_centroids = tf.gather(training_samples, centroid_indexes)     # gets the samples, which correspond the indexes

current_centroids = tf.placeholder("float32")
############################################

# get the set of the samples closest to the centroids
expanded_vectors   = tf.expand_dims(training_samples, 0)              # for broadcasting
expanded_centroids = tf.expand_dims(current_centroids, 1)             # for broadcasting

distances = tf.reduce_sum( tf.square(                                 # list of distances from the samples to both centroids
            tf.subtract(expanded_vectors, expanded_centroids)), 2)
nearest_indexes = tf.argmin(distances, 0)                             # for every sample the closest centroid index: 0 or 1
############################################

# update the centroid values
nearest_indexes_int32 = tf.to_int32(nearest_indexes)                             # dynamic partition doesn't support int64
partitions = tf.dynamic_partition(training_samples, nearest_indexes_int32, 2)    # groups training_samples based on the nearest_indexes: those, which belong to centroid 1 and those, which belong to centroid 2
new_centroids = tf.concat(axis=0, values=[tf.expand_dims(tf.reduce_mean(partition, 0), 0) for partition in partitions]) # calculates the mean across the dimensions of every one out of the two groups. This becomes to be a new centroid.

#update_centroids = tf.assign(current_centroids, new_centroids)                # updating centroids with the new values

############################################

with tf.Session() as session:
   session.run(init)

   initial_centroids_     = session.run(initial_centroids)
#   print initial_centroids_


   nearest_indexes_result = session.run(nearest_indexes, feed_dict={current_centroids: initial_centroids_})
   current_centroids_     = session.run(new_centroids,   feed_dict={current_centroids: initial_centroids_})
#   print nearest_indexes_result
#   print current_centroids_

   for i in xrange(9):  # 9 iterations is more than sufficient for convergence. [FIXME] Ideally, to be replaced with "while"
      nearest_indexes_result = session.run(nearest_indexes, feed_dict={current_centroids: current_centroids_})
      current_centroids_     = session.run(new_centroids,   feed_dict={current_centroids: current_centroids_})
#      print nearest_indexes_result
#      print current_centroids_


# printing the output with a human interface
for index in range(len(training_list)):  
   if ("client" in training_list[index][3].lower()) & (nearest_indexes_result[index] == 1): 
      polarity = 1
      continue
     
for index in range(len(training_list)):
   if (nearest_indexes_result[index] == polarity): print training_list[index], training_samples[index], " BUSINESS"
   else:                                    print training_list[index], training_samples[index], " PERSONAL"

