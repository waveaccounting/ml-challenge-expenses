#!/usr/bin/env python

import tensorflow as tf
import numpy as np

from file_parser import *

# converting text into vectors
#################################################
def predict_category_based_on_description(description):

   description = description.lower()

   Total_Number_of_Records = Travel['NUM_RECORDS'] + Meals_and_Entertainment['NUM_RECORDS'] + Computer_Hardware['NUM_RECORDS'] + Computer_Software['NUM_RECORDS'] + Office_Supplies['NUM_RECORDS']
   P_Travel                  = Travel['NUM_RECORDS']/Total_Number_of_Records
   P_Meals_and_Entertainment = Meals_and_Entertainment['NUM_RECORDS']/Total_Number_of_Records
   P_Computer_Hardware       = Computer_Hardware['NUM_RECORDS']/Total_Number_of_Records
   P_Computer_Software       = Computer_Software['NUM_RECORDS']/Total_Number_of_Records
   P_Office_Supplies         = Office_Supplies['NUM_RECORDS']/Total_Number_of_Records
   
# Example: P(Travel | Taxi ride) = P(Taxi ride | Travel)*P(Travel)/(SUM_{all categories}P(Taxi ride | some_category)*P(some_category))
# = P(Taxi | Travel)*P(ride | Travel)* P(other words NOT present | Travel) * P(Travel)
#   / (SUM_{all categories}P(Taxi | some_category)*P(Ride | some_category)*P(some_category))

   P_Travel_given_description = 1.0
   P_Meals_and_Entertainment_given_description = 1.0
   P_Computer_Hardware_given_description = 1.0
   P_Computer_Software_given_description = 1.0
   P_Office_Supplies_given_description = 1.0

   description_words = description.split(" ") # all the words in the current description

##### numerator
   for word in Key_Words:
      if word in description_words: 
         if (word in Travel): P_Travel_given_description = P_Travel_given_description * Travel[word]/Travel['NUM_RECORDS']
         else:                P_Travel_given_description = Epsilon
        
         if (word in Meals_and_Entertainment): P_Meals_and_Entertainment_given_description = P_Meals_and_Entertainment_given_description * Meals_and_Entertainment[word]/Meals_and_Entertainment['NUM_RECORDS']
         else: P_Meals_and_Entertainment_given_description = Epsilon
         
         if (word in Computer_Hardware): P_Computer_Hardware_given_description = P_Computer_Hardware_given_description * Computer_Hardware[word]/Computer_Hardware['NUM_RECORDS']
         else: P_Computer_Hardware_given_description = Epsilon

         if (word in Computer_Software): P_Computer_Software_given_description = P_Computer_Software_given_description * Computer_Software[word]/Computer_Software['NUM_RECORDS']
         else: P_Computer_Software_given_description = Epsilon

         if (word in Office_Supplies): P_Office_Supplies_given_description = P_Office_Supplies_given_description * Office_Supplies[word]/Office_Supplies['NUM_RECORDS']
         else: P_Office_Supplies_given_description = Epsilon

      else:                         
         if (word in Travel): P_Travel_given_description = P_Travel_given_description * (1 - (Travel[word]/Travel['NUM_RECORDS']))
         if (word in Meals_and_Entertainment): P_Meals_and_Entertainment_given_description = P_Meals_and_Entertainment_given_description * (1 - (Meals_and_Entertainment[word]/Meals_and_Entertainment['NUM_RECORDS']))
         if (word in Computer_Hardware): P_Computer_Hardware_given_description = P_Computer_Hardware_given_description * (1 - (Computer_Hardware[word]/Computer_Hardware['NUM_RECORDS']))
         if (word in Computer_Software): P_Computer_Software_given_description = P_Computer_Software_given_description * (1 - (Computer_Software[word]/Computer_Software['NUM_RECORDS']))
         if (word in Office_Supplies): P_Office_Supplies_given_description = P_Office_Supplies_given_description * (1 - (Office_Supplies[word]/Office_Supplies['NUM_RECORDS']))
     
   P_Travel_given_description = P_Travel_given_description * P_Travel
   P_Meals_and_Entertainment_given_description = P_Meals_and_Entertainment_given_description * P_Meals_and_Entertainment
   P_Computer_Hardware_given_description = P_Computer_Hardware_given_description * P_Computer_Hardware
   P_Computer_Software_given_description = P_Computer_Software_given_description * P_Computer_Software
   P_Office_Supplies_given_description = P_Office_Supplies_given_description * P_Office_Supplies

##### denominator is the same for all the classes

   SUM = P_Travel_given_description + P_Meals_and_Entertainment_given_description + P_Computer_Hardware_given_description + P_Computer_Software_given_description + P_Office_Supplies_given_description;

##### combining numerator and denominator
   P_Travel_given_description /= SUM
   P_Meals_and_Entertainment_given_description /= SUM
   P_Computer_Hardware_given_description /= SUM
   P_Computer_Software_given_description /= SUM
   P_Office_Supplies_given_description /= SUM

   max_probability = max(P_Travel_given_description, P_Meals_and_Entertainment_given_description, P_Computer_Hardware_given_description, P_Computer_Software_given_description, P_Office_Supplies_given_description)

#   print P_Travel_given_description
#   print P_Meals_and_Entertainment_given_description
#   print P_Computer_Hardware_given_description
#   print P_Computer_Software_given_description
#   print P_Office_Supplies_given_description

   if (P_Travel_given_description == max_probability): return "Travel"
   if (P_Meals_and_Entertainment_given_description == max_probability): return "Meals and Entertainment"
   if (P_Computer_Hardware_given_description == max_probability): return "Computer - Hardware"
   if (P_Computer_Software_given_description == max_probability): return "Computer - Software"
   if (P_Office_Supplies_given_description == max_probability): return "Office Supplies"


def category_key_words(training_list, category_name): # returns the array with the number of occurences per word
   Category_Array = {}

   for record in training_list:
      if (record[1] == category_name):
        
         # NUM_RECORDS number of records for the current category
         if "NUM_RECORDS" in Category_Array: Category_Array['NUM_RECORDS'] += 1
         else:                               Category_Array['NUM_RECORDS']  = float(1)

         # The number of occurences per word for category_name
         words = record[3].lower().split(" ")
         for word in words:
            if (word != "to") and (word != "the") and (word != "with"):
               Key_Words[word] = 1
               if word in Category_Array: Category_Array[word] += 1
               else:                      Category_Array[word]  = float(1)

   return Category_Array

#################################################
# Main program
Epsilon = 0.01 # in order to resolve potential issues of division by 0
#training_list,   training_header   = parse_csv_data("training_data_example.csv")
training_list,   training_header   = parse_csv_data("validation_data_example.csv")

Key_Words = {} # all key words (in all categories)

# number of records with the word for every particular category
Travel                  = category_key_words(training_list, "Travel")
Meals_and_Entertainment = category_key_words(training_list, "Meals and Entertainment")
Computer_Hardware       = category_key_words(training_list, "Computer - Hardware")
Computer_Software       = category_key_words(training_list, "Computer - Software")
Office_Supplies         = category_key_words(training_list, "Office Supplies")

if "NUM_RECORDS" not in Travel:                  Travel['NUM_RECORDS'] = 0
if "NUM_RECORDS" not in Meals_and_Entertainment: Meals_and_Entertainment['NUM_RECORDS'] = 0
if "NUM_RECORDS" not in Computer_Hardware:       Computer_Hardware['NUM_RECORDS'] = 0
if "NUM_RECORDS" not in Computer_Software:       Computer_Software['NUM_RECORDS'] = 0
if "NUM_RECORDS" not in Office_Supplies:         Office_Supplies['NUM_RECORDS'] = 0

success = 0;

for record in training_list:
   print record   
   if (predict_category_based_on_description(record[3]) == record[1]): 
      success+=1
      print "Predicted Category: ", predict_category_based_on_description(record[3]), ": CORRECT"
   else:
      print "Predicted Category: ", predict_category_based_on_description(record[3]), ": INCORRECT"

print "### Success Rate is:", (success*100/len(training_list)), "%"
