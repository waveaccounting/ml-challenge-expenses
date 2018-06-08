# This file has utility functions to load csv files and perform the following functions
#
# 1. Load CSV files into Spark and provide SQL based data extraction capability
# 2. Helps in dropping colums from a dataset
# 
# More capabilities will be added as the complexity of data proessing increases

import pyspark

# This function helps reading CSV files in local or distributed filesystem as Spark DataFrames
# inputs:
# SC - spark context
# filepath - path of the csv file to be ingested
#
def Ingest_CSV_in_Spark(sc, filepath):
    from pyspark.sql import SQLContext
    from pyspark import SparkContext
   
    sqlContext = SQLContext(sc)
    data = sqlContext.read.format('com.databricks.spark.csv').options(header='true', inferschema='true').load(filepath)
    return data

# This function drops colums from a dataframe
# inputs:
# Dataframe : dataframe from which colums should be dropped
# List_Colums_to_Drop : list of column names to be dropped form a data frame
def Drop_Coulmns_from_DataFrame(dataframe, List_Colums_to_Drop):
    dataframe = dataframe.select([column for column in dataframe.columns if column not in List_Colums_to_Drop])
    return dataframe

    