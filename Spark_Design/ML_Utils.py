# This file has utility functions to prepare teh data for machine learning and for calling the learning algorithms
# 
# More capabilities will be added as the complexity of data proessing increases

import pyspark


# Model Pipeline
#
# Create a pipleline for text classification. Our pipeline includes three steps:

#    1. regexTokenizer: Tokenization (with Regular Expression)
#    2. stopwordsRemover: Remove Stop Words
#    3. countVectors: Count vectors (“document-term vectors”) 
def Construct_Pipeline():
    from pyspark.ml.feature import RegexTokenizer, StopWordsRemover, CountVectorizer
    from pyspark.ml import Pipeline
    from pyspark.ml.feature import StringIndexer
    from pyspark.ml.feature import HashingTF, IDF

    
    # regular expression tokenizer
    regexTokenizer = RegexTokenizer(inputCol="expense description", outputCol="words", pattern="\\W")

    # stop words
    add_stopwords = ["http","https","amp","rt","t","c","the"] 

    stopwordsRemover = StopWordsRemover(inputCol="words", outputCol="filtered").setStopWords(add_stopwords)

    # bag of words count
    countVectors = CountVectorizer(inputCol="filtered", outputCol="features", vocabSize=10000, minDF=5)
    
    # Convert expense classes to labels
    label_stringIdx = StringIndexer(inputCol = "category", outputCol = "label")
    
    # Get term frequencies
    hashingTF = HashingTF(inputCol="filtered", outputCol="rawFeatures", numFeatures=10000)
    
    # get the inverse document frequency
    idf = IDF(inputCol="rawFeatures", outputCol="features", minDocFreq = 5) #minDocFreq: remove sparse terms
    
    # Construct pipeline 
    pipeline = Pipeline(stages=[regexTokenizer, stopwordsRemover, hashingTF, idf, label_stringIdx])
    
    return pipeline

# Transform data to Ml_Features using the pipeline
# inputs
# pipeline : preconstructed pipeline object4
# Data : Data to be transformed
def Construct_ML_Dataset(pipeline,Data):
    pipelineFit = pipeline.fit(Data)
    dataset = pipelineFit.transform(Data)
    return dataset

# Function for training a model
# Model takes training data set, splits it to training (70%) and test set(30%).
# Model can also be specified, choices are Logistic regression,(LR) Random Forestes(RF) and Logistic regress with cross validation(LRCV)
# Inputs:
# Training_Dataset - Model will be built on this data
# Model_Type - (String) LR, LRCV, RF
def Train_Model(Training_Dataset, Model_Type):
    
    # set seed for reproducibility
    (trainingData, testData) = Training_Dataset.randomSplit([0.7, 0.3], seed = 100)
    print("Training Dataset Count: " + str(trainingData.count()))
    print("Test Dataset Count: " + str(testData.count()))

    if Model_Type == "LR":
        from pyspark.ml.classification import LogisticRegression
        
        lr = LogisticRegression(maxIter=20, regParam=0.3, elasticNetParam=0)
        model = lr.fit(trainingData)
        
    elif Model_Type == "LRCV":
        from pyspark.ml.classification import LogisticRegression
        from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
        from pyspark.ml.evaluation import MulticlassClassificationEvaluator
       
        # define evaluator for cross validation 
        evaluator = MulticlassClassificationEvaluator(predictionCol="prediction")
        
        # estimator for cross validation
        lr = LogisticRegression(maxIter=20, regParam=0.3, elasticNetParam=0)
        
        # Create ParamGrid for Cross Validation
        paramGrid = (ParamGridBuilder()
                     .addGrid(lr.regParam, [0.1, 0.3, 0.5]) # regularization parameter
                     .addGrid(lr.elasticNetParam, [0.0, 0.1, 0.2]) # Elastic Net Parameter (Ridge = 0)
        #            .addGrid(model.maxIter, [10, 20, 50]) #Number of iterations
        #            .addGrid(idf.numFeatures, [10, 100, 1000]) # Number of features
                     .build())
        
        # Create 5-fold CrossValidator
        cv = CrossValidator(estimator=lr, \
                            estimatorParamMaps=paramGrid, \
                            evaluator=evaluator, \
                            numFolds=5)
        model = cv.fit(trainingData)
        
    else:
        from pyspark.ml.classification import RandomForestClassifier
        
        rf = RandomForestClassifier(labelCol="label", \
                            featuresCol="features", \
                            numTrees = 100, \
                            maxDepth = 4, \
                            maxBins = 32)

        # Train model with Training Data
        model = rf.fit(trainingData)
   
    return model, testData

# Utility function to validate models based on the validation data.
# Validation data can either be the data that was split from the training dataset or the complete validation dataset
def Validate_Model(model,test_data):
    from pyspark.ml.evaluation import MulticlassClassificationEvaluator
       
    # define evaluator
    evaluator = MulticlassClassificationEvaluator(predictionCol="prediction")
    
    predictions = model.transform(test_data)
    
    predictions.filter(predictions['prediction'] == 0) \
    .select("expense description","category","probability","label","prediction") \
    .orderBy("probability", ascending=False) \
    .show(truncate = 30)
    
    # evaluator = MulticlassClassificationEvaluator(predictionCol="prediction")
    # evaluator.evaluate(predictions)
           
    return evaluator, predictions