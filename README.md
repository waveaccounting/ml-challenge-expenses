ML Challenge Markdown
# Wave Machine Learning Engineer Challenge

## Project Description
Continue improvements in automation and enhancing the user experience are keys to what make Wave successful.  Simplifying the lives of our customers through automation is a key initiative for the machine learning team.  Your task is to solve the following questions around automation.

### What your learning application must do:

1. Your application must be able read provided comma separated files. 

2. Similarly, your application must accept a separate comma separated file as validation data with the same format.
3. You can make the following assumptions:
	* Columns will always be in that order.
	* There will always be data in each column.
 	* There will always be a header line.

### Questions to answer:
1. Train a learning model that assigns each expense transaction to one of the set of predefined categories and evaluate it against the validation data provided.  The set of categories are those found in the "category" column in the training data. Report on accuracy and at least one other performance metric.
2. Mixing of personal and business expenses is a common problem for small business.  Create an algorithm that can separate any potential personal expenses in the training data.  Labels of personal and business expenses were deliberately not given as this is often the case in our system.  There is no right answer so it is important you provide any assumptions you have made.
3. (Bonus) Train your learning algorithm for one of the above questions in a distributed fashion, such as using Spark.  Here, you can assume either the data or the model is too large/efficient to be process in a single computer.

### Documentation:

Please modify `README.md` to add:

1. Instructions on how to run your application
2. A paragraph or two about what what algorithm was chosen for which problem, why (including pros/cons) and what you are particularly proud of in your implementation, and why
3. Overall performance of your algorithm(s)

# ---------------------------------------------------------------
# Questions answered
# ---------------------------------------------------------------

I have answered all the three questions. 

## 1. Expense Classification

### Algorithm used
1. Random forest classifier
2. 300 dimension Wordvectors for feature engineering (fast text and glove)


### Data columns used
1. Expense Category - This column has the target labels
2. Expense Description - This column provides the prediction features

### Directions to Run the Algorithm
1. Downlaod the code base to a working directory
2. Launch Jupyter notebook from working directory
3. Open Expense_Classification_With_WordVectors.ipynb file, change the training and validation file paths under "Split Raw data into Predictors and Response" section of the notebook, to a path applicable in end user machine
4. Download pretrained fast text word vector from (https://s3-us-west-1.amazonaws.com/fasttext-vectors/wiki-news-300d-1M.vec.zip)
5. Download pretrained glove word vector from (http://nlp.stanford.edu/data/glove.6B.zip)
6. Unzip the vectors into working directory and modify the paths under "Load Pretrained Fast_Text Wordvectors" and "Load Pretrained Glove Wordvectors" section of the notebook
7. Execute each section

### Algorithm workflow
1. I chose wordvectors to extract features from the expenses description field of the data. Given the fact that the data is limited, Wordvectors allow us to extract maximun information from the expenses description data.
1.1 Each expense description is treated as a sentence
1.2 Each sentence is split into constituent words
1.3 Each word of a sentence is assigned a vector from the pretrained word vectors
1.4 The word vectors of each sentence is added together to form a sentence vector
1.5 The sentence vector is of 300 dimensions

2. The expense category column is modified to a categorical column, so that it can act as target labels

3. The expense category column is nominated as target and sentence vectors as predictors

4.  I used the scikit learn's train_test_split utility from model_selection library, to split the training data set to 70% training data and 30% test data

(steps 1-4 are repeated twice, once for Fasttext Wordvectors and once for Glove Wordvectors)

5. Test data is not used in training

6. As we have dense vector represnting each sentence, I used Random Forest algorithm for classification. I could have used multi-class logistic regression with cross validation as well.

Random forests for classification builds Multiple decision trees (not only one). Each tree output a label.Then, the final label is decided by a majority voting process.

Each tree is generated through a subset of the initial training set randomly selected (usually, 1/3 of the samples are used, depending on the default value of the implementation used). Each tree is also generated using just a few features (e.g. if your features are frequency counts of words, it will consider just a small subset of words, randomly selected among the initial ones). Usualy, 3-5 features are used...but, again...it depends on the default value of your implementation.

The splitting criteria on each tree node is to use a condition over typically one feature which divides the dataset in two equal parts. The criteria to select which is the feature to consider in each node is, by default, the gini criterion (see here to know more about this). This process stops whenever all the remaining data samples have the same label. This node will be a leaf one which will assign labels to test samples.

7. The trained model is tested with the test data

8. Model is then validated using validation data

9. Random forests are trained based on Glove and Fasttext Wordvector features. Model is finalised based on its performance on both the feature vectors   

### Algorithm performance measure
1. Model score and classifcation accuracy are used as performance measure for selecting a model

2. Model with no tuning predicts classes with 83% accuracy

## 2. Business Vs Personal Expense

### Algorithm Workflow:
I have not written a code for solving this problem, rather I have proposed a complete solution methodology. The proposed solution is written in the form of a white paper and saved as "Business_Personal_Classification_Approach.pdf". I will eloborate the approach during our discussion.

## 3. Distributed Learning with Spark - Expenses Classification

### Algorithm used
1. Parallel multi-class logisitic regression and Random Forests
2. Bag of word count vectors, TF IDF features 

### Data columns used
1. Expense Category - This column has the target labels
2. Expense Description - This column provides the prediction features

### Directions to Run the Algorithm
1. Spark machine learning code is under the folder Spark_Design
2. Downlod the code base to a working directory
3. Launch Jupyter notebook from working directory
4. Open Spark_ML_Launchpad.ipynb file, change the training and validation file paths under "Split Raw data into Predictors and Response" section of the notebook, to a path applicable in end user machine
5. Make sure Spark is installed and .bashrc file is updated to have Spark_HOME environment variable. 
6. Execute the Spark_ML_Launchpad.ipynb one cell at a time to evaluate results

### Algorithm approach
1. Data Ingestion and Extraction, loading a CSV file is straightforward with Spark csv packages. 

https://github.com/databricks/spark-csv. This package allows us to convert csv files to Spark dataframes. Hence it offers an advantage of dealing with data organized as a native Spark data structure. All the data management utilities are packaged under Data_to_Spark_Utils.py file. 

2. Once CSV files are ingested, only 2 columns are retained, the expenses category and the expense description. My approach to solve this problem is to use an NLP based approach. Expense category column provides the labels/ classes, expense description column provides the features. Features are extracted from the expense descriptions and are used to train a classifier 

3. Next step is to build a machine learning pipeline for the Spark machine learning library. https://spark.apache.org/docs/2.2.0/ml-pipeline.html. 

Pipeline construction, data preparation, model specification, training and validation functions are present in the ML_Utils.py file. 

For this exercise, we consider the following:
regexTokenizer: Tokenization (with Regular Expression)
stopwordsRemover: Remove Stop Words
countVectors: Count vectors (“document-term vectors”)
TF - IDF Features 

The features are used to construct a pipeline. Once the pipeline is constructed, the raw data is given to the pipeline to extract features and labels. This then is passed to classifiers. 

For this exercise, Logistic regression, Logistic regression with cross validation and random forests are implemented. 

The evaluation metrics generated by the multiclass evaluators in Spark help in judging the performance of the models. 

### Algorithm performance measure
1. Summary Stats used are given below and a representative result
F(1) Score         = 0.3333333333333333
Weighted Precision = 0.25
Weighted Recall    = 0.5
Accuracy           = 0.5

2. Model accuracy and predication capability is bad. 

3. Hence, I conclude that for a smaller corpus, a better classifier can be built based on Wordvectors as against the rudimentory features like bag of words and tf-idf.  

