ML Challenge Markdown
# Wave Machine Learning Engineer Challenge
Applicants for the Software Engineer (and Senior), Machine Learning(https://wave.bamboohr.co.uk/jobs/view.php?id=1) role at Wave must complete the following challenge, and submit a solution prior to the onsite interview. 

The purpose of this exercise is to create something that we can work on together during the onsite. We do this so that you get a chance to collaborate with Wavers during the interview in a situation where you know something better than us (it's your code, after all!) 

There isn't a hard deadline for this exercise; take as long as you need to complete it. However, in terms of total time spent actively working on the challenge, we ask that you not spend more than a few hours, as we value your time and are happy to leave things open to discussion in the onsite interview.

Please use whatever programming language, libraries and framework you feel the most comfortable with.  Preference here at Wave is Python.

Feel free to email [dev.careers@waveapps.com](dev.careers@waveapps.com) if you have any questions.

## Project Description
Continue improvements in automation and enhancing the user experience are keys to what make Wave successful.  Simplifying the lives of our customers through automation is a key initiative for the machine learning team.  Your task is to solve the following questions around automation.

### What your learning application must do:

1. Your application must be able read provided comma separated files. 

2. Similarly, your application must accept a separate comma separated file as validation data with the same format.
3. You can make the following assumptions:
	* Columns will always be in that order.
	* There will always be data in each column.
 	* There will always be a header line.

An example input files named `training_data_example.csv`, `validation_data_example.csv` and `employee.csv` are included in this repo.  A sample code `file_parser.py` is provided in Python to help get you started with loading all the files.  You are welcome to use if you like.

1. Your application must parse the given files.
2. Your application should train only on the training data but report on its performance for both data sets.
3. You are free to define appropriate performance metrics, in additional to any predefined, that fit the problem and chosen algorithm.
4. You are welcome to answer one or more of the following questions.  Also, you are free to drill down further on any of these questions by providing additional insights.

Your application should be easy to run, and should run on either Linux or Mac OS X.  It should not require any non open-source software.

There are many ways and algorithms to solve these questions; we ask that you approach them in a way that showcases one of your strengths. We're happy to tweak the requirements slightly if it helps you show off one of your strengths.

### Questions to answer:
1. Train a learning model that assigns each expense transaction to one of the set of predefined categories and evaluate it against the validation data provided.  The set of categories are those found in the "category" column in the training data. Report on accuracy and at least one other performance metric.
2. Mixing of personal and business expenses is a common problem for small business.  Create an algorithm that can separate any potential personal expenses in the training data.  Labels of personal and business expenses were deliberately not given as this is often the case in our system.  There is no right answer so it is important you provide any assumptions you have made.
3. (Bonus) Train your learning algorithm for one of the above questions in a distributed fashion, such as using Spark.  Here, you can assume either the data or the model is too large/efficient to be process in a single computer.

### Documentation:

Please modify `README.md` to add:

1. Instructions on how to run your application
2. A paragraph or two about what what algorithm was chosen for which problem, why (including pros/cons) and what you are particularly proud of in your implementation, and why
3. Overall performance of your algorithm(s)

## Submission Instructions

1. Fork this project on github. You will need to create an account if you don't already have one.
2. Complete the project as described below within your fork.
3. Push all of your changes to your fork on github and submit a pull request. 
4. You should also email [dev.careers@waveapps.com](dev.careers@waveapps.com) and your recruiter to let them know you have submitted a solution. Make sure to include your github username in your email (so we can match applicants with pull requests.)

## Alternate Submission Instructions (if you don't want to publicize completing the challenge)
1. Clone the repository.
2. Complete your project as described below within your local repository.
3. Email a patch file to [dev.careers@waveapps.com](dev.careers@waveapps.com)

## Evaluation
Evaluation of your submission will be based on the following criteria. 

1. Did you follow the instructions for submission? 
2. Did you apply an appropriate machine learning algorithm for the problem and why you have chosen it?
3. What features in the data set were used and why?
4. What design decisions did you make when designing your models? Why (i.e. were they explained)?
5. Did you separate any concerns in your application? Why or why not?
6. Does your solution use appropriate datatypes for the problem as described? 

# Asif's Notes

## Instructions

My solution is python 3 based and I have presented my solution in 2 ways:

- Jupyter Notebooks for more interactivity and details
- A [luigi](https://github.com/spotify/luigi) based solution that kind of simulates (not quite) a real production type run 

Before proceeding, please install/setup the pre-requisites below. I developed this on a Mac. For linux, the steps should be mostly same, except the XGBoost installation might be a bit different.

### Pre-requisites

- Clone/download this repository and make it the working directory
- Install python 3 `brew install python3`
- Install virtual environment `pip3 install virtualenv`
- Create a virtual env `virtualenv venv` and activate it `source venv/bin/activate`
- Do the following as a pre-requisite for XGBoost
```
brew install gcc
brew install gcc@5
```
- Now install the packages in [requirements.txt](https://github.com/asif31iqbal/ml-challenge-expenses/blob/master/requirements.txt)

- Download the pretrained vector `glove.6B.zip` from [here](https://nlp.stanford.edu/projects/glove/) and extract off the `glove.6B.100d` file into the working directory. **Note that this file is not included in the repo since it's too big. You do need to download it**

At this point, you can run `jupter notebook` and run the notebooks interactively. You can also run the luigi solution by doing
```
cd luigi
./run.sh
```
## Solution Overview

There are 4 Jupyter notebooks:
- [classification.ipynb](https://github.com/asif31iqbal/ml-challenge-expenses/blob/master/clasification.ipynb)
- [classification_attemp_with_deep_learning.ipynb](https://github.com/asif31iqbal/ml-challenge-expenses/blob/master/classification_attemp_with_deep_learning.ipynb)
- [clustering_personal_business.ipynb](https://github.com/asif31iqbal/ml-challenge-expenses/blob/master/clustering_personal_business.ipynb)
- [spark_random_forest.ipynb](https://github.com/asif31iqbal/ml-challenge-expenses/blob/master/spark_random_forest.ipynb)

And a folder called [luigi](https://github.com/asif31iqbal/ml-challenge-expenses/tree/master/luigi) that has a luigi based solution. Let's go through an overview of each of these.

### classification.ipynb

This notebook tries to address the first question in the problem set - classifying the data points into the right category. **Please follow the cell by cell detailed guideline to walk thruogh it**. Key summary points are:
- Intuitively the **expense description** field seems to be the best determiner
- Tried several **shallow learning** classifiers - Logistic Regression, Naive Bayes, SVM, Gradient Boosting (XGBoost) and Random Forest and attempted to take an ensemble of the ones that performed best
- Feature engineering involved vectorizing text (**expense description** field) as TF-IDF ventors and using other fields like day_of_week, expense amount, employee ID and tax name. I didn't use n-grams here, but that's possibly an improvement option
- Tried the classifiers in a two-fold approach - with all features and with only text features
- Tried with Grid Search cross validation (used `LeaveOneOut` since pretty small data) and tuning several parameters like regularization factor, number of estimators etc
- Most classifiers worked better with only text features, indicating that the **expense description** text field is the key field for classification here, and that goes right with common intuition
- XGBoost didn't perform as well as I had anticipated
- My personal pick of the classifiers in this case would be Random forest based on the performance (although I tried a further ensemle of Random Forest with SVM amd Logistic Regression)
- Target categories are skeweed in distribution (like **Meals and Entertainment** being the majority of it), so **accuracy** alone is not enough. Considered **precision** and **recall** as additional useful performance measures
- Achieved an accuracy of **100%** on the training data and **91.67%** on the validation data,and an average of **93%** precision and **92%** recall and **91%** f1-score
- I do understand that with bigger data and more intelligent feature engineering, the performance and prediction can change and nothing here is necessarily conclusive.

### classification_attemp_with_deep_learning.ipynb

This notebook tries to address the same problem as the previous one, but this time using **Deep Learning** using **Keras**. This notebook is not documented in detail, but I get a chance I can explain more in person. Key summary points are:
- Tried a simple 3-layer ANN with all features (non-text and text, with text vectorized using scikit learn's TfIdfVectorizer)
- Tried a simple 3-layer ANN with only text features (with text vectorized using Keras's text to matrix binary vectorizer)
- Tried a RNN with LSTM with a pre-trained embedding layer and only text features (with text vectorized using Keras's text to matrix binary vectorizer)
- The accuracy, precision, recall and f1-score that was acieved was the exact same as the one earned with the previous shallow learning approach

### clustering_personal_business.ipynb

This notebook tries to address the second problem in the problem set - identifying expenses as either personal or business. **Please follow the cell by cell detailed guideline to walk thruogh it**. Key summary points:

- Decided to use **K-means**
- Extracted holiday (not in a very proper way), along with employee ID and role, and most importantly, again, the text features. This time concatenated **expense description** and **category**
- Tried with tf-idf vectorizer and a pre-trained embedding
- 15 data points in business cluster and 9 in personal cluster
- Resulting clusters kind of makes sense as has been described in the notebook
- Tried the cluster model on the validation data as well and that sort of makes sense as well

### spark_random_forest.ipynb

This notebook tries to address the bonus of problem of running one solutio via spark. For this I tried implementing the classification problem. This notebook is not very well documented, I can explain in person if I get the chance. Key summary points:

- Although in the original solution I tried several approaches, i only used Random firests here to keep things simple as I am very new to using Spark's ML library
- Only used text TF-IDF features
- Training Accuracy **87.5%**, validation accuracy **83.33%**, precision **100%**, **83%** and f1-score **91%**
- Tons of improvement possible

## Luigi Solution

[Luigi](Luigi is a Python package that helps you build complex pipelines of batch jobs). I used it to manifest how a real production type pipeline can be developed. I have only included the prediction (classification) pipeline here, not the clustering. However, the clustering can be easily incorporated with some additional effort. The solution is in the `luigi` subfolder. The code is basically taken from the Classification notebook but aranged in a more modular format.

### Classes

- **PreProcessor** pre-processes and vectorizes the data and generates vectorized files
        - train_processed.csv:  vectorized training data
        - validation_processed.csv: vectorized validation data
        - train_processed_tfidf.csv: vectorized training data with only tf-idf features
        - validation_processed_tfidf.csv: vectorized validation data with only tf-idf features
        - labels.pkl: a labels files including training and validation labels
- **Predictor** does the actual classification and generates result files
	- report.txt: report contaning accuracy and other measurements
	- comparison.csv: file showing actual and predicted categories side by side

Sample output files are provided with the solution. You can delete those files and rerun `run.sh` to run the entore workflow.

The solution is nowhere close to production-ready and can be further modularized and enhanced in a million way. This was just a manifestation of how pipelines can be developed with modularized tasks.


## Conclusion

I have tried different approaches of addressing the solutions. Overall performance were fairly good, however, the datasets were very small and nothing I experimented here is conclusive. I have learnt from my experience and real work that models that perform well on small data are not necessarily best for tackling real big data. Also, there is a myriad of ways of improving things.

I am not particularly proud of anything, however, I am glad that I went through the exercise and learnt a lot in the process. I hope I get a chance to present myself in person and discuss in more detail.
