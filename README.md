A detailed solution to the classification task (as described below) is detailed within the classification notebook. To run the notebook with any csv file of the same format, perform the following steps:

1. Change the name of the csv files in cell [2] of the notebook (right below the Data header) to the name of the required files.
2. Run the notebook by clicking Cell -> Run All.

I used two algorithms for the classification problem: random forest and xgboost. 

1. Random Forest Classifier: This is an ensemble learning technique that works by constructing several decision trees at training time and outputting - for classification problems - the class that is the mode of the classes of the individual trees. It works well with a mixture of numerical and categorical features (such as the dataset used here). One of its advantages is that the trees can be decorrelated, which automatically helps with multi-collinearity of features; a disadvantage is that it is not easy to interpret visually. The performance of this model on the validation data was impressive, with an accuracy and f1-score of 1.0. 

2. XGBoost Classifier: It implements gradient boosted decision trees for classification. There are many hyperparameters that can be tuned (including regularization strength and learning rate), giving it a lot flexibility. Another cool aspect (which I did not exploit) is it allows you to implement your own custom loss functions. Much like random forests, xgboost is also very capable of handling categorical features. Now I am very new to xgboost, so I honestly can't think of any cons aside from the confusion that may arise from dealing with so many hyperparameters. Its performance was also equally impressive: an accuracy and f1-score of 1.0. 

Both classifiers performed seemingly well on the toy validation sets, so there isn't much to say in terms of comparing the two models (at least not on this dataset). However, xgboost is known to utilize a plethora of hyperparameters that can be fine-tuned to achieve better performance than random forests.

In conclusion, I am overall pleased with the content and results of this analysis. While I only tackled the classification task, I believe I did a thorough job with feature selection and model evaluation, and in making the project descriptive and readable. 

- Mayank Bhatia

==========================

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
