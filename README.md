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
"# waveappstest" 
"# waveappstest" 


# Submission Documentation (Nabil's entry)

## Pre-requisites
1. Python on system with environmental variables set (I use Python 2.7).
2. Some none-standard libraries might need to be installed if not already there. The main ones are the Scikit-Learn and the IPython 	    distributed computing libraries .
3. To make things easier there are no file path settings to follow, just ensure that all files are in the same path/folder. However, for    Q3, make sure that copies of the source csv files you provided in the IPython engine default local path at
   `/Python27/Lib/site-packages/ipyparallel/`
4. I use PyCharm IDE for working with Python, so a .idea folder is there if you like.


## How-to
There are 4 .py files independent of each other, so you may run them separately in any order you wish: 
1. `main.py` is the code that implements the main task, namely a sample of an ML miniapp that implements Q1 in your challenge.
2. `feature_selection.py` is an exercise that shows why I chose the features to use in Q1.
3. `expense_type_algo.py` is a possible solution for Q2, and.It uses a variant of the csv files you provided with some          	    additional info I put in manually based on some assumptions, which I will explain if you want to go further and meet with 		    me.
4. `mainWcluster.py` Is my attempt to solve Q3. It is quite basic, as I am not overly familiar with IPython and I had trouble getting 	     python distributed tools like PySPARK on my laptop. It is essentially the same code as in Q1, but running in an ippyparallel client     block.

## What algorithms and why

`Q1`
The candidate algorithms for this problem were `Naive Bayes(Gaussian)`, `Gradient Boost`, `SVC SVM`, `Linear Regression`, `Logisitic Regression` and `Linear Discriminant Analysis`.` Cross validation` was then conducted to see which would give higher accuracy using the validation set, and the one with highest score is then tested as per question instructions. These were selected because in practice I find them to be the best to deal with a sample of few instances and features, as more complex algorithms and ensembles tend to overfit and bias in such cases (Although realistically how a library or code is actually implementing an algorithm and on what platform tend to factor heavily).

`Q2`
For this problem I used an NLP(ish) text tokenizing approach to create classification features. I've always wanted to try this type of mining, and this seemed like a good example to try because the only real lead in the training/validatng csv files for differentiating between expenses was the expense description field, and as it's a 'free form' text field a text feature extraction based algorithm seemed a good option. For simplicity's sake and allowing for practical implementation concerns, the classification is split into `business` and needing `review` for reasons again I will explain if we get to talk.
A promising possible supplement to this algorithm is to factor in employees(i.e. their IDs) in the features as well, since the it is a sensible supposition that certain employees would incur higher business expenses than others (e.g. sales team would have more travel expenses). 

`What am I proud of here?`
I'm not particularly "proud", at least not yet :) . I've learned to be cautious about ML implementation and what its output means/implies. Things are not always as they seem and practicality concerns can easily break a solution even if it initially works great, so I tend to look for more data to try and a LOT more time in pre-processing before I pat myself on the back. Although it felt pretty rad to get to use text feature extraction for the first time and get a relatively good result so fast. 

## Overall algorithm performance

`Q1`
Running the code produces a metrics matrix csv file detailing performance. Prediction is around 89 percent accurate using `Gradient Boost`, however `SVM SVC` achieves similar results and would be more efficient and significantly faster (This will particularly matter in the case of distributed computing like in Q3). I should mention that out of curiousity I applied a `Random Forest` ensemble and managed to achieve over 97 percent accuracy, but again the dataset is just much too small for more complex algorithms.

`Q2`
This is my first experience with the text feature extraction sub-library, so I am not able to make the code test predictions properly yet. However I did try it manually with 10 arbitrary description text entries and it correctly classified 8 of them(Note that dictionary keywords content is necessary for successful prediction, and it would only 'learn' it from the text available in the csv). It obviously needs work and more appropriate testing, but it looks promising and in this field you have to be creative.