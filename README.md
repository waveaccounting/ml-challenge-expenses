<<<<<<< HEAD
1. Instructions on how to run your application:

I did the work iPython notebooks as I thought it would be useful to
interleave the code with explanation, instead of explaining the models 
in README. If you don't have Jupyter installed please follow steps in
(https://jupyter.readthedocs.io/en/latest/install.html)

There are 3 iPython notebooks for each of the task along with supplementary
python files in this folder:
	1. Classification.ipynb: Assigning each transaction to pre-defined category.
	2. Clustering.ipynb: Separate potential personal expenses in the provided data.
	3. Spark.ipynb: Same clustering algorithm as above performed using pyspark APIs

I used two different ways to represent text, one of the way would require 
you to download word embeddings, please download and keep in same folder:
	> wget http://nlp.stanford.edu/data/glove.6B.zip
	> unzip glove.6B.zip

Python packages you need:
	1. numpy
	2. pandas
	3. sklearn
	4. pyspark
	5. matplotlib

Once you open the Jupyter notebooks, you would already see results, 
you can re-run by clicking 'run'.

2. A paragraph or two about what what algorithm was chosen for which problem, 
	why (including pros/cons) and what you are particularly proud of in your implementation, and why

	1. I used Ensemble of decision trees (Extreme randomized trees) and also 
		gradient boosting (xgboost) for predicting the type of
		expense. Both these methods gave equally good results. I think decision trees are 
		perfect for this problem because there is inherent if-else reasoning in the data 
		(for example: if there is a word 'flight' in description, it is 'travel' category). 
		Randomized trees, instead of searching for the hard best hypothesis, generate 
		lot of random ones, then average their results, which reduces the variance introduced
		by randomization hence generalize well. But the random trees have low interpretability 
		compared to standard decision trees and are still subject to overfitting.
		Based on past experience I think Gradient boosting is very powerful method (though with 
		this dataset being simple it doesn't show better performance than random forests), it 
		combines weak learners to get expressive classifiers adaptively. 
		By combining these highly biased weak learners, boosting works as a bias-reduction technique.
		However, boosting's performance is highly subjected to hyper-parameter tuning and there 
		are lot of parameters to tune.
		I am particularly proud of the methodological way I implemented the solution for problem,
		startig with understanding the data, feature selection (getting rid of some features), 
		feature representation according to data type, model selection, cross validation, 
		performance measure.

	2. I used K-means clustering for separating the personal expenses from training data. 
		I think it is a challenging problem, mainly because all of the available features can't
		be represented in Eucledian space which is required for distance measure of algorithms.
		Therefore, I would have to rely on word embeddings of 'description' and 'category' (though 
		'amount' is available, I noticed it is not much help). Though word embeddings were very
		helpful in clustering, they rely on the semantics of the word and there is not
		many words available in description to identify 'personal expenses'. 
		Training the embeddings on a specific corpus which is more suitable to this problem 
		(maybe finance reports and news ?) could have helped. And I think there is not enough
		data to learn a projection into different dimension, or autoencoders could have helped.
		Also I feel the best way to tackle this problem is semi-supervised learning, even few labeled
		examples would make a great difference.

3. Overall performance of your algorithm(s)
	1. 91.66 % accuracy on validation set, other performance metrics like 
		precision, recall, f1-score are mentioned in the Jupyter notebooks.
	2. Since there is no quantitative performance metric here, Individual clusters are displayed 
		in the Jupyter notebook as tables. 
	3. Though it doesn't have meaningful interpretation for separating personal expenses, 
		I also measured the sum of squared error between the points and cluster center, which is 25.7618.
=======
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
>>>>>>> efbbf62f6b5c489f0cbf612d3acac816a96dccd7
