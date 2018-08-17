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
