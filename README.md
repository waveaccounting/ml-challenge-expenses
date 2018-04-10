#Wave ML Challenge Report

# 1. Running the application
The application can be run either with python3 (using script main_wave_ml_challenge.py)
or with jupyter notebook (using main_wave_ml_challenge.ipynb).
First install the requirements using:

pip3 install --upgrade -r Requirements.txt
python3 main_wave_ml_challenge.py

This will print out the best classifier selected using cross validation along with the
cross validation metrics with all different hyperparameter configurations.
It will also output the training_data with hypothetical cluster labels to discriminate
between business and personal expenses.

# 2. Algorithm 
Prediction of expense catefory is a classification problem. The dataset is a 
mixed bag of continuous and categorical features. 
The training and validatoon datasets are joined with employee dataset to get the 'role'.
First continuous features are standardized and categorical features i.e. 'role' and 'tax name' 
are binarized. Upon observation it can be stipulated that 'expense description' feature
plays a really important role in determination of category; therefore to leverage it, we 
use the tokenization and bag of words approach to generate binary features.
To build the classifier, support vector machine (SVM) model was used and its hyperparameters
were learned via cross validation.  To counteract overfitting which is often the 
case with small datasets (model doesn't generalize well for unobserved samples)
regularization parameter learned through 5 fold cross valiation was also non-zero. Another 
hyperparameter learned was 'kernel'; since data is not linearly separable, a non-linear kernel
called radial basis function performed best. 

As per ML literature and blogs, Bagging Estimators is a good approach with limited data, and
it was tried with base estimator set to decision tree. Bagging trains mulitple models 
    randomly selected proportion of dataset in terms of features and samples. 
    This often helps counteract the data limitation and curse ofdimensionality problems. It has a similar effect as 
    oversampling.
   
For the second task, kmeans clustering approach was used. In order to discriminate between business and personal expenses when no labeled data is present is an unsupervised
    learning problem. 
    To tackle this, first features needs to be prepared that are apt for a clustering algorithm input. 
    There are 2 real valued features 'pre-tax amount' and 'tax amount' that are used along with the categorical features
    'tax name', 'role','category'. The categorical features need to binarized as we did with classification model in 
    order to map them to real values. 

   Also note 'expense description' can be vital for the clustering algorithm but it leads a high dimensional 
    feature space and with such a small dataset it wouldn't be feasible. But for a sufficiently large dataset, 
    same approach can be followed as in classification feature engineering ie generate vocabulary and binarize.



# Performance
To measure performance of classification model, three metrics were used 'precision',
'recall' and 'f1-score'. For the best Bagging model: precision= 12%, recall=25% and f1-score=14%

For expense type clustering model, the script outputs the learned cluster index in 
cluster_index column. 