# Wave Machine Learning Engineer Challenge

1. Instructions on how to run the application

There are three main functions: get_data() is to parse the files and prepare data for models; category_prediction() is to run models for assigning individual expense transaction to one of predefined categories (Two modes are provided for use. Mode 1 uses logistic regression on Spark, while mode 2 applies the neural network.); expense_type_prediction() is for predicting the type (personal or business) of each transaction.
To further explain the models and various components, some comments are presented in the code including assumptions for assigning the original expense types.

2. A paragraph or two about what what algorithm was chosen for which problem, why (including pros/cons) and what you are particularly proud of in your implementation, and why

In this practice, logistic regression and the neural network were applied. Logistic regression is a simple but powerful traditional algorithm that works well on large dataset. Since only a small sized sample was given, the prediction is not optimal. On the other hand, the neural network has a greater potential to solve complicated problems but higher computing power/time is required and the result may not be as easily interpreted as those from traditional algorithms. In addition, it is unstable with small dataset.
Despite the ability of choosing different algorithms, one example of error handling is added to ensure robustness of the code.

3. Overall performance of your algorithm(s)

The limitation here is mainly the size of the given dataset (only 24 training entries and 12 validation entries).
For category prediction on the validation dataset: logistic regression results in 0.75 for precision, recall and f1 score, while the weighted scores range from 0.66 to 0.75. On the other hand, the neural network has highest accuracy of 0.5 and lowest loss of 1.3.
For expense type prediction on the validation dataset: accuracy tops at 0.83 while loss 0.45.
