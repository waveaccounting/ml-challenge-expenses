ML Challenge Markdown
# Wave Machine Learning Engineer Challenge

### Questions to answer:
1. Train a learning model that assigns each expense transaction to one of the set of predefined categories and evaluate it against the validation data provided.  The set of categories are those found in the "category" column in the training data. Report on accuracy and at least one other performance metric.

2. Mixing of personal and business expenses is a common problem for small business.  Create an algorithm that can separate any potential personal expenses in the training data.  Labels of personal and business expenses were deliberately not given as this is often the case in our system.  There is no right answer so it is important you provide any assumptions you have made.

3. (Bonus) Train your learning algorithm for one of the above questions in a distributed fashion, such as using Spark.  Here, you can assuming either the data or the model is too large/efficient to be process in a single computer.

### Answer

Question 1 has been answered for which 2 models were developed. 

1. Similarity Model: This needs no training. It calculates vectors for the categories in a 50-d space by summing and normalizing the 50-d pretrained word vectors. Then vectors for 'expense description' are calculated and the category assigned is the one that is closest as measured by cosine similarity in this 50-d space.<br/>  
The accuracy on both training and validation set turns out to be 0.75. The precision,recall and f1 calculations are in the notebook.<br/> 
The model serves as a good strong baseline for any complex models to beat. Because it gives decent results even without any examples; it can also help solve the cold start problem for new categories. 

2. LSTM Model: This is text classification model developed with keras.It also uses the pretrained 50-d vectors and only the 'expenses description' field.<br/>         
While there may be value in other fields like employee type but for this exercise I have chosen not to use those in the interest of time. Another reason for not using any other features is that there is a good chance that it will lead to overfitting given the smallness of our dataset. 


### Documentation:

1. Instructions on how to run your application<br/>
The application has been tested with Python 3.6.1. Each python notebook has commands to install the dependencies if they are not already available. Jupyter lab would be perfect to use for running the notebooks but jupyter notebook should also work.<br/>

2. A paragraph or two about what what algorithm was chosen for which problem, why (including pros/cons) and what you are particularly proud of in your implementation, and why <br/>
See Answer section for a description of models. <br/>
It may be worthwhile to take note of the analyze_predictions method in app_utils.py. It has been my experience that establishing deep metrics helps iterate quickly on the models. It particular confusion matrix can quickly point to the place where the model is struggling (and therefore where more data will help most).

3. Overall performance of your algorithm(s)<br/>
The LSTM model achieves an accuracy of 100% on both the training and validation set. It is promising but we must be cautious as there isn't really enough data to have any reasonable degree of confidence about the performance of the model. 

