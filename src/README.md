# Wave Machine Learning Engineer Challenge

There are two algorithms included in this code. The first trains a Fully 
Connected DNN model with arbitrary number of layers and neurons, which will be
trained on the data provided in the code package. The second is a simple algorithm 
used to seperate personal expenses from company expenses. 

The next section will detail how to run the algorithms and the following section 
will detail the decisions and details that went into the code, for your evaluation
and consideration.

## Running

The code was written in python 3.6, to install all the dependancies use pip.

`pip install -r requirements.txt`
 
 
 
### Train Model
 
I have provided a stripped down and modified version of a 
coding enviornment I have built and often use to explore different neural network architectures.
To train the model use the command

`python train.py`


To follow the progress of the training, I have logged the accuracy, loss, regalarization
loss and classification loss to a Tensorboard write out. You can access these with the 
command:

`tensorboard logdir=.`

and open the browser to the specified address, this should be your localhost at port 6006.
The parameters defining the model are defined in 'parameters.yml'. These include:



#### model_param

Incude parameters that pertain to all models.

**model_type:** This defines the model to be trained during the training process.
	The choosen model should be designed to ingest the correct type features and labels.
		
    
#### Model Specific Parameters

Each defined model will have a set of unique parameters to define it's
architechture. These are defined under the heading of their model type. Only 
one model has been included in this exercise.
 
#### fc

The only included model, this is a simple fully connected DNN with some 
optimization and architectual tricks included

**batch_norm:** include Batch Normalization on the FC layers? (T/F)

**drop_out:** include drop out on the FC layers? (T/F)

**drop_out_rate:** if drop_out is True this defines the drop out rate
	used throughout the model.
	
**layers:** defines the architecture of the model. The input data will 
	be fed into **n** FC layers of variable size. The final feature layer will 
	be fed into a scoring layer to classify the category of the entry. The parameter 
	takes in an array of size **n** each value representing the number of neurons 
	for that layer. for example [1000,1000,1000] defines a three layer model, each 
	layer with 1000 neurons
	
**reg_type:** Allows for choice of regularization [None/L1/L2]

**reg_strength:** If regularization is used this value determines 
	the scaling strength in the loss function.
	
**residual:** include residual learning in like sized layers? (T/F)

    
### solver_param

Include parameters that determine where data pertaining to the model will be stored
and the hyperparameters of the training process.

**id:** The name of the model

**batch_size:**  the number of samples used for each training iteration

**learning_rate:** the scaling factor used during back propogation of the 
	loss function

**read_out:** frequency of read outs during training (iterations). 
	Note: the script will also read out after each epoch. 

**model_check_point:** frequency in epochs of how often to checkpoint the 
	model and save the current weights.
	
**validation_check_point:** frequency in epochs of how often to check the 
	validation data.
	
**epoch:** number of times to go through the entire training data set.

The model can be optimized over all of these parameters, depending on the 
size of the data set and the character of the information within it.
  

### Seperate Personal Expenses

This algorithm will read in the training data from this repository and 
seperate out the personal expenses and read them out to 'personal.csv'.
To run the script use the command

`python seperate_personal_expenses.py`

The script identifies entries with 'personal expense' tags in the description.
These tags can be modifier in the python file *seperate_personal_expenses.py*.


## Algorithm Details

The following section details some of the decisions in the design of the 
code I have submitted and some further thoughts for improvement and exploration
into the models and data.

### Train Model
 
For this type of problem many methods could have been used and in a real 
professional situation, should be tested. I decided to implement a fully connected
neural network framework to showcase my ability to work within tensorflow and 
demonstrate some basic knowledge of neural networks. 

In implementing a neural network we need to consider three things:

1) The input data

2) The model architecture

3) The loss function


For categorical data, the input data has been transformed into a one hot encoding
format for model ingestion, this allows each category to seperately influence the 
model and avoids any spurious numerical relationship from developing. 
Numerical data has been transformed to a range of [0,1] within the training data. This 
choice was made to allow the network weights of the first layer to live in the same 
magnetude and hence allow for more efficient optimization of the model.

From employee records the 'role' category was added to the training and validation 
data for each entry, I decided name and address of an employee likely held no helpful information for the 
purposes of predicting the expense category.


The model I implemented is flexible allowing for optimization of the architecture,
it also allows the inclusion of state of the art optimization methods such as 
drop out, batch norm and residual learning. The model itself is a simple fully connected 
deep neural network.

I used cross-entropy to drive the loss function with the inclusion of a regularization 
term. This is a standard loss function for multi-label classification, something like 
margin loss could also be used and in optimizing a larger dataset it would be useful to 
explore the behaviour of both functions. 

The dataset itself is to small to do anything but overfit the training data. The
flexibility of the model allows for optimization when more data is provided.




### Seperate Personal Expenses

The solution I implemented for this problem is very naive, here I would like to 
discuss the nature of the problem and some possible solutions.


In implementing a solution to identifying personal expenses we must understand 
how this identification is to be used to determine how to tune a solution. Are we 
flagging expenses for review or are the positives then automatically excluded? 
What constitutes a personal expense and does this vary from company to company? These 
types of questions will influence whether to tune a network  to eliminate false positives 
or protect against false negatives.



As for a solution, an interesting method to explore would be to build a corpus of expense 
descriptions and build a word embedding based on this corpus. We could then use this word 
embedding to explore the word vector space for interesting properties to decide whether 
an expense is personal or not. Additionally, if we identified a set of personal expenses 
we could use this word embedding to form a supervised learning problem that would generalize 
better to unseen data. 

 
 

 