To run the code, please type the following.

```
python3 main.py
```

# Choice of Algorithm and Potential Improvements

I have chosen logistic regression with lasso, using 5-fold cross-validation to select the optimal regularization factor.

For a real-life project where accuracy is critical, I will probably implement an emsemble of
techniques, including logistic regression, SVM, neural network, and random forest, and have them
vote against each other.

The following features are used for the LR model.
- Expense month
- Expense day of the week
- Expense amount
- Sales tax state and rate combined into one variables
- Employee ID
- Employee address
- Employee role
- Unigrams from Expense description

Eventually, Lasso has narrowed down the following predictors.
- Some unigrams (e.g. computer for Hardware, airplane for Travel)
- Certain week of day (e.g. Tuesday for Computer - Hardware)
- Certain month (e.g. December for Meals and Entertainment)

Expense description seems to be the most predictive item. The algorithm can be potentially improved
by the following.
- Use also bi- or trigrams.
- Use word2vec pre-trained on a large corpus.
- Less granular employee address (e.g. only the city/state instead of the full address)

For a personal-vs-business classifier (not implemented in code), I would focus on the following
indicators.
- Presence of certain words (e.g. coffee, taxi)
- Lack of certain words (e.g. team)
- Expense amount (usually rather small)
- Expense date on weekend or public holiday

# Overall performance of your algorithm(s)

Train accuracy: 1.000; Test accuracy: 0.917.
