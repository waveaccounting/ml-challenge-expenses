
## Instructions on how to run your application
1. Install [Anaconda](https://www.anaconda.com/)
2. Checkout my branch
3. Start Jupiter Notebook on my branch folder --> `ipython notebook --ip=127.0.0.1`
4. Open `category__expense_desc-Personal_Business.ipynb` notebook.


## A paragraph or two about what algorithm was chosen for which problem, why (including pros/cons) and what you are particularly proud of in your implementation, and why
I've chosen `SGDClassifier` algorithm as my final decision to solve this text processing (NLP) classification problem as `SGDClassifier` stands out among all three text processing algorhtims I recently studied after cross validating and tuning the parameter against training data. Those three algorithms are OneVsRestClassifier with LinearSVC, MultinomialNB, and SGDClassifier. 

What I'm pround of this solution is that I came from no experience of ML and took 2-3 days to study the basics, then was able to compare 3 different algorithims with cross validating and parameter tuning to get a sence of which performs better towards which problem (my take would be this is a text processing classification problem). I might just scratch the surface as I know text processing can go really deep into Neural Networks, but I'm pretty satisfied with I'm accomplished within these couple days.


## Overall performance of your algorithm(s)
After examining all three algorithms, They all have accuracy of 0.91666 (after tuning two of them).
However, sincee SGDClassifier has better result with k-fold cross validation, mean:0.884615 (std:0.210663),

| K-Fold metric        | mean           | (std)  |
| ------------- |:-------------:| -----:|
| SVC      | 0.846154  | (0.230769) |
| NB      | 0.692308       |   (0.417799) |
| SGD | 0.884615       |    (0.210663) |


I'd pick SGDClassifier algorithm as my first choice to train the model.


As for potential personal and business expenses, the assumption I make is that from the expense description, if there's certain keywords, like personal, family, families, I'd clean the data by wiping out the expense description to reduce the word count. 

Since there's no such keywords in training data, the results come back the same. I might need to study more on this area as I'm not so sure about the approach to take from here.


## Next Steps:
  - since this is text processing classificaiton and NLP problem, we can even tune with stop words and stemming.
  - find more text processing classification algorithm and compare them through the same process.
  - try Neural Networks approach using deep learning.
