# Extracting best features for classification using Chi-squared based Univariate Selection
import pandas
import numpy
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2

# load data & pre-process
csvfile = 'training_data_example_and_validation.csv'
names = ['date', 'category', 'employee id', 'expense description', 'pre-tax amount', 'tax name', 'tax amount']
dataframe = pandas.read_csv(csvfile, names=names)
dataframe['date'] = pandas.factorize(dataframe['date'])[0]
dataframe['category'] = pandas.factorize(dataframe['category'])[0]
dataframe['employee id'] = pandas.factorize(dataframe['employee id'])[0]
dataframe['expense description'] = pandas.factorize(dataframe['expense description'])[0]
dataframe['pre-tax amount'] = pandas.factorize(dataframe['pre-tax amount'])[0]
dataframe['tax name'] = pandas.factorize(dataframe['tax name'])[0]
dataframe['tax amount'] = pandas.factorize(dataframe['tax amount'])[0]

array = dataframe.values
Y = array[:, 1]  # Use category field as target
X = array[:, [0, 2, 3, 4, 5, 6]]  # Use other fields as potential predictors
predictor_names = ['date', 'employee id', 'expense description', 'pre-tax amount', 'tax name', 'tax amount']
# feature extraction
test = SelectKBest(score_func=chi2, k=4)
fit = test.fit(X, Y)
# summarize scores
numpy.set_printoptions(precision=3)
counter = 0
# show how each feature scored
for x in fit.scores_:
    print x, '  ', predictor_names[counter]
    counter = counter + 1
features = fit.transform(X)
# summarize selected features
print(features[0:3, :])
