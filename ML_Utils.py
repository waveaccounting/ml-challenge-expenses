from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import cross_val_score
from sklearn.model_selection import train_test_split

import pandas as pd
import numpy as np

def Train_Model(Training_Features,Category_Labels):
    X_Train, X_Test, y_Train,y_Test = train_test_split(Training_Features,Category_Labels,random_state = 100)
    
    clf = RandomForestClassifier(max_depth=5,n_estimators=15,random_state = 100)
    clf.fit(X_Train, y_Train)
    train_score = clf.score(X_Train, y_Train)
    
    return clf, train_score,X_Test,y_Test

def Validate_Model(model,Validation_Features,Validation_Labels):
    df_results = pd.DataFrame(data=np.zeros(shape=(1,3)), columns = ['classifier', 'train_score', 'test_score'] )
    
    test_score  = model.score(Validation_Features, Validation_Labels)
    
    df_results.loc[1,'classifier'] = "Random Forest"
    # df_results.loc[1,'train_score'] = train_score
    df_results.loc[1,'test_score'] = test_score
    
    print  ("Prediction Probabilities")
    print  ("========================")
    print  (model.predict_proba(Validation_Features))
    
    # print  (model.predict(Validation_Features))
    
    return df_results

    
    
    