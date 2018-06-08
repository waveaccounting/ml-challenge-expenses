import pandas as pd
import Text_Proc_Utils as TPU

# This function returns a dataframe with 2 columns. Category of expenses column as a categorical variable 
# and expense description as string. 
def Get_Data(File_Path):
    expenses = pd.DataFrame.from_csv(File_Path,index_col= None)
    
    expenses.category = expenses.category.astype("category")
    
    Sentences = expenses['expense description'].tolist()
    
    return Sentences, expenses.category

# This function takes the expenses decription sentences and returns sentence vectors
def Get_Feature_Vectors(Sentences,model):
    V=[]
    for sentence in Sentences:
        V.append(TPU.sent_vectorizer(sentence, model))
    return V

