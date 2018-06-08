import numpy as np
from gensim.test.utils import datapath, get_tmpfile
from gensim.models import KeyedVectors

# Function to vectorize a sentence one word at a time. 
# Returns sentence vector which is the addition of individual word vectors in a given sentence
# Implementation can either return sum of word vectors or sum of word vectors divided by number of words in a sentence
def sent_vectorizer(sent, model):
    sent_vec =[]
    numw = 0
    for w in sent:
        try:
            if numw == 0:
                sent_vec = model[w]
            else:
                sent_vec = np.add(sent_vec, model[w])
            numw+=1
        except:
            pass
    
    # return np.asarray(sent_vec) / numw
    return np.asarray(sent_vec)

# Function to return the pre-trained glove model from the specified path
def Get_Word2Vec_Model_From_Glove(Glove_File_Path):
    from gensim.scripts.glove2word2vec import glove2word2vec
    
    glove_file = datapath(Glove_File_Path)
    tmp_file = get_tmpfile("test_word2vec.txt")
    glove2word2vec(glove_file, tmp_file)
    model = KeyedVectors.load_word2vec_format(tmp_file)
    
    return model

# Function to return pre-trained gensim wordvector from specified path
def Get_Word2Vec_Model(Word2Vec_File_Path):
    from gensim.models import KeyedVectors
    
    model = KeyedVectors.load_word2vec_format(Word2Vec_File_Path)
    return model