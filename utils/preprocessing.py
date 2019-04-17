import numpy as np
import re
from keras.preprocessing.sequence import pad_sequences
from utils.config import *
import time


def load_dataset(path = 'data/train.ft.txt'):
    """
        Arguments: 
            path: it is the path of actual dataset.
            
        this function reads the dataset and return the raw corpus
        
        Return:
            corpus: array of raw lines
    
    """
    print('Reading Dataset...')
    start = time.time()
    corpus = []
    count = 0
    with open(path, encoding = 'utf-8', errors = 'ignore') as f:
        for i in f:
            if count < TRAINING_DATASET_SIZE:
                corpus.append(i)
                count += 1 
    end = time.time()
    print('It took', end-start, 'secs to load.')
    return corpus


def splitXY(corpus):
    """
        Arguments:
            corpus: it is the raw coprus which contains raw text
            
        this function that reads corpus line by line and splits it into X and y
    
        Return:
            X: it is the training sentences
            y: it is the boolean value for good or bad review (0 for bad review and 1 for good review)
    """
    print('Splitting...')
    start = time.time()
    X = []
    y = []
    for i in corpus:
        X.append(i.split(' ', 1)[1])
        if int(list(i.split(' ', 1)[0])[9]) == 1:
            y.append(0)
        else:
            y.append(1)
    y = np.array(y)
    print('It took', time.time() - start , 'secs to split.')
    return X, y

def clean(text):
    """
        Arguments: 
            text: it is the string that is to be cleaned
        
        this is a helper function that cleans the string
        
        Return: 
            text: cleaned string
    """
    text = text.lower()
    text = re.sub(r'i\'m', 'i am', text)
    text = re.sub(r'he\'s', 'he is', text)
    text = re.sub(r'she\'s', 'she is', text)
    text = re.sub(r'that\'s', 'that is', text)
    text = re.sub(r'what\'s', 'what is ', text)
    text = re.sub(r'wher\'s', 'where is ', text)
    text = re.sub(r'\'ll', ' will', text)
    text = re.sub(r'\'ve', ' have', text)
    text = re.sub(r'\'er', ' are', text)
    text = re.sub(r'\'ve', ' have', text)
    text = re.sub(r'\'d', ' would', text)
    text = re.sub(r'won\'t', 'will not', text)
    text = re.sub(r'can\'t', 'cannot', text)
    text = re.sub(r"[-()\"\\#@/;:$*<>=|.,?&_^%'!~+]", '', text)
    return text

def cleanX(X):
    """
        Arguments:
            X: training sentences
            
        this functions cleans X
        
        Return:
            clean_X: array after cleaning
    """
    print('Cleaning...')
    start = time.time()
    clean_X = []
    for i in X:
        clean_X.append(clean(i).split())
    print('It took', time.time() - start , 'secs to clean.')
    return clean_X

def buildVocab(X):
    """
        Arguments: 
            X: cleaned sentences
        
        this function takes clean sentences array and build vocabulary
        
        Return:
            vocab: the vocabulary
    """
    print('Building Vocab..')
    start = time.time()
    wordcount = {}  # helper dictionary for maintaining word count
    for i in X:
        for j in i:
            if j in wordcount:
                wordcount[j] +=1
            else:
                wordcount[j] = 1
                
    vocab = {}
    n=0
    for i in wordcount:
        if wordcount.get(i) < VOCAB_THRESHOLD:
            continue
        else:
            if wordcount.get(i) in wordcount:
                continue
            else:
                n += 1
                vocab[i] = n
    print('It took', time.time() - start , 'secs to build vocab.')
    return vocab

def tokenizeAndPadd(X, vocab, vocab_size):
    """
        Arguments:
            X: cleaned sentences array
            vocab: vocublary
            vocab_size: size of the vocabulary
            
        this function tokenize and padd the cleaned sentences
        
        Return:
            X_final = final array that is to be feeded into the model
    """
    print('Tokenizing and Padding... ')
    start = time.time()
    X_final = []
    for i in X:
        temp = []
        for j in i:
            if j in vocab:
                temp.append(vocab.get(j))
            else:
                temp.append(vocab_size + 1)
        X_final.append(temp)
        
    X_final = pad_sequences(X_final, maxlen = MAXLEN_INPUT)
    print('It took', time.time() - start , 'secs to tokenize and pad.')
    return X_final

    
def gloveEmbedding(vocab, path = 'data/glove.6B.100d.txt'): #change
    """
		Arguments:
			path: path of glove txt file
			vocab: vocabulary
            
        This is a function to load GloVe Embeedding and making Embedding Matrix		
		
		Returns:
			embedding_matrix: Embedding Matrix
    """
    print('Loading Glove word embedding... ')
    start = time.time()
    word2vec = {}
    with open(path, encoding = 'utf-8', errors = 'ignore') as f:
        for line in f:
            values = line.split()
            word = values[0]
            vec = np.asarray(values[1:], dtype = 'float32')
            word2vec[word] = vec
        print('Found %s word vectors' % len(word2vec))
        print('Filling pre-trained embeddings...')
        num_words = len(vocab) + 1
        embedding_matrix = np.zeros((num_words, EMBEDDING_DIM))
        for word, i in vocab.items():
            embedding_vector = word2vec.get(word)
            if embedding_vector is not None:
                embedding_matrix[i] = embedding_vector
    
    print('It took', time.time() - start , 'secs to build embedding matrix.')
    return embedding_matrix


    

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    