from keras.models import Sequential
from keras.layers import Dense, Embedding, Dropout, Flatten, Conv1D, MaxPooling1D
from keras.layers import CuDNNLSTM as LSTM
from keras.callbacks import ModelCheckpoint
from keras import backend as K
from keras.preprocessing.sequence import pad_sequences
import time
from utils.config import *
from utils.preprocessing import *

class SentimentAnalysis:
    def __init__(self, vocab, vocab_size, embedding_matrix, weight_path = 'weights/SentimentAnalysisWeights.h5', weights_load_path = 'weights/SentimentAnalysisWeights.h5'):
        self.vocab_size = vocab_size
        self.vocab = vocab
        self.embedding_matrix = embedding_matrix
        self.weight_path = weight_path
        self.weights_load_path = weights_load_path
    
    def build_model(self):
        K.clear_session()
        model = Sequential()
        model.add(Embedding(self.vocab_size+1, 100, weights = [self.embedding_matrix], input_length = MAXLEN_INPUT, trainable = False))
        model.add(Dropout(0.2))
        model.add(Conv1D(CONV1D_FILTER_SIZE, 2))
        model.add(Dropout(0.4))
        model.add(LSTM(128))
        model.add(Dense(64, activation = 'relu'))
        model.add(Dropout(0.1))
        model.add(Dense(1, activation = 'sigmoid'))
        return model
    
    def train(self, X, y, model):
        model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics=['accuracy'])
        checkpoint = ModelCheckpoint(self.weight_path, monitor='val_acc', verbose=1, save_best_only=True, save_weights_only=True, mode='max')
        callback_list = [checkpoint]
        model.fit(X, y, batch_size=BATCH_SIZE, epochs=EPOCHS, validation_split=0.2, callbacks=callback_list, shuffle=True)
        
    def predict(self, input_text, model):
        start = time.time()
        clean_input = clean(input_text)
        temp=[]
        for i in clean_input.split():
            if i in self.vocab:
                temp.append(self.vocab.get(i))
            else:
                temp.append(self.vocab_size + 1)
        tokenized_input = pad_sequences([temp], maxlen = MAXLEN_INPUT)
        model.load_weights(self.weights_load_path)
        model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics=['accuracy'])
        prediction = model.predict(tokenized_input)
        end = time.time() 
        print('It took', end - start , 'secs to predict.')
        return prediction
    
        
