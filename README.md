# AmazonReview-Sentiment-Analysis
Sentiment Analysis using simple LSTM and Conv1D layers.

This is model that is used to get the sentiment rating(0-1 range). 0 means bad/worse and 1 means good sentiment.

## Dependencies
* Python 3+
* Keras with tensorflow backend
* nvidia Gpu (for training purpose as it use CuDNNLSTM layer that is accelerated by CuDNN library by nvidia)
* Numpy

## How to use
1. Fork this repo
2. Download the dataset from <a href='https://www.kaggle.com/bittlingmayer/amazonreviews'>here</a> .
3. Download the GloVe Word embeddings from <a href='http://nlp.stanford.edu/data/glove.6B.zip'> here</a>.
4. Save both data and GloVe embeddings in ```data``` folder.
5. If training, make changes in file ```utils/config.py``` if you want.
6. Use the ```train.ipynb``` notebook for training.
7. If using for test-predictions, download the weights from <a href='https://drive.google.com/open?id=1oqrnHQkSxdOI-SI8UsplTERjvZTjGjuV'>here</a> and save it in ```weights``` folder.
8. Use ```inference.ipynb``` notebook.
