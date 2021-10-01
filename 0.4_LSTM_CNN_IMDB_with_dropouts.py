# CREDITS: https://machinelearningmastery.com/sequence-classification-lstm-recurrent-neural-networks-python-keras/

# import lib    
import numpy
import keras
from keras.datasets import imdb
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
# CNN related imports
from keras.layers.convolutional import Conv1D, MaxPooling1D

# print keras version
keras.__version__
# 2.4.3

# CREDITS: https://keras.io/datasets/#imdb-movie-reviews-sentiment-classification
# load dataset
top_words = 5000
(X_train, y_train), (X_test,y_test) = imdb.load_data(num_words=top_words)

print(X_train[1])
""" [1, 194, 1153, 194, 2, 78, 228, 5, 6, 1463, 4369, 2, 134, 26, 4, 715, 8, 118, 1634, 14, 394, 20, 13, 119, 954, 189, 102, 5, 207, 110, 3103, 21, 14, 69, 188, 8, 30, 23, 7, 4, 249, 126, 93, 4, 114, 9, 2300, 1523, 5, 
647, 4, 116, 9, 35, 2, 4, 229, 9, 340, 1322, 4, 118, 9, 4, 130, 4901, 19, 4, 1002, 5, 89, 29, 952, 46, 37, 
4, 455, 9, 45, 43, 38, 1543, 1905, 398, 4, 1649, 26, 2, 5, 163, 11, 3215, 2, 4, 1153, 9, 194, 775, 7, 2, 2, 349, 2637, 148, 605, 2, 2, 15, 123, 125, 68, 2, 2, 15, 349, 165, 4362, 98, 5, 4, 228, 9, 43, 2, 1157, 15, 
299, 120, 5, 120, 174, 11, 220, 175, 136, 50, 9, 4373, 228, 2, 5, 2, 656, 245, 2350, 5, 4, 2, 131, 152, 491, 18, 2, 32, 2, 1212, 14, 9, 6, 371, 78, 22, 625, 64, 1382, 9, 8, 168, 145, 23, 4, 1690, 15, 16, 4, 1355, 5, 28, 6, 52, 154, 462, 33, 89, 78, 285, 16, 145, 95] """

print(type(X_train[1]))
# <class 'list'>

print(len(X_train[1]))
# 189

X_train.shape
# (25000,)

max(numpy.max(X_test))
# 4998

# padding
padding_length = 600

X_train = sequence.pad_sequences(X_train, maxlen=padding_length)
X_test = sequence.pad_sequences(X_test, maxlen=padding_length)

# view a datapoint after padding
print(X_train.shape)
# (25000, 600)

print(X_train[1])
""" [   0    0    0    0    0    0    0    0    0    0    0    0    0    0
    0    0    0    0    0    0    0    0    0    0    0    0    0    0
    0    0    0    0    0    0    0    0    0    0    0    0    0    0
    0    0    0    0    0    0    0    0    0    0    0    0    0    0
    0    0    0    0    0    0    0    0    0    0    0    0    0    0
    0    0    0    0    0    0    0    0    0    0    0    0    0    0
    0    0    0    0    0    0    0    0    0    0    0    0    0    0
    0    0    0    0    0    0    0    0    0    0    0    0    0    0
    0    0    0    0    0    0    0    0    0    0    0    0    0    0
    0    0    0    0    0    0    0    0    0    0    0    0    0    0
    0    0    0    0    0    0    0    0    0    0    0    0    0    0
    0    0    0    0    0    0    0    0    0    0    0    0    0    0
    0    0    0    0    0    0    0    0    0    0    0    0    0    0
    0    0    0    0    0    0    0    0    0    0    0    0    0    0
    0    0    0    0    0    0    0    0    0    0    0    0    0    0
    0    0    0    0    0    0    0    0    0    0    0    0    0    0
    0    0    0    0    0    0    0    0    0    0    0    0    0    0
    0    0    0    0    0    0    0    0    0    0    0    0    0    0
    0    0    0    0    0    0    0    0    0    0    0    0    0    0
    0    0    0    0    0    0    0    0    0    0    0    0    0    0
    0    0    0    0    0    0    0    0    0    0    0    0    0    0
    0    0    0    0    0    0    0    0    0    0    0    0    0    0
    0    0    0    0    0    0    0    0    0    0    0    0    0    0
    0    0    0    0    0    0    0    0    0    0    0    0    0    0
    0    0    0    0    0    0    0    0    0    0    0    0    0    0
    0    0    0    0    0    0    0    0    0    0    0    0    0    0
    0    0    0    0    0    0    0    0    0    0    0    0    0    0
    0    0    0    0    0    0    0    0    0    0    0    0    0    0
    0    0    0    0    0    0    0    0    0    0    0    0    0    0
    0    0    0    0    0    1  194 1153  194    2   78  228    5    6
 1463 4369    2  134   26    4  715    8  118 1634   14  394   20   13
  119  954  189  102    5  207  110 3103   21   14   69  188    8   30
  116    9   35    2    4  229    9  340 1322    4  118    9    4  130
 4901   19    4 1002    5   89   29  952   46   37    4  455    9   45
   43   38 1543 1905  398    4 1649   26    2    5  163   11 3215    2
    4 1153    9  194  775    7    2    2  349 2637  148  605    2    2
   15  123  125   68    2    2   15  349  165 4362   98    5    4  228
    9   43    2 1157   15  299  120    5  120  174   11  220  175  136
   50    9 4373  228    2    5    2  656  245 2350    5    4    2  131
  152  491   18    2   32    2 1212   14    9    6  371   78   22  625
   64 1382    9    8  168  145   23    4 1690   15   16    4 1355    5
   28    6   52  154  462   33   89   78  285   16  145   95] """

# create model

embedding_layer_outputs = 32
model = Sequential()
model.add(Embedding(top_words,embedding_layer_outputs,input_length=padding_length))
model.add(Dropout(0.2))
model.add(Conv1D(filters=32, kernel_size=3, padding='same',activation='relu'))
model.add(MaxPooling1D(pool_size=2))
model.add(LSTM(100))
model.add(Dropout(0.2))
model.add(Dense(1,activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam',metrics=['accuracy'])
print(model.summary())

""" 
Model: "sequential_4"
_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
embedding_4 (Embedding)      (None, 600, 32)           160000
_________________________________________________________________
dropout_2 (Dropout)          (None, 600, 32)           0
_________________________________________________________________
conv1d_1 (Conv1D)            (None, 600, 32)           3104
_________________________________________________________________
max_pooling1d_1 (MaxPooling1 (None, 300, 32)           0
_________________________________________________________________
lstm_4 (LSTM)                (None, 100)               53200
_________________________________________________________________
dropout_3 (Dropout)          (None, 100)               0
_________________________________________________________________
dense_4 (Dense)              (None, 1)                 101
=================================================================
Total params: 216,405
Trainable params: 216,405
Non-trainable params: 0
_________________________________________________________________
None
"""

# Refer: https://datascience.stackexchange.com/questions/10615/number-of-parameters-in-an-lstm-model for knowing how to compute param

model.fit(X_train,y_train, epochs=10, batch_size=64)
               
""" 
... Epoch 1/10
... 391/391 [==============================] - 9s 22ms/step - loss: 0.1786 - accuracy: 0.9352
... Epoch 2/10
... 391/391 [==============================] - 9s 22ms/step - loss: 0.1489 - accuracy: 0.9473
... Epoch 3/10
... 391/391 [==============================] - 9s 22ms/step - loss: 0.1258 - accuracy: 0.9572
... Epoch 4/10
... 391/391 [==============================] - 9s 22ms/step - loss: 0.1091 - accuracy: 0.9642
... Epoch 5/10
... 391/391 [==============================] - 9s 22ms/step - loss: 0.0691 - accuracy: 0.9800
... 391/391 [==============================] - 9s 22ms/step - loss: 0.0505 - accuracy: 0.9856
... 391/391 [==============================] - 9s 22ms/step - loss: 0.0353 - accuracy: 0.9912
... <tensorflow.python.keras.callbacks.History object at 0x00000199D400E3C8>
"""

scores = model.evaluate(X_test,y_test, verbose=0)
print(scores)
# [0.41930627822875977, 0.8704800009727478]

print("Accuracy: %.2f%%" % (scores[1]*100))
# Accuracy: 87.05%