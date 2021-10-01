# CREDITS: https://machinelearningmastery.com/sequence-classification-lstm-recurrent-neural-networks-python-keras/

# import lib    
import numpy
import keras
from keras.datasets import imdb
from keras.models import Sequential
from keras.layers import Dense, LSTM
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
from tensorflow.keras import models

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
model.add(LSTM(100))
model.add(Dense(1,activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam',metrics=['accuracy'])
print(model.summary())

""" Model: "sequential_15"
_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
embedding_15 (Embedding)     (None, 600, 32)           160000
_________________________________________________________________
lstm_15 (LSTM)               (None, 100)               53200
_________________________________________________________________
dense_15 (Dense)             (None, 1)                 101
=================================================================
Total params: 213,301
Trainable params: 213,301
Non-trainable params: 0
_________________________________________________________________
None """

# Refer: https://datascience.stackexchange.com/questions/10615/number-of-parameters-in-an-lstm-model for knowing how to compute param

model.fit(X_train,y_train, epochs=10, batch_size=64)
               
""" 391/391 [==============================] - 14s 36ms/step - loss: 0.5521 - accuracy: 0.7158
Epoch 2/10
391/391 [==============================] - 14s 35ms/step - loss: 0.3169 - accuracy: 0.8732
Epoch 3/10
391/391 [==============================] - 14s 36ms/step - loss: 0.2468 - accuracy: 0.9032
Epoch 4/10
391/391 [==============================] - 14s 35ms/step - loss: 0.2230 - accuracy: 0.9152
391/391 [==============================] - 13s 34ms/step - loss: 0.1981 - accuracy: 0.9246
Epoch 6/10
391/391 [==============================] - 14s 35ms/step - loss: 0.1825 - accuracy: 0.9316
Epoch 7/10
391/391 [==============================] - 14s 35ms/step - loss: 0.1666 - accuracy: 0.9368
Epoch 8/10
391/391 [==============================] - 14s 35ms/step - loss: 0.1470 - accuracy: 0.9449
Epoch 9/10
391/391 [==============================] - 14s 35ms/step - loss: 0.1443 - accuracy: 0.9468
Epoch 10/10
391/391 [==============================] - 13s 34ms/step - loss: 0.1307 - accuracy: 0.9521
<tensorflow.python.keras.callbacks.History object at 0x0000013394103688> """

scores = model.evaluate(X_test,y_test, verbose=0)
print(scores)
# [0.4422934651374817, 0.8656799793243408]

print("Accuracy: %.2f%%" % (scores[1]*100))
# Accuracy: 86.57%