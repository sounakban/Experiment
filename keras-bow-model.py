
# coding: utf-8

# In[1]:


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


# In[3]:


import itertools
import os

# get_ipython().magic('matplotlib inline')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.preprocessing import LabelBinarizer, LabelEncoder
from sklearn.metrics import confusion_matrix

from my_layer import ParaNet_layer1, ParaNet_layer2
# import keras
from keras.models import Model
from keras.layers import Input, Dense, Activation, Dropout, Lambda
from keras.preprocessing import text, sequence
from keras import utils

# This code was tested with TensorFlow v1.4
print("You have TensorFlow version", tf.__version__)


# In[5]:


# The CSV was generated from this query: https://bigquery.cloud.google.com/savedquery/513927984416:c494494324be4a80b1fc55f613abb39c
# The data is also publicly available at this Cloud Storage URL: https://storage.googleapis.com/tensorflow-workshop-examples/stack-overflow-data.csv
# data = pd.read_csv("so-export-0920.csv")
data = pd.read_csv("stack-overflow-data.csv")


# In[6]:


data.head()


# In[7]:


# Confirm that we have a balanced dataset
# Note: data was randomly shuffled in our BigQuery query
data['tags'].value_counts()


# In[7]:


# Split data into train and test
train_size = int(len(data) * .8)
print ("Train size: %d" % train_size)
print ("Test size: %d" % (len(data) - train_size))


# In[8]:


train_posts = data['post'][:train_size]
train_tags = data['tags'][:train_size]

test_posts = data['post'][train_size:]
test_tags = data['tags'][train_size:]


# In[9]:


max_words = 1000
tokenize = text.Tokenizer(num_words=max_words, char_level=False)


# In[10]:


tokenize.fit_on_texts(train_posts) # only fit on train
x_train = tokenize.texts_to_matrix(train_posts)
x_test = tokenize.texts_to_matrix(test_posts)


# In[11]:


# Use sklearn utility to convert label strings to numbered index
encoder = LabelEncoder()
encoder.fit(train_tags)
y_train = encoder.transform(train_tags)
y_test = encoder.transform(test_tags)


# In[12]:


# Converts the labels to a one-hot representation
num_classes = np.max(y_train) + 1
y_train = utils.to_categorical(y_train, num_classes)
y_test = utils.to_categorical(y_test, num_classes)


# In[13]:


# Inspect the dimenstions of our training and test data (this is helpful to debug)
print('x_train shape:', x_train.shape)
print('x_test shape:', x_test.shape)
print('y_train shape:', y_train.shape)
print('y_test shape:', y_test.shape)


# In[14]:


# This model trains very quickly and 2 epochs are already more than enough
# Training for more epochs will likely lead to overfitting on this dataset
# You can try tweaking these hyperparamaters when using this model with your own data
batch_size = 32
epochs = 5


# In[15]:


# Build the model
input = Input(shape=(max_words,))
output = Dense(512)(input)
output = Activation('relu')(output)
# output = Dropout(0.5)(output)
# num_of_splits = 128
# total_nodes = 512
# output = Lambda(ParaNet_layer1, arguments={'layer_splits':100, 'nodes_per_split':5})(output)
# output = Lambda(ParaNet_layer2, arguments={'layer_splits':128})(output)
# output = Dense(128)(input)
# output = Activation('relu')(output)
# output = Dropout(0.5)(output)
output = Dense(num_classes)(output)
output = Activation('softmax')(output)

model = Model(inputs=input, outputs=output)
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])



# In[16]:


# model.fit trains the model
# The validation_split param tells Keras what % of our training data should be used in the validation set
# You can see the validation loss decreasing slowly when you run this
# Because val_loss is no longer decreasing we stop training to prevent overfitting
history = model.fit(x_train, y_train,
                    batch_size=batch_size,
                    epochs=epochs,
                    verbose=1,
                    validation_split=0.1)


# In[17]:


# Evaluate the accuracy of our trained model
score = model.evaluate(x_test, y_test,
                       batch_size=batch_size, verbose=1)
print('Test score:', score[0])
print('Test accuracy:', score[1])


# In[18]:


# Here's how to generate a prediction on individual examples
text_labels = encoder.classes_

for i in range(10):
    prediction = model.predict(np.array([x_test[i]]))
    predicted_label = text_labels[np.argmax(prediction)]
    print(test_posts.iloc[i][:50], "...")
    print('Actual label:' + test_tags.iloc[i])
    print("Predicted label: " + predicted_label + "\n")


# In[19]:


y_softmax = model.predict(x_test)

y_test_1d = []
y_pred_1d = []

for i in range(len(y_test)):
    probs = y_test[i]
    index_arr = np.nonzero(probs)
    one_hot_index = index_arr[0].item(0)
    y_test_1d.append(one_hot_index)

for i in range(0, len(y_softmax)):
    probs = y_softmax[i]
    predicted_index = np.argmax(probs)
    y_pred_1d.append(predicted_index)


# In[21]:


# # This utility function is from the sklearn docs: http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html
# def plot_confusion_matrix(cm, classes,
#                           title='Confusion matrix',
#                           cmap=plt.cm.Blues):
#     """
#     This function prints and plots the confusion matrix.
#     Normalization can be applied by setting `normalize=True`.
#     """
#
#     cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
#
#     plt.imshow(cm, interpolation='nearest', cmap=cmap)
#     plt.title(title, fontsize=30)
#     plt.colorbar()
#     tick_marks = np.arange(len(classes))
#     plt.xticks(tick_marks, classes, rotation=45, fontsize=22)
#     plt.yticks(tick_marks, classes, fontsize=22)
#
#     fmt = '.2f'
#     thresh = cm.max() / 2.
#     for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
#         plt.text(j, i, format(cm[i, j], fmt),
#                  horizontalalignment="center",
#                  color="white" if cm[i, j] > thresh else "black")
#
#     plt.ylabel('True label', fontsize=25)
#     plt.xlabel('Predicted label', fontsize=25)
#
#
# # In[22]:
#
#
# cnf_matrix = confusion_matrix(y_test_1d, y_pred_1d)
# plt.figure(figsize=(24,20))
# plot_confusion_matrix(cnf_matrix, classes=text_labels, title="Confusion matrix")
# plt.show()
