#!/usr/bin/env python
# coding: utf-8

# In[1]:


#INTRODUCTION
import tensorflow as tf


# In[2]:


#Get Dataset from mnist

from tensorflow.keras.datasets import mnist
(x_train,y_train),(x_test,y_test)=mnist.load_data()


# In[3]:


#see shape of the dataset images. It contains 6000 images with 28 as height and 28 as width
x_train.shape


# In[4]:


#Plot image on graph using matplotlib

from matplotlib import pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
plt.imshow(x_train[0],cmap='binary')  #You can keep changing the array no. from 0 to any no. between 0 and 6000
plt.show()


# In[5]:


#Diplaying labels of corresponding image
y_train[0]                             # this is the label for image at x_train[0], labels aka ur expected ans for the image


# In[6]:


# one hot encoding

# It basically means encoding ur labels into a list of 10 elements where the element of the corresponding class
# is changed to 1 and rest others are kept as 0

#eg for label 5 we have [0,0,0,0,0,1,0,0,0,0] and for 10 we have [0,0,0,0,0,0,0,0,0,1]

#for this we use to_categorical function in keras instead of making a forloop and contnuously creating the encoded labels


from tensorflow.keras.utils import to_categorical
y_train_encoded=to_categorical(y_train)
y_test_encoded=to_categorical(y_test)


# In[7]:


print(y_train_encoded[1])
print(y_test_encoded[1])


# In[8]:


#NEURALNETWORKS

#Unroll N dimensional arrays to vectors
import numpy as np
x_train_reshape=np.reshape(x_train,(60000,784))
x_test_reshape=np.reshape(x_test,(10000,784))


#display normalized pixel values


# In[9]:


#display pixel values
print(set(x_train_reshape[0]))


# In[10]:


#data normalization. For this we calculate the mean and std deviation of our dataset
x_mean=np.mean(x_train_reshape)
print(x_mean)
x_deviation=np.std(x_train_reshape)
print(x_deviation)

epsilon=1e-10


# In[11]:


x_train_normal=(x_train_reshape - x_mean)/(x_deviation + epsilon)
x_test_normal= (x_test_reshape - x_mean)/(x_deviation + epsilon)


# In[12]:


print(set(x_test_normal[0]))


# In[13]:


#creation of the model

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

model=Sequential([
    Dense(128,activation='relu', input_shape=(784,)),
    Dense(128, activation='relu'),
    Dense(10, activation='softmax')
])


# In[14]:


#compile your model  (we will use sgd optimizer)

model.compile(
    optimizer='sgd',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()


# In[15]:


#Training of the model

model.fit(x_train_normal, y_train_encoded,epochs=4)


# In[24]:


#evaluation of model to check if it has not simply memorized the values but actually trained

loss,accuracy=model.evaluate(x_test_normal, y_test_encoded)
print('accuracy=',accuracy*100)
print('loss=',loss*100)


# In[25]:


#Final Predictions

preds=model.predict(x_test_normal)
preds.shape


# In[39]:


import matplotlib
plt.figure(figsize=(12,12))

start_index=0

for i in range(25):
    plt.subplot(5,5,i+1)
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])
    
    
    pred=np.argmax(preds[start_index+i])
    
    gt=y_test[start_index + i]       #gt=ground truth
    
    
    col='g'
    if pred != gt:
        col='r'
        
        
        plt.xlabel=('i={}, pred={}, gt={}'.format(start_index+i, pred, gt))
        plt.imshow(x_test[start_index+i], cmap='binary')
    plt.show()
    


# In[34]:


plt.plot(preds[8])


# In[ ]:




