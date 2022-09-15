#!/usr/bin/env python
# coding: utf-8

# In[1]:


##Image recoginization use CNN :  
#dataset : inbuilt dataset : fashion_mnist
#framework :  tensorflow and keras 

#!pip install tensorflow


# In[2]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import warnings 
warnings.filterwarnings("ignore")
import tensorflow


# In[3]:


#inbuilt dataset : fashion_mnist which define in tensorflow.keras.datasets
#library

(X_train,Y_train),(X_test,Y_test)=tensorflow.keras.datasets.fashion_mnist.load_data()


# In[4]:


X_train.shape,Y_train.shape #60000 no. of images and 28*28 pixels


# In[5]:


X_train


# In[6]:


X_test.shape,Y_test.shape


# In[7]:


#access 1st row means 1st image 
X_train[0]


# In[8]:


#class label of 1st image
Y_train[0]


# In[9]:


#to show image
plt.imshow(X_train[0])  #show 1st image dataset
plt.axis('off') 
plt.show()


# In[10]:


#to show image
plt.imshow(X_train[1])  #show 2nd image dataset
plt.axis('off') 
plt.show()


# In[11]:


#Loads the Fashion-MNIST dataset.
'''
This is a dataset of 60,000 28x28 grayscale images of 10 fashion categories,
along with a test set of 10,000 images. This dataset can be used as a
drop-in replacement for MNIST. The class labels are:

Label	Description
0	'T-shirt/top'
1	Trouser
2	Pullover
3	Dress
4	Coat
5	Sandal
6	Shirt
7	Sneaker
8	Bag
9	Ankle boot'''


# In[12]:


#class_labels user defined list object
class_labels=['T-shirt/top','Trouser','Pullover','Dress','Coat','Sandal',
'Shirt','Sneaker','Bag','Ankle boot']
print(class_labels)


# In[13]:


#To show 25 images randomly 
plt.figure(figsize=(16,16))
j=1
for i in np.random.randint(0,1000,30):
    plt.subplot(6,5,j);j=j+1
    plt.imshow(X_train[i]) #0-255
    plt.axis('off')
    plt.title("Record No. {}-{}-{}".format(i+1,class_labels[Y_train[i]],Y_train[i]))


# In[14]:


#CNN means Convolutional neural network CNN : 
#Note : In CNN , we have to give 4 dimension input data(images) compulsory

#check dimension of dataset 
print("Dimension of Training data : ",X_train.ndim)
#We have 3 dimensional dataset
print("Shape of Training Data : ",X_train.shape)


# In[15]:


#change the dimesion of training data
#We have to converts 3D dimension dataset into 4D dimension dataset
#so we use inbuilt method of numpy :  expand_dims()

X_train=np.expand_dims(X_train,-1)#expand_dims(data,axis)

#check dimension of training data after expand dimension
print("Dimension : ",X_train.ndim)
#Check shape of training data after expand dimension
print(X_train.shape)


# In[16]:


#change the dimesion of training data
#We have to converts 3D dimension dataset into 4D dimension dataset
#so we use inbuilt method of numpy :  expand_dims()

X_test=np.expand_dims(X_test,-1)#expand_dims(data,axis)

#check dimension of training data after expand dimension
print("Dimension : ",X_test.ndim)
#Check shape of training data after expand dimension
print(X_test.shape)


# In[17]:


#feature scaling : -
#Feature Scaling : - 
##Feature Scaling on input data(training data and testing data) 
#apply min_max_scaler() means value  between 0 to 1 
#here min value=0 and max value=255
X_train=X_train/255
X_test=X_test/255


# In[18]:


#X_train[0] #access 1st image


# In[19]:


#training error>=testing error :  model is perfect means model is not 
#overfit
#Split Dataset (To split training dataset into (80%  train data and 
#20% :- validation data for check overfitting model)
#means take 80% data for training and 20% for validation from X_train and 
#Y_train 
#call train_test_split
from sklearn.model_selection import train_test_split
X_train,X_val,Y_train,Y_val=train_test_split(X_train,Y_train,test_size=0.2,
                                            random_state=1)


# In[20]:


X_train.shape


# In[21]:


X_val.shape


# In[22]:


#Convolutional Neural Network - model Building 
model=tensorflow.keras.models.Sequential([
 tensorflow.keras.layers.Conv2D(filters=32,kernel_size=3,strides=(1,1),
                padding='valid',activation='relu',input_shape=[28,28,1]),
    tensorflow.keras.layers.MaxPooling2D(pool_size=(2,2)),
    tensorflow.keras.layers.Flatten(),
    tensorflow.keras.layers.Dense(units=128,activation='relu'), #hidden layer
     tensorflow.keras.layers.Dense(units=10,activation='softmax') #output layer
])


# In[23]:


#check summary
model.summary()


# In[24]:


#compile model
model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])


# In[25]:


#train model
model.fit(X_train,Y_train,epochs=10,batch_size=512,verbose=1,
          validation_data=(X_val,Y_val))


# In[28]:


#training and testing score
model.evaluate(X_train,Y_train)
model.evaluate(X_val,Y_val)


# In[30]:


#test the model
Y_pred=model.predict(X_test).round(2)


# In[31]:


Y_pred


# In[32]:


#visualize
#To show 25 images randomly 
plt.figure(figsize=(16,16))
j=1
for i in np.random.randint(0,1000,25):
    plt.subplot(5,5,j);j=j+1
    plt.imshow(X_test[i].reshape(28,28)) #0-255
    plt.axis('off')
    plt.title('Actual={}/{}\nPredicted ={}/{}'.
              format(class_labels[Y_test[i]],Y_test[i],
    class_labels[np.argmax(Y_pred[i])],np.argmax(Y_pred[i])))

