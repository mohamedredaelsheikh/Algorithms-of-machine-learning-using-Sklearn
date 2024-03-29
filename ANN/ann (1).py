# -*- coding: utf-8 -*-
"""ANN.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1TZutfcr5RrczMrZYSyX2_ByDbqHXYhE4

#### What is ANN?

Artificial Neural Networks (ANN) is a supervised learning system built of a large number of simple elements, called neurons or perceptrons. Each neuron can make simple decisions, and feeds those decisions to other neurons, organized in interconnected layers.

![download.png](attachment:740a1fa7-9975-4213-9a1c-af89313fdc57.png)![image.png](attachment:image.png)

![image.png](attachment:image.png)

#### What is Activation Function?

If we do not apply a Activation function then the output signal would simply be a simple linear function.A linear function is just a polynomial of one degree.

- Sigmoid
- Tanh
- ReLu
- LeakyReLu
- SoftMax

#### What is Back Propagation?

![image.png](attachment:image.png)

#### Steps for building your first ANN

- Data Preprocessing
- Add input layer
- Random w init
- Add Hidden Layers
- Select Optimizer, Loss, and Performance Metrics
- Compile the model
- use model.fit to train the model
- Evaluate the model
- Adjust optimization parameters or model if needed
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Flatten, Dense
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

dataset = pd.read_csv('Customer_Churn_Modelling.csv')

dataset.head()

# Extract the important Columns
X = dataset.drop(labels=['CustomerId', 'Surname', 'RowNumber', 'Exited'], axis = 1)
y = dataset['Exited']
X.head()

y.head()

from sklearn.preprocessing import LabelEncoder

#Encode the Geography to nuber to make processe
label1= LabelEncoder()
X['Geography'] = label1.fit_transform(X['Geography'])

X.head()

label2  = LabelEncoder()
X['Gender'] = label2.fit_transform(X['Gender'])
X.head()

#get_dummies  Convert categorical variable into dummy/indicator variables.
# As Geography is countries the label Encodl not work will 
X = pd.get_dummies(X, drop_first=True, columns=['Geography'])
X.head()

"""### Feature Standardization """

from sklearn.preprocessing import StandardScaler

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0, stratify = y)

# Scale the data to be in small range of number it will fast the process
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

X_train

"""### Build ANN """

model = Sequential()
# Desw : Fully connectet layer 
model.add(Dense(X.shape[1], activation='relu', input_dim = X.shape[1]))
model.add(Dense(128, activation='relu'))
model.add(Dense(1, activation = 'sigmoid'))

X.shape[1]

"""# For next  cell see vidio to understand
1-https://www.youtube.com/watch?v=SwWQpSVQlis

"""

# see the vidio up

model.compile(optimizer='adam', loss = 'binary_crossentropy', metrics=['accuracy'])

"""# What Is a Batch?
The batch size is a hyperparameter that defines the number of samples to work through before updating the internal model parameters.

Think of a batch as a for-loop iterating over one or more samples and making predictions. At the end of the batch, the predictions are compared to the expected output variables and an error is calculated. From this error, the update algorithm is used to improve the model, e.g. move down along the error gradient.

A training dataset can be divided into one or more batches
## What Is an Epoch?
The number of epochs is a hyperparameter that defines the number times that the learning algorithm will work through the entire training dataset.

One epoch means that each sample in the training dataset has had an opportunity to update the internal model parameters. An epoch is comprised of one or more batches. For example, as above, an epoch that has one batch is called the batch gradient descent learning algorithm.

You can think of a for-loop over the number of epochs where each loop proceeds over the training dataset. Within this for-loop is another nested for-loop that iterates over each batch of samples, where one batch has the specified “batch size” number of samples.

The number of epochs is traditionally large, often hundreds or thousands, allowing the learning algorithm to run until the error from the model has been sufficiently minimized. You may see examples of the number of epochs in the literature and in tutorials set to 10, 100, 500, 1000, and larger.
"""

model.fit(X_train, y_train, batch_size = 10, epochs = 10)

y_pred = model.predict_classes(X_test)

y_pred

model.evaluate(X_test, y_test)

from sklearn.metrics import confusion_matrix, accuracy_score

confusion_matrix(y_test, y_pred)

accuracy_score(y_test, y_pred)

