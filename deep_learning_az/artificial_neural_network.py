#%%

# Artificial Neural Network
#TODO (AC): translate this to pytorch


# Importing the libraries
import numpy as np
import pandas as pd
import tensorflow as tf
tf.__version__

# Part 1 - Data Preprocessing

# path = r'G:\My Drive\DATA_SCIENCE\deep_learning\Part 1 - Artificial Neural Networks'
path = r'C:\temp\local_learning'
fl = os.path.join(path, 'Churn_Modelling.csv')
# Importing the dataset
dataset = pd.read_csv(fl)

#%%
X = dataset.iloc[:, 3:-1].values
y = dataset.iloc[:, -1].values
print(X)
print(y)



#%%

# IMPUTE DATA WHERE DATA IS MISSING
# NOTE: AUTHOR HAS ALREADY DONE THIS



#%%

# Encoding categorical data
# Label Encoding the "Gender" column
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
X[:, 2] = le.fit_transform(X[:, 2])
print(X)




#%%

# One Hot Encoding the "Geography" column
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct = ColumnTransformer(transformers=[('encoder', 
                                      OneHotEncoder(), 
                                      [1])], 
                                remainder='passthrough')
X = np.array(ct.fit_transform(X))
print(X)




#%%
# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 17325)



#%%
# Feature Scaling (FUNDAMENTAL)
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)



#%%
# Part 2 - Building the ANN

# Initializing the ANN
ann = tf.keras.models.Sequential()

# Adding the input layer and the first hidden layer
ann.add(tf.keras.layers.Dense(units=6, activation='relu'))

# Adding the second hidden layer
ann.add(tf.keras.layers.Dense(units=6, activation='relu'))

# Adding the output layer - for 3-class output, 3 neurons. here we only need one
# sigmoid activation function gives probability of output, rather than rectifier [0, max(0,X)]
ann.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))

# NOTE: LOOK AT DIFFERENT ACTIVATION FUNCTIONS AND WHAT THEY DO




#%%
# Part 3 - Training the ANN

# Compiling the ANN
# optimizer
ann.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Training the ANN on the Training set
ann.fit(X_train, y_train, batch_size = 32, epochs = 100)
# Batch size is the number of predictions in the batches (32 rows at a time / 10000 rows)

# NOTE: LOOK AT DIFFERENT OPTIMIZERS AND LOSS FUNCTIONS AND WHAT THEY DO

#%%


# Part 4 - Making the predictions and evaluating the model

# Predicting the result of a single observation

"""
Homework:
Use our ANN model to predict if the customer with the following informations will leave the bank: 
Geography: France
Credit Score: 600
Gender: Male
Age: 40 years old
Tenure: 3 years
Balance: $ 60000
Number of Products: 2
Does this customer have a credit card? Yes
Is this customer an Active Member: Yes
Estimated Salary: $ 50000
So, should we say goodbye to that customer?

Solution:
"""

# Predicting a single new observation
geography = [1, 0, 0]
credit = 600
Gender = 1
Age = 40
Tenure = 3
Balance = 60000
Products = 2
CreditCard = 1
ActiveMember = 1
EstimatedSalary = 50000


print(ann.predict(sc.transform([[1, 0, 0, 600, 1, 40, 3, 600, 2, 1, 1, 50000]])) > 0.5)
# must produce a lists to have this work
# print(ann.predict(sc.tranform([geography + [credit, Gender, 
#                                 Age, Tenure, Balance, 
#                                 Products, CreditCard, 
#                                 ActiveMember, EstimatedSalary]                               
#                               ]         
#                             )
#                 ) > 0.5
#     )

"""
Therefore, our ANN model predicts that this customer stays in the bank!
Important note 1: Notice that the values of the features were all input 
                  in a double pair of square brackets. That's because the 
                  "predict" method always expects a 2D array as the format 
                  of its inputs. And putting our values into a double pair 
                  of square brackets makes the input exactly a 2D array.
                  
Important note 2: Notice also that the "France" country was not input 
                  as a string in the last column but as "1, 0, 0" in the 
                  first three columns. That's because of course the predict 
                  method expects the one-hot-encoded values of the state, 
                  and as we see in the first row of the matrix of features X, 
                  "France" was encoded as "1, 0, 0". And be careful to include 
                  these values in the first three columns, because the dummy 
                  variables are always created in the first columns.
"""
#%%
# Predicting the Test set results
y_pred = ann.predict(X_test)
y_pred = (y_pred > 0.5)
print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1))

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, y_pred)
print(cm)
accuracy_score(y_test, y_pred)