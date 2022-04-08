# -*- coding: utf-8 -*-
"""
Created on Thu Apr  7 14:55:16 2022

@author: Mohamed Elsheikh
"""
# we will create a linear regresion model 

# import libraries 
import pandas as pd 
import matplotlib.pyplot as plt 

# reading the data set 
data_set = pd.read_csv('Salary_Data.csv')
x=data_set.iloc[:,[0]].values
y=data_set.iloc[:,[1]].values

# split data set into train and test data set 
 from sklearn.model_selection import train_test_split
 x_train,x_test,y_train,y_test =train_test_split(x,y,test_size=0.30)
 #linear regresion model 
 from sklearn.linear_model import LinearRegression
 # creat an object from  class linearregression 
 regressor = LinearRegression()
 
 regressor.fit(x_train,y_train )
 
 # model is created
 # test of the model
 y_pred=regressor.predict(x_test)
 ##using mean square error
 from sklearn.metrics import mean_squared_error
 error = mean_squared_error(y_test,y_pred )
 # visualization the train set by using plt
 plt.scatter(x_train,y_train,color='red')
 plt.plot(x_train,regressor.predict(x_train),color='blue')
 plt.title('Salary vs experience (training set)')
 plt.xlabel('Years of experiments')
 plt.ylabel('Salary')
 plt.show()
  #visualization the test set result
 plt.scatter(x_test,y_test,color='red')
 plt.plot(x_train,regressor.predict(x_train),color='blue')
 plt.title('Salary vs experience (testing set)')
 plt.xlabel('Years of experiments')
 plt.ylabel('Salary')
 plt.show()
   
 salary=regressor.predict([[10]])
 print(salary)
 
 