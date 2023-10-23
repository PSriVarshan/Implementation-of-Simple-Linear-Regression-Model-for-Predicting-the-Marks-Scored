# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the standard Libraries.
2. Set variables for assigning dataset values.
3. Import linear regression from sklearn.
4. Assign the points for representing in the graph. 
5. .Predict the regression for marks by using the representation of the graph. 
6. Compare the graphs and hence we obtained the linear regression for the given datas.
## Program:
### Program to implement the simple linear regression model for predicting the marks scored.

#### Developed by: Sri Varshan P
#### RegisterNumber: 212222240104

### IMPORT REQUIRED PACKAGE
```py
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error,mean_squared_error
import matplotlib.pyplot as plt
dataset=pd.read_csv('student_scores.csv')
print(dataset)
```
### READ CSV FILES
```py
dataset=pd.read_csv('student_scores.csv')
print(dataset.head())
print(dataset.tail())
```
### COMPARE DATASET
```py
x=dataset.iloc[:,:-1].values
print(x)
y=dataset.iloc[:,1].values
print(y)
```
### PRINT PREDICTED VALUE
```py
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=1/3,random_state=0)
from sklearn.linear_model import LinearRegression
reg=LinearRegression()
reg.fit(x_train,y_train)
y_pred = reg.predict(x_test)
print(y_pred)
print(y_test)
```
### GRAPH PLOT FOR TRAINING SET
```py
plt.scatter(x_train,y_train,color='purple')
plt.plot(x_train,reg.predict(x_train),color='black')
plt.title("Hours vs Scores(Training set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
```
### GRAPH PLOT FOR TESTING SET
```py
plt.scatter(x_test,y_test,color='red')
plt.plot(x_train,reg.predict(x_train),color='black')
plt.title("Hours vs Scores(Testing set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
```
### PRINT THE ERROR
```py
mse=mean_absolute_error(y_test,y_pred)
print('Mean Square Error = ',mse)
mae=mean_absolute_error(y_test,y_pred)
print('Mean Absolute Error = ',mae)
rmse=np.sqrt(mse)
print("Root Mean Square Error = ",rmse)
```

## Output:
### To Read All CSV Files

![image](https://github.com/PSriVarshan/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/114944059/88e2784e-e0e0-4bcf-bea4-96a328c26de2)


### To Read Head and Tail Files

![image](https://github.com/PSriVarshan/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/114944059/12f46ae8-c329-432e-b680-119ff82a842a)


### Compare Dataset

![image](https://github.com/PSriVarshan/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/114944059/e7801ab1-bee5-4c8e-8cf8-d08afd237840)


### Predicted Value

![image](https://github.com/PSriVarshan/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/114944059/e584ddde-9a3e-44c9-b488-41877ae4edaa)


### Graph For Training Set

![image](https://github.com/PSriVarshan/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/114944059/a5dcc324-46df-44ad-a6e7-6102da5899a0)



### Graph For Testing Set

![image](https://github.com/PSriVarshan/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/114944059/cc50d111-5302-420a-b4d2-4e1ba8e2f73e)


### Error

![image](https://github.com/PSriVarshan/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/114944059/5712c50b-d302-4bb3-a88c-4b3e75f18e85)


## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
