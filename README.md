# Developing a Neural Network Regression Model

## AIM

To develop a neural network regression model for the given dataset.

## THEORY

A neural network is a computer program inspired by how our brains work. It's used to solve problems by finding patterns in data. Imagine a network of interconnected virtual "neurons." Each neuron takes in information, processes it, and passes it along.

The scope of the project includes data preprocessing, training, and evaluation of the neural network. However, it's important to acknowledge potential limitations, such as computational resources and constraints on model complexity.Performance evaluation will be carried out using appropriate regression metrics such as Mean Squared Error (MSE), Root Mean Squared Error (RMSE), or R-squared value. This will allow us to quantitatively measure the accuracy of the model's predictions against actual target values.

## Neural Network Model
![image](https://github.com/balaji-21005757/basic-nn-model/assets/94372294/0f57244e-8594-42b7-9740-7c13f654567c)


## DESIGN STEPS

### STEP 1:

Loading the dataset

### STEP 2:

Split the dataset into training and testing

### STEP 3:

Create MinMaxScalar objects ,fit the model and transform the data.

### STEP 4:

Build the Neural Network Model and compile the model.

### STEP 5:

Train the model with the training data.

### STEP 6:

Plot the performance plot

### STEP 7:

Evaluate the model with the testing data.

## PROGRAM
```
Developed by : Balaji K
Register number : 212221230011
```
### Importing Modules:
```
from google.colab import auth
import gspread
from google.auth import default
import pandas as pd
import matplotlib.pyplot as plt


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential as seq
from tensorflow.keras.layers import Dense as den
from tensorflow.keras.metrics import RootMeanSquaredError as rmse
```
### Authenticate & Create Dataframe using Data in Sheets:
```
auth.authenticate_user()
creds,_=default()
gc=gspread.authorize(creds)

worksheet=gc.open('DL EXP 1 DATA').sheet1
data = worksheet.get_all_values()

dataset1 = pd.DataFrame(data[1:],columns=data[0])
dataset1 = dataset1.astype({'Input':'float'})
dataset1 = dataset1.astype({'Output':'float'})
dataset1.head()
```
### Assign X and Y values:
```
X = dataset1[['Input']].values
y = dataset1[['Output']].values
```
### Normalize the values & Split the data:
```
scaler = MinMaxScaler()
scaler.fit(X)
X_n = scaler.fit_transform(X)
x_train,x_test,y_train,y_test = train_test_split(X_n,y,test_size = 0.3,random_state = 3)
```
### Create a Neural Network Model & Train it: 
```
my_model = seq([
    den(9,activation = 'relu',input_shape=[1]),
    den(16,activation = 'relu'),
    den(1),
])

my_model.compile(optimizer = 'rmsprop',loss = 'mse')

my_model.fit(x_train,y_train,epochs=1000)
my_model.fit(x_train,y_train,epochs=1000)
```
### Plot the Loss:
```
loss_plot = pd.DataFrame(my_model.history.history)
loss_plot.plot()
```
### Evaluate the model:
```
error = rmse()
pred = my_model.predict(x_test)
error(y_test,pred)
```
### Predict for some value:
```
x_n1 = [[7]]
x_n_n = scaler.transform(x_n1)
my_model.predict(x_n_n)
```
## Dataset Information
![image](https://github.com/balaji-21005757/basic-nn-model/assets/94372294/43f4b99f-d475-4489-b84a-d698f10017b5)

## OUTPUT

### Training Loss Vs Iteration Plot

![image](https://github.com/balaji-21005757/basic-nn-model/assets/94372294/0231571c-dacb-4e58-824a-cd6de617663a)


### Test Data Root Mean Squared Error

![image](https://github.com/balaji-21005757/basic-nn-model/assets/94372294/330d7e0e-82c9-414c-96bc-0e539a65fe4f)


### New Sample Data Prediction

![image](https://github.com/balaji-21005757/basic-nn-model/assets/94372294/1c125dab-8ad1-4891-92e5-de64509ad95d)



## RESULT
Thus to develop a neural network regression model for the dataset created is successfully executed.
