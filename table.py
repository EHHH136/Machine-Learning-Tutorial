'''
The Key Steps to create and use a machine learning model include a few steps:

1. Describing the Prediction target and Predictive features
2. Importing the model
3. Fitting the data to the Model / Training the Model
4. Making Predictions 

The code below is a simple example of how to do those things

'''


import pandas as pd
location = "C:/Users/Asus/Desktop/Python/Code/Machine Learning Tutorial/username.csv"

# Assigning the Data Set to a variable for ease of use
tab = pd.read_csv(location) 

# picking a prediction target from the Data Set
y = tab.Identifier

# picking predictive features from the Data Set
features = ['PredictiveData']
X = tab[features] # Assigning those predictive features to a variable



from sklearn.tree import DecisionTreeRegressor

# Creating a model with a constant random state to ensure the same results each time
idModel = DecisionTreeRegressor(random_state = 1)

# Fitting the data to the model / Training the model
idModel.fit(X, y)

# Making Prediction for the prediction target from the predictive feature.
# In this case we are using the same Predictive Features to Train and Predict, this isn't usually the case
print(idModel.predict(X))


#now in order to test the accuract of the model we are going to use an approach of mean absoulte error

#first we will split the dataset we have into two random halves 
from sklearn.model_selection import train_test_split

train_X, val_X, train_y, val_y = train_test_split(X,y,random_state = 0)

TestingModel = DecisionTreeRegressor(random_state = 1)

# then we will fit the training data to the Model
TestingModel.fit(train_X,train_y)

test_predictions = TestingModel.predict(val_X)

from sklearn.metrics import mean_absolute_error

print("The mean absolute error is : ", mean_absolute_error(val_y, test_predictions))