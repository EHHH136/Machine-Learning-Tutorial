import pandas as pd


location = "C:/Users/Asus/Desktop/Python/Code/username.csv"

tab = pd.read_csv(location)

y = tab.Identifier
features = ['PredictiveData']

X = tab[features]

from sklearn.tree import DecisionTreeRegressor

idModel = DecisionTreeRegressor(random_state = 1)

idModel.fit(X, y)

print(idModel.predict(X))