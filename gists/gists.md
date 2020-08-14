## Split the Data


```python
import numpy as np 
import pandas as pd
```


```python
path = "./"
df_train = pd.read_csv(path+"", sep=",", header=1, encoding="utf-8")
df_eval = pd.read_csv(path+"", sep=",", header=1, encoding="utf-8")
```


```python
from sklearn.model_selection import train_test_split

# df = df_train.fillna(-1)

X = df.drop('COL_TO_PREDICT', axis=1) # Predictor -- can remove more than 1 column
y = df['COL_TO_PREDICT'] # Predicted

Xtrain, Xval, ytrain, yval = train_test_split(X,y, train_size=0.5, random_state=0)

Xtrain.shape, Xval.shape, ytrain.shape, yval.shape 
```

## Filter Selection

### Optimal number of features


```python
from sklearn.feature_selection import SelectKBest, SelectPercentile, f_regression


k_vs_score = [] 
initial_features = 2
max_n_features = len(X.columns)
step = 2


for k in range(initial_features, max_n_features, step):
  selector = SelectKBest(score_func=f_regression, k=k)

  Xtrain2 = selector.fit_transform(Xtrain, ytrain)
  Xval2 = selector.transform(Xval)

  model = RandomForestRegressor(n_estimators=100, n_jobs=-1, random_state=0)
  model.fit(Xtrain2, ytrain)

  p = model.predict(Xval2)

  score = mean_absolute_error(yval, p)
  print("k = {} - MAE = {}".format(k, score))

  k_vs_score.append(score)
```


```python
pd.Series(
    k_vs_score, 
    index = range(initial_features, max_n_features, step)).plot(figsize=(10,7), 
    xticks=range(initial_features, max_n_features, step)
);
```

### Send to selector


```python
best_k=4

selector = SelectKBest(score_func=f_regression, k=best_k)

selector.fit(X_train,y_train)

X_val.columns[selector.get_support()]
```


```python
pd.Series(selector.scores_, index=X_train.columns).dropna().sort_values(ascending=True).plot.barh(figsize=(10,6));
```

## Modeling


```python
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor


k_vs_score = []

for k in range(2, Xtrain.shape[1], 2):
  #selector_model = LinearRegression(normalize=True)
  #selector_model = Ridge(alpha=0.1, normalize=True)
  selector_model = Lasso(alpha=1.,normalize=True)
  #selector_model = RandomForestRegressor(random_state=1, n_jobs=-1)
  selector = SelectFromModel(selector_model, max_features=k, threshold=-np.inf)

  selector.fit(Xtrain, ytrain)

  #Xtrain2 = np.zeros((Xtrain.shape[0], 7))
  Xtrain2 = selector.transform(Xtrain)
  #Xtrain2[:, -1] = Xtrain['START YEAR'].values

  #Xval2 = np.zeros((Xval.shape[0], 7))
  Xval2 = selector.transform(Xval)
  #Xval2[:, -1] = Xval['START YEAR'].values

  #print(Xtrain.shape, Xtrain2.shape)

  #Xtrain.columns[selector.get_support()]

  model = RandomForestRegressor(criterion='mse', n_estimators=1000, random_state=0, n_jobs=-1, max_depth=9)
  #mdl = XGBRegressor(objective="reg:squarederror", seed=0)
  model.fit(Xtrain2, ytrain)

  p = model.predict(Xval2)

  score = mean_absolute_error(yval, p)
  print("k = {} - MAE = {}".format(k, score))

  mask = selector.get_support()
  print(Xtrain.columns[mask])

  k_vs_score.append(score)
  #break
```

## Lazy Model


```python
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import Lasso
from sklearn.ensemble import RandomForestRegressor


selector_model = Lasso(alpha=1.,normalize=True)
selector = SelectFromModel(selector_model, max_features=5, threshold=-np.inf)

selector.fit(Xtrain, ytrain)
Xtrain2 = selector.transform(Xtrain)

model = RandomForestRegressor(criterion='mse', n_estimators=1000, max_depth=9, n_jobs=-1, random_state=0)
model.fit(Xtrain2, ytrain)

Xval2 = selector.transform(Xval)
model.predict(Xval2)
```

## Evaluation

Evaluation is just possible if there is a separate data set for evaluation


```python
Xtest = df_eval[X.columns]
# Xtest.fillna(-1, inplace=True)

Xtest2 = selector.transform(Xtest)

ypred = model.predict(Xtest2)
```
