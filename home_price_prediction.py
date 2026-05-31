import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
from xgboost import XGBRegressor

# Sample dataset
data = {
    "income": [50000,60000,70000,80000,90000,100000,110000,120000],
    "schools": [2,3,4,4,5,5,6,6],
    "hospitals": [1,2,2,3,3,4,4,5],
    "crime_rate": [8,7,6,5,4,3,2,1],
    "price": [150000,180000,220000,260000,300000,350000,400000,450000]
}

df = pd.DataFrame(data)

X = df.drop("price", axis=1)
y = df["price"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = XGBRegressor(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=3,
    random_state=42
)

model.fit(X_train, y_train)

predictions = model.predict(X_test)

print("Predicted Prices:", predictions)
print("MAE:", mean_absolute_error(y_test, predictions))
print("R2 Score:", r2_score(y_test, predictions))