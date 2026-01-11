import pandas as pd
from catboost import CatBoostRegressor
from sklearn.metrics import mean_absolute_percentage_error

train_data = pd.read_csv('data/train.csv')
test_data = pd.read_csv('data/test.csv')


X_train = train_data.drop(columns=['amount'])
y_train = train_data['amount']
X_test = test_data.drop(columns=['amount'])
y_test = test_data['amount']



model = CatBoostRegressor(cat_features=['store_type', 'assortment'])
model.fit(X_train, y_train)


score = mean_absolute_percentage_error(y_test, model.predict(X_test))
print(f'Error on test dataset: {100 * score:.1f}%')


naming = 'retail_model.cbm'
model.save_model(naming)
print(f'Model saved as {naming}')