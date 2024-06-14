import pickle
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import pandas as pd

df = pd.read_csv("../data/train.csv")
df['TotalSquareFeet'] = (df['BsmtFinSF1'] + df['BsmtFinSF2'] +
                         df['1stFlrSF'] + df['2ndFlrSF'] + df['TotalBsmtSF'])

df['TotalBath'] = (df['FullBath'] + (0.5 * df['HalfBath']) +
                   df['BsmtFullBath'] + (0.5 * df['BsmtHalfBath']))

df['TotalPorchArea'] = (df['OpenPorchSF'] + df['3SsnPorch'] +
                        df['EnclosedPorch'] + df['ScreenPorch'] + df['WoodDeckSF'])

df['SqFtPerRoom'] = df['GrLivArea'] / \
    (df['TotRmsAbvGrd'] + df['FullBath'] + df['HalfBath'] + df['KitchenAbvGr'])

# Select only the columns that you need for the model
features = ['OverallQual', 'TotalSquareFeet', 'TotalBath',
            'GrLivArea', 'SqFtPerRoom', 'GarageCars', 'YearBuilt']
X = df[features]
y = df["SalePrice"]
X = pd.get_dummies(X, drop_first=True, dtype=int)
for col in X.columns:
    if pd.api.types.is_numeric_dtype(X[col].dtype):
        if X[col].isnull().any():
            X[col].fillna(X[col].median(), inplace=True)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=10)
y_test

trainer = LinearRegression()
trainer.fit(X_train, y_train)
print(trainer.score(X_test, y_test))

with open('model.pkl', 'wb') as file:
    pickle.dump(trainer, file)
