
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

print("\n РЕГРЕССИЯ ")

car = pd.read_csv("CarPrice_Assignment.csv")

features = [
    "horsepower",
    "citympg",
    "enginesize",
    "curbweight",
    "highwaympg",
    "wheelbase"
]

X = car[features]
y = car["price"]

print("Количество автомобилей:", len(car))
print("Используемые признаки:", features)

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

model = LinearRegression()
model.fit(X_train, y_train)

predictions = model.predict(X_test)

r2 = r2_score(y_test, predictions)
mae = mean_absolute_error(y_test, predictions)

print("\nРЕЗУЛЬТАТЫ РЕГРЕССИИ")
print("R2:", round(r2, 4))
print("MAE:", round(mae, 2))

horsepower_model = LinearRegression()

horsepower_model.fit(
    X_train[["horsepower"]],
    y_train
)

horsepower_predictions = horsepower_model.predict(
    X_train[["horsepower"]]
)

plt.figure(figsize=(9, 5))

plt.scatter(
    car["horsepower"],
    car["price"],
    alpha=0.7,
    label="Автомобили"
)

sort_index = np.argsort(X_train["horsepower"].values)

plt.plot(
    X_train["horsepower"].values[sort_index],
    horsepower_predictions[sort_index],
    color="red",
    label="Линия регрессии"
)

plt.title("Зависимость цены автомобиля от мощности двигателя")
plt.xlabel("Horsepower")
plt.ylabel("Price")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

print("\n КЛАССИФИКАЦИЯ")

adult = pd.read_csv("adult.csv")

adult.columns = adult.columns.str.strip()

for column in adult.columns:
    if adult[column].dtype == "object":
        adult[column] = adult[column].map(
            lambda value: value.strip() if isinstance(value, str) else value
        )

adult.replace("?", np.nan, inplace=True)

y = (adult["income"] == ">50K").astype(int)
X = adult.drop(columns=["income"])

numeric_features = X.select_dtypes(include=["number"]).columns
categorical_features = [
    column for column in X.columns
    if column not in numeric_features
]

print("Количество записей:", len(adult))
print("Числовых признаков:", len(numeric_features))
print("Категориальных признаков:", len(categorical_features))
print("Пропущенных значений:", adult.isna().sum().sum())

numeric_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer([
    ("num", numeric_transformer, numeric_features),
    ("cat", categorical_transformer, categorical_features)
])

model = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", LogisticRegression(max_iter=1000))
])

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

model.fit(X_train, y_train)

predictions = model.predict(X_test)

accuracy = accuracy_score(y_test, predictions)
precision = precision_score(y_test, predictions)
recall = recall_score(y_test, predictions)

print("\nРЕЗУЛЬТАТЫ КЛАССИФИКАЦИИ")
print("Accuracy:", round(accuracy, 4))
print("Precision:", round(precision, 4))
print("Recall:", round(recall, 4))

print("\nМАТРИЦА ОШИБОК")

cm = confusion_matrix(y_test, predictions)

print(cm)

ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["<=50K", ">50K"]
).plot(cmap="Blues")

plt.title("Матрица ошибок")
plt.tight_layout()
plt.show()

print("\nРабота программы завершена.")
