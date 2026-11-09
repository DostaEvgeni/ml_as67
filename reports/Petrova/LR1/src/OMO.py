import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv('auto-mpg.csv')

df['horsepower'] = pd.to_numeric(df['horsepower'], errors='coerce')
df['horsepower'] = df['horsepower'].fillna(df['horsepower'].mean())

df['age'] = 1983 - (1900 + df['model year'])

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)

print("1. Первые 5 строк датасета:")
print(df.head())

print("\n2. Информация о типах столбцов (.info()):")
df.info()

fig, ax = plt.subplots(figsize=(8, 5))
ax.scatter(df['weight'], df['mpg'], color='#e74c3c', alpha=0.7, edgecolors='k')
ax.set_title('Зависимость расхода топлива (mpg) от веса (weight)')
ax.set_xlabel('Вес (weight)')
ax.set_ylabel('Расход топлива (mpg)')
ax.grid(True, linestyle='--', alpha=0.5)
plt.show()

df_encoded = pd.get_dummies(df, columns=['origin'], dtype=int)

cyl_counts = df['cylinders'].value_counts().sort_index()
fig, ax = plt.subplots(figsize=(8, 5))
ax.bar(cyl_counts.index.astype(str), cyl_counts.values, color='#3498db', edgecolor='k')
ax.set_title('Распределение количества цилиндров (cylinders)')
ax.set_xlabel('Количество цилиндров')
ax.set_ylabel('Количество автомобилей')
ax.grid(axis='y', linestyle='--', alpha=0.5)
plt.show()