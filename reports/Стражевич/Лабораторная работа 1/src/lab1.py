import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

#0. ЗАГРУЗКА ДАННЫХ
columns = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness',
           'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome']

df = pd.read_csv('pima-indians-diabetes.csv', encoding='utf-8-sig',
                 skiprows=9, header=None, names=columns)

print("Размер датасета:", df.shape)
print(df.head())


#1. СТАТИСТИКА
print("\n--- ПУНКТ 1: Описательная статистика ---")
print(df.describe())

print("\nТипы данных:")
print(df.dtypes)

print("\nНули в ключевых столбцах (это пропуски):")
for col in ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']:
    print(f"  {col}: {(df[col] == 0).sum()}")


#2. ЗАМЕНА НУЛЕЙ МЕДИАНОЙ
print("\n--- ПУНКТ 2: Замена нулей медианой ---")
cols_to_fix = ['Glucose', 'BloodPressure', 'SkinThickness']

for col in cols_to_fix:
    med = df.loc[df[col] != 0, col].median()
    df.loc[df[col] == 0, col] = med
    print(f"  {col}: медиана = {med:.2f}")

print("Нулей после замены:")
for col in cols_to_fix:
    print(f"  {col}: {(df[col] == 0).sum()}")


#3. ГИСТОГРАММЫ BMI и AGE
print("\n--- ПУНКТ 3: Гистограммы ---")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].hist(df['BMI'], bins=25, color='skyblue', edgecolor='black')
axes[0].set_title('Распределение BMI')
axes[0].set_xlabel('BMI')
axes[0].set_ylabel('Частота')

axes[1].hist(df['Age'], bins=25, color='salmon', edgecolor='black')
axes[1].set_title('Распределение возраста')
axes[1].set_xlabel('Age')
axes[1].set_ylabel('Частота')

plt.tight_layout()
plt.savefig('hist_bmi_age.png', dpi=120, bbox_inches='tight')
plt.show()


#4. МАТРИЦА КОРРЕЛЯЦИИ
print("\n--- ПУНКТ 4: Матрица корреляции ---")

subset = df[['Glucose', 'BMI', 'Age', 'Outcome']]
corr = subset.corr()
print(corr.round(3))

fig, ax = plt.subplots(figsize=(7, 6))
im = ax.imshow(corr.values, cmap='coolwarm', vmin=-1, vmax=1)
ax.set_xticks(range(len(corr.columns)))
ax.set_yticks(range(len(corr.columns)))
ax.set_xticklabels(corr.columns, rotation=45, ha='right')
ax.set_yticklabels(corr.columns)

for i in range(len(corr.columns)):
    for j in range(len(corr.columns)):
        ax.text(j, i, f'{corr.values[i, j]:.2f}',
                ha='center', va='center', color='black', fontsize=11)

plt.colorbar(im, ax=ax)
ax.set_title('Матрица корреляции')
plt.tight_layout()
plt.savefig('corr_matrix.png', dpi=120, bbox_inches='tight')
plt.show()

target_corr = corr['Outcome'].drop('Outcome').sort_values(ascending=False)
print("\nСамый сильный предиктор:", target_corr.idxmax(),
      f"(r = {target_corr.max():.3f})")


#5. КРУГОВАЯ ДИАГРАММА OUTCOME
print("\n--- ПУНКТ 5: Распределение Outcome ---")

counts = df['Outcome'].value_counts().sort_index()
print("Без диабета (0):", counts[0])
print("С диабетом  (1):", counts[1])

fig, ax = plt.subplots(figsize=(7, 7))
ax.pie(counts, labels=['Без диабета (0)', 'С диабетом (1)'],
       autopct='%1.1f%%', colors=['#7fcdbb', '#e34a33'],
       startangle=90, shadow=True)
ax.set_title('Распределение пациенток по наличию диабета')
plt.tight_layout()
plt.savefig('pie_outcome.png', dpi=120, bbox_inches='tight')
plt.show()


#6. СТАНДАРТИЗАЦИЯ
print("\n--- ПУНКТ 6: Стандартизация ---")

features = df.drop('Outcome', axis=1)
target = df['Outcome']

scaler = StandardScaler()
features_scaled = pd.DataFrame(
    scaler.fit_transform(features),
    columns=features.columns
)

print("Средние после стандартизации (должны быть ≈ 0):")
print(features_scaled.mean().round(4))

df_scaled = features_scaled.copy()
df_scaled['Outcome'] = target.values
df_scaled.to_csv('diabetes_cleaned_scaled.csv', index=False)

print("\nГотово! Файлы сохранены рядом со скриптом.")