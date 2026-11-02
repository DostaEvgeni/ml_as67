import pandas as pd

from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC

from sklearn.model_selection import train_test_split
from sklearn.metrics import (recall_score, precision_score, accuracy_score,
                             classification_report)
from sklearn.model_selection import cross_val_score

# 1. загрузка данных и One-Hot Encoding
df = pd.read_csv('mushrooms.csv', sep=',')

X = df.drop('class', axis='columns')
y = df['class']

X = pd.get_dummies(X, drop_first=True)

# 2. разделение на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(
    X, y, train_size=0.7, random_state=42, stratify=y)

# 3. обучение моделей
knn_model = KNeighborsClassifier(n_neighbors=5)
knn_model.fit(X_train, y_train)
knn_y_pred = knn_model.predict(X_test)

dec_tree_model = DecisionTreeClassifier(random_state=42)
dec_tree_model.fit(X_train, y_train)
dec_tree_y_pred = dec_tree_model.predict(X_test)

svm_model = SVC(random_state=42)
svm_model.fit(X_train, y_train)
svm_y_pred = svm_model.predict(X_test)

# 4. precision и recall для класса "ядовитый" (p)
print('precision k-NN =', precision_score(y_test, knn_y_pred, pos_label='p'))
print('recall k-NN =', recall_score(y_test, knn_y_pred, pos_label='p'))

print('precision Decision Tree =', precision_score(y_test, dec_tree_y_pred, pos_label='p'))
print('recall Decision Tree =', recall_score(y_test, dec_tree_y_pred, pos_label='p'))

print('precision SVM =', precision_score(y_test, svm_y_pred, pos_label='p'))
print('recall SVM =', recall_score(y_test, svm_y_pred, pos_label='p'))


print("\nОценка модели kNN:\n", classification_report(y_test, knn_y_pred))
print("Оценка модели DT:\n", classification_report(y_test, dec_tree_y_pred))
print("Оценка модели SVC:\n", classification_report(y_test, svm_y_pred))


print('accuracy k-NN =', accuracy_score(y_test, knn_y_pred))
print('accuracy Decision Tree =', accuracy_score(y_test, dec_tree_y_pred))
print('accuracy SVM =', accuracy_score(y_test, svm_y_pred))

# исследование k-NN при разных k
ks = range(1, 21)

for k in ks:
    model = KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(model, X_train, y_train, cv=5)
    print(f"k = {k:2}: средняя точность = {scores.mean():.3f}")
