import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, root_mean_squared_error

import matplotlib.pyplot as plt

def regression_solution() -> None:
    FEATURES : list[str] = ['Length1', 'Length2', 'Length3', 'Height', 'Width']

    TARGET_FEATURE : str = 'Weight'
    fish_dataset : pd.DataFrame = pd.read_csv('Fish.csv', sep=',')

    # Обучение модели линейной регрессии
    X_train, X_test, y_train, y_test = train_test_split(
        fish_dataset[FEATURES],
        fish_dataset[TARGET_FEATURE],
        test_size=0.2,
        random_state=42
    )
    model = LinearRegression()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    # Оценка модели
    r2 = r2_score(y_test, y_pred)
    rmse = root_mean_squared_error(y_test, y_pred)
    print(f"R^2: {r2:.3f}")
    print(f"RMSE: {rmse:.3f}")
    
    # Визуализация данных
    fig, ax = plt.subplots()
    plt.scatter(x=fish_dataset[FEATURES[2]],
                y=fish_dataset[TARGET_FEATURE],
                alpha=0.4,
                c='green'
                )
    # Добавление линии регрессии
    z : np.ndarray = np.polyfit(
                fish_dataset[FEATURES[2]], 
                fish_dataset[TARGET_FEATURE], 
                1
                )
    p = np.poly1d(z)
    plt.plot(fish_dataset[FEATURES[2]], 
             p(fish_dataset[FEATURES[2]]), 
             color='red', 
             label='Линия регрессии')
    plt.title(f"Диаграмма рассеивания '{FEATURES[2]}' и '{TARGET_FEATURE}' с линией регрессии")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    regression_solution()