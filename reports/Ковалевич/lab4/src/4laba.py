import torch
from torch import nn
from torch import optim

import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, f1_score

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Платформа обучения: ", device)


class MLP(nn.Module):
    def __init__(self, in_features: int, activation: str = 'relu') -> None:
        super().__init__()
        act = nn.ReLU() if activation == 'relu' else nn.Tanh()
        self.linear_stack = nn.Sequential(
            nn.Linear(in_features=in_features, out_features=8),  # скрытый слой, 8 нейронов
            act,                                                 # ReLU или Tanh
            nn.Linear(in_features=8, out_features=1),            # выход: 1 нейрон (логит)
        )

    def forward(self, x):
        return self.linear_stack(x)


def train_and_evaluate(activation: str, data, epochs: int = 200) -> float:
    X_train_t, X_test_t, y_train_t, y_test_t = data

    torch.manual_seed(42)  # одинаковая инициализация для сравнения
    model = MLP(in_features=X_train_t.shape[1], activation=activation).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    print("=" * 45)
    print(f"\nАктивация скрытого слоя: {activation}")
    print("\nОбучаемая модель: \n", model)
    print("\tКоличество эпох: ", epochs)
    print("\tОптимизатор(Adam): ")
    for name, val in optimizer.defaults.items():
        print("\t\t", name, " : ", val)
    print("\tФункция потерь(бинарная): ", criterion)
    print("=" * 45)

    # Цикл обучения
    for epoch in range(epochs):
        model.train()
        y_pred = model(X_train_t)
        loss = criterion(y_pred, y_train_t)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 50 == 0:
            print(f"Эпоха {epoch + 1}/{epochs}, loss = {loss.item():.4f}")

    print("\nОценка модели нейронной сети: ")
    model.eval()
    with torch.no_grad():
        logits = model(X_test_t)
        predicted_classes = (logits > 0).float()

    y_true = y_test_t.cpu().numpy().ravel()
    y_hat = predicted_classes.cpu().numpy().ravel()

    print(classification_report(y_true, y_hat, target_names=['съедобный (e)', 'ядовитый (p)']))
    print("f1-score =", f1_score(y_true, y_hat))
    return accuracy_score(y_true, y_hat)


def solution() -> None:
    # 1. загрузка данных и кодирование признаков
    df = pd.read_csv('mushrooms.csv', sep=',')
    y = df['class'].map({'e': 0, 'p': 1})  # ядовитый = 1
    X = pd.get_dummies(df.drop(columns='class'), drop_first=True).astype(float)

    # 2. разделение выборки
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, train_size=0.7, random_state=42, stratify=y)

    # 3. стандартизация
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 4. тензоры
    X_train_t = torch.tensor(X_train_scaled, dtype=torch.float32).to(device)
    X_test_t = torch.tensor(X_test_scaled, dtype=torch.float32).to(device)
    y_train_t = torch.tensor(y_train.values, dtype=torch.float32).view(-1, 1).to(device)
    y_test_t = torch.tensor(y_test.values, dtype=torch.float32).view(-1, 1).to(device)
    data = (X_train_t, X_test_t, y_train_t, y_test_t)

    # 5. эксперимент: ReLU vs Tanh
    results = {}
    for activation in ['relu', 'tanh']:
        results[activation] = train_and_evaluate(activation, data)

    print("\n" + "=" * 45)
    print("Сравнение итоговой точности (accuracy):")
    for activation, acc in results.items():
        print(f"\t{activation:5s}: {acc:.4f}")


if __name__ == "__main__":
    solution()