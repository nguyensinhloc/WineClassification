import numpy as np
from sklearn.datasets import load_wine
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


class LogisticRegression:
    def __init__(self, learning_rate=0.01, num_iterations=1000):
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.weights = None
        self.bias = None

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def initialize_parameters(self, dim):
        self.weights = np.zeros((dim, 1))
        self.bias = 0

    def compute_cost(self, A, Y):
        m = Y.shape[0]
        cost = -1 / m * np.sum(Y * np.log(A) + (1 - Y) * np.log(1 - A))
        return cost

    def gradient_descent(self, X, Y):
        m, dim = X.shape

        for i in range(self.num_iterations):
            # Forward propagation
            Z = np.dot(X, self.weights) + self.bias
            A = self.sigmoid(Z)

            # Compute cost
            cost = self.compute_cost(A, Y)

            # Compute gradients
            dw = 1 / m * np.dot(X.T, (A - Y))
            db = 1 / m * np.sum(A - Y)

            # Update parameters
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

            # Print cost every 100 iterations
            if i % 100 == 0:
                print(f"Cost after iteration {i}: {cost}")

    def predict(self, X):
        Z = np.dot(X, self.weights) + self.bias
        Y_pred = self.sigmoid(Z)
        return (Y_pred > 0.5).astype(int)

    def fit(self, X, Y):
        dim = X.shape[1]
        self.initialize_parameters(dim)
        self.gradient_descent(X, Y)


def main():
    # Load Wine dataset
    wine = load_wine()
    X = wine.data
    y = wine.target

    # Chuyển đổi thành bài toán binary classification
    # Ví dụ: phân loại giữa class 0 và các class khác
    y_binary = (y == 0).astype(int)

    # Chia dữ liệu train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_binary, test_size=0.2, random_state=42
    )

    # Chuẩn hóa dữ liệu
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Reshape labels
    y_train = y_train.reshape(-1, 1)
    y_test = y_test.reshape(-1, 1)

    # Huấn luyện mô hình Logistic Regression
    lr = LogisticRegression(learning_rate=0.001, num_iterations=1000)
    lr.fit(X_train, y_train)

    # Dự đoán
    y_pred = lr.predict(X_test)

    # Đánh giá mô hình
    print("\nModel Evaluation:")
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))


if __name__ == "__main__":
    main()
