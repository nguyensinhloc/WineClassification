import matplotlib.pyplot as plt
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
        self.cost_history = []  # Thêm danh sách lưu trữ cost

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
            self.cost_history.append(cost)  # Lưu giá trị cost

            # Compute gradients
            dw = 1 / m * np.dot(X.T, (A - Y))
            db = 1 / m * np.sum(A - Y)

            # Update parameters
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

            # In thông tin chi tiết
            if i % 100 == 0:
                print(f"Lần lặp {i}: Loss = {cost}")
                print(f"Trọng số hiện tại: {self.weights.flatten()[:5]}...")

    def predict(self, X):
        Z = np.dot(X, self.weights) + self.bias
        Y_pred = self.sigmoid(Z)
        return (Y_pred > 0.5).astype(int)

    def fit(self, X, Y):
        dim = X.shape[1]
        self.initialize_parameters(dim)
        self.gradient_descent(X, Y)


def plot_loss_curve(cost_history):
    plt.figure(figsize=(10, 5))
    plt.plot(cost_history)
    plt.title('Loss Function qua các Iteration')
    plt.xlabel('Iteration')
    plt.ylabel('Cost')
    plt.show()


def plot_classification_result(y_true, y_pred):
    plt.figure(figsize=(8, 6))
    cm = confusion_matrix(y_true, y_pred)
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Ma Trận Nhầm Lẫn')
    plt.colorbar()
    plt.xlabel('Nhãn Dự Đoán')
    plt.ylabel('Nhãn Thực Tế')
    plt.show()


def main():
    # Load Wine dataset
    wine = load_wine()
    X = wine.data
    y = wine.target

    # In thông tin dữ liệu ban đầu
    print("Dữ liệu ban đầu:")
    print(f"- Số lượng mẫu: {X.shape[0]}")
    print(f"- Số chiều đặc trưng: {X.shape[1]}")
    print(f"- Phân phối nhãn lớp: {np.bincount(y)}")

    # Chuyển đổi thành bài toán binary classification
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
    print("\nKết quả đánh giá mô hình:")
    print(f"- Độ chính xác: {accuracy_score(y_test, y_pred)}")
    print("\n- Ma trận nhầm lẫn:")
    print(confusion_matrix(y_test, y_pred))
    print("\n- Báo cáo phân loại:")
    print(classification_report(y_test, y_pred))

    # Vẽ biểu đồ
    plot_loss_curve(lr.cost_history)
    plot_classification_result(y_test, y_pred)


if __name__ == "__main__":
    main()
