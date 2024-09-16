# Bước 1: Import các thư viện
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
import pickle

# Bước 2: Tải dữ liệu hoa diên vĩ
D = load_iris()
X = D.data  # Đặc trưng (feature)
y = D.target  # Nhãn (class)

# Bước 3: Chia dữ liệu thành tập huấn luyện và kiểm tra
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=105)
print("Kích thước tập dl gốc: ")
print("X:", X.shape, "y: ", y.shape)
print("Kích thước tập dl huấn luyện: ")
print("X_train:", X_train.shape, "y_train: ", y_train.shape)
print("Kích thước tập dl kiểm thứ: ")
print("X_test:", X_test.shape, "y_test: ", y_test.shape)

# Bước 4: Tạo mô hình Naive Bayes
model = GaussianNB()

# Bước 5: Huấn luyện mô hình trên tập huấn luyện
model.fit(X_train, y_train)

# Bước 6: Dự đoán trên tập kiểm tra
y_hat = model.predict(X_test)

# Bước 7: Đánh giá độ chính xác của mô hình
accuracy = accuracy_score(y_test, y_hat)
print(f"Độ chính xác của mô hình Naive Bayes: {accuracy * 100:.2f}%")

# Lưu mô hình
with open('nb_iris_model.pkl', 'wb') as f:
    pickle.dump(model, f)

# In ra phiên bản của sklearn
import sklearn
print(sklearn.__version__)
