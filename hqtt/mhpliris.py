# Bước 1: Import các thư viện
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# Bước 2: Tải dữ liệu hoa diên vĩ
iris = load_iris()
X = iris.data  # Đặc trưng (feature)
y = iris.target  # Nhãn (class)

# Bước 3: Chia dữ liệu thành tập huấn luyện và kiểm tra
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=15)

# Bước 4: Tạo mô hình KNN với k = 3 (có thể thay đổi giá trị k)
knn = KNeighborsClassifier(n_neighbors=3)

# Bước 5: Huấn luyện mô hình trên tập huấn luyện
knn.fit(X_train, y_train)

# Bước 6: Dự đoán trên tập kiểm tra
y_pred = knn.predict(X_test)

# Bước 7: Đánh giá độ chính xác của mô hình
accuracy = accuracy_score(y_test, y_pred)
print(f"Độ chính xác của mô hình: {accuracy * 100:.2f}%")
import joblib

# Lưu mô hình
# model_filename = 'knn_model.pkl'
# joblib.dump(knn, model_filename)
#
import pickle
# Save model
with open('knn_k7iris_model.pkl', 'wb') as f:
    pickle.dump(knn, f)

# Load model
# with open('knn_model.pkl', 'rb') as f:
#     model = pickle.load(f)
# print(f"Mô hình đã được lưu vào file {model_filename}")
import sklearn
print(sklearn.__version__)


