# Bước 1: Import các thư viện
from sklearn.datasets import load_iris  # Tải dữ liệu Iris
from sklearn.model_selection import train_test_split  # Chia tập dữ liệu
from sklearn.neural_network import MLPClassifier  # Mô hình ANN
from sklearn.metrics import accuracy_score, confusion_matrix  # Đánh giá mô hình
import pickle  # Lưu và tải mô hình

# Bước 2: Tải dữ liệu Iris
D = load_iris()
X = D.data  # Đặc trưng (feature)
y = D.target  # Nhãn (class)

# Bước 3: Chia dữ liệu thành tập huấn luyện và kiểm tra
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Bước 4: Tạo mô hình ANN
model = MLPClassifier(hidden_layer_sizes=(5,15,50), max_iter=100)  # Mạng nơ-ron với 3 lớp ẩn

# Bước 5: Huấn luyện mô hình trên tập huấn luyện
model.fit(X_train, y_train)

# Bước 6: Dự đoán trên tập kiểm tra
y_pred = model.predict(X_test)

# Bước 7: Đánh giá độ chính xác của mô hình
accuracy = accuracy_score(y_test, y_pred)
print(f"Độ chính xác của mô hình ANN: {accuracy * 100:.2f}%")

# Bước 8: Tính toán và in ma trận nhầm lẫn
cm = confusion_matrix(y_test, y_pred)
print("\nMa trận nhầm lẫn:")
print(cm)

# Bước 9: Lưu mô hình vào file
with open('ann_iris_model.pkl', 'wb') as f:
    pickle.dump(model, f)
print("Mô hình ANN đã được lưu vào file 'ann_iris_model.pkl'")

# Bước 10: Tải mô hình từ file để sử dụng (tùy chọn)
# with open('ann_iris_model.pkl', 'rb') as f:
#     loaded_model = pickle.load(f)
#     print("Mô hình đã được tải thành công")
