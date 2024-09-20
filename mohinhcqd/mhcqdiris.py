# Bước 1: Import các thư viện
from sklearn.datasets import load_iris  # Tải dữ liệu Iris
from sklearn.model_selection import train_test_split  # Chia tập dữ liệu
from sklearn.tree import DecisionTreeClassifier  # Cây quyết định
from sklearn.metrics import accuracy_score, classification_report
import pickle  # Lưu và tải mô hình

# Bước 2: Tải dữ liệu Iris
D = load_iris()
X = D.data  # Đặc trưng (feature)
y = D.target  # Nhãn (class)

# Bước 3: Chia dữ liệu thành tập huấn luyện và kiểm tra
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Bước 4: Tạo mô hình cây quyết định
model = DecisionTreeClassifier(random_state=15)

# Bước 5: Huấn luyện mô hình trên tập huấn luyện
model.fit(X_train, y_train)

# Bước 6: Dự đoán trên tập kiểm tra
y_pred = model.predict(X_test)

# Bước 7: Đánh giá độ chính xác của mô hình
accuracy = accuracy_score(y_test, y_pred)
print(f"Độ chính xác của mô hình cây quyết định: {accuracy * 100:.2f}%")

# Bước 8: Tạo báo cáo phân loại
report = classification_report(y_test, y_pred, target_names=D.target_names)
print("\nBáo cáo phân loại:")
print(report)

# Bước 8: Lưu mô hình vào file
with open('cqd_iris_model.pkl', 'wb') as f:
    pickle.dump(model, f)
print("Mô hình cây quyết định đã được lưu vào file 'decision_tree_iris_model.pkl'")

# Bước 9: Tải mô hình từ file để sử dụng
# with open('decision_tree_iris_model.pkl', 'rb') as f:
#     loaded_model = pickle.load(f)
#     print("Mô hình đã được tải thành công")
