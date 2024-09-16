from sklearn.datasets import fetch_openml
import pandas as pd
import joblib

# Tải dataset từ OpenML
boston = fetch_openml(name='boston', version=1, as_frame=True)

# Tạo DataFrame
data = boston.frame

# Hiển thị vài dòng đầu tiên của dataset
print(data.head())

# Chọn RM (số phòng trung bình mỗi nhà) làm đại diện cho diện tích
X = data[['RM']]
y = data['MEDV']  # MEDV là biến mục tiêu (giá nhà)

# Chia tập dữ liệu thành tập huấn luyện và tập kiểm tra
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Xây dựng mô hình Hồi quy tuyến tính
from sklearn.linear_model import LinearRegression

# Khởi tạo và huấn luyện mô hình
model = LinearRegression()
model.fit(X_train, y_train)

# Dự đoán và đánh giá mô hình
from sklearn.metrics import mean_squared_error, r2_score

# Thực hiện dự đoán trên tập kiểm tra
y_pred = model.predict(X_test)
#hoặc chính là y_hat
# Đánh giá mô hình

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f'Mean Squared Error: {mse}')
print(f'R^2 Score: {r2}')

# Trực quan hóa mối quan hệ
import matplotlib.pyplot as plt

# Vẽ biểu đồ giữa số phòng trung bình và giá nhà
plt.scatter(X_test, y_test, color='blue', label='Giá thực tế')
plt.plot(X_test, y_pred, color='red', label='Giá dự đoán')
plt.xlabel('Số phòng trung bình (RM)')
plt.ylabel('Giá nhà (MEDV)')
plt.title('Mối quan hệ giữa Diện tích (RM) và Giá nhà')
plt.legend()
plt.show()



# Lưu mô hình vào file 'linear_regression_model.pkl'
import os
print("Thư mục làm việc hiện tại:", os.getcwd())
os.makedirs('models', exist_ok=True)
joblib.dump(model, 'models/linear_regression_model.pkl')


# # Tải mô hình từ file
# loaded_model = joblib.load('linear_regression_model.pkl')
#
# # Sử dụng mô hình đã tải để dự đoán
# y_loaded_pred = loaded_model.predict(X_test)
#
# # Kiểm tra độ chính xác
# print(f'Mean Squared Error (loaded model): {mean_squared_error(y_test, y_loaded_pred)}')
# print(f'R^2 Score (loaded model): {r2_score(y_test, y_loaded_pred)}')

# Tải mô hình từ file 'models/linear_regression_model.pkl'
# loaded_model = joblib.load('models/linear_regression_model.pkl')
# print("Mô hình đã được tải thành công từ file 'models/linear_regression_model.pkl'")
#
# # Sử dụng mô hình đã tải để dự đoán trên tập kiểm tra
# y_loaded_pred = loaded_model.predict(X_test)
#
# # Kiểm tra độ chính xác của mô hình đã tải
# from sklearn.metrics import mean_squared_error, r2_score
# mse_loaded = mean_squared_error(y_test, y_loaded_pred)
# r2_loaded = r2_score(y_test, y_loaded_pred)
#
# print(f'Mean Squared Error (loaded model): {mse_loaded}')
# print(f'R^2 Score (loaded model): {r2_loaded}')
