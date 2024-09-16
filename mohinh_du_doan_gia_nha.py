import pandas as pd
import numpy as np

#Buoc 1: Doc du lieu Boston house price dataset vao bo nho
data_url = "http://lib.stat.cmu.edu/datasets/boston"
raw_df = pd.read_csv(data_url, sep="\s+", skiprows=22, header=None)
X = np.hstack([raw_df.values[::2, :], raw_df.values[1::2, :2]])
y = raw_df.values[1::2, 2]
print("Kich thuoc tap thuoc tinh X = ", X.shape)
print("Kich thuoc tap gia tri dich y = ", y.shape)
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
# #Buoc 2: Huan luyen mo hinh HQTT voi tap du lieu tren
# from sklearn.linear_model import LinearRegression
# #Khởi tạo mô hình
# model = LinearRegression()
# #huấn luyện mô hình
# model.fit(X, y)
#Buoc 3: Lưu mô hình vào file
import joblib
# Lưu mô hình vào file
import os
print("Thư mục làm việc hiện tại:", os.getcwd())
os.makedirs('models', exist_ok=True)
joblib.dump(model, 'models/linear_regression_model.pkl')
print("Luu mo hinh thanh cong...")