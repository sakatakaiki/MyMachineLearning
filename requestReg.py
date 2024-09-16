import requests

# URL của API Flask
url = 'http://192.168.1.100:5000/predict'
# Dữ liệu đầu vào, gồm các biến từ dataset Boston House Price
data = {
    'CRIM': 0.1,    # Tỉ lệ tội phạm
    'ZN': 25.0,     # Tỉ lệ đất cho khu dân cư
    'INDUS': 5.0,   # Tỉ lệ đất công nghiệp
    'CHAS': 0,      # Sát sông (1 nếu có, 0 nếu không)
    'NOX': 0.5,     # Nồng độ oxit nitơ (NOX)
    'RM': 8.0,      # Số phòng trung bình
    'AGE': 30.0,    # Tỉ lệ nhà xây dựng trước năm 1940
    'DIS': 5.0,     # Khoảng cách đến trung tâm việc làm
    'RAD': 1,       # Chỉ số tiếp cận đường cao tốc
    'TAX': 300.0,   # Thuế tài sản
    'PTRATIO': 15.0,# Tỉ lệ học sinh-giáo viên
    'B': 380.0,     # Tỉ lệ người da đen
    'LSTAT': 4.0    # Tỉ lệ người nghèo
}

# Gửi yêu cầu POST tới API Flask và nhận phản hồi
response = requests.post(url, json=data)

# In kết quả dự đoán
print(response.json())





































# url = 'http://192.168.1.100:5000/predict'
# data = {'RM': 8.0}
# response = requests.post(url, json=data)
# print(response.json())
