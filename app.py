from flask import Flask, request, jsonify
import joblib
from flask_cors import CORS  #xử lý vấn đề CORS khi gọi API từ trình duyệt
import numpy as np

app = Flask(__name__)  # Khởi tạo ứng dụng Flask
CORS(app)  # Kích hoạt CORS cho toàn bộ ứng dụng, cho phép các domain khác truy cập API

# Tải mô hình hồi quy tuyến tính đã được huấn luyện trước đó
model = joblib.load('/root/models/linear_regression_model.pkl')


@app.route('/predict', methods=['POST'])  # Định nghĩa một endpoint cho việc dự đoán, chỉ nhận phương thức POST
def predict():
    # Lấy dữ liệu JSON từ yêu cầu gửi đến
    data = request.json

    # Kiểm tra nếu không có dữ liệu gửi lên
    if not data:
        return jsonify({'error': 'No input data provided'}), 400  # Trả về lỗi nếu dữ liệu không có

    # Các biến cần thiết cho mô hình từ dataset Boston House Price
    try:
        # Lấy các giá trị của các biến từ dữ liệu JSON, nếu không có sẽ gán giá trị mặc định là 0
        CRIM = data.get('CRIM', 0)
        ZN = data.get('ZN', 0)
        INDUS = data.get('INDUS', 0)
        CHAS = data.get('CHAS', 0)
        NOX = data.get('NOX', 0)
        RM = data.get('RM', 0)
        AGE = data.get('AGE', 0)
        DIS = data.get('DIS', 0)
        RAD = data.get('RAD', 0)
        TAX = data.get('TAX', 0)
        PTRATIO = data.get('PTRATIO', 0)
        B = data.get('B', 0)
        LSTAT = data.get('LSTAT', 0)

        # Tạo một mảng numpy chứa các giá trị đầu vào theo thứ tự các biến đã được mô hình huấn luyện
        input_data = np.array([[CRIM, ZN, INDUS, CHAS, NOX, RM, AGE, DIS, RAD, TAX, PTRATIO, B, LSTAT]])

        # Dự đoán giá nhà dựa trên dữ liệu đầu vào và lấy giá trị dự đoán đầu tiên (vì chỉ dự đoán 1 giá trị)
        prediction = model.predict(input_data)[0]

        # Trả về kết quả dự đoán dưới dạng JSON
        return jsonify({'prediction': prediction})

    except Exception as e:  # Bắt lỗi nếu có bất kỳ lỗi nào xảy ra
        return jsonify({'error': str(e)}), 500  # Trả về lỗi cùng thông điệp lỗi

# Khởi chạy ứng dụng Flask trên tất cả các địa chỉ IP (0.0.0.0) và sử dụng cổng 5000
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)





























# from flask import Flask, request, jsonify
# import joblib
# from flask_cors import CORS  # Thêm import này
#
# app = Flask(__name__)
# CORS(app)  # Kích hoạt CORS cho toàn bộ ứng dụng
#
# # Tải mô hình hồi quy tuyến tính
# model = joblib.load('/home/ftpuser/models/linear_regression_model.pkl')
#
#
# @app.route('/predict', methods=['POST'])
# def predict():
#     data = request.json
#     rm = data.get('RM')
#
#     if rm is None:
#         return jsonify({'error': 'Missing RM value'}), 400
#
#     # Dự đoán giá nhà
#     prediction = model.predict([[rm]])[0]
#
#     return jsonify({'prediction': prediction})
#
#
# if __name__ == '__main__':
#     app.run(host='0.0.0.0', port=5000)
