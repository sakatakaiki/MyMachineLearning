from flask import Flask, request, jsonify
import pickle  # Use pickle instead of joblib
import numpy as np
from flask_cors import CORS  # Xử lý CORS cho API

app = Flask(__name__)
CORS(app)  # Kích hoạt CORS

# Tải mô hình K-Neighbors đã được huấn luyện với pickle
with open('/root/apps/models/nb_iris_model.pkl', 'rb') as f:
    model = pickle.load(f)


@app.route('/predict', methods=['POST'])
def predict():
    data = request.json

    if not data:
        return jsonify({'error': 'No input data provided'}), 400

    try:
        # Lấy dữ liệu đầu vào cho các đặc trưng của tập Iris
        sepal_length = data.get('sepal_length', 0)
        sepal_width = data.get('sepal_width', 0)
        petal_length = data.get('petal_length', 0)
        petal_width = data.get('petal_width', 0)

        # Tạo mảng numpy từ dữ liệu đầu vào
        input_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])

        # Dự đoán nhãn (class) cho dữ liệu đầu vào
        prediction = model.predict(input_data)[0]

        # Tạo các nhãn class cho dự đoán
        target_names = ['setosa', 'versicolor', 'virginica']
        predicted_class = target_names[prediction]

        return jsonify({'predicted_class': predicted_class})

    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
