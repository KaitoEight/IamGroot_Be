import numpy as np
from PIL import Image
import tensorflow as tf
import json
from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
import os
from flask_cors import CORS

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": ["http://localhost:5500", "https://*.github.io","127.0.0.1:3000"]}})  # Cho phép localhost và GitHub Pages

# Initialize variables
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'jpg', 'jpeg', 'png'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Ensure upload folder exists
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Load disease database
try:
    with open("../../GUI Application/dis.json", encoding="utf-8") as DIS_json:
        dis_DB_cure = json.load(DIS_json)
except FileNotFoundError:
    app.logger.error("Không tìm thấy dis.json tại ../../GUI Application/dis.json")
    raise
except json.JSONDecodeError:
    app.logger.error("dis.json có định dạng JSON không hợp lệ")
    raise

# Load TFLite model
try:
    model_path_tflite = "../../GUI Application/tf_lite_Optimize_DEFAULT_model.tflite"
    interpreter = tf.lite.Interpreter(model_path=model_path_tflite)
    interpreter.allocate_tensors()
except Exception as e:
    app.logger.error(f"Lỗi khi tải mô hình TFLite: {str(e)}")
    raise

# Class labels
class_labels = [
    'Pepper bell Bacterial spot',
    'Pepper bell healthy',
    'Potato Early blight',
    'Potato Late blight',
    'Potato healthy',
    'Tomato Early blight',
    'Tomato Late blight',
    'Tomato Leaf Mold',
    'Tomato Septoria leaf spot',
    'Tomato Spider mites Two spotted spider mite',
    'Tomato Target Spot',
    'Tomato Tomato mosaic virus',
    'Tomato healthy',
    'potato hollow heart'
]

# Kiểm tra định dạng tệp
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Load và tiền xử lý ảnh
def load_and_preprocess_image(image_path, target_size=(600, 600)):
    try:
        img = Image.open(image_path)
        img = img.resize(target_size)
        img_array = np.array(img, dtype=np.float32) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        return img_array
    except Exception as e:
        app.logger.error(f"Lỗi khi xử lý ảnh {image_path}: {str(e)}")
        raise

# Hàm phân loại ảnh bằng mô hình TFLite
def classify_image_tflite(interpreter, image_path, class_labels):
    try:
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()

        img_array = load_and_preprocess_image(image_path, target_size=(input_details[0]['shape'][1], input_details[0]['shape'][2]))
        interpreter.set_tensor(input_details[0]['index'], img_array)
        interpreter.invoke()
        predictions = interpreter.get_tensor(output_details[0]['index'])

        predicted_class_index = np.argmax(predictions)
        predicted_class = class_labels[predicted_class_index]
        confidence = predictions[0][predicted_class_index]

        return predicted_class, confidence
    except Exception as e:
        app.logger.error(f"Lỗi khi phân loại ảnh {image_path}: {str(e)}")
        raise

# Endpoint cho phân loại ảnh
@app.route('/classify', methods=['POST'])
def classify_image():
    if 'file' not in request.files:
        app.logger.warning("Yêu cầu không chứa tệp")
        return jsonify({'error': 'No file part in the request'}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        app.logger.warning("Không có tệp được chọn")
        return jsonify({'error': 'No selected file'}), 400
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        try:
            file.save(file_path)
            predicted_class, confidence = classify_image_tflite(interpreter, file_path, class_labels)
            disease_name = predicted_class.lower()
            cure_info = "Leaf is healthy. No cure information available." if "healthy" in disease_name else dis_DB_cure.get(disease_name, "No cure information available.")

            # Xóa tệp sau khi xử lý
            os.remove(file_path)

            app.logger.info(f"Phân loại thành công: {predicted_class}, confidence: {confidence}")
            return jsonify({
                'predicted_class': predicted_class,
                'confidence': float(confidence * 100),  # Chuyển thành phần trăm
                'cure_info': cure_info
            }), 200
        
        except Exception as e:
            # Xóa tệp nếu có lỗi
            if os.path.exists(file_path):
                os.remove(file_path)
            app.logger.error(f"Lỗi khi phân loại: {str(e)}")
            return jsonify({'error': str(e)}), 500
    
    app.logger.warning(f"Định dạng tệp không hợp lệ: {file.filename}")
    return jsonify({'error': 'Invalid file format. Allowed formats: jpg, jpeg, png'}), 400

# Endpoint kiểm tra trạng thái
@app.route('/health', methods=['GET'])
def health_check():
    app.logger.info("Yêu cầu kiểm tra trạng thái thành công")
    return jsonify({'status': 'healthy'}), 200

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)