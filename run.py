import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
# Load model
model = load_model('D:\ExpertGroot\smol_test_withdata\run\model.h5')  # hoặc 'VGG16.h5', 'EfficientNetB3.h5'

# Load class labels
with open('models/models/big models/list of the classes.txt', 'r', encoding='utf-8') as f:
    class_names = [line.strip() for line in f.readlines()]

# Hàm dự đoán 1 ảnh
def predict_image(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    predictions = model.predict(img_array)
    predicted_index = np.argmax(predictions)
    predicted_class = class_names[predicted_index]
    confidence = np.max(predictions)

    print(f"Image: {img_path}")
    print(f"Predicted class: {predicted_class} ({confidence*100:.2f}%)")

# Ví dụ dự đoán một ảnh
predict_image('D:\ExpertGroot\smol_test_withdata\test\Pepper bell Bacterial spot\pepper_bell_bacterial_spot59.jpg')
