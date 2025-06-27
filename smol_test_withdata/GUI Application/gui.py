import numpy as np
from PIL import Image, ImageTk
import tensorflow as tf
import json
import tkinter as tk
from tkinter import filedialog

# Initialize variables
disease_name = ""
DIS_json = open("dis.json", encoding="utf-8")
dis_DB_cure = json.loads(DIS_json.read())

# Function to load and preprocess an image
def load_and_preprocess_image(image_path, target_size=(600, 600)):
    img = Image.open(image_path)
    img = img.resize(target_size)
    img_array = np.array(img, dtype=np.float32) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# Function to classify an image using a TFLite model
def classify_image_tflite(interpreter, image_path, class_labels):
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

# Function to update the GUI with classification results
def update_gui(image_path, true_label, predicted_class, confidence, cure_info):
    for widget in root.winfo_children():
        if widget != button:
            widget.destroy()

    img = Image.open(image_path)
    img = img.resize((300, 300), Image.ANTIALIAS)
    img = ImageTk.PhotoImage(img)
    img_label = tk.Label(root, image=img)
    img_label.image = img
    img_label.pack()

    result_label = tk.Label(root, text=f"Predicted Class: {predicted_class}", font=("Helvetica", 16, "bold"))
    result_label.pack()

    true_label_label = tk.Label(root, text=f"True Label: {true_label}", font=("Helvetica", 12))
    true_label_label.pack()

    confidence_label = tk.Label(root, text=f"Confidence: {confidence * 100:.2f}%", font=("Helvetica", 12))
    confidence_label.pack()

    # Format the cure text to fit the GUI
    word_list = cure_info.split()
    new_text = ""
    word_count = 0
    for word in word_list:
        if word_count % 20 == 0 and word_count != 0:
            new_text += "\n"
        new_text += word + " "
        word_count += 1

    cure_info = new_text

    cure_label = tk.Label(root, text=f"Cure Information: {cure_info}", font=("Arial", 16))
    cure_label.pack()

# Function to handle image classification and GUI update
def classify_and_update(image_path, interpreter, class_labels, dis_DB_cure):
    true_label = image_path.split("/")[-2]  # Extracting the true label from the image path
    predicted_class, confidence = classify_image_tflite(interpreter, image_path, class_labels)
    disease_name = predicted_class.lower()
    cure_info = "Leaf is healthy. No cure information available." if "healthy" in disease_name else dis_DB_cure.get(disease_name, "No cure information available.")
    update_gui(image_path, true_label, predicted_class, confidence, cure_info)

# Function to select an image using file dialog
def select_image():
    path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg;*.png;*.jpeg")])
    classify_and_update(path, interpreter, class_labels, dis_DB_cure)

# Main script execution
if __name__ == "__main__":
    model_path_tflite = "tf_lite_Optimize_DEFAULT_model.tflite"
    interpreter = tf.lite.Interpreter(model_path=model_path_tflite)
    interpreter.allocate_tensors()

    class_labels = ['Pepper bell Bacterial spot',
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
                    'potato hollow heart']

    root = tk.Tk()
    root.title("Leaf Disease Classifier")
    root.geometry("800x600")

    button = tk.Button(root, text="Select Image", height=3, width=20, font=("Helvetica", 12, "bold"), bg="blue", fg="white", command=select_image)
    button.pack(pady=20)

    root.mainloop()
