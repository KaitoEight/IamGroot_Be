# Import necessary libraries
import numpy as np
from PIL import Image
import tensorflow as tf
import json
from tkinter.filedialog import askopenfilename

# Initialize variables
disease_name = ""
DIS_json = open("dis.json", encoding="utf-8")
dis_DB_cure = json.loads(DIS_json.read())


# Function to load and preprocess an image
def load_and_preprocess_image(image_path, target_size=(224, 224)):
    # Open and resize the image
    img = Image.open(image_path)
    img = img.resize(target_size)

    # Convert the image to a NumPy array and normalize pixel values
    img_array = np.array(img, dtype=np.float32) / 255.0

    # Add an extra dimension to the array
    img_array = np.expand_dims(img_array, axis=0)

    return img_array


# Function to classify an image using a TFLite model
def classify_image_tflite(interpreter, image_path, class_labels):
    # Get input and output details from the TFLite interpreter
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Load and preprocess the image
    img_array = load_and_preprocess_image(image_path,
                                          target_size=(input_details[0]['shape'][1], input_details[0]['shape'][2]))

    # Set the input tensor for the interpreter
    interpreter.set_tensor(input_details[0]['index'], img_array)

    # Run inference
    interpreter.invoke()

    # Get the predictions from the output tensor
    predictions = interpreter.get_tensor(output_details[0]['index'])

    # Get the index of the predicted class
    predicted_class_index = np.argmax(predictions)

    # Get the predicted class label and confidence
    predicted_class = class_labels[predicted_class_index]
    confidence = predictions[0][predicted_class_index]

    return predicted_class, confidence


# Main script execution
if __name__ == "__main__":
    # Load the TFLite model
    model_path_tflite = "tf_lite_Optimize_DEFAULT_model.tflite"
    interpreter = tf.lite.Interpreter(model_path=model_path_tflite)
    interpreter.allocate_tensors()

    # Define class labels for the model
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
                    'potato hollow heart']  # Replace with actual class labels

    # Prompt user to select an image file
    image_path = str(askopenfilename())
    print(image_path)

    # Extract the true label from the image path
    path_list = image_path.split("/")
    path_list.reverse()
    true_label = path_list[1]

    # Perform image classification using the TFLite model
    predicted_class, confidence = classify_image_tflite(interpreter, image_path, class_labels)

    # Print the results
    print(f"Predicted Class: {predicted_class}")
    print(f"True Label: {true_label}")
    print(f"Confidence (%): {confidence * 100}")

    # Update disease_name based on the predicted class
    disease_name = predicted_class.lower()

    # Check if the predicted class indicates a healthy leaf
    if "healthy" in disease_name:
        print("The leaf is healthy")
        say_text = "The leaf is healthy"
        cure = "The leaf is healthy. Don't worry"
        dis_name = "The leaf is healthy"
    else:
        # Retrieve and print the cure information for the predicted disease
        cure_of_disease = dis_DB_cure[disease_name]
        print(cure_of_disease)
