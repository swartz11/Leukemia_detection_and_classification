import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.efficientnet import preprocess_input

def preprocess_image(image_path, target_size):
    """Load and preprocess image for ResNet"""
    img = load_img(image_path, target_size=target_size)
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)
    return img

def get_class_name(class_index):
    """Map class index to human-readable name"""
    class_names = ['Benign', 'Early', 'Pre', 'Pro']
    return class_names[class_index]