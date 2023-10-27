import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
from PIL import Image
import json
from typing import Tuple, Dict

def load_model(model_filepath: str):
    '''
    Load a pre-trained model from a file.
    
    Args:
        - model_filepath (str): The file path to the pre-trained model.
        
    Returns:
        - tf.keras.models.Model: The loaded pre-trained model.
    '''
    
    return tf.keras.models.load_model(model_filepath, custom_objects={'KerasLayer':hub.KerasLayer})

def load_and_preprocess_image(image_path: str):
    '''
    Load and preprocess an image from a file.
    
    The function loads the image from the specified path, resizes it to the
    targt size, normalizes pixel values, and returns it as a TensorFlow tensor.
    
    Args:
        - image_path (str): The file path to the image.
        
    Returns:
        - tf.Tensor: The preprocessed image as a TensorFlow tensor.
    '''

    # Load image
    image = Image.open(image_path)
    image = np.asarray(image)
    
    # Process image
    image_size = 224
    image = tf.cast(image, tf.float32)
    image = tf.image.resize_with_pad(image, image_size, image_size)
    image /= 255
    image = image.numpy().squeeze()
    image = tf.expand_dims(image, axis=0)
    
    return image

def predict(image: tf.Tensor, model: tf.keras.models.Model, top_k: int) -> Tuple[np.ndarray, np.ndarray]:
    '''
    Make predictions using a pre-trained model.
    
    Args:
        - image (tf.Tensor): The preprocessed image as a TensorFlow tensor.
        - model (tf.keras.models.Model): The pre-trained model for making predictions.
        - top_k (int): The number of top predictions to return.
        
    Returns:
        - Tuple[np.ndarray, np.ndarray]: A tuple of NumPy arrays containing the top
          probabilities and top class indices.
    '''

    print('Predicting...')
    ps = model.predict(image) # Predictions
    
    top_probs, top_classes = tf.math.top_k(ps, top_k)
    top_probs, top_classes = top_probs.numpy()[0], top_classes.numpy()[0]
    
    return top_probs, top_classes

def load_category_names(category_names: str) -> Tuple[Dict[str, str], int]:
    '''
    Load category names from a JSON file and return the category names dictionary and
    the first key.
    
    Args:
        - category_names (str): The file path to the JSON file containing category names
    
    Returns:
        - Tuple[Dict[str, str], int]: A tuple containing the loaded category names
          dictionary and the first key. If the file is not found, returns an empty
          dictionary. If the JSON format is invalid, returns an empty dictionary.
    '''
    
    try:
        with open(category_names, "r") as file:
            category_names = json.load(file)
            
            # Get first key of JSON file
            keys = list(map(int, category_names.keys()))
            first_key = min(keys)
            
        return category_names, first_key
    
    except FileNotFoundError:
        print("Category names file not found.")
        return {}
    
    except json.JSONDecodeError:
        print("Invalid JSON format in category names file.")
        return {}