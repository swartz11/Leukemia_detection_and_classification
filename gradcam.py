import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.efficientnet import preprocess_input
import os

def generate_gradcam(model, img_array, layer_name="top_activation"):
    """Generate Grad-CAM heatmap for all classes with contour lines"""
    # Create gradient model
    grad_model = tf.keras.models.Model(
        inputs=model.inputs,
        outputs=[model.get_layer(layer_name).output, model.output]
    )
    
    heatmaps = []
    preds = model.predict(img_array)
    
    for target_class in range(4):  
        with tf.GradientTape() as tape:
            conv_outputs, predictions = grad_model(img_array)
            loss = predictions[:, target_class]
        
        grads = tape.gradient(loss, conv_outputs)[0]
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        weights = tf.reduce_mean(grads, axis=(0, 1))
        
        # Combined heatmap calculation
        heatmap = tf.reduce_sum(conv_outputs[0] * (0.6 * weights + 0.4 * pooled_grads[..., tf.newaxis]), axis=-1)
        heatmap = tf.maximum(heatmap, 0)
        heatmap = heatmap / (tf.math.reduce_max(heatmap) + 1e-8)
        
        heatmaps.append(heatmap.numpy())
    
    return heatmaps, preds

def create_heatmap_visualization(img_array, heatmap, pred, class_name):
    """Create visualization for a single heatmap with contour lines"""
    # Resize and prepare heatmap
    heatmap = cv2.resize(heatmap, (img_array.shape[2], img_array.shape[1]))
    heatmap_display = np.uint8(255 * heatmap)
    
    # Medical-optimized colormap
    heatmap_colored = cv2.applyColorMap(heatmap_display, cv2.COLORMAP_JET)
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
    
    img_display = (img_array[0] - img_array[0].min()) / (img_array[0].max() - img_array[0].min())
    img_display = np.uint8(255 * img_display)
    
    superimposed = cv2.addWeighted(img_display, 0.65, heatmap_colored, 0.35, 0)
    
    _, thresh = cv2.threshold(heatmap_display, 30, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    cv2.drawContours(superimposed, contours, -1, (255, 255, 255), 2)  
    cv2.drawContours(superimposed, contours, -1, (0, 0, 0), 1)       
    
    superimposed = cv2.copyMakeBorder(superimposed, 5, 5, 5, 5, cv2.BORDER_CONSTANT, value=[200, 200, 200])
    
    # Save the visualization
    os.makedirs("static/heatmaps", exist_ok=True)
    output_filename = f"{class_name.lower()}_heatmap_{np.random.randint(10000)}.jpg"
    output_path = f"static/heatmaps/{output_filename}"
    cv2.imwrite(output_path, cv2.cvtColor(superimposed, cv2.COLOR_RGB2BGR))
    
    return output_filename

def calculate_activation_percentage(heatmap, threshold=0.15):
    """Calculate percentage of activated area with more precise calculation"""
    # Convert threshold to match heatmap scale (0-1)
    threshold_value = threshold * 255
    heatmap_scaled = (heatmap * 255).astype(np.uint8)
    
    # Calculate activated area
    activated_pixels = np.sum(heatmap_scaled > threshold_value)
    total_pixels = heatmap_scaled.size
    
    return (activated_pixels / total_pixels) * 100

def clear_heatmaps_folder():
    """Clear previous heatmap files"""
    folder = 'static/heatmaps'
    if os.path.exists(folder):
        for filename in os.listdir(folder):
            file_path = os.path.join(folder, filename)
            try:
                if os.path.isfile(file_path):
                    os.unlink(file_path)
            except Exception as e:
                print(f"Error deleting file {file_path}: {e}")