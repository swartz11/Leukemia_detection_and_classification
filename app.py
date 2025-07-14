
import os
from flask import Flask, render_template, request, redirect, url_for, send_from_directory
from werkzeug.utils import secure_filename
from tensorflow.keras.models import load_model
import numpy as np
from utils import preprocess_image, get_class_name
from gradcam import generate_gradcam, create_heatmap_visualization, calculate_activation_percentage
from report_generator import generate_report, clear_reports_folder
from datetime import datetime

app = Flask(__name__)

# Configuration
UPLOAD_FOLDER = 'static/uploads'
HEATMAP_FOLDER = 'static/heatmaps'
REPORT_FOLDER = 'static/reports'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
MODEL_PATH = 'models/resnet50.keras'

# Ensure folders exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(HEATMAP_FOLDER, exist_ok=True)
os.makedirs(REPORT_FOLDER, exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['HEATMAP_FOLDER'] = HEATMAP_FOLDER
app.config['REPORT_FOLDER'] = REPORT_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB limit

# Load model
model = load_model(MODEL_PATH, compile=False)
class_names = ['Benign', 'Early', 'Pre', 'Pro']

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        
        file = request.files['file']
        
        if file.filename == '':
            return redirect(request.url)
        
        if file and allowed_file(file.filename):
            # Clear previous files
            clear_reports_folder()
            
            # Save uploaded file
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            # Preprocess image
            img_array = preprocess_image(filepath, target_size=(224, 224))
            
            # Generate predictions and heatmaps
            heatmaps, preds = generate_gradcam(model, img_array)
            predicted_class = np.argmax(preds[0])
            confidence = np.max(preds) * 100
            class_name = get_class_name(predicted_class)
            
            # Create visualizations for all classes
            heatmap_paths = []
            activation_percentages = []
            
            for i in range(len(class_names)):
                heatmap_path = create_heatmap_visualization(
                    img_array, heatmaps[i], preds[0][i], class_names[i]
                )
                activation = calculate_activation_percentage(heatmaps[i])
                heatmap_paths.append(heatmap_path)
                activation_percentages.append(activation)
            
            # Generate medical report
            report_text, report_filename = generate_report(
                filepath,
                class_name,
                f"{confidence:.2f}%",
                activation_percentages,
                preds[0].tolist(),
                class_names
            )
            
            return render_template('result.html', 
                                 image_file=filename,
                                 prediction=class_name,
                                 confidence=f"{confidence:.2f}%",
                                 heatmap_paths=heatmap_paths,
                                 activation_percentages=activation_percentages,
                                 preds=preds[0].tolist(),
                                 class_names=class_names,
                                 predicted_class_idx=predicted_class,
                                 report_text=report_text,
                                 report_filename=report_filename)
    
    return render_template('index.html')

@app.route('/reports/<filename>')
def download_report(filename):
    return send_from_directory(app.config['REPORT_FOLDER'], filename, as_attachment=True)

@app.route('/clear_session', methods=['POST'])
def clear_session():
    # Clear all temporary files
    for folder in [app.config['UPLOAD_FOLDER'], app.config['HEATMAP_FOLDER'], app.config['REPORT_FOLDER']]:
        for filename in os.listdir(folder):
            file_path = os.path.join(folder, filename)
            try:
                if os.path.isfile(file_path):
                    os.unlink(file_path)
            except Exception as e:
                print(f"Error deleting file {file_path}: {e}")
    return redirect(url_for('upload_file'))

if __name__ == '__main__':
    app.run(debug=True)