import os
import google.generativeai as genai
from datetime import datetime
from dotenv import load_dotenv



GEMINI_API_KEY = os.getenv("HF_API_KEY")
genai.configure(api_key=GEMINI_API_KEY)
load_dotenv()
# Set up the model
generation_config = {
    "temperature": 0.7,
    "top_p": 1,
    "top_k": 1,
    "max_output_tokens": 2048,
}

safety_settings = [
    {
        "category": "HARM_CATEGORY_HARASSMENT",
        "threshold": "BLOCK_MEDIUM_AND_ABOVE"
    },
    {
        "category": "HARM_CATEGORY_HATE_SPEECH",
        "threshold": "BLOCK_MEDIUM_AND_ABOVE"
    },
    {
        "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
        "threshold": "BLOCK_MEDIUM_AND_ABOVE"
    },
    {
        "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
        "threshold": "BLOCK_MEDIUM_AND_ABOVE"
    },
]

model = genai.GenerativeModel(model_name="gemini-1.5-flash",
                              generation_config=generation_config,
                              safety_settings=safety_settings)

def generate_report(image_path, prediction, confidence, activation_percentages, preds, class_names):
    """
    Generate a comprehensive medical report using Gemini API
    """
    
    prompt = f"""
    You are an expert hematologist specializing in Acute Lymphoblastic Leukemia (ALL). 
    Analyze the following diagnostic results and provide a detailed medical report:
    
    Patient's Blood Smear Analysis Results:
    - Predicted Class: {prediction}
    - Confidence Level: {confidence}
    - Class Probabilities: {dict(zip(class_names, preds))}
    - Activation Percentages for each class: {dict(zip(class_names, activation_percentages))}
    
    The image shows white blood cells from a peripheral blood smear. Based on these results:
    
    1. Provide a professional interpretation of the diagnosis in medical terms.
    2. Explain what the predicted class means in the context of ALL progression.
    3. Analyze the activation heatmap percentages and what they indicate about cell abnormalities.
    4. Describe the typical morphological characteristics expected for this stage.
    5. Provide clinical recommendations for next steps (additional tests, treatments, etc.).
    6. Include any relevant differential diagnoses to consider.
    
    Format your response as a formal medical report with the following sections:
    - Clinical Findings
    - Interpretation
    - Morphological Analysis
    - Recommendations
    - Differential Diagnosis
    
    Use professional medical terminology but keep explanations clear for clinicians.
    """
    
    try:
        response = model.generate_content(prompt)
        report = response.text
        
       
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_filename = f"report_{timestamp}.txt"
        report_path = os.path.join("static/reports", report_filename)
        
        os.makedirs("static/reports", exist_ok=True)
        with open(report_path, "w") as f:
            f.write(f"Acute Lymphoblastic Leukemia Diagnostic Report\n")
            # f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write(report)
        
        return report, report_filename
    except Exception as e:
        print(f"Error generating report: {e}")
        return None, None

def get_medical_definition(term):
    """
    Get a medical definition for a term using Gemini
    """
    prompt = f"Provide a concise medical definition of {term} in the context of hematology and Acute Lymphoblastic Leukemia. Keep it under 100 words."
    
    try:
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        print(f"Error getting definition: {e}")
        return f"Could not retrieve definition for {term}"

def clear_reports_folder():
    """Clear previous report files"""
    folder = 'static/reports'
    if os.path.exists(folder):
        for filename in os.listdir(folder):
            file_path = os.path.join(folder, filename)
            try:
                if os.path.isfile(file_path):
                    os.unlink(file_path)
            except Exception as e:
                print(f"Error deleting file {file_path}: {e}")