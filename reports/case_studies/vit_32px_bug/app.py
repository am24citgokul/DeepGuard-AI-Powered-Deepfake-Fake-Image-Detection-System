from flask import Flask, request, jsonify, render_template
from transformers import ViTImageProcessor, ViTForImageClassification
from PIL import Image
import torch
import logging

# Initialize Flask app
app = Flask(__name__)

# Configure logging
logging.basicConfig(level=logging.INFO)

# Define the model and processor directory
model_dir = "./model/ai_vs_real_image_detection"  # Adjust this path as needed

try:
    # Try loading from local directory first, fallback to HuggingFace repo if needed
    try:
        model = ViTForImageClassification.from_pretrained(model_dir, local_files_only=True)
        processor = ViTImageProcessor.from_pretrained(model_dir)
        logging.info("Model and Processor loaded successfully from local directory!")
    except Exception as local_err:
        logging.warning(f"Local model loading failed ({local_err}), downloading from HuggingFace...")
        hf_model_name = "dima806/ai_vs_real_image_detection"
        model = ViTForImageClassification.from_pretrained(hf_model_name)
        processor = ViTImageProcessor.from_pretrained(hf_model_name)
        model.save_pretrained(model_dir)
        processor.save_pretrained(model_dir)
        logging.info("Model downloaded and cached locally!")

except Exception as e:
    logging.error(f"Error loading model or processor: {e}")
    raise RuntimeError("Failed to load model or processor")

@app.route('/')
def home():
    return render_template('index.html')

# Define the prediction endpoint
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Check if an image file is included in the request
        if 'image' not in request.files:
            return jsonify({"error": "No image file found in the request"}), 400
        
        # Read the image file from the request
        file = request.files['image']
        image = Image.open(file.stream).convert("RGB")

        # Preprocess the image
        inputs = processor(images=image, return_tensors="pt")

        # Perform inference
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            predicted_class_id = logits.argmax(-1).item()
        
        # Map the predicted class ID to the corresponding label
        predicted_label = model.config.id2label[predicted_class_id]
        confidence = torch.softmax(logits, dim=-1)[0][predicted_class_id].item()

          # Rename prediction labels
        if predicted_label.lower() == "fake":
            predicted_label = "AI Generated Image"
        elif predicted_label.lower() == "real":
            predicted_label = "Real Image"

        logging.info(f"Prediction: {predicted_label}, Confidence: {confidence:.4f}")

        # Return the prediction as JSON
        return jsonify({
            "prediction": predicted_label,
            "confidence": confidence
        })
    
    except Exception as e:
        logging.error(f"Prediction error: {e}")
        return jsonify({"error": str(e)}), 500

# Run the Flask app
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5050, debug=False)
