from flask import Flask, request, jsonify, render_template, send_file
import torch
from model import VlixG1
from PIL import Image
import io
import numpy as np
import time

app = Flask(__name__)

# Initialize model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = VlixG1().to(device)
model.load_state_dict(torch.load('vlixg_model.pth', map_location=device))
model.eval()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/generate', methods=['POST'])
def generate():
    start_time = time.time()
    
    data = request.get_json()
    text = data.get('text', 'A simple shape')
    
    try:
        # Generate image
        generated = model.generate_from_text(text, device)
        generated = (generated * 255).astype('uint8')
        
        # Create PIL image
        img = Image.fromarray(generated.squeeze(), mode='L')
        
        # Save to bytes
        img_byte_arr = io.BytesIO()
        img.save(img_byte_arr, format='PNG')
        img_byte_arr.seek(0)
        
        # Log performance
        print(f"Generated in {time.time() - start_time:.2f}s")
        
        return send_file(img_byte_arr, mimetype='image/png')
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
