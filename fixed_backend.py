from flask import Flask, request, jsonify, send_file, render_template
from flask_cors import CORS
from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image
import io
import os
import base64
from werkzeug.utils import secure_filename
import requests
from io import BytesIO

app = Flask(__name__)
CORS(app)

# Configuración
UPLOAD_FOLDER = 'uploads'
PROCESSED_FOLDER = 'processed'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'mp4', 'avi', 'mov'}

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PROCESSED_FOLDER, exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['PROCESSED_FOLDER'] = PROCESSED_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100MB max

# Cargar modelo YOLO (usa yolov8n.pt como base)
model = YOLO('yolov8n.pt')

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def detect_color(image, bbox):
    """Detecta el color dominante en el área del bbox"""
    x1, y1, x2, y2 = map(int, bbox)
    roi = image[y1:y2, x1:x2]
    
    if roi.size == 0:
        return "Desconocido"
    
    # Convertir a HSV para mejor detección de color
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    avg_hue = np.mean(hsv[:, :, 0])
    avg_sat = np.mean(hsv[:, :, 1])
    avg_val = np.mean(hsv[:, :, 2])
    
    # Clasificar color basado en HSV
    if avg_sat < 50:
        if avg_val > 200:
            return "Blanco"
        elif avg_val < 50:
            return "Negro"
        else:
            return "Gris"
    
    if avg_hue < 10 or avg_hue > 170:
        return "Rojo"
    elif 10 <= avg_hue < 25:
        return "Naranja"
    elif 25 <= avg_hue < 35:
        return "Amarillo"
    elif 35 <= avg_hue < 85:
        return "Verde"
    elif 85 <= avg_hue < 130:
        return "Azul"
    elif 130 <= avg_hue < 170:
        return "Morado"
    
    return "Multicolor"

def classify_shape(contour):
    """Clasifica la forma geométrica basada en el contorno"""
    peri = cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, 0.04 * peri, True)
    vertices = len(approx)
    
    if vertices == 3:
        return "Triángulo"
    elif vertices == 4:
        x, y, w, h = cv2.boundingRect(approx)
        aspect_ratio = w / float(h)
        if 0.95 <= aspect_ratio <= 1.05:
            return "Cuadrado"
        else:
            return "Rectángulo"
    elif vertices == 5:
        return "Pentágono"
    elif vertices == 6:
        return "Hexágono"
    elif vertices > 6:
        return "Círculo"
    
    return "Forma irregular"

def process_image(image):
    """Procesa una imagen y detecta objetos con formas y colores"""
    # Detección con YOLO
    results = model(image)
    
    detections = []
    annotated_image = image.copy()
    
    for result in results:
        boxes = result.boxes
        for box in boxes:
            # Obtener coordenadas
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            conf = float(box.conf[0])
            cls = int(box.cls[0])
            
            # Detectar color
            color = detect_color(image, [x1, y1, x2, y2])
            
            # Extraer región para análisis de forma
            roi = image[int(y1):int(y2), int(x1):int(x2)]
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            shape = "Objeto"
            if contours:
                largest_contour = max(contours, key=cv2.contourArea)
                shape = classify_shape(largest_contour)
            
            # Anotar imagen
            cv2.rectangle(annotated_image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            label = f"{shape} {color} ({conf:.2f})"
            cv2.putText(annotated_image, label, (int(x1), int(y1)-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            detections.append({
                'shape': shape,
                'color': color,
                'confidence': conf,
                'bbox': [float(x1), float(y1), float(x2), float(y2)]
            })
    
    return annotated_image, detections

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/detect/image', methods=['POST'])
def detect_image():
    try:
        image = None
        
        # Opción 1: Imagen subida
        if 'file' in request.files:
            file = request.files['file']
            if file and allowed_file(file.filename):
                image_bytes = file.read()
                nparr = np.frombuffer(image_bytes, np.uint8)
                image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        # Opción 2: URL de imagen
        elif 'url' in request.json:
            url = request.json['url']
            response = requests.get(url)
            image_bytes = BytesIO(response.content)
            pil_image = Image.open(image_bytes)
            image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
        
        # Opción 3: Imagen base64
        elif 'image_base64' in request.json:
            base64_str = request.json['image_base64'].split(',')[1]
            image_bytes = base64.b64decode(base64_str)
            nparr = np.frombuffer(image_bytes, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            return jsonify({'error': 'No se proporcionó imagen válida'}), 400
        
        # Procesar imagen
        annotated_image, detections = process_image(image)
        
        # Convertir imagen procesada a base64
        _, buffer = cv2.imencode('.jpg', annotated_image)
        img_base64 = base64.b64encode(buffer).decode('utf-8')
        
        return jsonify({
            'success': True,
            'detections': detections,
            'processed_image': f'data:image/jpeg;base64,{img_base64}'
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/detect/video', methods=['POST'])
def detect_video():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No se proporcionó archivo'}), 400
        
        file = request.files['file']
        if not allowed_file(file.filename):
            return jsonify({'error': 'Formato de archivo no permitido'}), 400
        
        # Guardar video
        filename = secure_filename(file.filename)
        input_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(input_path)
        
        # Procesar video
        cap = cv2.VideoCapture(input_path)
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        output_filename = f'processed_{filename}'
        output_path = os.path.join(app.config['PROCESSED_FOLDER'], output_filename)
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        all_detections = []
        frame_count = 0
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            annotated_frame, detections = process_image(frame)
            out.write(annotated_frame)
            
            if frame_count % 30 == 0:  # Guardar detecciones cada 30 frames
                all_detections.append({
                    'frame': frame_count,
                    'detections': detections
                })
            
            frame_count += 1
        
        cap.release()
        out.release()
        
        return jsonify({
            'success': True,
            'video_path': f'/download/{output_filename}',
            'detections': all_detections,
            'total_frames': frame_count
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/download/<filename>')
def download_file(filename):
    return send_file(
        os.path.join(app.config['PROCESSED_FOLDER'], filename),
        as_attachment=True
    )

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
