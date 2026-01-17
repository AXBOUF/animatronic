from flask import Flask, render_template, jsonify
import cv2
import numpy as np
from collections import deque
import threading
import time

app = Flask(__name__)

class EyeballTracker:
    def __init__(self):
        self.cap = cv2.VideoCapture(2)
        
        # Lower resolution for better performance
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        self.eye_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_eye.xml'
        )
        
        # Enhanced smoothing with larger history buffer
        self.history_size = 20
        self.dx_history = deque(maxlen=self.history_size)
        self.dy_history = deque(maxlen=self.history_size)
        
        # Current gaze direction (heavily smoothed)
        self.dx = 0.0
        self.dy = 0.0
        
        # Multi-level smoothing
        self.raw_dx = 0.0
        self.raw_dy = 0.0
        self.smoothing_factor = 0.20  # Higher = faster response
        
        # Start background tracking thread
        self.running = True
        self.thread = threading.Thread(target=self._tracking_loop, daemon=True)
        self.thread.start()
        
    def detect_face_position(self, frame):
        """Enhanced face/eye detection"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)
        
        faces = self.face_cascade.detectMultiScale(
            gray, 
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(80, 80)
        )
        
        if len(faces) > 0:
            largest_face = max(faces, key=lambda rect: rect[2] * rect[3])
            x, y, w, h = largest_face
            
            # Try to detect eyes
            face_roi = gray[y:y+h, x:x+w]
            eyes = self.eye_cascade.detectMultiScale(face_roi, 1.1, 10)
            
            frame_h, frame_w = frame.shape[:2]
            
            if len(eyes) >= 2:
                eyes_sorted = sorted(eyes, key=lambda e: e[0])
                left_eye = eyes_sorted[0]
                right_eye = eyes_sorted[1]
                
                eye_center_x = x + (left_eye[0] + left_eye[2]//2 + right_eye[0] + right_eye[2]//2) // 2
                eye_center_y = y + (left_eye[1] + left_eye[3]//2 + right_eye[1] + right_eye[3]//2) // 2
                
                dx = (eye_center_x - frame_w // 2) / (frame_w // 2)
                dy = (eye_center_y - frame_h // 2) / (frame_h // 2)
            else:
                face_center_x = x + w // 2
                face_center_y = y + h // 2
                
                dx = (face_center_x - frame_w // 2) / (frame_w // 2)
                dy = (face_center_y - frame_h // 2) / (frame_h // 2)
            
            return dx, dy
        
        return None, None
    
    def _smooth_direction(self, target_dx, target_dy):
        """Advanced multi-stage smoothing"""
        if target_dx is not None:
            # Add to history
            self.dx_history.append(target_dx)
            self.dy_history.append(target_dy)
            
            if len(self.dx_history) >= 5:  # Wait for enough samples
                # Exponential weighted average (recent values weighted more)
                weights = np.exp(np.linspace(0, 1, len(self.dx_history)))
                weights = weights / weights.sum()
                
                self.raw_dx = np.average(list(self.dx_history), weights=weights)
                self.raw_dy = np.average(list(self.dy_history), weights=weights)
        
        # Apply smooth interpolation
        self.dx += (self.raw_dx - self.dx) * self.smoothing_factor
        self.dy += (self.raw_dy - self.dy) * self.smoothing_factor
    
    def _tracking_loop(self):
        """Background thread for continuous tracking"""
        while self.running:
            ret, frame = self.cap.read()
            if ret:
                frame = cv2.flip(frame, 1)
                dx, dy = self.detect_face_position(frame)
                self._smooth_direction(dx, dy)
            
            time.sleep(0.010)  # ~100fps tracking for lower latency
    
    def get_direction(self):
        """Get current smoothed direction"""
        return self.dx, self.dy
    
    def cleanup(self):
        """Clean up resources"""
        self.running = False
        self.cap.release()

tracker = EyeballTracker()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/direction')
def get_direction():
    """API endpoint for getting current gaze direction"""
    dx, dy = tracker.get_direction()
    return jsonify({'dx': dx, 'dy': dy})

HTML_TEMPLATE = '''
<!DOCTYPE html>
<html>
<head>
    <title>Digital Animatronic Eyeball</title>
    <style>
        body {
            margin: 0;
            padding: 20px;
            font-family: Arial, sans-serif;
            background: #1a1a1a;
            color: white;
            display: flex;
            flex-direction: column;
            align-items: center;
            min-height: 100vh;
        }
        h1 {
            text-align: center;
            margin-bottom: 30px;
            font-size: 2.5em;
        }
        .eyeballs-container {
            display: flex;
            gap: 60px;
            justify-content: center;
            margin-bottom: 30px;
        }
        .eyeball-wrapper {
            position: relative;
        }
        .eyeball-canvas {
            border-radius: 50%;
            background: white;
            box-shadow: 0 10px 30px rgba(0,0,0,0.5);
        }
        .info {
            text-align: center;
            padding: 15px;
            background: #2a2a2a;
            border-radius: 10px;
            font-family: monospace;
            font-size: 14px;
            max-width: 400px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.3);
        }
        .info-row {
            margin: 8px 0;
            display: flex;
            justify-content: space-between;
            padding: 5px 10px;
        }
        .label {
            color: #888;
        }
        .value {
            color: #4CAF50;
            font-weight: bold;
        }
    </style>
</head>
<body>
    <h1>👀 Digital Animatronic Eyes</h1>
    
    <div class="eyeballs-container">
        <div class="eyeball-wrapper">
            <canvas id="left-eyeball" class="eyeball-canvas" width="400" height="400"></canvas>
        </div>
        <div class="eyeball-wrapper">
            <canvas id="right-eyeball" class="eyeball-canvas" width="400" height="400"></canvas>
        </div>
    </div>
    
    <div class="info">
        <div class="info-row">
            <span class="label">Horizontal:</span>
            <span class="value" id="dx">0.000</span>
        </div>
        <div class="info-row">
            <span class="label">Vertical:</span>
            <span class="value" id="dy">0.000</span>
        </div>
        <div class="info-row">
            <span class="label">Render FPS:</span>
            <span class="value" id="fps">0</span>
        </div>
    </div>
    
    <script>
        const leftCanvas = document.getElementById('left-eyeball');
        const rightCanvas = document.getElementById('right-eyeball');
        const leftCtx = leftCanvas.getContext('2d');
        const rightCtx = rightCanvas.getContext('2d');
        
        const eyeballSize = 400;
        const scleraRadius = 160;
        const pupilRadius = 50;
        const irisRadius = 75;
        const maxPupilOffset = 65;
        
        // Client-side smoothing variables
        let targetDx = 0, targetDy = 0;
        let currentLeftDx = 0, currentLeftDy = 0;
        let currentRightDx = 0, currentRightDy = 0;
        const clientSmoothness = 0.18;  // Higher = faster response
        
        // FPS counter
        let lastTime = performance.now();
        let frameCount = 0;
        
        function drawEyeball(ctx, dx, dy, isLeft) {
            // Subtle parallax effect
            const parallaxFactor = isLeft ? -0.08 : 0.08;
            const adjustedDx = dx + parallaxFactor;
            
            // Clear canvas
            ctx.clearRect(0, 0, eyeballSize, eyeballSize);
            
            const centerX = eyeballSize / 2;
            const centerY = eyeballSize / 2;
            
            // Draw sclera with subtle gradient
            const scleraGradient = ctx.createRadialGradient(
                centerX, centerY - 20, 0,
                centerX, centerY, scleraRadius
            );
            scleraGradient.addColorStop(0, '#ffffff');
            scleraGradient.addColorStop(0.7, '#fafafa');
            scleraGradient.addColorStop(1, '#efefef');
            
            ctx.fillStyle = scleraGradient;
            ctx.beginPath();
            ctx.arc(centerX, centerY, scleraRadius, 0, 2 * Math.PI);
            ctx.fill();
            
            // Subtle sclera border
            ctx.strokeStyle = '#d5d5d5';
            ctx.lineWidth = 2;
            ctx.stroke();
            
            // Calculate pupil position
            let pupilX = centerX + adjustedDx * maxPupilOffset;
            let pupilY = centerY + dy * maxPupilOffset;
            
            // Keep pupil within bounds
            const distFromCenter = Math.sqrt(
                Math.pow(pupilX - centerX, 2) + Math.pow(pupilY - centerY, 2)
            );
            const maxDist = scleraRadius - irisRadius;
            
            if (distFromCenter > maxDist) {
                const scale = maxDist / distFromCenter;
                pupilX = centerX + (pupilX - centerX) * scale;
                pupilY = centerY + (pupilY - centerY) * scale;
            }
            
            // Draw iris with detailed gradient
            const irisGradient = ctx.createRadialGradient(
                pupilX, pupilY, pupilRadius,
                pupilX, pupilY, irisRadius
            );
            irisGradient.addColorStop(0, '#0d3d66');
            irisGradient.addColorStop(0.3, '#1a5490');
            irisGradient.addColorStop(0.7, '#2a78c8');
            irisGradient.addColorStop(1, '#1a5490');
            
            ctx.fillStyle = irisGradient;
            ctx.beginPath();
            ctx.arc(pupilX, pupilY, irisRadius, 0, 2 * Math.PI);
            ctx.fill();
            
            // Draw pupil with slight gradient
            const pupilGradient = ctx.createRadialGradient(
                pupilX, pupilY, 0,
                pupilX, pupilY, pupilRadius
            );
            pupilGradient.addColorStop(0, '#000000');
            pupilGradient.addColorStop(1, '#1a1a1a');
            
            ctx.fillStyle = pupilGradient;
            ctx.beginPath();
            ctx.arc(pupilX, pupilY, pupilRadius, 0, 2 * Math.PI);
            ctx.fill();
            
            // Main highlight
            ctx.fillStyle = 'rgba(255, 255, 255, 0.9)';
            ctx.beginPath();
            ctx.arc(pupilX - 20, pupilY - 20, 18, 0, 2 * Math.PI);
            ctx.fill();
            
            // Secondary highlight
            ctx.fillStyle = 'rgba(255, 255, 255, 0.5)';
            ctx.beginPath();
            ctx.arc(pupilX + 15, pupilY + 18, 10, 0, 2 * Math.PI);
            ctx.fill();
            
            // Tertiary tiny highlight
            ctx.fillStyle = 'rgba(255, 255, 255, 0.3)';
            ctx.beginPath();
            ctx.arc(pupilX - 8, pupilY + 25, 5, 0, 2 * Math.PI);
            ctx.fill();
        }
        
        function animate() {
            // Ultra-smooth interpolation for each eye separately
            currentLeftDx += (targetDx - currentLeftDx) * clientSmoothness;
            currentLeftDy += (targetDy - currentLeftDy) * clientSmoothness;
            
            currentRightDx += (targetDx - currentRightDx) * clientSmoothness;
            currentRightDy += (targetDy - currentRightDy) * clientSmoothness;
            
            // Draw both eyeballs
            drawEyeball(leftCtx, currentLeftDx, currentLeftDy, true);
            drawEyeball(rightCtx, currentRightDx, currentRightDy, false);
            
            // Update FPS
            frameCount++;
            const now = performance.now();
            if (now - lastTime >= 1000) {
                document.getElementById('fps').textContent = frameCount;
                frameCount = 0;
                lastTime = now;
            }
            
            requestAnimationFrame(animate);
        }
        
        function updateDirection() {
            fetch('/direction')
                .then(response => response.json())
                .then(data => {
                    targetDx = data.dx;
                    targetDy = data.dy;
                    
                    document.getElementById('dx').textContent = data.dx.toFixed(3);
                    document.getElementById('dy').textContent = data.dy.toFixed(3);
                })
                .catch(err => console.error('Error:', err));
        }
        
        // Fetch direction at 60Hz for low latency
        setInterval(updateDirection, 16);
        
        // Start animation loop at 60fps
        requestAnimationFrame(animate);
        
        // Initial draw
        drawEyeball(leftCtx, 0, 0, true);
        drawEyeball(rightCtx, 0, 0, false);
    </script>
</body>
</html>
'''

# Create templates directory and save HTML
import os
os.makedirs('templates', exist_ok=True)
with open('templates/index.html', 'w') as f:
    f.write(HTML_TEMPLATE)

if __name__ == '__main__':
    print("=" * 60)
    print("Digital Animatronic Eyeball System - Low Latency Edition")
    print("=" * 60)
    print("\nOptimizations:")
    print("  • 100fps tracking rate")
    print("  • 60Hz data polling")
    print("  • Faster smoothing (0.20 server, 0.18 client)")
    print("  • 60 FPS rendering")
    print("\nStarting server...")
    print("Open your browser and go to:")
    print("\n  http://localhost:8080\n")
    print("Press Ctrl+C to stop")
    print("=" * 60)
    
    try:
        app.run(host='0.0.0.0', port=8080, debug=False, threaded=True)
    finally:
        tracker.cleanup()