import socketio
import eventlet
import numpy as np
from flask import Flask
from keras.models import load_model
import base64
from io import BytesIO
from PIL import Image
import cv2
import keras.metrics as metrics
import keras.saving


sio = socketio.Server()
app = Flask(__name__)  
speed_limit = 20


def mse(y_true, y_pred):
    return metrics.mean_squared_error(y_true, y_pred)

keras.saving.register_keras_serializable()(mse)  


try:
    model = load_model('model/model.h5', custom_objects={'mse': mse})
    print("✅ Model Loaded Successfully!")
except Exception as e:
    print(f"❌ Error Loading Model: {str(e)}")
    raise SystemExit 

# Image Preprocessing Function
def img_preprocess(img):
    img = img[50:300,:,:]
    img = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
    img = cv2.GaussianBlur(img,  (3, 3), 0)
    img = cv2.resize(img, (200, 66))
    img = img/255
    return img

# Telemetry Event Handler
@sio.on('telemetry')
def telemetry(sid, data):
    speed = float(data['speed'])
    image = Image.open(BytesIO(base64.b64decode(data['image'])))
    image = np.asarray(image)
    image = img_preprocess(image)
    image = np.array([image])  

    # Predict Steering Angle
    steering_angle = float(model.predict(image)[0][0])
    throttle = 1.0 - (speed / speed_limit)  

    print(f"🚗 Steering: {steering_angle:.4f}, Throttle: {throttle:.2f}, Speed: {speed:.2f}")
    send_control(steering_angle, throttle)

# Connection Handler
@sio.on('connect')
def connect(sid, environ):
    print("🔗 Client Connected")
    send_control(0, 0)

# Send Steering & Throttle Data
def send_control(steering_angle, throttle):
    sio.emit('steer', data={
        'steering_angle': str(steering_angle),
        'throttle': str(throttle)
    })

# Run the Flask-SocketIO Server
if __name__ == '__main__':
    app = socketio.Middleware(sio, app)
    eventlet.wsgi.server(eventlet.listen(('', 4567)), app)
