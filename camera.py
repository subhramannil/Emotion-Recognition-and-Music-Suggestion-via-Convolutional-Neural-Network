import cv2
import numpy as np
from tensorflow.keras.models import load_model
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

emotion_model = load_model("resnet50_model.keras")
logger.info("Model loaded")

emotion_mapping = {
    0: 'angry',
    1: 'disgusted',
    2: 'fearful',
    3: 'happy',
    4: 'neutral',
    5: 'sad',
    6: 'surprised'
}

face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
if face_cascade.empty():
    raise Exception("Haar cascade load failed")


class VideoCamera:
    def __init__(self):
        self.emotion = "neutral"

    def get_emotion(self):
        return self.emotion
    def gen_frames(self):
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            logger.error("Webcam not opened")
            return
        logger.info("Webcam streaming started")
        while True:
            success, frame = cap.read()
            if not success:
                logger.warning("Failed to grab frame")
                break
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.3, 5)
            for (x, y, w, h) in faces:
                roi_color = frame[y:y+h, x:x+w]
                roi_resized = cv2.resize(roi_color, (224, 224))  # ResNet expects 224x224 RGB
                roi = roi_resized.astype('float32') / 255.0
                roi = np.expand_dims(roi, axis=0)


                preds = emotion_model.predict(roi, verbose=0)
                emotion_idx = np.argmax(preds)
                self.emotion = emotion_mapping.get(emotion_idx, 'neutral')
                label = self.emotion.capitalize()

                cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
                cv2.putText(frame, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX,
                            1, (0, 255, 0), 2)

            # Encode frame as JPEG
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

        cap.release()


        """
        python -m venv tf_env
        .\tf_env\Scripts\activate
        pip --version
        python -c "import tensorflow as tf; print('TensorFlow version:', tf.__version__)"
        from tensorflow.keras.models import load_model
        model_path = "path/to/custom_cnn_model.h5"
        emotion_model = load_model(model_path)
        print("Model loaded successfully in TensorFlow 2.18.0")
        python app.py
        """