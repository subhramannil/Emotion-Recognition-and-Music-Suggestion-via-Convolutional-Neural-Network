from flask import Flask, render_template, Response, jsonify
from camera import VideoCamera
import pandas as pd
import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Initialize the webcam
try:
    video_stream = VideoCamera()
    logger.info("Video stream initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize video stream: {e}")
    video_stream = None

# Ensure songs directory exists
os.makedirs('songs', exist_ok=True)

@app.route('/')
def index():
    emotion = "neutral"
    if video_stream:
        emotion = video_stream.get_emotion()

    csv_path = f'songs/{emotion}.csv'
    try:
        df = pd.read_csv(csv_path, usecols=['Name', 'Album', 'Artist'])
        logger.info(f"Loaded {len(df)} songs for {emotion}")
    except Exception as e:
        logger.error(f"Error loading CSV: {e}")
        df = pd.DataFrame(columns=['Name', 'Album', 'Artist'])

    headings = ['Name', 'Album', 'Artist']
    return render_template("index.html", 
                           headings=headings, 
                           data=df.to_dict(orient='records'), 
                           emotion=emotion.capitalize())

@app.route('/video_feed')
def video_feed():
    if not video_stream:
        return "Camera not available", 500
    return Response(video_stream.gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/emotion_data')
def emotion_data():
    emotion = video_stream.get_emotion()
    csv_path = f'songs/{emotion}.csv'
    try:
        df = pd.read_csv(csv_path, usecols=['Name', 'Album', 'Artist'])
    except:
        df = pd.DataFrame(columns=['Name', 'Album', 'Artist'])

    return jsonify({
        "emotion": emotion.capitalize(),
        "songs": df.to_dict(orient='records')
    })

if __name__ == '__main__':
    app.run(debug=True, host="0.0.0.0")
