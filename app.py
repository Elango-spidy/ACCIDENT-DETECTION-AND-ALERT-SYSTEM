from flask import Flask, request, jsonify, render_template, make_response
import cv2
import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
from werkzeug.utils import secure_filename
import os
from datetime import datetime
import requests
import telegram
import base64
import io

# Create a Telegram bot using the BotFather and get the bot token
BOT_TOKEN = '6131817346:AAFmd-64C8z34l1_mS13hB9X2-Rnr8-0JDQ'

# Get the chat ID of the Telegram channel
chat_id = '-976671016'

# Set the url for sending messages
send_message_url = f'https://api.telegram.org/bot{BOT_TOKEN}/sendMessage'

# Set the url for sending photos
send_photo_url = f'https://api.telegram.org/bot{BOT_TOKEN}/sendPhoto'

app = Flask(__name__, template_folder='template')

# Load the model
model = load_model('models/accident_dete_opti.h5')

@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def predict():
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            return jsonify({'error': 'No file part'})
        file = request.files['file']
        # if user does not select file, browser also
        # submit an empty part without filename
        if file.filename == '':
            return jsonify({'error': 'No selected file'})
        if file:
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))

            # Load the video file
            cap = cv2.VideoCapture(filepath)

            # Change this line to set the number of frames to process
            cap.set(cv2.CAP_PROP_FPS, 2)

            flag = False
            while True:
                ret, frame = cap.read()
                if not ret:  # check if the frame is valid
                    break
                resize = tf.image.resize(frame, (256, 256))
                prediction = model.predict(np.expand_dims(resize / 255, 0))
                if prediction < 0.5:
                    predict = "Accident detected"
                    cv2.imwrite('static/Result.jpg', frame)
                    # Encode the image as a base64 string
                    ret, buffer = cv2.imencode('.jpg', frame)
                    image_data = base64.b64encode(buffer).decode('utf-8')
                    # Set the image source to a base64-encoded data URL for rendering in the response
                    res_image = f"data:image/jpeg;base64,{image_data}"

                    flag = True
                    break

            cap.release()

            if flag:
                # photo_data = {'photo': open('static/Result.jpg', 'rb')}
                # response = requests.post(send_photo_url, data={'chat_id': chat_id, 'caption': 'Accident was detected!'}, files=photo_data)
                # if response.status_code == 200:
                response = make_response(render_template('success.html', res_image= res_image))
                return response
            else:
                return render_template('noaccident.html')


if __name__ == '__main__':
    app.config['UPLOAD_FOLDER'] = 'static'
    app.run(debug=True)