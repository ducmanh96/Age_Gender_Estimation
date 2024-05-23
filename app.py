# Import necessary libraries
from flask import Flask, render_template, Response, request, redirect, url_for, send_file
import os
import cv2
from openvino.runtime import Core
import numpy as np
import threading

# Initialize Flask app
app = Flask(__name__)

# Initialize OpenVINO Runtime
ie = Core()

# Load the pre-trained models for face detection
face_detection_model = ie.read_model(model="model/face-detection-adas-0001.xml")
compiled_face_detection_model = ie.compile_model(model=face_detection_model, device_name="CPU")

# Load the pre-trained models for age and gender recognition
age_gender_model = ie.read_model(model="model/age-gender-recognition-retail-0013.xml")
compiled_age_gender_model = ie.compile_model(model=age_gender_model, device_name="CPU")

# Initialize variables for video capture and output
video_cap = None
camera_cap = None
out = None
output_path = None
frame_width, frame_height = None, None

# Lock for synchronizing inference requests
inference_lock = threading.Lock()

# Function to capture an image from the camera
@app.route('/capture_image', methods=['GET'])
def capture_image():
    global camera_cap, frame_width, frame_height
    temp_image_path = 'your_image.jpg'

    if camera_cap is None:
        return "Camera is not started"

    success, frame = camera_cap.read()
    if success:
        image_with_info = draw_detection_info(frame)
        cv2.imwrite(temp_image_path, image_with_info)
        return send_file(temp_image_path, as_attachment=True)
    else:
        return "Failed to capture image"

# Function to draw detection information on an image frame
def draw_detection_info(frame):
    global frame_width, frame_height
    image_with_info = frame.copy()
    input_shape = compiled_face_detection_model.input(0).shape
    height, width = input_shape[2], input_shape[3]
    input_image = cv2.resize(frame, (width, height))
    input_image = input_image.transpose((2, 0, 1))
    input_image = np.expand_dims(input_image, axis=0)

    with inference_lock:
        face_detection_result = compiled_face_detection_model([input_image])

    detections = face_detection_result[compiled_face_detection_model.output(0)]

    for detection in detections[0][0]:
        confidence = detection[2]
        if confidence > 0.3:
            x1, y1, x2, y2 = (detection[3:7] * np.array([frame_width, frame_height, frame_width, frame_height])).astype(int)
            face = frame[y1:y2, x1:x2]
            input_face = cv2.resize(face, (62, 62))
            input_face = input_face.transpose((2, 0, 1))
            input_face = np.expand_dims(input_face, axis=0)

            with inference_lock:
                age_gender_result = compiled_age_gender_model([input_face])

            gender = "Male" if age_gender_result[compiled_age_gender_model.output("prob")][0][0][0] < 0.7 else "Female"
            age = age_gender_result[compiled_age_gender_model.output("age_conv3")][0][0][0][0] * 100
            text = f"Age: {age:.1f}, {gender}"
            cv2.rectangle(image_with_info, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(image_with_info, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    return image_with_info

# Generator function to generate video frames
def generate_frames():
    global video_cap, out, frame_width, frame_height, recent_age_predictions
    while True:
        if video_cap is None:
            break

        ret, frame = video_cap.read()
        if not ret:
            break

        input_shape = compiled_face_detection_model.input(0).shape
        height, width = input_shape[2], input_shape[3]
        input_image = cv2.resize(frame, (width, height))
        input_image = input_image.transpose((2, 0, 1))
        input_image = np.expand_dims(input_image, axis=0)

        with inference_lock:
            face_detection_result = compiled_face_detection_model([input_image])

        detections = face_detection_result[compiled_face_detection_model.output(0)]

        for detection in detections[0][0]:
            confidence = detection[2]
            if confidence > 0.3:
                x1, y1, x2, y2 = (detection[3:7]* np.array([frame.shape[1], frame.shape[0], frame.shape[1], frame.shape[0]])).astype(int)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                face = frame[y1:y2, x1:x2]

                if face.size:
                    input_face = cv2.resize(face, (62, 62))
                    input_face = input_face.transpose((2, 0, 1))
                    input_face = np.expand_dims(input_face, axis=0)

                    with inference_lock:
                        age_gender_result = compiled_age_gender_model([input_face])

                    gender = "Male" if age_gender_result[compiled_age_gender_model.output("prob")][0][0][0] < 0.5 else "Female"
                    age = age_gender_result[compiled_age_gender_model.output("age_conv3")][0][0][0][0] * 100

                    # Add predicted age to the list of recent age predictions
                    recent_age_predictions.append(age)

                    # Limit the number of recent age predictions to avoid excessive computation
                    MAX_RECENT_AGE_PREDICTIONS = 10
                    recent_age_predictions = recent_age_predictions[-MAX_RECENT_AGE_PREDICTIONS:]

                    # Compute the average of recent age predictions
                    average_age_prediction = np.mean(recent_age_predictions)

                    # Round the predicted age
                    rounded_age_prediction = round(average_age_prediction)

                    text = f"Age: {rounded_age_prediction}, {gender}"
                    cv2.putText(frame, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        out.write(frame)
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        

# Index route to render the HTML template
@app.route('/')
def index():
    return render_template('index.html')

# Route to handle video upload
@app.route('/upload_video', methods=['POST'])
def upload_video():
    global video_cap, out, output_path, frame_width, frame_height
    if 'video_file' not in request.files:
        return redirect(request.url)

    video_file = request.files['video_file']

    if video_file.filename == '':
        return redirect(request.url)

    if video_file:
        video_file_path = os.path.join('uploads', video_file.filename)
        video_file.save(video_file_path)

        if video_cap is not None:
            video_cap.release()
        if out is not None:
            out.release()

        video_cap = cv2.VideoCapture(video_file_path)
        frame_width = int(video_cap.get(3))
        frame_height = int(video_cap.get(4))
        frame_fps = int(video_cap.get(5))
        output_video_path = video_file_path.replace(".mp4", "_output.avi") 
        out = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*'XVID'), frame_fps, (frame_width, frame_height))
        output_path = output_video_path

    return redirect(url_for('index'))

# Route to provide video feed
@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

# Route to download the processed video
@app.route('/download_video')
def download_video():
    global output_path
    if output_path is not None:
        return send_file(output_path, as_attachment=True, mimetype='video/x-msvideo')
    else:
        return "No video to download"

# Route to provide video feed from the camera
@app.route('/video_feed_camera')
def video_feed_camera():
    return Response(generate_camera_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

# Generator function to generate frames from the camera
def generate_camera_frames():
    global camera_cap, frame_width, frame_height
    while True:
        if camera_cap is not None:
            success, frame = camera_cap.read()
            if not success:
                break
            detect_age_gender(frame)
            ret, buffer = cv2.imencode('.jpg', frame)
            if not ret:
                continue
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

# Function to start the camera
def start_camera():
    global camera_cap, frame_width, frame_height
    if camera_cap is None:
        camera_cap = cv2.VideoCapture(0)
        frame_width = int(camera_cap.get(3))
        frame_height = int(camera_cap.get(4))

# Function to stop the camera
def stop_camera():
    global camera_cap
    if camera_cap is not None:
        camera_cap.release()
        camera_cap = None

# Route to start the camera
@app.route('/start_camera')
def start_camera_route():
    start_camera()
    return redirect(url_for('index'))

# Route to stop the camera
@app.route('/stop_camera')
def stop_camera_route():
    stop_camera()
    return redirect(url_for('index'))

# Global variable to store recent age predictions
recent_age_predictions = []

# Function to detect age and gender from a frame
def detect_age_gender(frame):
    global frame_width, frame_height, recent_age_predictions
    input_shape = compiled_face_detection_model.input(0).shape
    height, width = input_shape[2], input_shape[3]
    input_image = cv2.resize(frame, (width, height))
    input_image = input_image.transpose((2, 0, 1))
    input_image = np.expand_dims(input_image, axis=0)

    with inference_lock:
        face_detection_result = compiled_face_detection_model([input_image])

    detections = face_detection_result[compiled_face_detection_model.output(0)]

    for detection in detections[0][0]:
        confidence = detection[2]
        if confidence > 0.3:
            x1, y1, x2, y2 = (detection[3:7] * np.array([frame_width, frame_height, frame_width, frame_height])).astype(int)
            face = frame[y1:y2, x1:x2]
            
            # Check the size of the image before resizing
            if not face.size:
               
                continue  # Skip if no face is detected

            input_face = cv2.resize(face, (62, 62))
            input_face = input_face.transpose((2, 0, 1))
            input_face = np.expand_dims(input_face, axis=0)

            with inference_lock:
                age_gender_result = compiled_age_gender_model([input_face])

            gender = "Male" if age_gender_result[compiled_age_gender_model.output("prob")][0][0][0] < 0.7 else "Female"
            age = age_gender_result[compiled_age_gender_model.output("age_conv3")][0][0][0][0] * 100

            # Add predicted age to the list of recent age predictions
            recent_age_predictions.append(age)

            # Limit the number of recent age predictions to avoid excessive computation
            MAX_RECENT_AGE_PREDICTIONS = 10
            recent_age_predictions = recent_age_predictions[-MAX_RECENT_AGE_PREDICTIONS:]

            # Compute the average of recent age predictions
            average_age_prediction = np.mean(recent_age_predictions)

            # Round the predicted age
            rounded_age_prediction = round(average_age_prediction)

            text = f"Age: {rounded_age_prediction}, {gender}"
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)


if __name__ == "__main__":
    os.makedirs('uploads', exist_ok=True)
    app.run(debug=True)
