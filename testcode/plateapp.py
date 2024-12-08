
from flask import Flask, Response, send_file, jsonify
import cv2
import threading
from ultralytics import YOLO
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import time
import os
import torch
from configParams import Parameters
import datetime
import psycopg2
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
params = Parameters()
from flask_cors import CORS,cross_origin


app = Flask(__name__)
CORS(app) 
DB_NAME = "license_plate_db"
DB_USER = "postgres"
DB_PASSWORD = "m102030m"
DB_HOST = "localhost"
DB_PORT = "5432"
video_capture = None
frame = None
lock = threading.Lock()

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")
model_object = YOLO("weights/best.pt")
modelCharX = torch.hub.load('yolov5', 'custom', "model/CharsYolo.pt", source='local', force_reload=True)

font_path = "vazir.ttf"  
persian_font = ImageFont.truetype(font_path, 20)

database_path = 'database/license_plate.db'
images_dir = 'images'
raw_images_dir = os.path.join(images_dir,'raw')
plate_images_dir = os.path.join(images_dir,'plate')

if os.path.exists(database_path):
    os.makedirs(images_dir,exist_ok=True)
    os.makedirs(raw_images_dir, exist_ok=True)
    os.makedirs(plate_images_dir, exist_ok=True)

def detectPlateChars(croppedPlate):
    """Detect characters on a cropped plate."""
    chars, confidences = [], []
    results = modelCharX(croppedPlate)
    detections = results.pred[0]
    detections = sorted(detections, key=lambda x: x[0])  # Sort by x coordinate
    for det in detections:
        conf = det[4]
        if conf > 0.5:
            cls = det[5].item()
            char = params.char_id_dict.get(str(int(cls)), '')
            chars.append(char)
            confidences.append(conf.item())
    charConfAvg = 0
    return chars, charConfAvg

def process_frame(img):
    tick = time.time()
    global last_char_display
    conn = sqlite3.connect(database_path)
    cursor = conn.cursor()
    
    results = model_object(img, conf=0.7, stream=True)

    for detection in results:
        bbox = detection.boxes
        for box in bbox:
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            plate_img = img[y1:y2, x1:x2]

            cls_names = int(box.cls[0])
            if cls_names == 1:
                chars, charConfAvg = detectPlateChars(plate_img)
                char_display = []
                if len(chars) == 8:
                    for english_char in chars:
                        char_display.append(english_char)

                    current_char_display = ''.join(char_display)
                    if current_char_display != last_char_display:
                        timestamp = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
                        raw_path = f"{raw_images_dir}/raw_{timestamp}.jpg"
                        plt_path = f"{plate_images_dir}/plt_{timestamp}.jpg"
                        cv2.imwrite(raw_path, img)
                        cv2.imwrite(plt_path, plate_img)

                        english_output = f"{char_display[6]}{char_display[7]}-{char_display[3]}{char_display[4]}{char_display[5]}-{char_display[2]}-{char_display[0]}{char_display[1]}"
                        cursor.execute(
                            "INSERT INTO plates (date, raw_image_path, plate_cropped_image_path, predicted_string) VALUES (?, ?, ?, ?)",
                            (timestamp, raw_path, plt_path, english_output)
                        )
                        conn.commit()

                        last_char_display = current_char_display

                    persian_output = f"{char_display[0]}{char_display[1]}-{char_display[2]}-{char_display[3]}{char_display[4]}{char_display[5]}-{char_display[6]}{char_display[7]}"
                    img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                    draw = ImageDraw.Draw(img_pil)
                    draw.text((x1, y1 - 30), persian_output, font=persian_font, fill=(255, 0, 0))
                    img = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

    tock = time.time()
    elapsed_time = tock - tick
    fps_text = "FPS: {:.2f}".format(1 / elapsed_time)
    fps_text_loc = (0, 50)
    cv2.putText(img, fps_text, fps_text_loc, fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(10, 50, 255), thickness=2, lineType=cv2.LINE_AA)

    conn.close()
    return img

# Function to process video
def process_video():
    global frame
    global last_char_display
    last_char_display = ''
    cap =cv2.VideoCapture("1.mp4")
    #cap = cv2.VideoCapture("rtsp://admin:Admin2020@@192.168.1.13")  # Replace with your video source or RTSP stream
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    output_videoname = f'videos/output_{timestamp}.mp4'
    os.makedirs("videos",exist_ok=True)
    width, height = 1280, 720
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    fps = 15
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter(output_videoname, fourcc, fps, (frame_width, frame_height))

    if not cap.isOpened():
        print("Failed to open video source.")
        return

    while cap.isOpened():
        ret, img = cap.read()
        if not ret:
            print("No frames read. Exiting...")
            break

        processed_frame = process_frame(img)

        out.write(processed_frame)

        with lock:
            frame = processed_frame

    cap.release()
    out.release()

cameraId = 1
streamendpoint = f'/camera/{cameraId}/stream'
@app.route(streamendpoint, methods=['GET'])
@cross_origin(supports_credentials=True)
def video_feed():
    def generate():
        global frame
        while True:
            with lock:
                if frame is None:
                    continue
                _, buffer = cv2.imencode('.jpg', frame)
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
            time.sleep(0.05) 

    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/cameras', methods=['GET'])
@cross_origin(supports_credentials=True)
def get_cameras():
    """
    Fetch and return all camera information from the 'cameras' table.
    """
    try:
        conn = psycopg2.connect(dbname=DB_NAME, user=DB_USER, password=DB_PASSWORD, host=DB_HOST, port=DB_PORT)
        cursor = conn.cursor()

        # Query the cameras table
        cursor.execute("SELECT cameraname, cameralocation, cameralink FROM cameras")
        cameras = cursor.fetchall()

        # Format the results
        cameras_list = [
            {"cameraname": row[0], "cameralocation": row[1], "cameralink": row[2]} for row in cameras
        ]

        return jsonify(cameras_list), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        if conn:
            cursor.close()
            conn.close()

if __name__ == '__main__':
    threading.Thread(target=process_video, daemon=True).start()
    app.run(port=5000)
