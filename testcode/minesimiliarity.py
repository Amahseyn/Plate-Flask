
from apscheduler.schedulers.background import BackgroundScheduler
import psycopg2
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
import requests
from datetime import datetime, timedelta
from collections import defaultdict
warnings.filterwarnings("ignore", category=FutureWarning)
params = Parameters()
from flask_cors import CORS, cross_origin
from datetime import datetime, timedelta

video_lock = threading.Lock()
frame = None 

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
dirpath = os.getcwd()
images_dir = 'images'
raw_images_dir = os.path.join(images_dir, 'raw')
plate_images_dir = os.path.join(images_dir, 'plate')


def detectPlateChars(croppedPlate):
    """Detect characters on a cropped plate."""
    chars, englishchars, confidences = [], [], []
    results = modelCharX(croppedPlate)
    detections = results.pred[0]
    detections = sorted(detections, key=lambda x: x[0])  # Sort by x coordinate
    clses = []
    for det in detections:
        conf = det[4]
        
        if conf > 0.5:
            cls = int(det[5].item())  # Ensure cls is an integer
            clses.append(int(cls))
            char = params.char_id_dict.get(str(int(cls)), '')  # Get character or empty string
            englishchar = params.char_id_dict1.get(str(int(cls)), '')  # Get English character or empty string
            chars.append(char)
            englishchars.append(englishchar)
            confidences.append(conf.item())
    state= False
    if len(chars)==8:
        if 10<=clses[2]<=42:
            for i in [0,1,3,4,5,6,7]:
                if clses[i]<10:
                    state = True


    # If conditions are not met, maintain the same return structure
    return state, chars, englishchars, confidences
last_detection_time = {}
recent_detections = []

def get_similarity_score(plate1, plate2):
    """Compute similarity score between two plates."""
    return sum(c1 == c2 for c1, c2 in zip(plate1, plate2))

def process_frame(img, cameraId):
    global last_char_display, last_detection_time, recent_detections
    tick = time.time()
    
    results = model_object(img, conf=0.7, stream=True)

    for detection in results:
        bbox = detection.boxes
        for box in bbox:
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            plate_img = img[y1:y2, x1:x2]

            cls_names = int(box.cls[0])
            if cls_names == 1:
                state, chars, englishchars, charConfAvg = detectPlateChars(plate_img)
                if not state or len(englishchars) != 8:
                    continue

                current_plate = ''.join(englishchars)
                current_time = datetime.now()

                # Clean up old detections
                recent_detections = [
                    entry for entry in recent_detections
                    if (current_time - entry['timestamp']).total_seconds() <= 10
                ]

                # Check for similar plates in the log
                most_similar_plate = None
                max_similarity = 0

                for entry in recent_detections:
                    similarity = get_similarity_score(current_plate, entry['plate'])
                    if similarity > max_similarity:
                        max_similarity = similarity
                        most_similar_plate = entry

                if max_similarity >= 6:
                    # Update the count for the similar plate
                    most_similar_plate['count'] += 1
                    most_similar_plate['timestamp'] = current_time
                else:
                    # Add the current plate to the log
                    recent_detections.append({
                        'plate': current_plate,
                        'timestamp': current_time,
                        'count': 1
                    })

                # Find the plate with the highest count
                best_plate = max(recent_detections, key=lambda x: x['count'])
                if best_plate['plate'] == current_plate:
                    # Save the detected plate to the database
                    english_output = f"{current_plate[:2]}-{current_plate[2]}-{current_plate[3:6]}-{current_plate[6:]}"
                    persian_output = f"{''.join(chars[:2])}-{chars[2]}-{''.join(chars[3:6])}-{''.join(chars[6:])}"
                    
                    # Save to the database and emit the result (same as in your original logic)
                    save_plate_to_database(img, plate_img, english_output, persian_output, cameraId, current_time)

def save_plate_to_database(img, plate_img, english_output, persian_output, cameraId, current_time):
    """Helper function to save the plate data to the database."""
    timestamp = current_time.strftime("%Y-%m-%d-%H-%M-%S")
    raw_filename = f"raw_{timestamp}.jpg"
    plate_filename = f"plt_{timestamp}.jpg"
    
    raw_path = os.path.join('static/images/raw', raw_filename)
    plate_path = os.path.join('static/images/plate', plate_filename)
    
    os.makedirs(os.path.dirname(raw_path), exist_ok=True)
    os.makedirs(os.path.dirname(plate_path), exist_ok=True)

    cv2.imwrite(raw_path, img)
    cv2.imwrite(plate_path, plate_img)

    raw_url = f"http://localhost:5000/static/images/raw/{raw_filename}"
    plate_url = f"http://localhost:5000/static/images/plate/{plate_filename}"

    conn = psycopg2.connect(dbname=DB_NAME, user=DB_USER, password=DB_PASSWORD, host=DB_HOST, port=DB_PORT)
    cursor = conn.cursor()
    cursor.execute(
        """
        INSERT INTO plates (date, raw_image_path, plate_cropped_image_path, predicted_string, camera_id)
        VALUES (%s, %s, %s, %s, %s)
        RETURNING id
        """,
        (timestamp, raw_url, plate_url, english_output, cameraId)
    )
    print(i)
    plate_id = cursor.fetchone()[0]
    conn.commit()
    cursor.close()
    conn.close()

    try:
        data = {
            "id": plate_id,  
            "date": timestamp,
            "raw_image_path": raw_url,
            "plate_cropped_image_path": plate_url,
            "predicted_string": english_output,
            "cameraid": cameraId
        }
        socketio.emit('plate_detected', data)
        print(f"Data emitted via SocketIO with ID: {plate_id}")
    except Exception as e:
        print(f"Error emitting data: {e}")


def video_feed(cameraId):
        global frame
        try:
            conn = psycopg2.connect(dbname=DB_NAME, user=DB_USER, password=DB_PASSWORD, host=DB_HOST, port=DB_PORT)
            cursor = conn.cursor()
            cursor.execute("SELECT cameralink FROM cameras WHERE id = %s", (cameraId,))
            camera_link = cursor.fetchone()
            cursor.execute("SELECT mine_id FROM mine_info WHERE cameraid = %s",(str(cameraId),))
            mine_id = cursor.fetchone()
            if camera_link is None:
                print("error: Camera not found, 404")

            camera_link = camera_link[0]  
            camera_link = "a09.mp4"
            cap = cv2.VideoCapture(camera_link)
            width, height = 1280, 720
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = 10
            cap.set(cv2.CAP_PROP_FPS, fps)
            if not cap.isOpened():
                print(f"Failed to open video stream from {camera_link}")
            while True:
                ret, img = cap.read()
                if not ret:
                    print("No frames read. Exiting...")
                    break
                img = cv2.resize(img,(width,height))
                processed_frame = process_frame(img,cameraId,mine_id)

                with lock:
                    frame = processed_frame
                time.sleep(0.05)
            #cap.release()
        except Exception as e:
            return print(f"error: {str(e)}, 500")
        finally:
            if conn:
                cursor.close()
                conn.close()

def get_db_connection():
    conn = psycopg2.connect(dbname=DB_NAME, user=DB_USER, password=DB_PASSWORD, host=DB_HOST, port=DB_PORT)
    return conn

# Function to send data to another server
# def send_data_to_server():
#     try:
#         conn = get_db_connection()
#         cursor = conn.cursor()
#         cursor.execute("SELECT * FROM plates WHERE sent = FALSE LIMIT 10;")  # Get unsent records
#         plates = cursor.fetchall()

#         if plates:
#             for plate in plates:
#                 plate_id, date, raw_image, plate_image, predicted_string, camera_id,_,valid = plate
#                 data = {
#                     "id": plate_id,
#                     "date": date,
#                     "raw_image": raw_image,
#                     "plate_image": plate_image,
#                     "predicted_string": predicted_string,
#                     "camera_id": camera_id,
#                     'permit':valid
#                 }
                
#                 success = False
#                 while not success:
#                     try:
#                         response = requests.post("http://127.0.0.1:5000/process_data", json=data)
#                         if response.status_code == 200:
#                             success = True
#                             cursor.execute("UPDATE plates SET sent = TRUE WHERE id = %s", (plate_id,))
#                             conn.commit()
#                         else:
#                             # If the response is not OK, wait and retry
#                             print(f"Error: Received status code {response.status_code} for plate {plate_id}. Retrying...")
#                             time.sleep(10)  # Wait 10 seconds before retrying
#                     except requests.exceptions.RequestException as e:
#                         print(f"Error sending data: {e}. Retrying...")
#                         time.sleep(10)  # Wait 10 seconds before retrying
                
#         cursor.close()
#         conn.close()
    
#     except Exception as e:
#         print(f"Error in sending data to server: {e}")

# Function to schedule sending data every 5 minutes
def schedule_data_transfer():
    scheduler = BackgroundScheduler()
    scheduler.add_job(send_data_to_server, 'interval', minutes=1)
    scheduler.start()
def check_vehicle_permit(license_plate, mine_id):
    """
    Check if a vehicle with a specific license plate has a valid permit for a given mine.

    Parameters:
        license_plate (str): The license plate of the vehicle.
        mine_id (int): The ID of the mine.

    Returns:
        bool: True if the permit is valid, False otherwise.
    """
    try:
        cursor.execute("""
            SELECT vehicle_id FROM vehicle_info WHERE license_plate = %s
        """, (license_plate,))
        vehicle_result = cursor.fetchone()

        if not vehicle_result:
            print(f"Vehicle with license plate '{license_plate}' not found.")
            return False
        
        vehicle_id = vehicle_result[0]

        cursor.execute("""
            SELECT start_date, end_date FROM vehicle_permit
            WHERE vehicle_id = %s AND mine_id = %s
        """, (vehicle_id, mine_id))
        permit_result = cursor.fetchone()

        if not permit_result:
            print(f"No permit found for vehicle '{license_plate}' at mine ID '{mine_id}'.")
            return False

        start_date, end_date = permit_result
        current_date = datetime.now().date()

        if datetime.strptime(start_date, "%Y-%m-%d").date() <= current_date <= datetime.strptime(end_date, "%Y-%m-%d").date():
            print(f"Vehicle '{license_plate}' has a valid permit for mine ID '{mine_id}'.")
            return True
        else:
            print(f"Permit for vehicle '{license_plate}' at mine ID '{mine_id}' is not valid.")
            return False

    except Exception as e:
        print(f"Error: {e}")
        return False

def main():
    camera_id = 1  # Example camera ID

    video_thread = threading.Thread(target=video_feed, args=(camera_id,))
    video_thread.start()

    schedule_data_transfer()
    try:
        while True:
            time.sleep(.05)
    except KeyboardInterrupt:
        print("Stopping threads and scheduler.")
        video_thread.join()

if __name__ == "__main__":
    main()