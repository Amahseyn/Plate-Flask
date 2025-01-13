import eventlet
eventlet.monkey_patch()
from flask import Flask, Response, send_file, jsonify,send_from_directory
from psycopg2.extras import RealDictCursor
from readsensor import *
from checks import *
from torchvision import transforms
from psycopg2 import sql, OperationalError, DatabaseError
from flask import request, jsonify, send_file
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
from datetime import datetime, timedelta
import datetime
import psycopg2
import random
import redis
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
params = Parameters()
from flask_cors import CORS, cross_origin
from datetime import datetime, timedelta
from flask_socketio import SocketIO, emit
torch.cuda.empty_cache()
check_gps_port()
app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}}, supports_credentials=True)
socketio = SocketIO(app, cors_allowed_origins="*")

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
model_object = YOLO("weights/best.pt",verbose = False).to(device)
modelCharX = torch.hub.load('yolov5', 'custom', "model/CharsYolo.pt", source='local', force_reload=True,verbose = False).to(device)

font_path = "vazir.ttf"
persian_font = ImageFont.truetype(font_path, 20)
dirpath = os.getcwd()
images_dir = 'images'
raw_images_dir = os.path.join(images_dir, 'raw')
plate_images_dir = os.path.join(images_dir, 'plate')
redis_client = redis.Redis(host='localhost', port=6379, db=0)
redis_client = redis.StrictRedis(host='localhost', port=6379, db=0)

def detectPlateChars(croppedPlate):
    """Detect characters on a cropped plate."""
    chars, englishchars, confidences = [], [], []
    results = modelCharX(croppedPlate)
    detections = results.pred[0]
    detections = sorted(detections, key=lambda x: x[0])
    clses = []
    for det in detections:
        conf = det[4]
        
        if conf > 0.5:
            cls = int(det[5].item())
            clses.append(int(cls))
            char = params.char_id_dict.get(str(int(cls)), '')
            englishchar = params.char_id_dict1.get(str(int(cls)), '')  
            chars.append(char)
            englishchars.append(englishchar)
            confidences.append(conf.item())
    state= False
    if len(chars)==8:
        if 10<=clses[2]<=42:
            state = True
            for i in [0,1,3,4,5,6,7]:
                if clses[i] >= 10: 
                    state = False
                    break
    return state, chars, englishchars, confidences




last_detection_time = {}
last_plate = None
repeat_threshold = 3  # Number of times a plate must be detected before adding to the database


# Initialize Redis connection
k = None
checktimedup = 5
repeateplate = 2
repeateplate -=1 
def process_frame(img, cameraId, conn, cursor, frame):
    with app.app_context():
        insertion=None
        global last_plate, plate_counter,k
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
                    char_display = [char for char in chars]
                    englishchardisplay = [char for char in englishchars]

                    if state:
                        englishoutput = f"{englishchardisplay[0]}{englishchardisplay[1]}-{englishchardisplay[2]}-{englishchardisplay[3]}{englishchardisplay[4]}{englishchardisplay[5]}-{englishchardisplay[6]}{englishchardisplay[7]}"
                        persian_output = f"{char_display[0]}{char_display[1]}-{char_display[2]}-{char_display[3]}{char_display[4]}{char_display[5]}-{char_display[6]}{char_display[7]}"
                        # Check Redis for recent plate detection
                        
                        if last_plate ==englishoutput:
                            k+=1
                            insertion=True
                            print("0")
                        else:
                            print("1")
                            last_plate= englishoutput
                            k = 0
                            last_seen = redis_client.get(englishoutput)
                            if last_seen:
                                print("2")
                                last_seen_time = datetime.fromisoformat(last_seen.decode('utf-8'))
                                diff = datetime.now() - last_seen_time
                                if diff < timedelta(minutes=checktimedup):
                                    print("3")
                                    if diff < timedelta(minutes=1):
                                        txt = f"Detected {int(diff.total_seconds())} seconds before"
                                    else:
                                        txt = f"Detected {int(diff.total_seconds() // 60)} minute(s) before"
                                    cv2.putText(img, txt, (0, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (10, 50, 255), 2) 
                                    cv2.rectangle(img, (x1, y1), (x2, y2), color=(0, 215, 255), thickness=1)
                                    img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                                    draw = ImageDraw.Draw(img_pil)
                                    draw.text((x1, y1 - 30), persian_output, font=persian_font, fill=(255, 0, 0))
                                    img = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

                                    insertion= False
                        if insertion==True and k >repeateplate:
                            txt = "Detected before"
                            cv2.putText(img, txt, (x1, y1-50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2) 
                            cv2.rectangle(img, (x1, y1), (x2, y2), color=(0, 215, 255), thickness=1)
                            img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                            draw = ImageDraw.Draw(img_pil)
                            draw.text((x1, y1 - 30), persian_output, font=persian_font, fill=(255, 0, 0))
                            img = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

                        if insertion==True and k == repeateplate:
                                value = datetime.now().isoformat()
                                redis_client.set(englishoutput, value)
                                redis_client.setex(englishoutput, int(checktimedup*60), value)
                                cv2.rectangle(img, (x1, y1), (x2, y2), color=(0, 215, 255), thickness=1)
                                img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                                draw = ImageDraw.Draw(img_pil)
                                draw.text((x1, y1 - 30), persian_output, font=persian_font, fill=(255, 0, 0))
                                img = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
                                timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
                                random_number = random.randint(1, 1000)
                                raw_filename = f"raw_{timestamp}_{random_number}.jpg"
                                plate_filename = f"plt_{timestamp}-{random_number}.jpg"

                                raw_url = f"static/images/raw/{raw_filename}"
                                plate_url = f"static/images/plate/{plate_filename}"

                                x1 = int(x1 * frame.shape[1] / 640)
                                x2 = int(x2 * frame.shape[1] / 640)
                                y1 = int(y1 * frame.shape[0] / 400)
                                y2 = int(y2 * frame.shape[0] / 400)
                                plate_img = frame[y1:y2, x1:x2]

                                cv2.imwrite(raw_url, img)
                                cv2.imwrite(plate_url, plate_img)

                                cursor.execute(
                                    """
                                    INSERT INTO plates (date, raw_image_path, plate_cropped_image_path, predicted_string, camera_id)
                                    VALUES (%s, %s, %s, %s, %s)
                                    RETURNING id
                                    """,
                                    (timestamp, raw_url, plate_url, englishoutput, cameraId)
                                )
                                conn.commit()
                                plate_id = cursor.fetchone()[0]
                                try:
                                    evenodd = 0
                                    try:
                                        if int(char_display[-4]) % 2 == 0:
                                            evenodd = 1
                                    except:
                                        evenodd = 0
                                    data = {
                                        "id": str(plate_id),
                                        "date": str(timestamp),
                                        "raw_image_path": str(raw_url),
                                        "plate_cropped_image_path": str(plate_url),
                                        "predicted_string": str(englishoutput),
                                        "cameraid": str(cameraId),
                                        "evenodd": str(evenodd)
                                    }
                                    socketio.emit('plate_detected', data)
                                    print(f"Data emitted via SocketIO with ID: {plate_id}")
                                except Exception as e:
                                    print(f"Error emitting data: {e}")
        fps_text = f"FPS: {1 / (time.time() - tick):.2f}"
        cv2.putText(img, fps_text, (0, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (10, 50, 255), 2)
        return img

@app.route('/camera/<int:cameraId>/<int:mod>/stream', methods=['GET'])
@cross_origin(supports_credentials=True)
def video_feed(cameraId, mod):
    def generate():
        global frame
        conn = None
        cap = None
        print(f"State is {mod}, Camera ID is {cameraId}")
        vis = (mod == 1)  
        reload_interval = 5 * 60  
        start_time = time.time()

        try:
            if conn==False or conn==None:
                conn = psycopg2.connect(dbname=DB_NAME, user=DB_USER, password=DB_PASSWORD, host=DB_HOST, port=DB_PORT)
                cursor = conn.cursor()

            # Fetch the camera link from the database
            cursor.execute("SELECT cameralink FROM cameras WHERE id = %s", (cameraId,))
            camera_link = cursor.fetchone()

            if camera_link is None:
                return jsonify({"error": "Camera not found"}), 404
            else:
                camera_link = camera_link[0]

            cap = cv2.VideoCapture("a09.mp4")

            if not cap.isOpened():
                print(f"Failed to open video stream from {camera_link}")
                return jsonify({"error": "Failed to open camera stream"}), 500

            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

            while cap.isOpened():
                if time.time() - start_time > reload_interval:
                    print("Reloading video stream...")
                    cap.release()
                    cap = cv2.VideoCapture(camera_link)
                    start_time = time.time()

                ret, frame = cap.read()
                if not ret:
                    print("No frames read. Exiting...")
                    break

                img = cv2.resize(frame, (640, 400))
                processed_frame = process_frame(img, cameraId,conn,cursor,frame) if vis else img

                with lock:
                    frame = processed_frame

                _, buffer = cv2.imencode('.jpg', frame)
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
                time.sleep(0.01)
        except Exception as e:
            print(f"Error: {str(e)}")
            return jsonify({"error": str(e)}), 500

        finally:
            if cap:
                cap.release()

            if conn:
                cursor.close()
                conn.close()
            print("Resources released successfully.")

    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')
@socketio.on('connect')
def handle_connect():
    print("Client connected")

@socketio.on('message')
def handle_message(data):
    print(f"Message received: {data}")
    # Optionally, you can emit a response back
    emit('response', {'status': 'Message received'})
@app.route('/camera_sort', methods=['POST'])
@cross_origin(supports_credentials=True)
def update_camera_ids():
    from flask import request, jsonify
    conn = None
    try:
        data = request.get_json()

        # Ensure the 'camera_ids' field is present in the request
        if 'camera_ids' not in data:
            return jsonify({"error": "Missing required field: camera_ids"}), 400

        camera_ids = list(map(int, data['camera_ids'].split(';')))

        # Ensure IDs are unique and within range
        if len(set(camera_ids)) != len(camera_ids) or len(set(camera_ids)) > 4:
            return jsonify({"error": "Camera IDs must be unique and no more than 4"}), 400

        conn = psycopg2.connect(dbname=DB_NAME, user=DB_USER, password=DB_PASSWORD, host=DB_HOST, port=DB_PORT)
        cursor = conn.cursor()

        # Fetch all existing camera IDs
        cursor.execute("SELECT id FROM cameras ORDER BY id")
        existing_ids = [row[0] for row in cursor.fetchall()]

        # Ensure the number of provided IDs matches the number of records
        if len(existing_ids) != len(camera_ids):
            return jsonify({"error": "Mismatch between the number of cameras and provided IDs"}), 400

        # Step 1: Assign temporary IDs to avoid conflict
        temp_ids = [id + 100 for id in existing_ids]
        for old_id, temp_id in zip(existing_ids, temp_ids):
            cursor.execute("UPDATE cameras SET id = %s WHERE id = %s", (temp_id, old_id))

        # Step 2: Assign the final desired IDs
        for temp_id, new_id in zip(temp_ids, camera_ids):
            cursor.execute("UPDATE cameras SET id = %s WHERE id = %s", (new_id, temp_id))

        # Commit the transaction
        conn.commit()

        # Fetch all updated data to return as a list of dictionaries
        cursor.execute("SELECT id, cameraname, cameralocation, cameralink FROM cameras ORDER BY id")
        columns = [desc[0] for desc in cursor.description]
        updated_cameras = [dict(zip(columns, row)) for row in cursor.fetchall()]

        return jsonify({"message": "Camera IDs updated successfully", "cameras": updated_cameras}), 200

    except Exception as e:
        conn.rollback()
        return jsonify({"error": str(e)}), 500

    finally:
        if conn:
            cursor.close()
            conn.close()

@app.route('/cameras', methods=['GET'])
@cross_origin(supports_credentials=True)
def get_cameras():
    try:
        conn = psycopg2.connect(dbname=DB_NAME, user=DB_USER, password=DB_PASSWORD, host=DB_HOST, port=DB_PORT)
        cursor = conn.cursor()

        cursor.execute("SELECT cameraname, cameralocation, cameralink ,id FROM cameras")
        cameras = cursor.fetchall()


        cameras_list = [
            {"cameraname": row[0], "cameralocation": row[1], "cameralink": row[2],"cameraid":row[3]} for row in cameras
        ]
        return jsonify(cameras_list), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        if conn:
            cursor.close()
            conn.close()
@app.route('/camera', methods=['POST'])
@cross_origin(supports_credentials=True)
def add_camera():
    from flask import request
    conn = None
    try:
        data = request.get_json()
        required_fields = ['cameraname', 'cameralocation', "cameralink"]
        for field in required_fields:
            if field not in data:
                return jsonify({"error": f"Missing required field: {field}"}), 400

        cameraname = data['cameraname']
        cameralocation = data['cameralocation']
        cameralink = data['cameralink']
        print(cameraname)

        conn = psycopg2.connect(dbname=DB_NAME, user=DB_USER, password=DB_PASSWORD, host=DB_HOST, port=DB_PORT)
        cursor = conn.cursor()

        cursor.execute(
            "INSERT INTO cameras (cameraname, cameralocation, cameralink) VALUES (%s, %s, %s)",
            (cameraname, cameralocation, cameralink)
        )
        conn.commit()

        return jsonify({"message": "Camera added successfully"}), 201

    except Exception as e:
        return jsonify({"error": str(e)}), 500

    finally:
        if conn:
            cursor.close()
            conn.close()


@app.route('/camera/<int:cameraId>', methods=['PUT'])
@cross_origin(supports_credentials=True)
def update_camera(cameraId):
    from flask import request
    conn = None
    try:
        data = request.get_json()
        conn = psycopg2.connect(dbname=DB_NAME, user=DB_USER, password=DB_PASSWORD, host=DB_HOST, port=DB_PORT)
        cursor = conn.cursor()

        if 'cameraname' in data:
            cursor.execute(
                "UPDATE cameras SET cameraname = %s WHERE id = %s",
                (data['cameraname'], cameraId)
            )
        if 'cameralocation' in data:
            cursor.execute(
                "UPDATE cameras SET cameralocation = %s WHERE id = %s",
                (data['cameralocation'], cameraId)
            )
        if 'cameralink' in data:
            cursor.execute(
                "UPDATE cameras SET cameralink = %s WHERE id = %s",
                (data['cameralink'], cameraId)
            )

        conn.commit()

        return jsonify({"message": "Camera updated successfully"}), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        if conn:
            cursor.close()
            conn.close()


@app.route('/camera/<int:cameraId>', methods=['DELETE'])
@cross_origin(supports_credentials=True)
def delete_camera(cameraId):
    conn = None
    try:
        conn = psycopg2.connect(dbname=DB_NAME, user=DB_USER, password=DB_PASSWORD, host=DB_HOST, port=DB_PORT)
        cursor = conn.cursor()

        cursor.execute("DELETE FROM cameras WHERE id = %s", (cameraId,))
        conn.commit()

        return jsonify({"message": "Camera deleted successfully"}), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        if conn:
            cursor.close()
            conn.close()
@app.route('/camera/<int:id>', methods=['PATCH'])
@cross_origin(supports_credentials=True)
def patch_camera(id):
    conn = None
    try:
        # Extract data from the incoming request
        data = request.get_json()
        
        # Fields to update - only update the fields that are provided
        cameraname = data.get('cameraname', None)
        cameralocation = data.get('cameralocation', None)
        cameralink = data.get('cameralink', None)

        # Connect to the database
        conn = psycopg2.connect(dbname=DB_NAME, user=DB_USER, password=DB_PASSWORD, host=DB_HOST, port=DB_PORT)

        cursor = conn.cursor()

        # Check if the camera exists
        cursor.execute("SELECT * FROM cameras WHERE id = %s", (id,))
        camera = cursor.fetchone()

        if not camera:
            return jsonify({"error": "Camera not found"}), 404

        # Update only the fields that were provided
        if cameraname:
            cursor.execute("UPDATE cameras SET cameraname = %s WHERE id = %s", (cameraname, id))
        if cameralocation:
            cursor.execute("UPDATE cameras SET cameralocation = %s WHERE id = %s", (cameralocation, id))
        if cameralink:
            cursor.execute("UPDATE cameras SET cameralink = %s WHERE id = %s", (cameralink, id))

        # Commit the changes
        conn.commit()

        return jsonify({"message": "Camera updated successfully"}), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500

    finally:
        if conn:
            cursor.close()
            conn.close()

@app.route('/license', methods=['GET'])
@cross_origin(supports_credentials=True)
def get_license_data():
    conn = None
    try:
        license_string = request.args.get('license')
        if not license_string:
            return jsonify({"error": "License string is required"}), 400

        conn = psycopg2.connect(dbname=DB_NAME, user=DB_USER, password=DB_PASSWORD, host=DB_HOST, port=DB_PORT)
        cursor = conn.cursor()

        query = """
            SELECT date, raw_image_path, plate_cropped_image_path
            FROM plates
            WHERE predicted_string = %s
        """
        cursor.execute(query, (license_string,))
        result = cursor.fetchone()

        if not result:
            return jsonify({"error": "No data found for the provided license string"}), 404

        date, raw_image_path, plate_image_path = result
        response = {
            "date": date,
            "raw_image": raw_image_path,
            "plate_image": plate_image_path
        }

        return jsonify(response), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        if conn:
            cursor.close()
            conn.close()
@app.route('/plates', methods=['GET'])
@cross_origin(supports_credentials=True)
def get_all_plates():
    conn = None
    try:
        # Get query parameters
        page = request.args.get('page', type=int, default=1)
        limit = request.args.get('limit', type=int, default=10)

        # Dynamic filters
        filters = []
        params = []

        # Add dynamic filters based on input arguments
        if 'platename' in request.args:
            search_value = request.args.get('platename').lower().replace('-', '')
            filters.append("REPLACE(LOWER(predicted_string), '-', '') LIKE %s")
            params.append(f"%{search_value}%")

        if 'date' in request.args:
            filters.append("date = %s")
            params.append(request.args.get('date'))

        if 'id' in request.args:
            filters.append("id = %s")
            params.append(request.args.get('id', type=int))

        # Connect to the database
        conn = psycopg2.connect(
            dbname=DB_NAME,
            user=DB_USER,
            password=DB_PASSWORD,
            host=DB_HOST,
            port=DB_PORT
        )
        cursor = conn.cursor()

        # Build the base query
        base_query = """
            SELECT id, date, predicted_string, raw_image_path, plate_cropped_image_path, camera_id
            FROM plates
        """

        # Add WHERE clause if there are filters
        if filters:
            base_query += " WHERE " + " AND ".join(filters)

        # Fetch the total count with filters
        count_query = "SELECT COUNT(*) FROM plates"
        if filters:
            count_query += " WHERE " + " AND ".join(filters)

        cursor.execute(count_query, tuple(params))
        total_count = cursor.fetchone()[0]

        # Handle pagination or fetch all records
        if page == 0:
            # Fetch all records without pagination
            query = base_query + " ORDER BY id DESC"
            cursor.execute(query, tuple(params))
        else:
            # Fetch records with pagination
            offset = (page - 1) * limit
            query = base_query + " ORDER BY id DESC LIMIT %s OFFSET %s"
            cursor.execute(query, tuple(params) + (limit, offset))
        plates = cursor.fetchall()

        # Function to check if a number is even or odd


        # Format the results
        plates_list = []
        for row in plates:
            predicted_string = row[2]
            evenodd_result = 0
            try :
                if int(predicted_string[-4])%2==0:
                    evenodd_result = 1
            except: 
                evenodd_result=0

            # Extract the specific segment for checking
            # if predicted_string:
            #     parts = predicted_string.split('-')
            #     if len(parts) >= 3:  # Ensure there are enough parts
            #         segment = parts[2]  # Targeting the third part
            #         evenodd_result = 0  # Check even/odd

            plates_list.append({
                "id": row[0],
                "datetime": row[1],
                "predicted_string": predicted_string,
                "evenodd": evenodd_result,  # Even/odd result for the segment
                "raw_image_path": row[3],
                "cropped_plate_path": row[4],
                "cameraid": row[5]
            })

        # Build the response
        response = {
            "count": total_count,
            "plates": plates_list,
        }

        return jsonify(response), 200

    except psycopg2.OperationalError as db_err:
        return jsonify({"error": f"Database connection failed: {db_err}"}), 500
    except psycopg2.DatabaseError as sql_err:
        return jsonify({"error": f"SQL error: {sql_err}"}), 500
    except Exception as e:
        return jsonify({"error": f"Unexpected error: {e}"}), 500
    finally:
        if 'cursor' in locals() and cursor:
            cursor.close()
        if 'conn' in locals() and conn:
            conn.close()


def get_db_connection():
    conn = psycopg2.connect(dbname=DB_NAME, user=DB_USER, password=DB_PASSWORD, host=DB_HOST, port=DB_PORT)
    return conn
@app.route('/penalties', methods=['GET'])
@cross_origin(supports_credentials=True)
def get_penalties():
    conn = None
    try:
        # Get query parameters
        page = request.args.get('page', type=int, default=1)
        limit = request.args.get('limit', type=int, default=10)
        time1 = request.args.get('time1', default=None)
        time2 = request.args.get('time2', default=None)

        # Connect to the database
        conn = psycopg2.connect(
            dbname=DB_NAME,
            user=DB_USER,
            password=DB_PASSWORD,
            host=DB_HOST,
            port=DB_PORT
        )
        cursor = conn.cursor()

        # Base query
        base_query = """
        SELECT id, platename, penaltytype, location, datetime, rawimagepth, plateimagepath,predicted_string
        FROM penalties
        """
        where_clause = []
        params = []

        # Build conditions for time1 and time2
        if time1 and time2:
            where_clause.append("datetime BETWEEN %s AND %s")
            params.extend([time1, time2])
        elif time1:
            where_clause.append("datetime >= %s")
            params.append(time1)
        elif time2:
            where_clause.append("datetime <= %s")
            params.append(time2)

        # Add WHERE clause if needed
        if where_clause:
            base_query += " WHERE " + " AND ".join(where_clause)

        # Handle pagination or fetch all records
        if page == 0:
            # Fetch all records without pagination
            final_query = base_query + " ORDER BY datetime DESC"
            cursor.execute(final_query, params)
            penalties = cursor.fetchall()
        else:
            # Fetch records with pagination
            offset = (page - 1) * limit
            final_query = (
                base_query + " ORDER BY datetime DESC LIMIT %s OFFSET %s"
            )
            cursor.execute(final_query, params + [limit, offset])
            penalties = cursor.fetchall()


        # Function to check if a character is even or odd


        # Format the result
        penalties_list = []
        for row in penalties:

            predicted_string = row[7]
            evenodd_result = 0
            try :
                if int(predicted_string[-4])%2==0:
                    evenodd_result = 1
            except: 
                evenodd_result=0
            penalties_list.append({
                "id": row[0],
                "platename": row[7],  # Replace platename with predicted_string
                "evenodd":evenodd_result,
                "penaltytype": row[2],
                "location": row[3],
                "datetime": row[4],
                "raw_image_path": row[5],
                "plate_image_path": row[6],  # Include plate_image_path
            })

        # Count the total number of matching records
        count_query = "SELECT COUNT(*) FROM penalties"
        if where_clause:
            count_query += " WHERE " + " AND ".join(where_clause)
        cursor.execute(count_query, params)
        total_count = cursor.fetchone()[0]

        response = {
            "count": total_count,
            "penalties": penalties_list,
        }

        return jsonify(response), 200

    except psycopg2.OperationalError as db_err:
        return jsonify({"error": f"Database connection failed: {db_err}"}), 500
    except psycopg2.DatabaseError as sql_err:
        return jsonify({"error": f"SQL error: {sql_err}"}), 500
    except Exception as e:
        return jsonify({"error": f"Unexpected error: {e}"}), 500
    finally:
        if 'cursor' in locals() and cursor:
            cursor.close()
        if 'conn' in locals() and conn:
            conn.close()

# POST: Add a penalty
def get_last_raw_image_path(platename):
    conn = None
    try:
        conn = get_db_connection()
        cur = conn.cursor()

        # Query to get the last raw image path for the specific platename
        query = """
        SELECT raw_image_path, plate_cropped_image_path FROM plates
        WHERE id = %s
        ORDER BY id DESC
        LIMIT 1;
        """
        cur.execute(query, (platename,))
        result = cur.fetchone()
        print(result)

        # If result is found, return the raw image path
        if result and len(result) >= 2:
            return result  # Assuming result[0] is the rawimagepath
        else:
            return None
        
    except Exception as e:
        print(f"Error fetching last image path for platename {platename}: {e}")
        return None
    finally:
        cur.close()
        conn.close()

# POST: Add a penalty

@app.route('/status', methods=['GET'])
@cross_origin(supports_credentials=True)
def get_status():
    conn = None
    try:
        conn = get_db_connection()
        cur = conn.cursor()

        # Count rows in plates table
        cur.execute("SELECT COUNT(*) FROM plates")
        plates_count = cur.fetchone()[0]

        # Count rows in penalties table
        cur.execute("SELECT COUNT(*) FROM penalties")
        penalties_count = cur.fetchone()[0]

        # Calculate plates - penalties
        difference = plates_count - penalties_count
        cur.execute("SELECT gpsport, api_key FROM configuration LIMIT 1")
        row = cur.fetchone()

        if row:
            gpsport, api_key = row
            print(f"First gpsport: {gpsport}, api_key: {api_key}")
        else:
            print("No rows found in the configuration table.")
        # Check GPS availability (Mocking a check function here)
        gps_available = check_gps_availability(gpsport)  # Implement this function

        # Check Internet connection (Mocking a check function here)
        internet_available = check_internet_connection(api_key)  # Implement this function

        cur.close()
        conn.close()

        # Return the status data
        return jsonify({
            'penalties_count': penalties_count,
            'notpenalty': difference,
            'gps_available': gps_available,
            'internet_available': internet_available
        }), 200

    except Exception as e:
        if conn:
            conn.close()
        return jsonify({'error': str(e)}), 500
@app.route('/configuration', methods=['GET'])
@cross_origin(supports_credentials=True)
def get_configuration():
    try:
        # Connect to the database
        conn = get_db_connection()
        cursor = conn.cursor()

        # Get gpsport and api_key from the configuration table
        cursor.execute("SELECT gpsport, api_key FROM configuration LIMIT 1")
        config_row = cursor.fetchone()

        if config_row:
            gpsport, api_key = config_row
        else:
            gpsport, api_key = None, None

        # Get camera ids and cameralink from the cameras table
        cursor.execute("SELECT id, cameralink FROM cameras")
        cameras = cursor.fetchall()

        # Get the server's IP address
        ip_address ="192.168.1.20"

        return jsonify({
            'gpsport': gpsport,
            'api_key': api_key,
            'ip_address': ip_address,
            'cameras': [{'id': row[0], 'cameralink': row[1]} for row in cameras]
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500
    finally:
        if conn:
            cursor.close()
            conn.close()

@app.route('/configuration', methods=['PATCH'])
@cross_origin(supports_credentials=True)

def update_configuration():
    try:
        # Get the data from the request
        data = request.get_json()

        gpsport = data.get('gpsport')
        api_key = data.get('api_key')
        cameras = data.get('cameras', [])

        # Connect to the database
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT 1 FROM cameras WHERE id = %s", (camera_id,))
        if cursor.fetchone() is None:
            return jsonify({'error': f'Camera with ID {camera_id} not found'}), 404

        # Update gpsport and api_key in the configuration table if provided
        if gpsport:
            cursor.execute("UPDATE configuration SET gpsport = %s WHERE id = 1", (gpsport,))
        if api_key:
            cursor.execute("UPDATE configuration SET api_key = %s WHERE id = 1", (api_key,))

        # Update cameralink for each camera in the request if provided
        for camera in cameras:
            camera_id = camera.get('id')
            cameralink = camera.get('cameralink')
            if camera_id and cameralink:
                cursor.execute("SELECT id FROM cameras WHERE id = %s", (camera_id,))
                if cursor.fetchone() is None:
                    return jsonify({'error': f'Camera with ID {camera_id} not found'}), 404

                cursor.execute("UPDATE cameras SET cameralink = %s WHERE id = %s", (cameralink, camera_id))

        # Commit the changes
        conn.commit()

        return jsonify({'message': 'Configuration and cameras updated successfully'})

    except Exception as e:
        return jsonify({'error': str(e)}), 500
    finally:
        if conn:
            cursor.close()
            conn.close()


@app.route('/location', methods=['GET'])
@cross_origin(supports_credentials=True)
def returnlocation():
        location = read_location_from_com3()
        locationstatus=None
        if location==None:
            conn = psycopg2.connect(dbname=DB_NAME, user=DB_USER, password=DB_PASSWORD, host=DB_HOST, port=DB_PORT)
            cursor = conn.cursor()

            # Execute the query to fetch the location
            cursor.execute("SELECT gpsport, location FROM configuration LIMIT 1")
            row = cursor.fetchone()
            gpsport , location = row
            locationstatus= False
        else:
            locationstatus = True
        return jsonify({'location': location,"status":locationstatus}), 200
@app.route('/penalty', methods=['POST'])
@cross_origin(supports_credentials=True)
def add_penalty():
    conn = None
    try:
        # Extract data from the incoming request
        data = request.get_json()
        platename = data['id']
        penaltytype = data['penaltytype']
        location = data['location']
        actualplate= data.get('predicted_string') if data else None
        # Connect to the database
        conn = get_db_connection()
        cur = conn.cursor()
        plate = actualplate
        if plate == None:
            cur.execute("SELECT predicted_string FROM plates WHERE id = %s ORDER BY id DESC LIMIT 1", (platename,))
            plate = cur.fetchone()
        # Check the last penalty timestamp for this platename
        cur.execute("SELECT datetime FROM penalties WHERE predicted_string = %s ORDER BY id DESC LIMIT 1", (plate,))
        last_penalty = cur.fetchone()

        # Get the current timestamp
        current_time = datetime.now()
        if last_penalty:
            # Convert the last penalty time to a datetime object
            last_penalty_time = datetime.strptime(last_penalty[0], "%Y-%m-%d-%H-%M-%S")
            time_difference = (current_time - last_penalty_time).total_seconds() / 60
            message = f"{last_penalty_time.hour}:{last_penalty_time.minute}"
            if time_difference < 15:

                return jsonify({'Time':message}), 400
 
        # Continue with the penalty addition if the time check passes
        result = get_last_raw_image_path(platename)

        if not result:
            return jsonify({'error': f'No image path found for platename {plate}'}), 400

        # Format the current timestamp
        current_time_str = current_time.strftime("%Y-%m-%d-%H-%M-%S")

        # Insert data into the 'penalties' table
        query = """
        INSERT INTO penalties (platename, penaltytype, location, datetime, rawimagepth, plateimagepath,predicted_string)
        VALUES (%s, %s, %s, %s, %s, %s,%s);
        """
        cur.execute(query, (platename, penaltytype, location, current_time_str, result[0], result[1],plate))

        # Commit the transaction
        conn.commit()

        # Close the cursor and connection
        cur.close()
        conn.close()

        # Return success response
        return jsonify({'message': 'Penalty added successfully'}), 201

    except Exception as e:
        if conn:
            conn.close()
        return jsonify({'error': str(e)}), 400
@app.route('/penalty/<int:id>', methods=['PUT'])
@cross_origin(supports_credentials=True)
def update_penalty(id):
    conn = None
    try:
        # Extract data from the incoming request
        data = request.get_json()
        platename = data.get('platename') if data else request.args.get('platename')
        penaltytype = data.get('penaltytype') if data else request.args.get('penaltytype')
        location = data.get('location') if data else request.args.get('location')

        print("Request JSON:", data)
        print("Query Parameters:", request.args)
        print("Extracted Data - platename:", platename, "penaltytype:", penaltytype, "location:", location)

        # Connect to the database
        conn = get_db_connection()
        cur = conn.cursor()
        current_time = datetime.now()
        datetime_value = current_time.strftime("%Y-%m-%d-%H-%M-%S")

        print("Executing Update...")
        query = """
        UPDATE penalties
        SET predicted_string = %s, penaltytype = %s, location = %s, datetime = %s
        WHERE id = %s;
        """
        cur.execute(query, (platename, penaltytype, location, datetime_value, id))
        print("Query:", query)
        print("Parameters:", (platename, penaltytype, location, datetime_value, id))

        # Commit the transaction
        conn.commit()

        if cur.rowcount == 0:
            print("No rows updated for ID:", id)
            return jsonify({'error': 'Penalty record not found'}), 404

        # Close the cursor and connection
        cur.close()
        conn.close()
        print("Penalty updated successfully for ID:", id)

        return jsonify({'message': 'Penalty record updated successfully'}), 200

    except Exception as e:
        import traceback
        print("Error occurred:", traceback.format_exc())
        return jsonify({'error': str(e)}), 400

@app.route('/penalty/<int:penalty_id>', methods=['GET'])
@cross_origin(supports_credentials=True)
def get_penalty_by_id(penalty_id):
    conn = None
    try:
        # Connect to the database
        conn = get_db_connection()
        cursor = conn.cursor()

        # Query the database for the specific penalty by ID
        cursor.execute(
            """
            SELECT id, platename, penaltytype, location, datetime, rawimagepth,predicted_string,plateimagepath
            FROM penalties
            WHERE id = %s
            """,
            (penalty_id,)
        )
        penalty = cursor.fetchone()

        # If no record is found
        if penalty is None:
            return jsonify({"error": f"No penalty found with id {penalty_id}"}), 404

        # Format the result
        penalty_data = {
            "id": penalty[0],
            "platename": penalty[6],
            "penaltytype": penalty[2],
            "location": penalty[3],
            "datetime": penalty[4],
            "raw_image_path": penalty[5],
            "cropped_plate":penalty[7]

        }

        return jsonify(penalty_data), 200

    except psycopg2.OperationalError as db_err:
        return jsonify({"error": f"Database connection failed: {db_err}"}), 500
    except psycopg2.DatabaseError as sql_err:
        return jsonify({"error": f"SQL error: {sql_err}"}), 500
    except Exception as e:
        return jsonify({"error": f"Unexpected error: {e}"}), 500
    finally:
        if 'cursor' in locals() and cursor:
            cursor.close()
        if 'conn' in locals() and conn:
            conn.close()

# DELETE: Delete a penalty record
@app.route('/penalty/<int:id>', methods=['DELETE'])
@cross_origin(supports_credentials=True)
def delete_penalty(id):
    conn = None
    try:
        # Connect to the database
        conn = get_db_connection()
        cur = conn.cursor()

        # Query to delete penalty record by ID
        query = "DELETE FROM penalties WHERE id = %s;"
        cur.execute(query, (id,))

        # Commit the transaction
        conn.commit()

        # Check if the record was deleted
        if cur.rowcount == 0:
            return jsonify({'error': 'Penalty record not found'}), 404

        # Close the cursor and connection
        cur.close()
        conn.close()

        return jsonify({'message': 'Penalty record deleted successfully'}), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 400
@app.route('/penalty/<int:id>', methods=['PATCH'])
@cross_origin(supports_credentials=True)
def patch_penalty(id):
    conn = None
    try:
        # Extract data from the incoming request
        data = request.get_json()
        platename = data.get('platename') if data else None
        penaltytype = data.get('penaltytype') if data else None
        location = data.get('location') if data else None

        print("Request JSON:", data)
        print("Extracted Data - platename:", platename, "penaltytype:", penaltytype, "location:", location)

        # Connect to the database
        conn = get_db_connection()
        cur = conn.cursor()
        current_time = datetime.now()
        datetime_value = current_time.strftime("%Y-%m-%d-%H-%M-%S")

        # Build the dynamic update query
        update_fields = []
        update_values = []

        if platename:
            update_fields.append("predicted_string = %s")
            update_values.append(platename)
        if penaltytype:
            update_fields.append("penaltytype = %s")
            update_values.append(penaltytype)
        if location:
            update_fields.append("location = %s")
            update_values.append(location)

        # Ensure there's something to update
        if not update_fields:
            return jsonify({'error': 'No fields provided for update'}), 400

        # Add datetime and id to the update values
        update_fields.append("datetime = %s")
        update_values.append(datetime_value)
        update_values.append(id)

        # Construct the query
        query = f"""
        UPDATE penalties
        SET {', '.join(update_fields)}
        WHERE id = %s;
        """
        print("Executing Update Query:", query)
        print("Update Values:", update_values)

        # Execute the query
        cur.execute(query, tuple(update_values))

        # Commit the transaction
        conn.commit()

        if cur.rowcount == 0:
            print("No rows updated for ID:", id)
            return jsonify({'error': 'Penalty record not found'}), 404

        # Close the cursor and connection
        cur.close()
        conn.close()
        print("Penalty updated successfully for ID:", id)

        return jsonify({'message': 'Penalty record updated successfully'}), 200

    except Exception as e:
        import traceback
        print("Error occurred:", traceback.format_exc())
        return jsonify({'error': str(e)}), 400



@app.route('/static/images/<path:filename>')
@cross_origin(supports_credentials=True)
def serve_image(filename):
    print(f"filename:{filename}")
    return send_from_directory('static/images', filename)

@app.before_request
def basic_authentication():
    if request.method.lower() == 'options':
        return Response()
    
if __name__ == '__main__':
    socketio.run(app, host='0.0.0.0', debug= True,port=5000)
