import eventlet
eventlet.monkey_patch()
from flask import Flask, Response, send_file, jsonify,send_from_directory
from psycopg2.extras import RealDictCursor
from readsensor import *
# import socket
# import json
from psycopg2 import sql, OperationalError, DatabaseError
# import socket
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
import datetime
import psycopg2
import warnings
import requests
warnings.filterwarnings("ignore", category=FutureWarning)
params = Parameters()
from flask_cors import CORS, cross_origin
from datetime import datetime, timedelta
from psycopg2 import sql, OperationalError, DatabaseError
from flask_socketio import SocketIO, emit


app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "http://localhost:3000"}}, supports_credentials=True)
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
# Global dictionary to track last detection times
def process_frame(img, cameraId):
    global last_char_display, last_detection_time
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
                state,chars,englishchars, charConfAvg = detectPlateChars(plate_img)
                char_display = []
                englishchardisplay=[]
                
                if state==True:
                    for persianchar in chars:
                        char_display.append(persianchar)
                    for englishchar in englishchars:
                        englishchardisplay.append(englishchar)
                    current_char_display = ''.join(englishchardisplay)
                    current_time = datetime.now()

                    # Check if the plate has been detected recently
                    if current_char_display in last_detection_time:
                        last_time = last_detection_time[current_char_display]
                        time_diff = (current_time - last_time).total_seconds() / 60  # Time difference in minutes

                        if time_diff < 5:
                            persian_output = f"{char_display[0]}{char_display[1]}-{char_display[2]}-{char_display[3]}{char_display[4]}{char_display[5]}-{char_display[6]}{char_display[7]}"
                            img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                            draw = ImageDraw.Draw(img_pil)
                            draw.text((x1, y1 - 30), persian_output, font=persian_font, fill=(255, 0, 0))
                            img = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
                            txtout = f"Detected  {last_time.strftime('%Y-%m-%d %H:%M:%S')}"
                            cv2.putText(img, txtout, (100,200), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(10, 50, 255), thickness=2, lineType=cv2.LINE_AA)
                            continue  # Skip writing this plate as it was recently detected

                    englishoutput = f"{englishchardisplay[0]}{englishchardisplay[1]}-{englishchardisplay[2]}-{englishchardisplay[3]}{englishchardisplay[4]}{englishchardisplay[5]}-{englishchardisplay[6]}{englishchardisplay[7]}"
                    persian_output = f"{char_display[0]}{char_display[1]}-{char_display[2]}-{char_display[3]}{char_display[4]}{char_display[5]}-{char_display[6]}{char_display[7]}"
                    img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                    draw = ImageDraw.Draw(img_pil)
                    draw.text((x1, y1 - 30), persian_output, font=persian_font, fill=(255, 0, 0))
                    img = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
                    # Save images to disk
                    timestamp = current_time.strftime("%Y-%m-%d-%H-%M-%S")
                    raw_filename = f"raw_{timestamp}.jpg"
                    plate_filename = f"plt_{timestamp}.jpg"
                    
                    raw_path = os.path.join('static/images/raw', raw_filename)
                    plate_path = os.path.join('static/images/plate', plate_filename)
                    
                    os.makedirs(os.path.dirname(raw_path), exist_ok=True)
                    os.makedirs(os.path.dirname(plate_path), exist_ok=True)

                    cv2.imwrite(raw_path, img)
                    cv2.imwrite(plate_path, plate_img)
                    # Save to database
                    raw_url = f"http://localhost:5000/static/images/raw/{raw_filename}"
                    plate_url = f"http://localhost:5000/static/images/plate/{plate_filename}"
                    #print(raw_url)
                    conn = psycopg2.connect(dbname=DB_NAME, user=DB_USER, password=DB_PASSWORD, host=DB_HOST, port=DB_PORT)
                    cursor = conn.cursor()
                    cursor.execute(
                        """
                        INSERT INTO plates (date, raw_image_path, plate_cropped_image_path, predicted_string, camera_id)
                        VALUES (%s, %s, %s, %s, %s)
                        RETURNING id
                        """,
                        (timestamp, raw_url, plate_url, englishoutput, cameraId)
                    )
                    plate_id = cursor.fetchone()[0]
                    conn.commit()
                    cursor.close()
                    conn.close()


                    last_detection_time[current_char_display] = current_time
                    
                    try:
                        data = {
                            "id": plate_id,  
                            "date": timestamp,
                            "raw_image_path": raw_url,
                            "plate_cropped_image_path": plate_url,
                            "predicted_string": englishoutput,
                            "cameraid": cameraId
                        }

                        # Emit the data via SocketIO with the ID from the database
                        socketio.emit('plate_detected', data)
                        print(f"Data emitted via SocketIO with ID: {plate_id}")
                    except Exception as e:
                        print(f"Error emitting data: {e}")

    # Add FPS overlay
    tock = time.time()
    elapsed_time = tock - tick
    fps_text = f"FPS: {1 / elapsed_time:.2f}"
    cv2.putText(img, fps_text, (0, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (10, 50, 255), 2)
    return img

@app.route('/camera/<int:cameraId>/stream', methods=['GET'])
@cross_origin(supports_credentials=True)
def video_feed(cameraId):
    def generate():
        global frame
        try:
            conn = psycopg2.connect(dbname=DB_NAME, user=DB_USER, password=DB_PASSWORD, host=DB_HOST, port=DB_PORT)
            cursor = conn.cursor()
            #cameraId=1
            # Fetch the camera link based on the cameraId
            cursor.execute("SELECT cameralink FROM cameras WHERE id = %s", (cameraId,))
            camera_link = cursor.fetchone()

            if camera_link is None:
                return jsonify({"error": "Camera not found"}), 404

            camera_link = camera_link[0]  # Extract link from tuple
            camera_link = 'http://admin:Maziar123@192.168.1.11/cgi-bin/mjpeg'
            # Open video stream
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
                return jsonify({"error": "Failed to open camera stream"}), 500
            while True:
                ret, img = cap.read()
                if not ret:
                    print("No frames read. Exiting...")
                    break
                img = cv2.resize(img,(width,height))
                processed_frame = process_frame(img,cameraId)

                with lock:
                    frame = processed_frame

                _, buffer = cv2.imencode('.jpg', frame)
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
                time.sleep(0.05)
            #cap.release()
        except Exception as e:
            return jsonify({"error": str(e)}), 500
        finally:
            if conn:
                cursor.close()
                conn.close()

                            

    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/cameras', methods=['GET'])
@cross_origin(supports_credentials=True)
def get_cameras():
    try:
        conn = psycopg2.connect(dbname=DB_NAME, user=DB_USER, password=DB_PASSWORD, host=DB_HOST, port=DB_PORT)
        cursor = conn.cursor()

        cursor.execute("SELECT cameraname, cameralocation, cameralink ,id FROM cameras")
        cameras = cursor.fetchall()


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


@app.route('/camera', methods=['POST'])
@cross_origin(supports_credentials=True)
def add_camera():
    from flask import request

    try:
        data = request.get_json()
        required_fields = ['cameraname', 'cameralocation', 'cameralink']
        for field in required_fields:
            if field not in data:
                return jsonify({"error": f"Missing required field: {field}"}), 400

        cameraname = data['cameraname']
        cameralocation = data['cameralocation']
        cameralink = data['cameralink']

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

@app.route('/license', methods=['GET'])
@cross_origin(supports_credentials=True)
def get_license_data():
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
            SELECT id, date, predicted_string, raw_image_path, plate_cropped_image_path,valid
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
            query = base_query + " ORDER BY id LIMIT %s OFFSET %s"
            cursor.execute(query, tuple(params) + (limit, offset))
        plates = cursor.fetchall()

        # Format the results
        plates_list = [
            {
                "id": row[0],
                "datetime": row[1],
                "predicted_string": row[2],
                "raw_image_path": row[3],
                "cropped_plate_path": row[4],
                "permit":row[5]
            }
            for row in plates
        ]

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
    try:
        # Get query parameters
        page = request.args.get('page', type=int, default=1)
        limit = request.args.get('limit', type=int, default=10)
        time1 = request.args.get('time1', default=None)
        time2 = request.args.get('time2', default=None)

        # Connect to the database
        conn = get_db_connection()
        cursor = conn.cursor()

        # Base query
        base_query = """
        SELECT id, platename, penaltytype, location, datetime, rawimagepath, plateimagepath
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

        # Fetch predicted_string from plates based on platename
        plate_query = "SELECT predicted_string FROM plates WHERE id = %s"

        # Format the result
        penalties_list = []
        for row in penalties:
            plateid = row[1]
            cursor.execute(plate_query, (plateid,))
            plate_result = cursor.fetchone()
            predicted_string = plate_result[0] if plate_result else None
            #print(f"predictedstring:{predicted_string}")
            penalties_list.append({
                "id": row[0],
                "platename": predicted_string,  # Replace platename with predicted_string
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
@app.route('/penalty', methods=['POST'])
@cross_origin(supports_credentials=True)
def add_penalty():
    try:
        # Extract data from the incoming request
        data = request.get_json()
        platename = data['id']
        penaltytype = data['penaltytype']
        location = data['location']

        
        # Get the last raw image path from the plates table based on platename
        result = get_last_raw_image_path(platename)

        if not result:
            return jsonify({'error': f'No image path found for platename {platename}'}), 400
        
        # Get the current timestamp
        current_time = datetime.now()
        current_time = current_time.strftime("%Y-%m-%d-%H-%M-%S")

        # Connect to the database
        conn = get_db_connection()
        cur = conn.cursor()

        # Insert data into the 'penalties' table
        query = """
        INSERT INTO penalties (platename, penaltytype, location, datetime, rawimagepath, plateimagepath)
        VALUES (%s, %s, %s, %s, %s, %s);
        """
        cur.execute(query, (platename, penaltytype, location, current_time, result[0], result[1]))  # Using result[1] for plateimagepath

        # Commit the transaction
        conn.commit()

        # Close the cursor and connection
        cur.close()
        conn.close()

        # Return success response
        return jsonify({'message': 'Penalty added successfully'}), 201

    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/location', methods=['GET'])
@cross_origin(supports_credentials=True)
def returnlocation():
        location = read_location_from_com3()
        return jsonify({'location': location}), 200
@app.route('/penalty/<int:id>', methods=['PUT'])
@cross_origin(supports_credentials=True)
def update_penalty(id):
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
        SET platename = %s, penaltytype = %s, location = %s, datetime = %s
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
    try:
        # Connect to the database
        conn = get_db_connection()
        cursor = conn.cursor()

        # Query the database for the specific penalty by ID
        cursor.execute(
            """
            SELECT id, platename, penaltytype, location, datetime, rawimagepath
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
            "platename": penalty[1],
            "penaltytype": penalty[2],
            "location": penalty[3],
            "datetime": penalty[4],
            "raw_image_path": penalty[5],
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
            update_fields.append("platename = %s")
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

### POST: Add a new mine record ###
@app.route('/mine', methods=['POST'])
@cross_origin(supports_credentials=True)
def create_mine():
    data = request.json
    try:
        conn = psycopg2.connect(dbname=DB_NAME, user=DB_USER, password=DB_PASSWORD, host=DB_HOST, port=DB_PORT)
        cursor = conn.cursor()

        cursor.execute("""
            INSERT INTO mine_info (mine_name, cameraid, location, owner_name, contact_number)
            VALUES (%s, %s, %s, %s, %s)
            RETURNING mine_id
        """, (data['mine_name'], data['cameraid'], data.get('location'), data.get('owner_name'), data.get('contact_number')))
        mine_id = cursor.fetchone()[0]
        conn.commit()

        return jsonify({"message": "Mine created successfully", "mine_id": mine_id}), 201

    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        if conn:
            cursor.close()
            conn.close()


@app.route('/mine/<int:mine_id>', methods=['PUT'])
@cross_origin(supports_credentials=True)
def update_mine(mine_id):
    data = request.json
    try:
        conn = psycopg2.connect(dbname=DB_NAME, user=DB_USER, password=DB_PASSWORD, host=DB_HOST, port=DB_PORT)
        cursor = conn.cursor()

        cursor.execute("""
            UPDATE mine_info
            SET mine_name = %s, cameraid = %s, location = %s, owner_name = %s, contact_number = %s
            WHERE mine_id = %s
        """, (data['mine_name'], data['cameraid'], data.get('location'), data.get('owner_name'), data.get('contact_number'), mine_id))
        conn.commit()

        return jsonify({"message": "Mine updated successfully"}), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        if conn:
            cursor.close()
            conn.close()


@app.route('/mine/<int:mine_id>', methods=['PATCH'])
@cross_origin(supports_credentials=True)
def patch_mine(mine_id):
    try:
        # Extract data from the request
        data = request.get_json()

        # Build the query dynamically based on the provided fields
        fields = []
        values = []
        for key, value in data.items():
            fields.append(f"{key} = %s")
            values.append(value)

        # Ensure at least one field is provided
        if not fields:
            return jsonify({'error': 'No fields to update provided'}), 400

        values.append(mine_id)  # Add mine_id for the WHERE clause
        query = f"UPDATE mine_info SET {', '.join(fields)} WHERE mine_id = %s;"

        # Connect to the database
        conn = get_db_connection()
        cur = conn.cursor()

        # Execute the query
        cur.execute(query, values)

        # Commit the transaction
        conn.commit()
        cur.close()
        conn.close()

        # Return success response
        return jsonify({'message': 'Mine partially updated successfully'}), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 400
@app.route('/mine', methods=['GET'])
@cross_origin(supports_credentials=True)
def get_mines():
    try:
        # Extract query parameters
        page = int(request.args.get('page', 1))  # Default to page 1 if not provided
        limit = int(request.args.get('limit', 10))  # Default limit is 10
        search = request.args.get('search', '')  # Search query (default empty)

        # If page is 0, return all records
        if page == 0:
            query = """
                SELECT * FROM mine_info
                WHERE mine_name ILIKE %s OR location ILIKE %s OR owner_name ILIKE %s;
            """
            search_term = f"%{search}%"

            conn = get_db_connection()
            cur = conn.cursor(cursor_factory=RealDictCursor)

            cur.execute(query, (search_term, search_term, search_term))
            all_records = cur.fetchall()
            count_query = "SELECT COUNT(*) FROM mine_info WHERE mine_name ILIKE %s OR location ILIKE %s OR owner_name ILIKE %s;"
            cur.execute(count_query, (search_term, search_term, search_term))
            total_count = cur.fetchone()['count']

            cur.close()
            conn.close()

            return jsonify({
                'data': all_records,
                'total_count': total_count
            }), 200

        # Otherwise, apply pagination
        offset = (page - 1) * limit
        query = """
            SELECT * FROM mine_info
            WHERE mine_name ILIKE %s OR location ILIKE %s OR owner_name ILIKE %s
            LIMIT %s OFFSET %s;
        """
        count_query = """
            SELECT COUNT(*) FROM mine_info
            WHERE mine_name ILIKE %s OR location ILIKE %s OR owner_name ILIKE %s;
        """
        search_term = f"%{search}%"

        # Connect to the database
        conn = get_db_connection()
        cur = conn.cursor(cursor_factory=RealDictCursor)

        # Execute the queries
        cur.execute(query, (search_term, search_term, search_term, limit, offset))
        records = cur.fetchall()

        cur.execute(count_query, (search_term, search_term, search_term))
        total_count = cur.fetchone()['count']

        cur.close()
        conn.close()

        # Return the data with pagination info
        return jsonify({
            'data': records,
            'total_count': total_count
        }), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 400

### DELETE: Remove a mine record ###
@app.route('/mine/<int:mine_id>', methods=['DELETE'])
@cross_origin(supports_credentials=True)
def delete_mine(mine_id):
    try:
        # Connect to the database
        conn = get_db_connection()
        cur = conn.cursor()

        # Delete the record from the mine_info table
        query = "DELETE FROM mine_info WHERE mine_id = %s;"
        cur.execute(query, (mine_id,))

        # Commit the transaction
        conn.commit()
        cur.close()
        conn.close()

        # Return success response
        return jsonify({'message': 'Mine deleted successfully'}), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/permit', methods=['POST'])
@cross_origin(supports_credentials=True)
def add_vehicle_permit():
    try:
        data = request.get_json()
        license_plate = data['license_plate']
        mine_id = data['mine_id']
        start_date = data['start_date']
        end_date = data['end_date']

        conn = get_db_connection()
        cur = conn.cursor()

        # Insert or fetch vehicle_id from vehicle_info
        cur.execute(
            """
            INSERT INTO vehicle_info (license_plate)
            VALUES (%s)
            ON CONFLICT (license_plate) DO NOTHING
            RETURNING vehicle_id;
            """,
            (license_plate,)
        )
        result = cur.fetchone()
        if result is None:
            cur.execute("SELECT vehicle_id FROM vehicle_info WHERE license_plate = %s", (license_plate,))
            vehicle_id = cur.fetchone()[0]
        else:
            vehicle_id = result[0]

        # Insert data into vehicle_permit
        cur.execute(
            """
            INSERT INTO vehicle_permit (vehicle_id, mine_id, start_date, end_date)
            VALUES (%s, %s, %s, %s);
            """,
            (vehicle_id, mine_id, start_date, end_date)
        )

        conn.commit()
        cur.close()
        conn.close()

        return jsonify({'message': 'Vehicle permit added successfully'}), 201

    except Exception as e:
        return jsonify({'error': str(e)}), 400

### PUT: Update a vehicle permit ###
@app.route('/permit/<int:permit_id>', methods=['PUT'])
@cross_origin(supports_credentials=True)
def update_vehicle_permit(permit_id):
    try:
        data = request.get_json()
        license_plate = data['license_plate']
        mine_id = data['mine_id']
        start_date = data['start_date']
        end_date = data['end_date']

        conn = get_db_connection()
        cur = conn.cursor()

        # Fetch the current vehicle_id linked to the permit
        cur.execute(
            """
            SELECT vehicle_id 
            FROM vehicle_permit
            WHERE permit_id = %s;
            """,
            (permit_id,)
        )
        result = cur.fetchone()
        if not result:
            return jsonify({'error': 'Permit ID not found'}), 404

        current_vehicle_id = result[0]

        # Check if the new license_plate exists in vehicle_info, or add it
        cur.execute(
            """
            INSERT INTO vehicle_info (license_plate)
            VALUES (%s)
            ON CONFLICT (license_plate) DO NOTHING
            RETURNING vehicle_id;
            """,
            (license_plate,)
        )
        result = cur.fetchone()
        if result:
            new_vehicle_id = result[0]
        else:
            # Fetch the vehicle_id for the existing license_plate
            cur.execute("SELECT vehicle_id FROM vehicle_info WHERE license_plate = %s", (license_plate,))
            new_vehicle_id = cur.fetchone()[0]

        # Update the vehicle_permit with the new vehicle_id and other details
        cur.execute(
            """
            UPDATE vehicle_permit
            SET vehicle_id = %s, mine_id = %s, start_date = %s, end_date = %s
            WHERE permit_id = %s;
            """,
            (new_vehicle_id, mine_id, start_date, end_date, permit_id)
        )

        conn.commit()
        cur.close()
        conn.close()

        return jsonify({'message': 'Vehicle permit updated successfully'}), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 400

### PATCH: Partially update a vehicle permit ###
@app.route('/permit/<int:permit_id>', methods=['PATCH'])
@cross_origin(supports_credentials=True)
def patch_vehicle_permit(permit_id):
    try:
        data = request.get_json()

        conn = get_db_connection()
        cur = conn.cursor()

        # Check if license_plate is in the data
        if 'license_plate' in data:
            license_plate = data.pop('license_plate')

            # Insert or fetch vehicle_id from vehicle_info
            cur.execute(
                """
                INSERT INTO vehicle_info (license_plate)
                VALUES (%s)
                ON CONFLICT (license_plate) DO NOTHING
                RETURNING vehicle_id;
                """,
                (license_plate,)
            )
            result = cur.fetchone()
            if result:
                vehicle_id = result[0]
            else:
                # Fetch the vehicle_id for the existing license_plate
                cur.execute("SELECT vehicle_id FROM vehicle_info WHERE license_plate = %s", (license_plate,))
                vehicle_id = cur.fetchone()[0]

            # Add vehicle_id to the update fields
            data['vehicle_id'] = vehicle_id

        # Dynamically build the query based on provided fields
        fields = []
        values = []
        for key, value in data.items():
            fields.append(f"{key} = %s")
            values.append(value)

        values.append(permit_id)
        query = f"UPDATE vehicle_permit SET {', '.join(fields)} WHERE permit_id = %s;"

        cur.execute(query, tuple(values))

        conn.commit()
        cur.close()
        conn.close()

        return jsonify({'message': 'Vehicle permit updated successfully'}), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 400


### DELETE: Remove a vehicle permit ###
@app.route('/permit/<int:permit_id>', methods=['DELETE'])
@cross_origin(supports_credentials=True)
def delete_vehicle_permit(permit_id):
    try:
        conn = get_db_connection()
        cur = conn.cursor()

        # Delete the vehicle_permit
        cur.execute("DELETE FROM vehicle_permit WHERE permit_id = %s;", (permit_id,))

        conn.commit()
        cur.close()
        conn.close()

        return jsonify({'message': 'Vehicle permit deleted successfully'}), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/permit', methods=['GET'])
@cross_origin(supports_credentials=True)
def get_vehicle_permits():
    try:
        # Get query parameters
        page = request.args.get('page', 1, type=int)
        limit = request.args.get('limit', 10, type=int)
        license_plate = request.args.get('license_plate', '')

        # Connect to the database
        conn = get_db_connection()
        cur = conn.cursor(cursor_factory=RealDictCursor)

        # Base query to fetch permits with mine_name
        base_query = """
            SELECT 
                vp.*, 
                vi.license_plate, 
                mi.mine_id, 
                mi.mine_name
            FROM 
                vehicle_permit vp
            JOIN 
                vehicle_info vi ON vp.vehicle_id = vi.vehicle_id
            JOIN 
                mine_info mi ON vp.mine_id = mi.mine_id
        """
        count_query = """
            SELECT COUNT(*)
            FROM 
                vehicle_permit vp
            JOIN 
                vehicle_info vi ON vp.vehicle_id = vi.vehicle_id
            JOIN 
                mine_info mi ON vp.mine_id = mi.mine_id
        """
        where_clause = ""
        params = []

        # Filter by license_plate if provided
        if license_plate:
            where_clause = "WHERE vi.license_plate ILIKE %s"
            params.append(f"%{license_plate}%")

        # Add pagination or fetch all records
        if page == 0:
            query = f"{base_query} {where_clause} ORDER BY permit_id DESC;"
            count_query = f"{count_query} {where_clause};"
        else:
            offset = (page - 1) * limit
            query = f"{base_query} ORDER BY permit_id DESC {where_clause} LIMIT %s OFFSET %s ;"
            params.extend([limit, offset])

        # Execute the query
        cur.execute(query, tuple(params))
        records = cur.fetchall()

        # Execute count query
        cur.execute(count_query, tuple(params[:1]))
        total_count = cur.fetchone()['count']

        # Close database connection
        cur.close()
        conn.close()

        # Return response
        return jsonify({
            'data': records,
            'total_count': total_count,
            'page': page,
            'limit': limit if page != 0 else total_count
        }), 200

    except Exception as e:
        # Handle errors
        return jsonify({'error': str(e)}), 400


@app.route('/images/<path:filename>')
@cross_origin(supports_credentials=True)
def serve_image(filename):
    print(f"filename:{filename}")
    return send_from_directory('static', filename)

@app.before_request
def basic_authentication():
    if request.method.lower() == 'options':
        return Response()
    
if __name__ == '__main__':
    socketio.run(app, host='0.0.0.0',debug=True, port=5000)
