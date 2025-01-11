import cv2
import psycopg2
from datetime import datetime, timedelta

# Database connection details
DB_NAME = "license_plate_db"
DB_USER = "postgres"
DB_PASSWORD = "m102030m"
DB_HOST = "localhost"
DB_PORT = "5432"
def get_camera_link(camera_id):
    """Fetch camera link from the database."""
    try:
        conn = psycopg2.connect(
            dbname=DB_NAME, user=DB_USER, password=DB_PASSWORD,
            host=DB_HOST, port=DB_PORT
        )
        cursor = conn.cursor()
        cursor.execute("SELECT cameralink FROM cameras WHERE id = %s", (camera_id,))
        result = cursor.fetchone()
        conn.close()
        return result[0] if result else None
    except Exception as e:
        print(f"Database error: {e}")
        return None

def record_camera(camera_id):
    """Record video from a single camera and save as MP4 until window is closed."""
    camera_link = get_camera_link(camera_id)
    if not camera_link:
        print(f"Camera with ID {camera_id} not found!")
        return

    cap = cv2.VideoCapture(camera_link)
    if not cap.isOpened():
        print(f"Failed to open camera stream for Camera {camera_id}")
        return

    # Set up MP4 video writer using the H.264 codec
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for MP4
    current_time = datetime.now()
    timestamp = current_time.strftime("%Y-%m-%d-%H-%M-%S")
    out = cv2.VideoWriter(f'camera_{camera_id}.mp4', fourcc, 20.0, (640, 400))

    print(f"Recording from Camera {camera_id}. Press 'q' to stop.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print(f"Failed to read frame from Camera {camera_id}")
            break

        frame = cv2.resize(frame, (640, 400))
        out.write(frame)  # Save frame to MP4 video file
        cv2.imshow(f'Camera {camera_id}', frame)

        # Press 'q' to quit the recording
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print(f"Recording for Camera {camera_id} stopped.")

def record_multiple_cameras(camera_ids):
    """Record from multiple cameras (one at a time)."""
    for camera_id in camera_ids:
        record_camera(camera_id)

# Example usage: Record from cameras with IDs 1 to 4 until manually stopped
record_multiple_cameras([3])
