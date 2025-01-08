from ultralytics import YOLO
import cv2
import math
import time
import datetime
from PIL import Image, ImageDraw, ImageFont 
import numpy as np
import sqlite3
from dbcode.createdb import insert_plate_data
import tkinter as tk
from tkinter import filedialog
from tkinter import messagebox
from PIL import Image, ImageTk
from tkinter import ttk
import threading
from tkinter import messagebox
import os 
import torch
from configParams import Parameters
from PIL import Image, ImageTk
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
params = Parameters()
# Flags for pause/resume and close
paused = False
running = True
import torch
if torch.cuda.is_available():
    print("Gpu is available")
    device = "cuda"
else:
    device = "cpu"
    print("cpu is available")
def toggle_pause():
    global paused
    paused = not paused

def close_program():
    global running
    running = False
    cap.release()
    video_writer.release()
    cv2.destroyAllWindows()
    window.quit()  

database_path = 'database/license_plate.db'
if os.path.exists(database_path):
    print("Database exist")


font_path = "vazir.ttf" 
persian_font = ImageFont.truetype(font_path, 20)

model_object = YOLO("weights/best.pt").to(device)
modelCharX = torch.hub.load('yolov5', 'custom', "model/CharsYolo.pt", source='local', force_reload=True).to(device)

classnames = ['car', 'plate']
charclassnames = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'D', 'Gh', 'H', 'J', 'L', 'M',
                          'N',
                          'P',
                          'PuV', 'PwD', 'Sad', 'Sin', 'T', 'Taxi', 'V', 'Y']
char_dict = {'0': '0', '1': '1', '2': '2', '3': '3', '4': '4', '5': '5', '6': '6', '7': '7', '8': '8',
                          '9': '9',
                          'A': '10', 'B': '11', 'P': '12', 'Taxi': '13', 'ث': '14', 'J': '15', 'چ': '16', 'ح': '17',
                          'خ': '18',
                          'D': '19', 'ذ': '20', 'ر': '21', 'ز': '22', 'ژ': '23', 'Sin': '24', 'ش': '25', 'Sad': '26',
                          'ض': '27',
                          'T': '28', 'ظ': '29', 'PuV': '30', 'غ': '31', 'ف': '32', 'Gh': '33', 'ک': '34', 'گ': '35',
                          'L': '36',
                          'م': '37', 'N': '38', 'H': '39', 'V': '40', 'Y': '41', 'PwD': '42'}

def start_detection():
    global source
    option = messagebox.askquestion("Input Source", "Do you want to use a streaming camera (Yes) or select a video file (No)?")
    
    if option == "yes":
        source = "rtsp://admin:123456@192.168.1.40:554/stream1"  # Camera index (0 is default for the primary webcam)
    else:
        source = filedialog.askopenfilename(
            title="Select Video File",
            filetypes=(("Video Files", "*.mp4;*.avi;*.mov"), ("All Files", "*.*"))
        )
        if not source:
            messagebox.showerror("Error", "No video selected!")
            return
    
    detection_thread = threading.Thread(target=run_detection)
    detection_thread.daemon = True
    detection_thread.start()
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
    #charConfAvg = round(statistics.mean(confidences) * 100) if confidences else 0
    charConfAvg = 0
    return chars, charConfAvg


# Initialize a variable to store the last char_display
last_char_display = ""

def run_detection():
    starttime = time.time()
    global running, paused, cap, video_writer, last_char_display  # Add last_char_display here
    cap = cv2.VideoCapture(source)  # 'source' should be defined earlier
    conn = sqlite3.connect(database_path)  # 'database_path' should be defined earlier
    cursor = conn.cursor()
    
    # Prepare video writer
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    output_videoname = f'videos/output_{timestamp}.mp4'
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    width, height = 640, 480
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = 10
    cap.set(cv2.CAP_PROP_FPS, fps)
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    video_writer = cv2.VideoWriter(output_videoname, fourcc, fps, (frame_width, frame_height))

    if not cap.isOpened():
        print("Failed to open the video source.")
        return

    while cap.isOpened() and running:
        if paused:
            cv2.waitKey(1)
            continue

        success, img = cap.read()
        if not success:
            print("Failed to read frame. Exiting...")
            break

        tick = time.time()
        #a = time.time()
        output = model_object(img, show=False, conf=0.7, stream=True)  # 'model_object' should be defined
        #print(f"model1:{time.time()-a}")

        for detection in output:
            bbox = detection.boxes
            #print(bbox)
            # Process the first valid bounding box only
            for box in bbox:
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                plate_img = img[y1:y2, x1:x2]
                #cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 3)
                
                #plate_img= cv2.resize(plate_img,(96,32))

                cls_names = int(box.cls[0])
                if cls_names==1:
                        chars, charConfAvg = detectPlateChars(plate_img)  # 'detectPlateChars' should be defined
                        char_display = []
                        if len(chars) == 8:
                            for english_char in chars:
                                char_display.append(english_char)

                            # Check if char_display is the same as the last inserted one
                            current_char_display = ''.join(char_display)
                            if current_char_display != last_char_display:
                                timestamp = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
                                raw_path = f"database/images/raw/raw_{timestamp}.jpg"
                                plt_path = f"database/images/plate/plt_{timestamp}.jpg"
                                cv2.imwrite(raw_path, img)
                                cv2.imwrite(plt_path, plate_img)

                                # Insert data into the database
                                english_output = f"{char_display[6]}{char_display[7]}-{char_display[3]}{char_display[4]}{char_display[5]}-{char_display[2]}-{char_display[0]}{char_display[1]}"
                                cursor.execute(
                                    "INSERT INTO plates (date, raw_image_path, plate_cropped_image_path, predicted_string) VALUES (?, ?, ?, ?)",
                                    (timestamp, raw_path, plt_path, english_output)
                                )
                                conn.commit()

                                # Update last_char_display to current one
                                last_char_display = current_char_display

                            # Create and display Persian output
                            persian_output = f"{char_display[0]}{char_display[1]}-{char_display[2]}-{char_display[3]}{char_display[4]}{char_display[5]}-{char_display[6]}{char_display[7]}"
                            img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                            draw = ImageDraw.Draw(img_pil)
                            draw.text((x1, y1 - 30), persian_output, font=persian_font, fill=(255, 0, 0))  # 'persian_font' should be defined
                            img = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

                # Break after processing the first bounding box


        # Calculate elapsed time and FPS
        tock = time.time()
        elapsed_time = tock - tick
        fps_text = "FPS: {:.2f}".format(1 / elapsed_time)
        fps_text_loc = (frame_width - 200, 50)
        print(fps_text)
        cv2.putText(img, fps_text, fps_text_loc, fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(10, 50, 255), thickness=2, lineType=cv2.LINE_AA)

        # Display the detection result
        cv2.imshow('detection', img)
        video_writer.write(img)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    video_writer.release()
    cv2.destroyAllWindows()
    print(f"duration time: {time.time() - starttime}")
import threading

def show_database():
    def fetch_and_display_data():
        try:
            db_window = tk.Toplevel(window)
            db_window.title("Database Contents")
            db_window.geometry("600x500")

            frame = tk.Frame(db_window)
            frame.pack(fill=tk.BOTH, expand=True)

            canvas = tk.Canvas(frame)
            scrollbar = tk.Scrollbar(frame, orient=tk.VERTICAL, command=canvas.yview)
            scrollable_frame = tk.Frame(canvas)

            scrollable_frame.bind(
                "<Configure>",
                lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
            )

            canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
            canvas.configure(yscrollcommand=scrollbar.set)

            canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
            scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

            conn = sqlite3.connect(database_path)
            cursor = conn.cursor()
            cursor.execute("SELECT ID, date, predicted_string, plate_cropped_image_path FROM plates ORDER BY ID DESC")
            rows = cursor.fetchall()
            conn.close()

            headers = ["ID", "Date", "Predicted String", "Plate Cropped Image Path"]
            for col, header in enumerate(headers):
                label = tk.Label(scrollable_frame, text=header, font=("Helvetica", 12, "bold"), borderwidth=2, relief="groove")
                label.grid(row=0, column=col, sticky="nsew", padx=2, pady=2)

            for i, row in enumerate(rows, start=1):
                for j, value in enumerate(row):
                    if j == 3:
                        image_path = value
                        try:
                            img = Image.open(image_path)
                            img.thumbnail((50, 50))
                            img_tk = ImageTk.PhotoImage(img)
                            label = tk.Label(scrollable_frame, image=img_tk, borderwidth=1, relief="solid")
                            label.image = img_tk
                            label.grid(row=i, column=j, sticky="nsew", padx=1, pady=1)
                        except Exception as e:
                            label = tk.Label(scrollable_frame, text="Error loading image", borderwidth=1, relief="solid")
                            label.grid(row=i, column=j, sticky="nsew", padx=1, pady=1)
                    else:
                        label = tk.Label(scrollable_frame, text=value, borderwidth=1, relief="solid")
                        label.grid(row=i, column=j, sticky="nsew", padx=1, pady=1)

                for col in range(len(headers)):
                    scrollable_frame.grid_rowconfigure(i, weight=1)
                    scrollable_frame.grid_columnconfigure(col, weight=1)

        except Exception as e:
            messagebox.showerror("Error", f"Failed to fetch database contents.\n{e}")

    # Run the data-fetching function in a separate thread
    thread = threading.Thread(target=fetch_and_display_data)
    thread.daemon = True  # Allows the thread to exit when the main program exits
    thread.start()


window = tk.Tk()
window.title("License Plate Detection")
window.geometry("500x200")

start_button = tk.Button(window, text="Start Detection", command=start_detection)
start_button.pack(pady=20)
stop_button = tk.Button(window, text="Stop", command=toggle_pause)
stop_button.pack(pady=10)
tk.Button(window, text="Show Database", command=show_database).pack(pady=5)

window.mainloop()