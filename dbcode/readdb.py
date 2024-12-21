
import psycopg2
import datetime


DB_NAME = "license_plate_db"
DB_USER = "postgres"
DB_PASSWORD = "m102030m"
DB_HOST = "localhost"
DB_PORT = "5432"
def fetch_plate_data():
    # Establish connection to the SQLite database
    conn = psycopg2.connect(dbname=DB_NAME, user=DB_USER, password=DB_PASSWORD, host=DB_HOST, port=DB_PORT)
    cursor = conn.cursor()
    
    cursor.execute(
        """
        SELECT * FROM vehicle_info

        """
    )

    rows = cursor.fetchall()

    # Fetch all the rows from the query result

    # Iterate through the fetched rows
    for row in rows:
        # id, time, time, raw_image, plate_cropped_image, predicted_string = row
        # print(f"ID: {id}")
        # print(f"time: {time}")
        # print(f"Time: {time}")
        # print(f"Predicted String: {predicted_string}")
        
        # # Optionally, save the images to files if needed
        # with open(f'raw_image_{id}.jpg', 'wb') as file:
        #     file.write(raw_image)
        
        # with open(f'plate_cropped_image_{id}.jpg', 'wb') as file:
        #     file.write(plate_cropped_image)
        
        # print(f"Images saved as raw_image_{id}.jpg and plate_cropped_image_{id}.jpg\n")
        print(row)

    # Close the cursor and connection
    cursor.close()
    conn.close()

# Call the function to fetch and display data
fetch_plate_data()