import serial
import requests
import re
import psycopg2

# Function to send reverse geocoding request
def reverse_geocode(lat, lon,apikey):
    BASE_URL = 'https://map.ir/reverse/'
    headers = {
        'x-api-key': apikey,
        'content-type': 'application/json'
    }
    params = {
        'lat': lat,
        'lon': lon
    }
    try:
        response = requests.get(BASE_URL, headers=headers, params=params)
        response.raise_for_status()
        data = response.json()
        location_name = data.get('address', None)  # Use None if no address is found
        return location_name
    except:
        # On any error, return None
        return None

# Function to extract coordinates from a string
def extract_coords(data):
    pattern = r'(\d+\.\d+),(\d+\.\d+)'  # Regex to capture latitude and longitude
    match = re.search(pattern, data)
    if match:
        lat = float(match.group(1))
        lon = float(match.group(2))
        return lat, lon
    return None, None

# Function to read from COM3 and return location name
def read_location_from_com3():
    try:
        DB_NAME = "license_plate_db"
        DB_USER = "postgres"
        DB_PASSWORD = "m102030m"
        DB_HOST = "localhost"
        DB_PORT = "5432"
        conn = psycopg2.connect(dbname=DB_NAME, user=DB_USER, password=DB_PASSWORD, host=DB_HOST, port=DB_PORT)
        cursor = conn.cursor()

        # Execute the query to fetch the location
        cursor.execute("SELECT gpsport, location,api_key FROM configuration LIMIT 1")
        row = cursor.fetchone()
        gpsport , lastlocation,api_key = row
        print(row)
        # Configure serial connection
        ser = serial.Serial(
            port=gpsport,
            baudrate=115200,
            bytesize=serial.EIGHTBITS,
            parity=serial.PARITY_NONE,
            stopbits=serial.STOPBITS_ONE,
            timeout=1
        )
        if ser.is_open:
            print(f"Connected to {ser.port} at {ser.baudrate} baudrate.")

        # Read and parse serial data
        for i in range(1):
            data = ser.readline()
            if data:
                # Decode the received data
                decoded_data = data.decode('utf-8', errors='replace').strip()
                print(f"Received: {decoded_data}")

                # Extract latitude and longitude
                lat, lon = extract_coords(decoded_data)
                if lat is None or lon is None:
                    print("Invalid coordinates in received data.")
                    return None

                # Get the location name using reverse geocoding
                location_name = reverse_geocode(lat,lon,api_key)
                if location_name is None:
                    print("Error occurred. Returning null.")
                    return None  # Return null on error

                print(f"Location: {location_name}")
                cursor.execute("""
                    UPDATE configuration
                    SET location = %s
                    WHERE id = (SELECT id FROM configuration LIMIT 1);
                """, (location_name,))
                return location_name
            

    except:
        
        return None
    finally:
        if 'ser' in locals() and ser.is_open:
            ser.close()
            print("Serial port closed.")
def check_gps_port():
    try:
        conn = psycopg2.connect(dbname="license_plate_db", user="postgres", password="m102030m", host="localhost", port="5432")
        cursor = conn.cursor()
        cursor.execute("SELECT gpsport, location, api_key FROM configuration LIMIT 1")
        row = cursor.fetchone()

        for port_number in range(1, 9):
            gpsport = f"COM{port_number}"
            try:
                ser = serial.Serial(
                    port=gpsport,
                    baudrate=115200,
                    bytesize=serial.EIGHTBITS,
                    parity=serial.PARITY_NONE,
                    stopbits=serial.STOPBITS_ONE,
                    timeout=1
                )
                if ser.is_open:
                    print(f"GPS device detected on {gpsport}")
                    cursor.execute("UPDATE configuration SET gpsport = %s WHERE id = (SELECT id FROM configuration LIMIT 1)", (gpsport,))
                    conn.commit()
                    ser.close()
                    print(f"Database updated with GPS port: {gpsport}")
                    return gpsport
            except (serial.SerialException, serial.SerialTimeoutException):
                continue

        print("No GPS device found.")
        conn.close()
    except psycopg2.Error as e:
        print(f"Database error: {e}")

if __name__ == "__main__":
    check_gps_port()
    # Fetch location name
    location = read_location_from_com3()
    print(f"Location fetched: {location}")