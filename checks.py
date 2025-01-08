import requests
import serial
def check_internet_connection(API_KEY):
    BASE_URL = 'https://map.ir/reverse/'
    headers = {
        'x-api-key': API_KEY,
        'content-type': 'application/json'
    }
    lat = 35.73247
    lon = 51.42268
    params = {
        'lat': lat,
        'lon': lon
    }
    try:
        response = requests.get(BASE_URL, headers=headers, params=params)
        response.raise_for_status()  # Raise an exception for HTTP errors
        return True
    except:
        return False
def check_gps_availability(gpsport):
    try:
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

            data = ser.readline()
            if data:
                return True
        else:
            return False
    except:
        return False