import serial
import requests
import re

# Define the API key and base URL
API_KEY = 'eyJ0eXAiOiJKV1QiLCJhbGciOiJSUzI1NiIsImp0aSI6IjJjZWVjODdlMGI4Mzg5N2M4YTI3MWEzZmQ4NDQ2YmIwZTJmODE2NmExYTBlYjc1YmQ1NTNjYTVhMzBmYTQ3MjM5OWVmOTU5OGVjYzgxMjk1In0.eyJhdWQiOiIzMDA4MyIsImp0aSI6IjJjZWVjODdlMGI4Mzg5N2M4YTI3MWEzZmQ4NDQ2YmIwZTJmODE2NmExYTBlYjc1YmQ1NTNjYTVhMzBmYTQ3MjM5OWVmOTU5OGVjYzgxMjk1IiwiaWF0IjoxNzM0MzgyODYyLCJuYmYiOjE3MzQzODI4NjIsImV4cCI6MTczNjg4ODQ2Miwic3ViIjoiIiwic2NvcGVzIjpbImJhc2ljIl19.DXpzXl6tVvPqBjs-5bvRQJN5uE09XKo015Iz8nRueWmGcx7oF-TKxnLIAWi2s0jCFVbh6XXBxto3vVDsNBTaZpo5vW1qcUR6g99X_gHtfEm5UKCW6Y4nemLrXz2ihnpS1CDKvYSB-r91aoqAOYfKGvnIxFc5PWxWkhlfRxzvV0WJveIbt7O5fof9qdTJCX-ARQPYaPqNHcC8aFFpiGu0e28TsppNxce78fQnObgnXXzfqYjoAvZ1Fiqg2bVDRgDGTeuxckWPzrjKCIx0EPH5McQpFl_ukFfXqkdgE-CvOBIWBmez5BxbDukjjc0seDJlu2wP4HiLuRSn7rY9pMAi3w'
BASE_URL = 'https://map.ir/reverse/'


# Function to send reverse geocoding request
def reverse_geocode(lat, lon):
    headers = {
        'x-api-key': API_KEY,
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
        location_name = data.get('address', 'No address found')
        return location_name  # Return location name
    except requests.exceptions.RequestException as e:
        print(f"Error: {e}")
        return "Error retrieving location"
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
    # try:
    #     # Configure serial connection
    #     ser = serial.Serial(
    #         port='COM3',
    #         baudrate=115200,
    #         bytesize=serial.EIGHTBITS,
    #         parity=serial.PARITY_NONE,
    #         stopbits=serial.STOPBITS_ONE,
    #         timeout=1
    #     )
    #     if ser.is_open:
    #         print(f"Connected to {ser.port} at {ser.baudrate} baudrate.")

        # Read and parse serial data
    return 0 
# if __name__ == "__main__":
#     # Fetch location name
#     location = read_location_from_com3()

#     # Pass location to another function or code
