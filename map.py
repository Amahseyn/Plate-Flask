import requests

# Define the API key and base URL
API_KEY = 'eyJ0eXAiOiJKV1QiLCJhbGciOiJSUzI1NiIsImp0aSI6IjJjZWVjODdlMGI4Mzg5N2M4YTI3MWEzZmQ4NDQ2YmIwZTJmODE2NmExYTBlYjc1YmQ1NTNjYTVhMzBmYTQ3MjM5OWVmOTU5OGVjYzgxMjk1In0.eyJhdWQiOiIzMDA4MyIsImp0aSI6IjJjZWVjODdlMGI4Mzg5N2M4YTI3MWEzZmQ4NDQ2YmIwZTJmODE2NmExYTBlYjc1YmQ1NTNjYTVhMzBmYTQ3MjM5OWVmOTU5OGVjYzgxMjk1IiwiaWF0IjoxNzM0MzgyODYyLCJuYmYiOjE3MzQzODI4NjIsImV4cCI6MTczNjg4ODQ2Miwic3ViIjoiIiwic2NvcGVzIjpbImJhc2ljIl19.DXpzXl6tVvPqBjs-5bvRQJN5uE09XKo015Iz8nRueWmGcx7oF-TKxnLIAWi2s0jCFVbh6XXBxto3vVDsNBTaZpo5vW1qcUR6g99X_gHtfEm5UKCW6Y4nemLrXz2ihnpS1CDKvYSB-r91aoqAOYfKGvnIxFc5PWxWkhlfRxzvV0WJveIbt7O5fof9qdTJCX-ARQPYaPqNHcC8aFFpiGu0e28TsppNxce78fQnObgnXXzfqYjoAvZ1Fiqg2bVDRgDGTeuxckWPzrjKCIx0EPH5McQpFl_ukFfXqkdgE-CvOBIWBmez5BxbDukjjc0seDJlu2wP4HiLuRSn7rY9pMAi3w'
BASE_URL = 'https://map.ir/reverse/'

# Function to send a reverse geocoding request
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
        response.raise_for_status()  # Raise an exception for HTTP errors
        data = response.json()
        print(f"Address: {data.get('address', 'No address found')}")
    except requests.exceptions.RequestException as e:
        print(f"Error: {e}")

# Example coordinates
latitude = 35.73247
longitude = 51.42268

for x in range(5):
    reverse_geocode(latitude, longitude)
