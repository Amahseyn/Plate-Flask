import requests

def reverse_geocode(latitude, longitude, api_key):
    # Define the URL for the Neshan API
    url = f"https://api.neshan.org/v5/reverse?lat={latitude}&lng={longitude}"

    # Define the headers with the API key
    headers = {
        "Api-Key": api_key
    }

    # Send the GET request to the API
    response = requests.get(url, headers=headers)

    # Check if the request was successful
    if response.status_code == 200:
        return response.json()  # Return the JSON response
    else:
        return {"error": f"Failed to retrieve data. Status code: {response.status_code}"}

# Example usage
latitude = 35.73247
longitude = 51.42268
api_key = "service.6c3b50bd19a84afb8c7a1e5127bfa0d2"  # Replace with your actual Neshan API key

result = reverse_geocode(latitude, longitude, api_key)

# Print the result
print(result)