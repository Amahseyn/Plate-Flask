import socketio

# Create SocketIO client instance
sio = socketio.Client()

@sio.event
def connect():
    print('Connected to WebSocket server')

@sio.event
def plate_detected(data):
    print('Received plate_detected event with data:', data)

sio.connect('ws://localhost:5000')  # Connect to Flask-SocketIO server

sio.wait()  # Wait for events to be handled
