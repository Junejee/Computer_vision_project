# Import libraries
import cv2 # For computer vision tasks
import argparse  # For parsing command-line arguments
from tkinter import * # For creating graphical user interfaces
from PIL import Image, ImageTk
import numpy as np # For numerical operations on arrays
from datetime import datetime # For working with dates and times
from ultralytics import YOLO # For object detection
from tracker import Tracker # For tracking objects
import json # For handling JSON data
import pandas as pd # For data analysis and manipulation

# Function to highlight faces detected in a frame
def highlightFace(net, frame, conf_threshold=0.30): #Highlight faces detected in a frame
    frameOpencvDnn = frame.copy() #Make a copy of the input frame to avoid altering the original image
    frameHeight = frameOpencvDnn.shape[0] # Get the height of the frame
    frameWidth = frameOpencvDnn.shape[1] # Get the width of the frame
    blob = cv2.dnn.blobFromImage(frameOpencvDnn, 1.0, (600, 600), [104, 117, 123], True, False)
    net.setInput(blob)  # Set the input to the neural network
    detections = net.forward() # Perform a forward pass to get the detections
    faceBoxes = [] # Initialize an empty list to store face bounding boxes
    for i in range(detections.shape[2]): # Iterate over all detections
        confidence = detections[0, 0, i, 2]  # Get the confidence score of the detection
        if confidence > conf_threshold: # If the confidence is above the threshold, process the detection
            x1 = int(detections[0, 0, i, 3] * frameWidth) # Calculate the x-coordinate of the top-left corner of the bounding box
            y1 = int(detections[0, 0, i, 4] * frameHeight) # Calculate the y-coordinate of the top-left corner of the bounding box
            x2 = int(detections[0, 0, i, 5] * frameWidth) # Calculate the x-coordinate of the bottom-right corner of the bounding box
            y2 = int(detections[0, 0, i, 6] * frameHeight)  #Calculate the y-coordinate of the bottom-right corner of the bounding box
            faceBoxes.append([x1, y1, x2, y2]) # Append the bounding box coordinates to the list
            cv2.rectangle(frameOpencvDnn, (x1, y1), (x2, y2), (0, 255, 0), int(round(frameHeight/150)), 8) # Draw a rectangle around the detected face
    return frameOpencvDnn, faceBoxes  # Return the frame with highlighted faces and the list of face bounding boxes

# Parse arguments for video file
parser = argparse.ArgumentParser() # Create a parser object to handle command-line arguments
parser.add_argument('--video', required=True) # Add the '--video' argument which is required and represents the path to the video file
args = parser.parse_args() # Parse the command-line arguments and store the result in the 'args' variable

# Load face, age, gender detection models
# File paths for the face detection model's configuration and weights
faceProto = "opencv_face_detector.pbtxt"

faceModel = "opencv_face_detector_uint8.pb"
# File paths for the age detection model's configuration and weights
ageProto = "age_deploy.prototxt"
ageModel = "age_net.caffemodel"
# File paths for the gender detection model's configuration and weights
genderProto = "gender_deploy.prototxt"
genderModel = "gender_net.caffemodel"

# Mean values for model normalization
MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
# List of age ranges corresponding to age detection model outputs
ageList = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
# List of gender categories corresponding to gender detection model outputs
genderList = ['Male', 'Female']

# Load the pre-trained face detection model using OpenCV
faceNet = cv2.dnn.readNet(faceModel, faceProto)
# Load the pre-trained age detection model using OpenCV
ageNet = cv2.dnn.readNet(ageModel, ageProto)
# Load the pre-trained gender detection model using OpenCV
genderNet = cv2.dnn.readNet(genderModel, genderProto)

# Load YOLO model for object detection
model = YOLO('yolov8s.pt')

# Initialize GUI window for displaying age and gender detection
root = Tk() # Create the main window for the application
root.title("Age and Gender Detection") #Set the title of the main window
shift_value = 110
#Define two areas as lists of tuples representing coordinates
area1 = [(366, 715), (830, 406), (854, 422), (428, 714)]
area2 = [(438, 715), (859, 431), (882, 450), (513, 716)]
# Function to handle mouse move event
def RGB(event, x, y, flags, param): #Define a callback function for mouse events
    if event == cv2.EVENT_MOUSEMOVE:
        colorsBGR = [x, y] # Capture the x and y coordinates of the mouse movement
        print(colorsBGR)  # Print the coordinates to the console

cv2.namedWindow('RGB') # Create a named window for displaying the video
cv2.setMouseCallback('RGB', RGB) # Set the callback function for mouse events on the 'RGB' window

# Open video file specified by user
cap = cv2.VideoCapture(args.video)
# Calculate the width and height of the video frames, scaled down by a factor of 4
frame_width = int(cap.get(3)) // 4
frame_height = int(cap.get(4)) // 4

# Create GUI window to display frames
label = Label(root) # Create a Label widget to display video frames in the Tkinter window
label.pack() # Add the Label widget to the Tkinter window
my_file = open("coco.txt", "r") # Open and read the class list from a file
data = my_file.read() # Read the contents of the file
class_list = data.split("\n") # Split the file content into a list of class names

count = 0  # Initialize a counter variable
tracker = Tracker() # Create an instance of the Tracker class

# Dictionaries and sets to track people entering and exiting
people_entering = {}
entering = set()

people_exiting = {}
exiting = set()

# List to store detection results and set to track recorded IDs
detections = []
recorded_ids = set()
last_detection = {}

while True:
    current_time = datetime.now() # Get the current time
    ret, frame = cap.read() # Read a frame from the video capture
    if not ret: # If no frame is read, exit the loop
        break

    frame = cv2.resize(frame, (1020, 720)) # Resize the frame to 1020x720 pixels
# Detect faces in the frame
    resultImg, faceBoxes = highlightFace(faceNet, frame)

    if not faceBoxes: # If no faces are detected, print a message
        print("No face detected")

    padding = 20 # Padding around detected faces
    for faceBox in faceBoxes:
        # Extract face region from the frame and create a blob for model input
        face = frame[max(0, faceBox[1] - padding):min(faceBox[3] + padding, frame.shape[0] - 1), max(0, faceBox[0] - padding):min(faceBox[2] + padding, frame.shape[1] - 1)]
        blob = cv2.dnn.blobFromImage(face, 1.0, (227, 227), MODEL_MEAN_VALUES, swapRB=False)
        #Predict gender
        genderNet.setInput(blob)
        genderPreds = genderNet.forward()
        gender = genderList[genderPreds[0].argmax()]
        print(f'Gender: {gender}')

        #Predict age
        ageNet.setInput(blob)
        agePreds = ageNet.forward()
        age = ageList[agePreds[0].argmax()]
        print(f'Age: {age[1:-1]} years')

        if len(faceBox) >= 4:
            prediction = {
                "id": len(detections) + 1,
                "gender": gender,
                "age": age,
                "position": {
                    "x1": faceBox[0],
                    "y1": faceBox[1],
                    "x2": faceBox[2],
                    "y2": faceBox[3]
                }
            }
            detections.append(prediction) # Append the prediction to the detections list

    count += 1
    #  # Skip processing if count is odd
    if count % 2 != 0:
        continue

    # Predict objects using YOLO model
    results = model.predict(frame)
    a = results[0].boxes.data.cpu().numpy() # Extract bounding box data
    px = pd.DataFrame(a).astype("float")
    list = []

    # Loop for predict object
    for index, row in px.iterrows():
        x1 = int(row[0])
        y1 = int(row[1])
        x2 = int(row[2])
        y2 = int(row[3])
        confidence = row[4]
        d = int(row[5])
        
        # Check if the detected class index is valid
        if 0 <= d < len(class_list):
            c = class_list[d]
        else:
            print(f"Index out of range: {d} for class_list with length {len(class_list)}")
            continue
        
        # If the detected class is 'person' and confidence is high, process it
        if 'person' in c and confidence > 0.5:
            # Append position to list
            list.append([x1, y1, x2, y2])
            # Draw bounding box around detected person
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            # Loop in facebox
            for faceBox in faceBoxes:
                # Check if the detected face is within the bounding box of the person
                if x1 <= faceBox[0] <= x2 and y1 <= faceBox[1] <= y2:
                    # Annotate the frame with gender and age
                    cv2.putText(frame, f'person, {gender}, {age}', (x1, y1 - 10), cv2.FONT_HERSHEY_COMPLEX, 0.7, (0, 255, 255), 2)
                                                         
                        
    # Update object positions using tracker
    bbox_id = tracker.update(list)
    for bbox in bbox_id:
        x3, y3, x4, y4, id = bbox
        results = cv2.pointPolygonTest(np.array(area2, np.int32), ((x4, y4)), False)
        if results >= 0:
            people_entering[id] = (x4, y4)
            cv2.rectangle(frame, (x3, y3), (x4, y4), (0, 0, 255), 2) # Draw rectangle for entering persons

        if id in people_entering:
            results = cv2.pointPolygonTest(np.array(area1, np.int32), ((x4, y4)), False)
            if results >= 0:
                cv2.rectangle(frame, (x3, y3), (x4, y4), (0, 255, 0), 2) # Draw rectangle for exiting persons
                cv2.circle(frame, (x4, y4), 5, (255, 0, 255), -1) # Draw a circle at the person's location
                cv2.putText(frame, str(id), (x3, y3 - 10), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 255), 1) # Annotate with ID
                entering.add(id)

        results2 = cv2.pointPolygonTest(np.array(area1, np.int32), ((x4, y4)), False)
        if results2 >= 0:
            people_exiting[id] = (x4, y4)
            cv2.rectangle(frame, (x3, y3), (x4, y4), (0, 255, 0), 2) # Draw rectangle for exiting persons

        if id in people_exiting:
            results3 = cv2.pointPolygonTest(np.array(area2, np.int32), ((x4, y4)), False)
            if results3 >= 0:
                cv2.rectangle(frame, (x3, y3), (x4, y4), (255, 0, 255), 2) # Draw rectangle for exiting person
                cv2.circle(frame, (x4, y4), 5, (255, 0, 255), -1) # Draw a circle at the person's location
                cv2.putText(frame, str(id), (x3, y3 - 10), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 255), 1) # Annotate with ID
                exiting.add(id)

    #  Draw a polygonal line around the defined area1
    cv2.polylines(frame, [np.array(area1, np.int32)], True, (255, 0, 0), 2)
    # Label area1 with the number '1'
    cv2.putText(frame, '1', (366, 715), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 0), 1)
    #  Draw a polygonal line around the defined area2
    cv2.polylines(frame, [np.array(area2, np.int32)], True, (255, 0, 0), 2)
    # Label area2 with the number '2'
    cv2.putText(frame, '2', (300, 715), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 0), 1)

    # Count the number of people entering, exiting, and inside the store
    i = len(entering) # Number of people currently entering the store
    o = len(exiting) # Number of people currently exiting the store
    inside_store = len(list) # Number of people currently inside the store
    print('i', i) # Print the number of people entering the store
    print('o', o) # Print the number of people exiting the store
    print('confidence', confidence) # Print the confidence value (last detected confidence)
    #  Display the number of people entering, exiting, and inside the store on the frame
    cv2.putText(frame, f"People entering the store: {str(i)}", (60, 80), cv2.FONT_HERSHEY_COMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(frame, f"People leaving the store: {str(o)}", (60, 110), cv2.FONT_HERSHEY_COMPLEX, 0.7, (0, 0, 255), 2)
    cv2.putText(frame, f"People in the store: {str(inside_store)}", (60, 140), cv2.FONT_HERSHEY_COMPLEX, 0.7, (255, 255, 0), 2)

    # Display the processed frame in the "RGB" window
    cv2.imshow("RGB", frame)

    # Wait for the ESC key (key code 27) to exit the loop
    if cv2.waitKey(1) & 0xFF == 27:
        break

# Release the video capture object
cap.release()
# Close all OpenCV windows
cv2.destroyAllWindows()

# Save the face detection results as a JSON file
with open('detections_yolov8s.json', 'w', encoding='utf-8') as f:
    json.dump(detections, f, ensure_ascii=False, indent=4)