
#People Counting and Face Recognition in Video-Based for Retail Analytic with YOLO

<img width="794" height="622" alt="image" src="https://github.com/user-attachments/assets/b749e2ca-7afb-4a65-9e72-5051c6069ca6" />



The following objectives are proposed:

1. Classify demographic customers by age range and gender with computer vision.
2. Tracking and detecting customers who enter inside or outside a retail store.
3. A DCNN model that learns how to perform efficiently and accurately by comparing the ground truth.
4. Report with dashboard: summary with visualization that can presented on computer desktops and mobile devices

#Dataset Used

The dataset used in this paper is the Adience dataset, available at the Kaggle website.  https://www.kaggle.com/datasets/ttungl/adience-benchmark-gender-and-age-classification
It comprises face photos in real-world imaging conditions like noise, lighting, pose, and appearance, collected from Flickr albums and distributed under the Creative Commons (CC) license. It has 26,580 photos of 2,284 subjects in eight age ranges and is about 1 GB in size.

The actual test video footage is from the iProx CCTV HD Dome 3MP and 4MP high-resolution CCTV cameras sold on hdcctvcameras.net. This video was recorded via a Hikvision 8-channel POE NVR. To view more high-resolution CCTV videos and photos. They are recorded in the home care product retail store.


#Installation and Operation

The installation procedure requires you to install OpenCV (cv2) and argparse, a standard Python library used for manipulating and parsing parameters, to be able to run this project. 
You can do this with pip

- pip install opencv-python

- pip install argparse

Download and extract the zip file: Download the zip file containing the code-model files and sample video images and extract it to a folder of your choice. Inside this folder you will find the following files:
 
-opencv_face_detector.pbtxt

-opencv_face_detector_uint8.pb

-age_deploy.prototxt

-age_net.caffemodel

-gender_deploy.prototxt

-gender_net.caffemodel

-HD CCTV.mp4

-tracker.py

-main.py

-detect.py


#Installation Library
 

OpenCV: Used for image and video processing.

Math: Used for mathematical calculations

Argparse: Used for manipulating parameters received from the user

Tkinter: Used for creating GUIs

Pillow (PIL): Used for manipulating and processing images.

Pandas: Used for manipulating data in tabular form

Numpy: Used for mathematical calculations and array manipulation

Ultralytics YOLO: Used for mathematical calculations and array manipulation

Tracker: Used for tracking objects

JSON: Used for reading and writing JSON files

OS: Used for manipulating file systems and operating system related operations

Datetime: Used for manipulating dates and times


	Creating python file for running program or main .py for using and test the video
Run the program from a command prompt with python your_script.py --video path_to_your_video_file , where your_script.py is the name of the python file you created and path_to_your_video_file is the path to the video file you want to capture, or python main.py --video "HD CCTV.mp4" , which is the file we're working with.
Program Operation details

	The program reads frames from a video file and uses a DNN (Deep et al.) model to detect faces in each frame. It detects the gender and age of the people in the frames using the trained model. It uses the YOLO model to detect and track people entering and leaving a specified area by displaying a coloured border around the detected people and their gender and age information. It shows the number of people entering and leaving the store and the number of people in the store at any given time. It displays the detection results on the screen and saves them to JSON.


 #Cautions and Limitation

Gender and age detection may be subject to errors due to lighting, obstacles, facial poses, and excessive distances. Age prediction is difficult due to several factors, so this program chooses to make it a classification problem instead of a prediction problem.

 #Conclusion
This project uses Deep Learning technology to detect and track people, including gender and age detection from videos. This technology can be applied in real-world situations, such as checking people entering and exiting stores or other locations
