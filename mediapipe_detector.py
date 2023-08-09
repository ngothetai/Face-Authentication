import mediapipe as mp
import cv2
import numpy as np
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

def detect(image, face_detection, draw_bbox=True):
    frame_height, frame_width, _ = image.shape # get shape of image frame
    cropped_img = image.copy() # create a copy original image
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    
    results = face_detection.process(image) # detect face by mediapipe library
    if results.detections:
        for face in results.detections:
            # Get bounding box location
            face_react = np.multiply(
        [
            face.location_data.relative_bounding_box.xmin,
            face.location_data.relative_bounding_box.ymin,
            face.location_data.relative_bounding_box.width,
            face.location_data.relative_bounding_box.height,
        ],
        [frame_width, frame_height, frame_width, frame_height],
    ).astype(int)
        x, y, w, h = tuple(face_react)
        
        # Get cropped image
        cropped_img = image.copy()[max(y,0):min(frame_height, y+h), max(x, 0):min(frame_width,x+w)]
    
    # cropped image option
    if draw_bbox:
        # Draw bounding box into image frame
        image = cv2.rectangle(img=image, pt1=(x, y), pt2=(x+w, y+h), color=(255, 0, 0), thickness=2)
        image = cv2.imshow('MediaPipe Face Detection', cv2.flip(image, 1))
    
    return image, cropped_img