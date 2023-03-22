from operator import rshift
import cv2 as cv 
import numpy as np
import mediapipe as mp 
mp_face_mesh = mp.solutions.face_mesh
from time import sleep

# From (https://raw.githubusercontent.com/google/mediapipe/master/mediapipe/modules/face_geometry/data/canonical_face_model_uv_visualization.png)

LEFT_IRIS = [474,475, 476, 477]
RIGHT_IRIS = [469, 470, 471, 472]

RIGHT_CORNERS = [33, 362]
LEFT_CORNERS = [133, 263]

OFFSET_THRESHOLD = 5

address = 'tcp://MM.local:3333'
print("Connecting to tcp video stream")
cap = cv.VideoCapture(0)
#Add a check to see if stream is opened correctly
with mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
) as face_mesh:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error reading frame")
            break
        frame = cv.flip(frame, 1)
        rgb_frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        img_h, img_w = frame.shape[:2]
        results = face_mesh.process(rgb_frame)
        if results.multi_face_landmarks:
            # Get Mesh Points
            mesh_points=np.array([np.multiply([p.x, p.y], [img_w, img_h]).astype(int) for p in results.multi_face_landmarks[0].landmark])
            
            left_iris_points = mesh_points[LEFT_IRIS]
            right_iris_points = mesh_points[RIGHT_IRIS]

            left_iris = left_iris_points[0][0] + left_iris_points[1][0] + left_iris_points[2][0] + left_iris_points[3][0]
            left_iris = left_iris//4
            right_iris = right_iris_points[0][0] + right_iris_points[1][0] + right_iris_points[2][0] + right_iris_points[3][0]
            right_iris = right_iris//4

            ### Calculate Eye Corners
            lcorners = mesh_points[LEFT_CORNERS]
            lcorners = [lcorners[0][0], lcorners[1][0]]
            rcorners = mesh_points[RIGHT_CORNERS]
            rcorners = [rcorners[0][0], rcorners[1][0]]

            # Calculate angle
            # Represented as % of distance the centre of iris is from left corner to right corner of eye
            total_dist = lcorners[1] - lcorners[0] + rcorners[1] - rcorners[0]
            offset = left_iris - lcorners[0] + right_iris - rcorners[0]
            angle = offset/total_dist
            print(angle)
            sleep(0.1)

        cv.imshow('img', frame)
        key = cv.waitKey(1)
        if key ==ord('q'):
            break
cap.release()
cv.destroyAllWindows()
