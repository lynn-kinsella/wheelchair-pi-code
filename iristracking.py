from operator import rshift
import cv2 as cv 
import numpy as np
import mediapipe as mp 
mp_face_mesh = mp.solutions.face_mesh

# Face Mesh indicies that compose Left Eye
LEFT_EYE =[ 362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385,384, 398 ]
# Face Mesh Indicies that compose Right Eye
RIGHT_EYE=[ 33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161 , 246 ] 

# From (https://raw.githubusercontent.com/google/mediapipe/master/mediapipe/modules/face_geometry/data/canonical_face_model_uv_visualization.png)
# Indicies for Middle Left Eye
LEFT_CENTRE=[380, 374, 373, 385, 386, 387]
# Indicies for Middle Right Eye
RIGHT_CENTRE=[144, 145, 153, 160, 159, 158]

LEFT_IRIS = [474,475, 476, 477]
RIGHT_IRIS = [469, 470, 471, 472]

OFFSET_THRESHOLD = 5

address = 'MM.local'
print("Connecting to tcp video stream")
cap = cv.VideoCapture('tcp://MM.local:3333')
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
            
            # Calculate Iris Position
            (l_ix, l_iy), l_radius = cv.minEnclosingCircle(mesh_points[LEFT_IRIS])
            (r_ix, r_iy), r_radius = cv.minEnclosingCircle(mesh_points[RIGHT_IRIS])
            left_iris = np.array([l_ix, l_iy], dtype=np.int32)
            right_iris = np.array([r_ix, r_iy], dtype=np.int32)
            #print("Iris: {}{}", left_iris, right_iris)

            # Calculate Eye Position
            (l_cx, l_cy), l_radius = cv.minEnclosingCircle(mesh_points[LEFT_CENTRE])
            (r_cx, r_cy), r_radius = cv.minEnclosingCircle(mesh_points[RIGHT_CENTRE])
            left_eye = np.array([l_cx, l_cy], dtype=np.int32)
            right_eye = np.array([r_cx, r_cy], dtype=np.int32)
            #print("Eye: {}{}", left_eye, right_eye)

            # Draw
            cv.circle(frame, left_iris, int(l_radius), (255,0,255), 1, cv.LINE_AA)
            cv.circle(frame, right_iris, int(r_radius), (255,0,255), 1, cv.LINE_AA)
            cv.circle(frame, left_eye, int(l_radius), (255,0,255), 1, cv.LINE_AA)
            cv.circle(frame, right_eye, int(r_radius), (255,0,255), 1, cv.LINE_AA)

            # Compute Direction
            left_offset = left_iris[0] - left_eye[0]      
            right_offset = right_iris[0] - right_eye[0]
            offset = left_offset + right_offset

            #print(offset)

            if(offset < -OFFSET_THRESHOLD):
                print("Left")
            elif(offset > OFFSET_THRESHOLD):
                print("Right")
            else:
                print("Straight")

        cv.imshow('img', frame)
        key = cv.waitKey(1)
        if key ==ord('q'):
            break
cap.release()
cv.destroyAllWindows()
