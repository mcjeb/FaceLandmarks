import cv2
import numpy as np

# confidence level for recognition
conf_lvl = .55

print("[INFO] loading facial landmark predictor...")
detector = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_default.xml')

dnn_path = 'models/'
dnn_detector = cv2.dnn.readNetFromCaffe(dnn_path + 'deploy.prototxt', dnn_path + 'res10_300x300_ssd_iter_140000.caffemodel')

LBFmodel = "models/lbfmodel.yaml"

landmark_detector = cv2.face.createFacemarkLBF()
landmark_detector.loadModel(LBFmodel)

# read in webcam camera
print("[INFO] Initializing camera...")
vid = cv2.VideoCapture(0)

while(True):
    # get frame from camera
    _, frame = vid.read()
    # resize to desired width
    width = 600
    # shape[0] = height; shape[1] = width of frame
    frame = cv2.resize(frame, 
                        (width, int(frame.shape[0]*width/int(frame.shape[1]))), 
                        interpolation=cv2.INTER_AREA)
    # convert image to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
    faces = detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=8, minSize=(50, 50))
    
    (h, w) = frame.shape[:2]
    imgBlob = cv2.dnn.blobFromImage(
                cv2.resize(frame, (300, 300)),  # input image after resize
                    size=(300, 300),            # spatial size of cnn
                    mean=(104.0, 177.0, 123.0), # mean value to subtract in RGB
                    crop = False)

    # detect faces with opencv dnn
    dnn_detector.setInput(imgBlob)
    detections = dnn_detector.forward()

    if len(faces) != 0:
        # assume only one face per image
        # therefore find bounding box with largest probability
        i = np.argmax(detections[0, 0, :, 2])
        confidence = detections[0, 0, i, 2]

        # filter out weak detections
        if confidence < conf_lvl:
            print('[ERROR] face detection not successful')
            continue
        else:
            # compute (x,y) box of face
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype('int')
            face = np.array([startX, startY, endX, endY])

        _, landmarks = landmark_detector.fit(gray, faces)
        
        for landmark in landmarks:
            for x,y in landmark[0]:
            # display landmarks on "image_cropped"
            # with blue colour in BGR and thickness 1
                cv2.circle(frame, (int(x), int(y)), 
                            radius=2, 
                            color=(255, 0, 0), 
                            thickness=-1)

    cv2.imshow('Video', frame)

    # break with q
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
  
# release the video after loop
vid.release()
#destroy all windows
cv2.destroyAllWindows()
