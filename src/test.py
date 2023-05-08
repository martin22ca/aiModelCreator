import cv2
from classRecogLib.faceRecog import loadDetectionModel,loadRecognitionModel,loadKNN,encodeFace,findFaces,predictClass

modelR = loadRecognitionModel('src/models/')
modelKnn = loadKNN('src/models/')
modelD = loadDetectionModel()

cam = cv2.VideoCapture(0)
while (cam.isOpened()):
    success, image = cam.read()
    if not success:
        print("Ignoring empty camera frame.")
        continue
    else:
        cv2.imshow('imag',image)
        cv2.waitKey(1)
        faces = findFaces(image, modelD)
        for idx,face in enumerate(faces):
            cv2.imshow('face'+str(idx),face)
            cv2.waitKey(1)
            encoding = encodeFace(face, modelR,2)
            print(predictClass(encoding,modelKnn))