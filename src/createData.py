import os 
import cv2
from  classRecogLib.faceRecog import findFaces,loadDetectionModel
from tqdm import tqdm

model = loadDetectionModel()
base = "src/shortDataset/"
baseM = "src/moddedPics/"
num = 0 
for name in tqdm(os.listdir(base)):
    ## Check Ai key db
    if len(os.listdir(base+name+'/')) != 0:
        os.mkdir(baseM+str(num)+'_'+str(name)+'/')
        pcsNum = 0
        for pic in os.listdir(base+name+'/'):
            image = cv2.imread(base+name+'/'+pic)
            try:
                faces = findFaces(image,model)
                for face in faces:
                    #cv2.imshow('face',face)
                    #cv2.waitKey()
                    cv2.imwrite(baseM+str(num)+'_'+str(name)+'/img'+str(pcsNum)+'.jpg',face)
                    pcsNum += 1
            except:
                continue
        num += 1