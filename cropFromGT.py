import glob
import cv2
from yoloLabelConverter import fromYolo

mainFolder = "/ddsmDataset/yolo_ddsm_modify_crop_stride800/"
newFolder = "/ddsmDataset/classification_BM_DDSM_crop/"

testFile = open(f"{mainFolder}test.txt",'r')
trainFile = open(f"{mainFolder}train.txt",'r')

trainFileList = trainFile.readlines()
testFileList = testFile.readlines()

trainFile.close()
testFile.close()

for rawImgPath in trainFileList:
    imgPath = rawImgPath.replace("\n","")
    fileName = imgPath.split("/")[-1].split(".")[0]
    img = cv2.imread(imgPath)
    roiFile = open(imgPath.replace(".jpg",".txt"),'r')
    roiFileList = roiFile.readlines()
    roiFile.close()
    count = 0
    for rawRoi in roiFileList:
        c, x, y, w, h = rawRoi.replace("\n","").split(" ")
        x1, y1, x2, y2 = fromYolo(img, x, y, w, h)
        cropImg = img[y1:y2,x1:x2]
        # BENIGN
        if c == "0":
            cv2.imwrite(f"{newFolder}train/BENIGN/{fileName}_{count}.jpg",cropImg)
        else:
            cv2.imwrite(f"{newFolder}train/MALIGNANT/{fileName}_{count}.jpg",cropImg)
        count += 1

for rawImgPath in testFileList:
    imgPath = rawImgPath.replace("\n","")
    fileName = imgPath.split("/")[-1].split(".")[0]
    img = cv2.imread(imgPath)
    roiFile = open(imgPath.replace(".jpg",".txt"),'r')
    roiFileList = roiFile.readlines()
    roiFile.close()
    count = 0
    for rawRoi in roiFileList:
        c, x, y, w, h = rawRoi.replace("\n","").split(" ")
        x1, y1, x2, y2 = fromYolo(img, x, y, w, h)
        cropImg = img[y1:y2,x1:x2]
        # BENIGN
        if c == "0":
            cv2.imwrite(f"{newFolder}val/BENIGN/{fileName}_{count}.jpg",cropImg)
        else:
            cv2.imwrite(f"{newFolder}val/MALIGNANT/{fileName}_{count}.jpg",cropImg)
        count += 1