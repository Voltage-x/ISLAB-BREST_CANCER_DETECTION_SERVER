from unicodedata import category
import cv2
import glob
import numpy as np

imgFolder = "/ddsmDataset/yolo_ddsm_modify_crop_stride800/"

cropMode = True

if cropMode:
    testText = "/ddsmDataset/yolo_ddsm_modify_crop_stride800/test.txt"
    textFile = open(testText,'r')
    textFileList = textFile.readlines()
    textFile.close()
    textFileName = [ele.replace("\n","").split("/")[4] for ele in textFileList]
    textFileFold = [ele.replace("\n","").split("/")[3] for ele in textFileList]


for txtPath in glob.glob("../labelsFromYOLOR/*.txt"):
    #print(txtPath)
    fileName = txtPath.split("/")[-1].replace(".txt", ".jpg")
    if cropMode:
        img = cv2.imread(imgFolder + textFileFold[textFileName.index(fileName)] + "/" + fileName,0)
    else:
        img = cv2.imread(imgFolder + fileName,0)
    txtFile = open(txtPath, 'r')
    height, width = img.shape[:2]
    count = 0
    txtLine = txtFile.readlines()
    txtFile.close()
    #txtFile = open(txtPath, 'w')
    for roi in txtLine:
        # print(roi)
        c, x, y, w, h = roi.replace("\n", "").split(" ")[0:5]
        #print(x, y, w, h)
        x1 = float(x)*width - float(w)*width/2
        y1 = float(y)*height - float(h)*height/2
        x2 = float(x)*width + float(w)*width/2
        y2 = float(y)*height + float(h)*height/2
        cropROI = img[int(y1):int(y2), int(x1):int(x2)]
        
        '''blackRow = np.where((cropROI[:, :] < 10))
        if len(blackRow[0]) > cropROI.shape[0]*cropROI.shape[1]*0.5:
            continue
        else:
            txtFile.write(roi)'''
        #print(x1, y1, x2, y2)
        if c == "0":
            categoryFold = "BENIGN/"
        elif c == "1":
            categoryFold = "MALIGNANT/"
        else:
            assert False
        try:
            cv2.imwrite("./cropResult/" + categoryFold + txtPath.split("/")[-1].split(".")[0] + "_" + str(count) + ".jpg", cropROI)
        except:
            print(txtPath)
        count += 1
    #txtFile.close()
