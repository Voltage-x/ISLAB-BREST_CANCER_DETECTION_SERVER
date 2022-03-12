import glob
import cv2

mergeFolder = "../labelsFromMerge/"
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
    print(txtPath)
    fileName = txtPath.split("/")[-1].replace(".txt", ".jpg")
    if cropMode:
        img = cv2.imread(imgFolder + textFileFold[textFileName.index(fileName)] + "/" + fileName,0)
    else:
        img = cv2.imread(imgFolder + fileName,0)
    txtFile = open(txtPath, 'r')
    height, width = img.shape[:2]
    count = 0
    mergeTxtFile = open(txtPath.replace(
        "labelsFromYOLOR", "labelsFromMerge"), 'w')
    for roi in txtFile.readlines():
        c, x, y, w, h, conf = roi.replace("\n", "").split(" ")
        #print(x, y, w, h)
        x1 = float(x)*width - float(w)*width/2
        y1 = float(y)*height - float(h)*height/2
        x2 = float(x)*width + float(w)*width/2
        y2 = float(y)*height + float(h)*height/2
        efficientNetTxt = open(txtPath.replace(
            "labelsFromYOLOR", "labelsFromEfficientNet").replace(".txt", "_"+str(count)+".txt"), 'r')
        ENc1, ENc2 = efficientNetTxt.readlines()[0].replace("\n", "").split(" ")
        newClass = ""
        # 0 is bad, 1 is good
        if float(conf) < 0.15:
            if float(ENc1) > float(ENc2):
                c = "0"
            else:
                c = "1"
        '''if float(conf) > 0.01:
            mergeTxtFile.write(c+" "+conf+" "+str(int(x1)) + " "+str(int(y1))+" "+str(int(x2))+" "+str(int(y2))+"\n")
        else:
            continue
            if float(ENc1) > float(ENc2):
                c = "0"
                conf = float(conf) * float(ENc1) * 10
            else:
                c = "1"
                conf = float(conf) * float(ENc2) * 10'''
        mergeTxtFile.write(c+" "+str(conf)+" "+str(int(x1)) + " "+str(int(y1))+" "+str(int(x2))+" "+str(int(y2))+"\n")
        count += 1
        efficientNetTxt.close()
    mergeTxtFile.close()
    txtFile.close()
