# In[0]:
import cv2
import glob
import numpy as np
from complexCase import getComplexList
import csv
import threading
import queue
import os
import random

fileList = glob.glob("../testOut_CROP_no_clahe/*.jpg")

cropFolder = "/ddsmDataset/test003/"

complexCaseList = getComplexList()

settingList = ['../mass_case_description_test_setR.csv', '../calc_case_description_test_set.csv',
               '../mass_case_description_train_setR.csv', '../calc_case_description_train_set.csv']
'''
print("Start cleaning previous files...")
for f in glob.glob(cropFolder + "*"):
    os.remove(f)
print("Clean finish.")
'''

mutex = threading.Lock()

trainText = open(cropFolder + "train.txt", 'w')
testText = open(cropFolder + "test.txt", 'w')

trainTextR = open(cropFolder + "trainR.txt", 'w')
testTextR = open(cropFolder + "testR.txt", 'w')

def proc(ele,splitNum):
    global complexCaseList
    global settingList
    global cropFolder
    # check if it is mask
    if ele.find("MASK") == -1 and ele.find("Mass") != -1:
        fileInfo = ele.split("/")[2].split(".")[0]

        # check if it is a complex image
        complexFlag = False
        for complex in complexCaseList:
            if ele.find(complex.replace("\n", "")) != -1:
                print("xxx")
                complexFlag = True
                break
        if complexFlag:
            return 0
        # get mask infomation
        maskList = glob.glob("../testOut_CROP_no_clahe/" + fileInfo + "_MASK*")

        # open image
        img = cv2.imread(ele, cv2.IMREAD_GRAYSCALE)
        imgRowOriginal, imgColOriginal = np.where(img != 0)
        height, width = img.shape

        # start cropping
        count = 0
        emptyCount = 0
        emptyList = []
        
        # cropsize
        cropsize = 608

        # record data
        cropROIText = open("/ddsmDataset/test003/emptyRecord/" + fileInfo + ".txt", 'w')
        
        # iter. crop
        for heightIndex in range(0, height, 32):
            for widthIndex in range(0, width, 32):
                # make empty flag if this crop no any mask
                emptyFlag = True
                # check if the crop is out of bound
                if heightIndex + cropsize > height-1 or widthIndex + cropsize > width-1:
                    continue
                cropImg = img[heightIndex: heightIndex +
                              cropsize, widthIndex: widthIndex+cropsize]
                # check if the crop have > 0.15 black area
                row, col = np.where(cropImg == 0)
                if len(row)/(cropsize*cropsize) > 0.15:
                    continue
                # split folder
                splitFolder = str(splitNum) + "/"
                # new file name
                newFileName = fileInfo + "_" + str(count)
                # store roi in this crop
                roiInfo = []
                
                newCategory = None
                
                # iter. mask
                for mask in maskList:
                    maskROInum = mask.split("/")[2].split(".")[
                        0].split("_")[6]
                    maskImg = cv2.imread(mask, cv2.IMREAD_GRAYSCALE)
                    cropMask = maskImg[heightIndex: heightIndex +
                                       cropsize, widthIndex: widthIndex+cropsize]
                    maskRowOriginal, maskColOriginal = np.where(maskImg != 0)
                    maskRow, maskCol = np.where(cropMask != 0)
                    
                    if len(maskRow) != 0:
                        emptyFlag = False

                    # check if mask region more then 0.25 of entire object
                    if len(maskRowOriginal) > len(imgRowOriginal) * 0.25:
                        print("Mask too large, pass.")
                        continue

                    # if mask region is not entire roi then break
                    if len(maskRowOriginal) != len(maskRow):
                        roiInfo.clear()
                        break

                    # get mask bound
                    yMin = maskRow[np.argmin(maskRow)]
                    yMax = maskRow[np.argmax(maskRow)]
                    xMin = maskCol[np.argmin(maskCol)]
                    xMax = maskCol[np.argmax(maskCol)]

                    # turn to yolo format

                    xCenter = str(round((xMin+xMax)/(2*cropsize), 6))
                    yCenter = str(round((yMin+yMax)/(2*cropsize), 6))
                    xWidth = str(round((xMax-xMin)/cropsize, 6))
                    yHeight = str(round((yMax-yMin)/cropsize, 6))

                    # get img type
                    '''imgtype = fileInfo.split("_")[0]
                    csvFileName = None
                    if imgtype == "Calc-Test":
                        csvFileName = settingList[1]
                    elif imgtype == "Calc-Training":
                        csvFileName = settingList[3]
                    elif imgtype == "Mass-Test":
                        csvFileName = settingList[0]
                    elif imgtype == "Mass-Training":
                        csvFileName = settingList[2]
                    else:
                        assert False'''

                    # open csv to get class
                    BIRADSclass = None
                    with open(settingList[0], newline='') as csvfile:
                        # read csv all rows
                        rows = csv.reader(csvfile)
                        headers = next(rows)
                        # per row
                        for picData in rows:
                            csvFileInfo = picData[11].split("/")[0]
                            csvROInum = picData[12].split("/")[0].split("_")[5]
                            if csvFileInfo == fileInfo and csvROInum == maskROInum:
                                if picData[9] == "BENIGN_WITHOUT_CALLBACK":
                                    continue
                                elif picData[9] == "BENIGN":
                                    BIRADSclass = "0"
                                elif picData[9] == "MALIGNANT":
                                    BIRADSclass = "1"
                                else:
                                    assert False
                                break
                    if BIRADSclass != None:
                        newCategory = "Test"
                        roiInfo.append([BIRADSclass,xCenter,yCenter,xWidth,yHeight])
                        continue
                    with open(settingList[2], newline='') as csvfile:
                        # read csv all rows
                        rows = csv.reader(csvfile)
                        headers = next(rows)
                        # per row
                        for picData in rows:
                            csvFileInfo = picData[11].split("/")[0]
                            csvROInum = picData[12].split("/")[0].split("_")[5]
                            if csvFileInfo == fileInfo and csvROInum == maskROInum:
                                if picData[9] == "BENIGN_WITHOUT_CALLBACK":
                                    continue
                                elif picData[9] == "BENIGN":
                                    BIRADSclass = "0"
                                elif picData[9] == "MALIGNANT":
                                    BIRADSclass = "1"
                                else:
                                    assert False
                                break
                    if BIRADSclass != None:
                        newCategory = "Train"
                        roiInfo.append([BIRADSclass,xCenter,yCenter,xWidth,yHeight])
                        continue

                # if roi exist
                if len(roiInfo) != 0:
                    # record
                    cropROIText.write(str(widthIndex) + " " + str(heightIndex) + " 1\n")
                    # write crop img
                    cv2.imwrite(cropFolder + splitFolder + newFileName + ".jpg", cropImg, [cv2.IMWRITE_JPEG_QUALITY, 100])
                    # new textfile to save roi
                    roiFile = open(cropFolder + splitFolder + newFileName + ".txt", 'w')
                    for ele in roiInfo:
                        roiFile.write(ele[0] + " " + ele[1] + " " + ele[2] + " " + ele[3] + " " + ele[4] + "\n")
                    roiFile.close()
                    # write to train or test textfile
                    mutex.acquire()
                    if newCategory == "Test":
                        testText.write(cropFolder + splitFolder + newFileName + ".jpg\n")
                        testTextR.write(cropFolder + splitFolder + newFileName + ".jpg\n")
                    elif newCategory == "Train":
                        trainText.write(cropFolder + splitFolder + newFileName + ".jpg\n")
                        trainTextR.write(cropFolder + splitFolder + newFileName + ".jpg\n")
                    else:
                        assert False
                    mutex.release()
                elif emptyFlag:
                    # record
                    cropROIText.write(str(widthIndex) + " " + str(heightIndex) + " 0\n")
                    # if this crop image no roi then record to list
                    emptyCount+=1
                    emptyList.append([newFileName, heightIndex, widthIndex])
                count += 1

        cropROIText.close()
        '''
        random.shuffle(emptyList)
        appendSize = (count-emptyCount) if (count-emptyCount) < len(emptyList) else len(emptyList)
        for ele in emptyList[:appendSize]:
            # write crop img
            cv2.imwrite(cropFolder + splitFolder + ele[0] + ".jpg", img[ele[1]: ele[1] + cropsize, ele[2]: ele[2] + cropsize], [cv2.IMWRITE_JPEG_QUALITY, 100])
            # new textfile to save roi
            roiFile = open(cropFolder + splitFolder + ele[0] + ".txt", 'w')
            roiFile.close()
            # write to train or test textfile
            mutex.acquire()
            if ele[0].find("Training") != -1:
                trainText.write(cropFolder + splitFolder + ele[0] + ".jpg\n")
            elif ele[0].find("Test") != -1:
                testText.write(cropFolder + splitFolder + ele[0] + ".jpg\n")
            else:
                assert False
            mutex.release()
        '''


class Worker(threading.Thread):
    def __init__(self, queue, num):
        threading.Thread.__init__(self)
        self.queue = queue
        self.num = num

    def run(self):
        while self.queue.qsize() > 0:
            # get msg from queue
            msg = self.queue.get()
            # start worker
            proc(msg, self.num)

my_queue = queue.Queue()

for ele in fileList:
    my_queue.put(ele)

print("Start cropping worker.")

# init Worker
my_worker1 = Worker(my_queue, 1)
my_worker2 = Worker(my_queue, 2)
my_worker3 = Worker(my_queue, 3)
my_worker4 = Worker(my_queue, 4)
#my_worker5 = Worker(my_queue, 5)


# start worker
my_worker1.start()
my_worker2.start()
my_worker3.start()
my_worker4.start()
#my_worker5.start()

# join worker
my_worker1.join()
my_worker2.join()
my_worker3.join()
my_worker4.join()
#my_worker5.join()

trainText.close()
testText.close()
trainTextR.close()
testTextR.close()

print("Done.")
# %%
