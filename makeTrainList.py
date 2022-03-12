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
from shutil import copyfile

fileList = glob.glob("../testOut_CROP/*.jpg")

cropFolder = "/ddsmDataset/test001/ddsm/"

complexCaseList = getComplexList()

settingList = ['../mass_case_description_test_setR.csv', '../calc_case_description_test_set.csv',
               '../mass_case_description_train_setR.csv', '../calc_case_description_train_set.csv']


mutex = threading.Lock()

trainText = open(cropFolder + "train.txt", 'w')
testText = open(cropFolder + "test.txt", 'w')

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
        maskList = glob.glob("../testOut_CROP/" + fileInfo + "_MASK*")

        # open image
        img = cv2.imread(ele, cv2.IMREAD_GRAYSCALE)
        height, width = img.shape
        newHeight, newWidth = height, width
        
        roiFile = open(cropFolder + fileInfo + ".txt", 'w')
        
        newCategory = None

        for mask in maskList:
            maskROInum = mask.split("/")[2].split(".")[0].split("_")[6]
            maskImg = cv2.imread(mask, cv2.IMREAD_GRAYSCALE)
            maskRowOriginal, maskColOriginal = np.where(maskImg != 0)

            # get mask bound
            yMin = maskRowOriginal[np.argmin(maskRowOriginal)]
            yMax = maskRowOriginal[np.argmax(maskRowOriginal)]
            xMin = maskColOriginal[np.argmin(maskColOriginal)]
            xMax = maskColOriginal[np.argmax(maskColOriginal)]

            # turn to yolo format            

            xCenter = str(round((xMin+xMax)/(2*newWidth), 6))
            yCenter = str(round((yMin+yMax)/(2*newHeight), 6))
            xWidth = str(round((xMax-xMin)/newWidth, 6))
            yHeight = str(round((yMax-yMin)/newHeight, 6))
                
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
                roiFile.write(str(BIRADSclass) + " " + str(xCenter) + " " + str(yCenter) + " " + str(xWidth) + " " + str(yHeight) + "\n")
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
                roiFile.write(str(BIRADSclass) + " " + str(xCenter) + " " + str(yCenter) + " " + str(xWidth) + " " + str(yHeight) + "\n")
                continue

        roiFile.close()
        
        if newCategory == None:
            os.remove(cropFolder + fileInfo + ".txt")
            return 0

        # copy image
        #img = cv2.resize(img, (newWidth, newHeight), interpolation=cv2.INTER_AREA)
        #cv2.imwrite(cropFolder + fileInfo + ".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 100])
        copyfile(ele, cropFolder + fileInfo + ".jpg")

        # write to train or test textfile
        mutex.acquire()
        if newCategory == "Train":
            trainText.write(cropFolder + fileInfo + ".jpg\n")
        elif newCategory == "Test":
            testText.write(cropFolder + fileInfo + ".jpg\n")
        else:
            assert False
        mutex.release()



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

print("Done.")
# %%
