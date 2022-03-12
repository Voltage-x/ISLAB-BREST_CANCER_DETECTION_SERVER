# In[0]:
import cv2
import glob
import numpy as np
import csv
import threading
import queue
import os
import random
import time

cropFolder = "/ddsmDataset/yolo_inbreast_crop_stride304/"

experimentFolder = "/ddsmDataset/yolo_inbreast_crop_stride304/extra/"

'''
print("Start cleaning previous files...")
for f in glob.glob(cropFolder + "*"):
    os.remove(f)
print("Clean finish.")
'''

countSkip = 0

mutex = threading.Lock()

trainText = open(cropFolder + "train.txt", 'w')
testText = open(cropFolder + "test.txt", 'w')
extraTestFile = open(cropFolder + "extraTest.txt", 'w')

def proc(ele, splitNum, category):
    global cropFolder
    global countSkip

    fileInfo = ele.split("/")[-1].split(".")[0]

    # open image
    img = cv2.imread(ele, cv2.IMREAD_GRAYSCALE)
    height, width = img.shape

    # start cropping
    count = 0

    # cropsize
    cropsize = 1600

    # open maskFile
    maskFile = open(ele.replace(".jpg", ".txt"), 'r')
    maskList = maskFile.readlines()

    #skip list
    roiSkip = []
    extraList = [] # experiment
    categoryList = [] # experiment
    for i in range(len(maskList)):
        roiSkip.append(0)
        extraList.append([]) # experiment
        categoryList.append("") # experiment
    st_time = time.time()
    # iter. crop
    for heightIndex in range(0, height, 800):
        for widthIndex in range(0, width, 800):
            # check if the crop is out of bound
            if heightIndex + cropsize > height-1 or widthIndex + cropsize > width-1:
                continue

            cropImg = img[heightIndex: heightIndex +
                          cropsize, widthIndex: widthIndex+cropsize]

            # check if the crop have > 0.15 black area
            row, _ = np.where(cropImg == 0)
            if len(row)/(cropsize*cropsize) > 0.8:
                continue

            # split folder
            splitFolder = str(splitNum) + "/"

            # new file name
            newFileName = fileInfo + "_" + str(count)

            # store roi in this crop
            roiInfo = []

            # iter. mask
            for roiIndex, mask in enumerate(maskList):
                maskCategory, yolox, yoloy, yolow, yoloh = mask.replace(
                    "\n", "").split(" ")
                categoryList[roiIndex] = maskCategory # experiment
                realx0 = float(yolox) * width - (float(yolow) * width / 2)
                realy0 = float(yoloy) * height - (float(yoloh) * height / 2)
                realx1 = float(yolox) * width + (float(yolow) * width / 2)
                realy1 = float(yoloy) * height + (float(yoloh) * height / 2)

                # if mask region is not entire roi then break
                if realx0 < widthIndex or realy0 < heightIndex or realx1 > widthIndex+cropsize or realy1 > heightIndex+cropsize:
                    #roiInfo.clear()
                    #continue
                    if (realx1 < widthIndex or realy1 < heightIndex) or (realx0 > widthIndex+cropsize or realy0 > heightIndex+cropsize):
                        continue
                    #print("==start==")
                    #print(realx0, realy0, realx1, realy1)
                    #print(widthIndex, heightIndex, widthIndex+cropsize, heightIndex+cropsize)
                    
                    realx0 = 0.0 if realx0 < widthIndex else (realx0-widthIndex)
                    realy0 = 0.0 if realy0 < heightIndex else (realy0-heightIndex)
                    realx1 = cropsize-1 if realx1 > widthIndex+cropsize else (realx1-widthIndex)
                    realy1 = cropsize-1 if realy1 > heightIndex+cropsize else (realy1-heightIndex)
                    newYolox = (realx0 + realx1) / (2*cropsize)
                    newYoloy = (realy0 + realy1) / (2*cropsize)
                    newYolow = (realx1 - realx0) / cropsize
                    newYoloh = (realy1 - realy0) / cropsize
                    newYolox = 0.000000 if newYolox < 0 else newYolox
                    newYolox = 0.999999 if newYolox > 1 else newYolox
                    newYoloy = 0.000000 if newYoloy < 0 else newYoloy
                    newYoloy = 0.999999 if newYoloy > 1 else newYoloy
                    newYolow = 0.000000 if newYolow < 0 else newYolow
                    newYolow = 0.999999 if newYolow > 1 else newYolow
                    newYoloh = 0.000000 if newYoloh < 0 else newYoloh
                    newYoloh = 0.999999 if newYoloh > 1 else newYoloh
                    if newYolow < 2 or newYoloh < 2:
                        roiSkip[roiIndex] = 1 # experiment
                        extraList[roiIndex].append([cropImg,[maskCategory, newYolox, newYoloy, newYolow, newYoloh]]) # experiment
                        #countSkip+=1
                        continue
                    #print(realx0, realy0, realx1, realy1)
                    #print("==end==")
                else:
                    newYolox = (float(yolox) * width - widthIndex) / cropsize
                    newYoloy = (float(yoloy) * height - heightIndex) / cropsize
                    newYolow = float(yolow) * width / cropsize
                    newYoloh = float(yoloh) * height / cropsize

                newYolox = 0.000000 if newYolox < 0 else newYolox
                newYolox = 0.999999 if newYolox > 1 else newYolox
                newYoloy = 0.000000 if newYoloy < 0 else newYoloy
                newYoloy = 0.999999 if newYoloy > 1 else newYoloy
                newYolow = 0.000000 if newYolow < 0 else newYolow
                newYolow = 0.999999 if newYolow > 1 else newYolow
                newYoloh = 0.000000 if newYoloh < 0 else newYoloh
                newYoloh = 0.999999 if newYoloh > 1 else newYoloh
                extraList[roiIndex].append([cropImg,[maskCategory, newYolox, newYoloy, newYolow, newYoloh]]) # experiment
                roiInfo.append(
                    [maskCategory, newYolox, newYoloy, newYolow, newYoloh])
            maskFile.close()

            # if roi exist
            if len(roiInfo) != 0:
                # write to train or test textfile
                cv2.imwrite(cropFolder + splitFolder +
                            newFileName + ".jpg", cropImg)
                newMaskFile = open(cropFolder + splitFolder +
                                   newFileName + ".txt", 'w')
                for roi in roiInfo:
                    newMaskFile.write(roi[0] + " " + str(roi[1]) + " " + str(
                        roi[2]) + " " + str(roi[3]) + " " + str(roi[4]) + "\n")
                newMaskFile.close()

                mutex.acquire()
                if category == "test":
                    testText.write(cropFolder + splitFolder +
                                   newFileName + ".jpg\n")
                elif category == "train":
                    trainText.write(cropFolder + splitFolder +
                                    newFileName + ".jpg\n")
                else:
                    assert False
                mutex.release()
            count += 1
    
    mutex.acquire()
    for extraIndex, flagx in enumerate(roiSkip):
        if category == "train":
                break
        #if flagx == 1:
        if True:
            for imgIndex, ext in enumerate(extraList[extraIndex]):
                cv2.imwrite(f"{experimentFolder}{fileInfo}_{extraIndex}_{imgIndex}.jpg",ext[0])
                extraRoi = open(f"{experimentFolder}{fileInfo}_{extraIndex}_{imgIndex}.txt",'w')
                extraRoi.write(ext[1][0] + " " + str(ext[1][1]) + " " + str(ext[1][2]) + " " + str(ext[1][3]) + " " + str(ext[1][4]) + "\n")
                extraRoi.close()
                extraTestFile.write(f"{experimentFolder}{fileInfo}_{extraIndex}_{imgIndex}.jpg\n")
    mutex.release()
    
    print(time.time()-st_time)
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
            proc(msg[0], self.num, msg[1])


my_queue = queue.Queue()

originalTrainText = open("/ddsmDataset/yolor_inbrest_noCrop/train.txt", 'r')
originalTestText = open("/ddsmDataset/yolor_inbrest_noCrop/test.txt", 'r')

for ele in originalTrainText.readlines():
    my_queue.put([ele.replace("\n", ""), "train"])
for ele in originalTestText.readlines():
    my_queue.put([ele.replace("\n", ""), "test"])

originalTrainText.close()
originalTestText.close()

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
# my_worker5.start()

# join worker
my_worker1.join()
my_worker2.join()
my_worker3.join()
my_worker4.join()
# my_worker5.join()

trainText.close()
testText.close()
extraTestFile.close()

#print(countSkip)
print("Done.")
# %%
