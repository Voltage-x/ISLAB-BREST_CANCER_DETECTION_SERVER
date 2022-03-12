# In[0]:
import cv2
import glob
import numpy as np
from complexCase import getComplexList
import csv
import threading
import queue
import os

print("Start clean")

fileList = glob.glob("../testOut_CROP/*.jpg")

cropFolder = "../smallCrop/"


class Worker(threading.Thread):
    def __init__(self, queue, num):
        threading.Thread.__init__(self)
        self.queue = queue
        self.num = num

    def run(self):
        while self.queue.qsize() > 0:
            # 取得新的資料
            msg = self.queue.get()

            # 處理資料
            os.remove(msg)


my_queue = queue.Queue()

print(len(glob.glob(cropFolder + "*")))

for ele in glob.glob(cropFolder + "*"):
    my_queue.put(ele)


# 建立兩個 Worker
my_worker1 = Worker(my_queue, 1)
my_worker2 = Worker(my_queue, 2)
my_worker3 = Worker(my_queue, 3)
my_worker4 = Worker(my_queue, 4)

# 讓 Worker 開始處理資料
my_worker1.start()
my_worker2.start()
my_worker3.start()
my_worker4.start()

# 等待所有 Worker 結束
my_worker1.join()
my_worker2.join()
my_worker3.join()
my_worker4.join()

print("Done.")

