import glob
import cv2
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
import numpy as np

labels = ['0~100', '100~200', '200~300', '300~400', '400~500', '500~600', '600~700', '700~800', '800~900', '900~1000', '>1000'] 

fileList = open("/ddsmDataset/yolor_inbrest_noCrop/train.txt",'r').readlines()
maxHeight = 0
maxWidth = 0
widthList = []
heightList = []
for rawPath in fileList:
	path = rawPath.replace("\n","")
	img = cv2.imread(path)
	height, width = img.shape[:2]
	roiList = open(path.replace(".jpg",".txt"),'r').readlines()
	for rawRoi in roiList:
		c, x, y, w, h = rawRoi.replace("\n","").split(" ")
		widthList.append(float(w)*width)
		heightList.append(float(h)*height)
		if float(w)*width > maxWidth:
			maxWidth = float(w)*width
			maxWidth_Height = float(h)*height
		if float(h)*height > maxHeight:
			maxHeight = float(h)*height
			maxHeight_Width = float(w)*width
fileList = open("/ddsmDataset/yolor_inbrest_noCrop/test.txt",'r').readlines()
for rawPath in fileList:
	path = rawPath.replace("\n","")
	img = cv2.imread(path)
	height, width = img.shape[:2]
	roiList = open(path.replace(".jpg",".txt"),'r').readlines()
	for rawRoi in roiList:
		c, x, y, w, h = rawRoi.replace("\n","").split(" ")
		widthList.append(float(w)*width)
		heightList.append(float(h)*height)
		if float(w)*width > maxWidth:
			maxWidth = float(w)*width
			maxWidth_Height = float(h)*height
		if float(h)*height > maxHeight:
			maxHeight = float(h)*height
			maxHeight_Width = float(w)*width
print(maxHeight, maxHeight_Width)
print(maxWidth_Height, maxWidth)
counts, edges, bars = plt.hist(widthList, bins=np.arange(0,1200,100)-50, rwidth = 0.5)
for ele in bars:
	print(ele)
plt.xticks(range(0,1200,100), labels)
plt.title("Distribution of result IOU")
plt.xlabel("IOU")
plt.ylabel("Number")
plt.show()