import glob
import shutil

fileList = open("/ddsmDataset/yolo_ddsm_modify_crop_stride800/test.txt",'r').readlines()

dstPath = "/home/islab/vincentwang/project/proc/"

for rawImgPath in fileList:
    imgPath = rawImgPath.replace("\n","")
    fileName = imgPath.split("/")[-1].split(".")[0]
    print(f"{dstPath}{fileName}.jpg")
    shutil.copyfile(imgPath,f"{dstPath}groundTruthImage/{fileName}.jpg")
    shutil.copyfile(imgPath.replace(".jpg",".txt"),f"{dstPath}groundTruthLabel/{fileName}.txt")