import cv2
import glob


txtFileList = []

for i in range(1, 5):
    txtFileList += glob.glob("/ddsmDataset/test005/"+str(i)+"/*.txt")

for txtpath in txtFileList:

    if txtpath.find("train.txt") != -1 or txtpath.find("test.txt") != -1:
        continue

    txtFile = open(txtpath, 'r')
    txtLines = txtFile.readlines()

    roiList = []

    for line in txtLines:
        cls, newYolox, newYoloy, newYolow, newYoloh = line.replace(
            "\n", "").split(" ")

        newYolox = 0.0000000000000000 if float(
            newYolox) < 0 else float(newYolox)
        newYolox = 0.9999999999999999 if float(
            newYolox) > 1 else float(newYolox)
        newYoloy = 0.0000000000000000 if float(
            newYoloy) < 0 else float(newYoloy)
        newYoloy = 0.9999999999999999 if float(
            newYoloy) > 1 else float(newYoloy)
        newYolow = 0.0000000000000000 if float(
            newYolow) < 0 else float(newYolow)
        newYolow = 0.9999999999999999 if float(
            newYolow) > 1 else float(newYolow)
        newYoloh = 0.0000000000000000 if float(
            newYoloh) < 0 else float(newYoloh)
        newYoloh = 0.9999999999999999 if float(
            newYoloh) > 1 else float(newYoloh)

        roiList.append([cls, newYolox, newYoloy, newYolow, newYoloh])

    txtFile.close()
    newtxtFile = open(txtpath, 'w')
    for roi in roiList:
        newtxtFile.write(cls + " " + str(newYolox) + " " + str(newYoloy) +
                         " " + str(newYolow) + " " + str(newYoloh) + "\n")
    newtxtFile.close()
