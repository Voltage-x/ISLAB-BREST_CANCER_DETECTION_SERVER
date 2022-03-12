import cv2, glob


txtFileList = glob.glob("/ddsmDataset/test005/1/*.txt")

for txtFile in txtFileList:
    
    if txtFile.find("train.txt") != -1 or txtFile.find("test.txt") != -1:
        continue

    txtLines = open(txtFile,'r').readlines()
    img = cv2.imread(txtFile.replace(".txt", ".jpg"))
    height,width = img.shape[:2]
    print(txtFile.replace(".txt", ".jpg"),height,width)
    
    for line in txtLines:
        cls, cx, cy, w, h = line.replace("\n", "").split(" ")
        x0 = float(cx)*width - float(w)*width/2
        y0 = float(cy)*height - float(h)*height/2
        x1 = float(cx)*width + float(w)*width/2
        y1 = float(cy)*height + float(h)*height/2
        cv2.rectangle(img, (int(x0), int(y0)), (int(x1), int(y1)), (0, 255, 0), 10)
    cv2.namedWindow("output", cv2.WINDOW_NORMAL)
    img = cv2.resize(img, (round(width/8),round(height/8)))
    cv2.imshow("output", img)
    cv2.waitKey(0)
    #cv2.destroyAllWindows()
    #video.release()
