import cv2


def fromYolo(img, x, y, w, h):
    height, width = img.shape[:2]
    x1 = float(x)*width - float(w)*width/2
    y1 = float(y)*height - float(h)*height/2
    x2 = float(x)*width + float(w)*width/2
    y2 = float(y)*height + float(h)*height/2
    return int(x1), int(y1), int(x2), int(y2)


def toYolo(img, x1, y1, x2, y2):
    height, width = img.shape[:2]
    x = round(((int(x1) + int(x2)) / 2) / width, 6)
    y = round(((int(y1) + int(y2)) / 2) / height, 6)
    w = round((int(x2) - int(x1)) / width, 6)
    h = round((int(y2) - int(y1)) / height, 6)
    return x, y, w, h
