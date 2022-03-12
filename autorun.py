import os


os.system("rm /home/islab/vincentwang/project/proc/labelsFromMerge/*")
os.system("rm /home/islab/vincentwang/project/proc/labelsFromEfficientNet/*")
os.system("rm /home/islab/vincentwang/project/proc/labelsFromYOLOR/*")
os.system("rm /home/islab/vincentwang/project/proc/code/cropResult/BENIGN/*")
os.system("rm /home/islab/vincentwang/project/proc/code/cropResult/MALIGNANT/*")
os.system("cp /home/islab/vincentwang/project/yolor/runs/test/yolor_p6_val_inbreast9/labels/* /home/islab/vincentwang/project/proc/labelsFromYOLOR/")
