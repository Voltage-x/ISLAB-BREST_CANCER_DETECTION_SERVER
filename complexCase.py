# In[0]:
import glob

fileList = glob.glob("../testOut/*")

lenx = 0


def getComplexList():
    skipList = []
    for index in range(len(fileList)):
        for dupIndex in range(index+1, len(fileList)):
            if fileList[index].split("/")[2].split("_P_")[1] == fileList[dupIndex].split("/")[2].split("_P_")[1]:
                if not fileList[index].split("/")[2].split("_P_")[1] in skipList:
                    skipList.append(fileList[index].split("/")[2])
                if not fileList[dupIndex].split("/")[2].split("_P_")[1] in skipList:
                    skipList.append(fileList[dupIndex].split("/")[2])
    print("get skiplist len: " + str(len(skipList)))
    return skipList
# %%
