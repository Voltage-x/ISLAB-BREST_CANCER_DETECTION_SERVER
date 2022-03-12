import glob

count = 0
for i in range(1,6):
    count += len(glob.glob("/ddsmDataset/" + str(i) + "/*.jpg"))

print(count)

count = 0
for i in range(1,6):
    count += len(glob.glob("/ddsmDataset/test002/" + str(i) + "/*.jpg"))

print(count)