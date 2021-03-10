import os
import shutil
import random

old_path = "/home/g1007540910/DATA/"
old_folder = "ImageNet2012"
new_folder = "ImageNet2012_O"
known_class_num = 500    #500
unnknown_class_000 = 0
unnknown_class_100 = 100
unnknown_class_200 = 200 #  300
unnknown_class_300 = 300 #  300
unnknown_class_400 = 400 #  300
unnknown_class_500 = 500 #  300

# random select known and unknown classes
random.seed(10)
class_folders = []
for class_folder in os.listdir(os.path.join(old_path,old_folder,'train')):
    if not class_folder.startswith("."):
        class_folders.append(class_folder)
selected_known_folders =random.sample(class_folders,known_class_num)
rest_folders = list(set(class_folders).difference(set(selected_known_folders)))
selected_unknown_folders_300 = random.sample(rest_folders,unnknown_class_300)
selected_unknown_folders_400 = random.sample(rest_folders,unnknown_class_400)
selected_unknown_folders_500 = random.sample(rest_folders,unnknown_class_500)
selected_unknown_folders_200 = random.sample(rest_folders,unnknown_class_200)
selected_unknown_folders_100 = random.sample(rest_folders,unnknown_class_100)



# for test set
print("Processing validation set 100....")
counter = 0
for class_folder in selected_known_folders:
    counter += 1
    shutil.copytree(
        os.path.join(old_path,old_folder,'val',class_folder),
        os.path.join(old_path,new_folder,'val100',class_folder))
    print(f"copying Val classes (known) \t \t {counter} / {known_class_num}")
targetdir = os.path.join(old_path,new_folder,'val100','n99999999')
os.makedirs(targetdir)
counter = 0
for class_folder in selected_unknown_folders_100:
    counter += 1
    workdir = os.path.join(old_path,old_folder,'val',class_folder)
    for file_ in os.listdir(workdir):
        srcFile = os.path.join(workdir, file_)
        targetFile = os.path.join(targetdir, file_)
        shutil.copyfile(srcFile, targetFile)
    print(f"copying Val classes (unknown) \t \t {counter} / {unnknown_class_100}")

print("done!!!")

# for test set
print("Processing validation set 200....")
counter = 0
for class_folder in selected_known_folders:
    counter += 1
    shutil.copytree(
        os.path.join(old_path,old_folder,'val',class_folder),
        os.path.join(old_path,new_folder,'val200',class_folder))
    print(f"copying Val classes (known) \t \t {counter} / {known_class_num}")
targetdir = os.path.join(old_path,new_folder,'val200','n99999999')
os.makedirs(targetdir)
counter = 0
for class_folder in selected_unknown_folders_200:
    counter += 1
    workdir = os.path.join(old_path,old_folder,'val',class_folder)
    for file_ in os.listdir(workdir):
        srcFile = os.path.join(workdir, file_)
        targetFile = os.path.join(targetdir, file_)
        shutil.copyfile(srcFile, targetFile)
    print(f"copying Val classes (unknown) \t \t {counter} / {unnknown_class_200}")

print("done!!!")


# for test set
print("Processing validation set 000....")
counter = 0
for class_folder in selected_known_folders:
    counter += 1
    shutil.copytree(
        os.path.join(old_path,old_folder,'val',class_folder),
        os.path.join(old_path,new_folder,'val000',class_folder))
    print(f"copying Val classes (known) \t \t {counter} / {known_class_num}")
print("done!!!")














# # for train set
# print("Processing training set....")
# counter = 0
# for class_folder in selected_known_folders:
#     counter +=1
#     shutil.copytree(
#         os.path.join(old_path,old_folder,'train',class_folder),
#         os.path.join(old_path,new_folder,'train',class_folder))
#     print(f"copying Train classes \t \t {counter} / {known_class_num}")
#

"""
# for test set
print("Processing validation set 300....")
counter = 0
for class_folder in selected_known_folders:
    counter += 1
    shutil.copytree(
        os.path.join(old_path,old_folder,'val',class_folder),
        os.path.join(old_path,new_folder,'val300',class_folder))
    print(f"copying Val classes (known) \t \t {counter} / {known_class_num}")
targetdir = os.path.join(old_path,new_folder,'val300','n99999999')
os.makedirs(targetdir)
counter = 0
for class_folder in selected_unknown_folders_300:
    counter += 1
    workdir = os.path.join(old_path,old_folder,'val',class_folder)
    for file_ in os.listdir(workdir):
        srcFile = os.path.join(workdir, file_)
        targetFile = os.path.join(targetdir, file_)
        shutil.copyfile(srcFile, targetFile)
    print(f"copying Val classes (unknown) \t \t {counter} / {unnknown_class_300}")

print("done!!!")



# for test set
print("Processing validation set 400....")
counter = 0
for class_folder in selected_known_folders:
    counter += 1
    shutil.copytree(
        os.path.join(old_path,old_folder,'val',class_folder),
        os.path.join(old_path,new_folder,'val400',class_folder))
    print(f"copying Val classes (known) \t \t {counter} / {known_class_num}")
targetdir = os.path.join(old_path,new_folder,'val400','n99999999')
os.makedirs(targetdir)
counter = 0
for class_folder in selected_unknown_folders_400:
    counter += 1
    workdir = os.path.join(old_path,old_folder,'val',class_folder)
    for file_ in os.listdir(workdir):
        srcFile = os.path.join(workdir, file_)
        targetFile = os.path.join(targetdir, file_)
        shutil.copyfile(srcFile, targetFile)
    print(f"copying Val classes (unknown) \t \t {counter} / {unnknown_class_400}")

print("done!!!")




# for test set
print("Processing validation set 500....")
counter = 0
for class_folder in selected_known_folders:
    counter += 1
    shutil.copytree(
        os.path.join(old_path,old_folder,'val',class_folder),
        os.path.join(old_path,new_folder,'val500',class_folder))
    print(f"copying Val classes (known) \t \t {counter} / {known_class_num}")
targetdir = os.path.join(old_path,new_folder,'val500','n99999999')
os.makedirs(targetdir)
counter = 0
for class_folder in selected_unknown_folders_500:
    counter += 1
    workdir = os.path.join(old_path,old_folder,'val',class_folder)
    for file_ in os.listdir(workdir):
        srcFile = os.path.join(workdir, file_)
        targetFile = os.path.join(targetdir, file_)
        shutil.copyfile(srcFile, targetFile)
    print(f"copying Val classes (unknown) \t \t {counter} / {unnknown_class_500}")

print("done!!!")
"""
