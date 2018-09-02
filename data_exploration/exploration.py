import glob, pylab, pandas as pd
import pydicom, numpy as np



df = pd.read_csv('stage_1_train_labels.csv')
df1 = pd.read_csv('stage_1_detailed_class_info.csv')

"d87954c5-4889-4c82-a110-99fe2aabd598"
filename = "./b59e042d-0ec6-455a-82c8-658ccb24e614.dcm"
# filename = get_testdata_files()
print(filename)
ds = pydicom.dcmread(filename) 
print(ds)

# print(ds.dir("setup"))
# print(ds.PatientSetupSequence[0])


im = dcm_data.pixel_array
print(type(im))
print(im.dtype)
print(im.shape)

# Image dimension 1024x1024
# Path to data: /mnt/additional/nitthilan/data/kaggle
# 
