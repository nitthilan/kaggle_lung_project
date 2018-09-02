import glob, pylab, pandas as pd
import pydicom, numpy as np

# Image dimension 1024x1024
# Path to data: "/mnt/additional/nitthilan/data/kaggle"
# 

base_folder = "/mnt/additional/nitthilan/data/kaggle/"

df_train_label = base_folder + 'stage_1_train_labels.csv'
df_train_class_info = base_folder + 'stage_1_detailed_class_info.csv'

print(df_train_label)
dftl = pd.read_csv(df_train_label)
dftci = pd.read_csv(df_train_class_info)

print(dftl.describe())

print(dftci.describe())

print(dftl.Target.unique())

dftl_np = dftl.values

print(dftl_np.shape)
print(dftl_np[:5])

num_three_boxes = 0
num_two_boxes = 0
num_one_boxes = 0
num_zero_boxes = 0
num_four_boxes = 0
for idx, feature in enumerate(dftl_np):
	# print(idx, feature)
	if(feature[5] and feature[0] == dftl_np[idx-1,0]
		and feature[0] == dftl_np[idx-2,0]
		and feature[0] == dftl_np[idx-3,0]):
		# print(idx, dftl_np[idx-3], dftl_np[idx-2], dftl_np[idx-1], dftl_np[idx])
		num_four_boxes += 1
	elif(feature[5] and feature[0] == dftl_np[idx-1,0]
		and feature[0] == dftl_np[idx-2,0]
		and feature[0] != dftl_np[idx+1,0]):
		# print(idx, dftl_np[idx-2], dftl_np[idx-1], dftl_np[idx])
		num_three_boxes += 1
	elif(feature[5] and feature[0] == dftl_np[idx-1,0]
		and feature[0] != dftl_np[idx-2,0]
		and feature[0] != dftl_np[idx+1,0]):
		num_two_boxes += 1
	elif(feature[5] and feature[0] != dftl_np[idx-1,0] 
		and feature[0] != dftl_np[idx+1,0]):
		num_one_boxes += 1
	elif(feature[5] == 0):
		num_zero_boxes += 1
	# else:
	# 	print(idx, dftl_np[idx-1], dftl_np[idx], dftl_np[idx+1])

	# check whether all the boxes are bound within a quandrant
	if(feature[5]):
		x = feature[1]; y = feature[2]; 
		width = feature[3]; height = feature[4]

		x1 = x+width; y1 = y+height

		# if(x < 512 and x1 > 512):
		# 	print(idx, "x", x, y, x1, y1)

		if(x < 1024/3 and x1 > 2*1024/3):
			print(idx, "x/3", x, y, x1, y1, 2*1024/3)
		# if(y < 1024/3 and y1 > 2*1024/3):
		# 	print(idx, "y/3", x, y, x1, y1)

		if(width > 512):
			print(idx, "wth", x, y, x1, y1, width, height)
		# if(height > 512):
		# 	print(idx, "hgt", x, y, x1, y1, width, height)
		
		# if(y < 512 and y1 > 512):
		# 	print(idx, "y", x, y, x1, y1)

		# if(x < 512 and y < 512):
		# 	if(x1 > 512 or y1 > 512):
		# 		print(idx, x, y, x1, y1)
		# elif(x > 512 and y < 512):
		# 	if(x1 < 512 or y1 > 512):
		# 		print(idx, x, y, x1, y1)
		# elif(x > 512 and y > 512):
		# 	if(x1 < 512 or y1 < 512):
		# 		print(idx, x, y, x1, y1)
		# elif(x < 512 and y > 512):
		# 	if(x1 > 512 or y1 < 512):
		# 		print(idx, x, y, x1, y1)


print(num_two_boxes, num_one_boxes, num_zero_boxes, 
	num_three_boxes, num_four_boxes,
	4*num_four_boxes+3*num_three_boxes+2*num_two_boxes+\
	num_one_boxes+num_zero_boxes)




# dcm_train_path = base_folder+""
# "d87954c5-4889-4c82-a110-99fe2aabd598"
# filename = "./b59e042d-0ec6-455a-82c8-658ccb24e614.dcm"
# # filename = get_testdata_files()
# print(filename)
# ds = pydicom.dcmread(filename) 
# print(ds)

# # print(ds.dir("setup"))
# # print(ds.PatientSetupSequence[0])


# im = dcm_data.pixel_array
# print(type(im))
# print(im.dtype)
# print(im.shape)


