# Read all the train and test data and store as npy for easy access
import numpy as np
import os
from os import listdir
from os.path import isfile, join
import pydicom, numpy as np
import time
from PIL import Image



base_folder = "/mnt/additional/nitthilan/data/kaggle/"
df_train_label = base_folder + 'stage_1_train_labels.csv'
df_train_class_info = base_folder + 'stage_1_detailed_class_info.csv'
df_train_base_folder = base_folder + "stage_1_train_images/"
df_test_base_folder = base_folder + "stage_1_test_images/"

df_preprocess_base_folder = base_folder + "preprocess_folder/"
df_train_preprocess_filename = df_preprocess_base_folder + "train.npz"
df_test_preprocess_filename = df_preprocess_base_folder + "test.npz"

def read_dcm(basefolder, output_filename):
	filelist = [f for f in listdir(basefolder) \
		if isfile(join(basefolder, f))]
	image_array = []
	filename_list = []
	for file in filelist:
		ds = pydicom.dcmread(basefolder+file) 
		# print(ds.pixel_array.shape, ds.pixel_array.dtype,
		# 	np.min(ds.pixel_array), np.max(ds.pixel_array))
		image_array.append(ds.pixel_array)
		filename_list.append(file)

	image_array = np.array(image_array)
	print(image_array.shape)
	np.savez(output_filename, image_array=image_array, 
		filename_list=filename_list)
	return image_array, filename_list

def load_images(filename):
	filevalue = np.load(filename)
	return filevalue["image_array"], filevalue["filename_list"]

def resize_images(x_train, resize_factor):
	(n,w,h) = x_train.shape
	w = int(w*1.0/resize_factor)
	h = int(h*1.0/resize_factor)
	print("Image dimensions", n,w,h)
	# x_train_resized = np.zeros((n,w,h,d))
	x_train_resized = []
	for idx in range(n):
		im = Image.fromarray(np.uint8(x_train[idx,:,:]))
		im = im.resize((w,h), Image.ANTIALIAS)
		im = np.asarray(im)
		# im = np.roll(im, 2, axis=-1)
		# im = np.transpose(im, [0, 3, 1, 2])
		# x_train_resized[idx,:,:,:] = im
		x_train_resized.append(im)
	return np.array(x_train_resized)

if not os.path.exists(df_preprocess_base_folder):
    os.makedirs(df_preprocess_base_folder)

# read_dcm(df_train_base_folder, df_train_preprocess_filename)
# read_dcm(df_test_base_folder, df_test_preprocess_filename) 
start = time.time()
image_array, filename_list = load_images(df_train_preprocess_filename)
# image_array = np.zeros((10,1024, 1024))
filename_list = []
end = time.time()
print(image_array.shape, len(filename_list), end - start)
# image_array = np.expand_dims(image_array, axis=3)
image_array_resized = resize_images(image_array, 2)
np.savez(df_preprocess_base_folder+"train_512.npz", image_array=image_array_resized, 
		filename_list=filename_list)
image_array_resized = resize_images(image_array, 4)
np.savez(df_preprocess_base_folder+"train_256.npz", image_array=image_array_resized, 
		filename_list=filename_list)

start = time.time()
image_array, filename_list = load_images(df_test_preprocess_filename)
end = time.time()
print(image_array.shape, len(filename_list), end - start)
image_array = np.expand_dims(image_array, axis=3)

image_array_resized = resize_images(image_array, 2)
np.savez(df_preprocess_base_folder+"test_512.npz", image_array=image_array_resized, 
		filename_list=filename_list)
image_array_resized = resize_images(image_array, 4)
np.savez(df_preprocess_base_folder+"test_256.npz", image_array=image_array_resized, 
		filename_list=filename_list)
