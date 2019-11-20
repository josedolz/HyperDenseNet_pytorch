from os.path import isfile, join
import os
import numpy as np
from utils import reconstruct_volume
from utils import load_data_train
from utils import load_data_test
import pdb

def getDataNames(imagesFolder):
	if os.path.exists(imagesFolder):
	    imageNames = [f for f in os.listdir(imagesFolder) if isfile(join(imagesFolder, f))]

	imageNames.sort()

	return imageNames



moda_1 = './Data/MRBrainS/DataNii/Training/T1'
moda_2 = './Data/MRBrainS/DataNii/Training/T1_IR'
moda_3 = './Data/MRBrainS/DataNii/Training/T2_FLAIR'
moda_g = './Data/MRBrainS/DataNii/Training/GT'


if os.path.exists(moda_1):
    imageNames = [f for f in os.listdir(moda_1) if isfile(join(moda_1, f))]

# image names are not sorted.

x_train, y_train, img_shape = load_data_train(moda_1, moda_2, moda_3, moda_g, imageNames[0]) # hardcoded to read the first file. Loop this to get all files
patch_1, patch_2, patch_3, patch_g, img_shape = load_data_test(moda_1, moda_2, moda_3, moda_g, imageNames[0]) # hardcoded to read the first file. Loop this to get all files

pdb.set_trace()
# To reconstruct the predicted volume
pred_classes = np.argmax(pred, axis=2)
pred_classes = pred_classes.reshape((len(pred_classes), 9, 9, 9))
bin_seg = reconstruct_volume(pred_classes, (img_shape[1], img_shape[2], img_shape[3]))

