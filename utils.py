import numpy as np
import nibabel as nib
from sklearn.feature_extraction.image import extract_patches as sk_extract_patches
import pdb
import itertools

def generate_indexes(patch_shape, expected_shape) :
    ndims = len(patch_shape)

    poss_shape = [patch_shape[i+1] * (expected_shape[i] // patch_shape[i+1]) for i in range(ndims-1)]

    idxs = [range(patch_shape[i+1], poss_shape[i] - patch_shape[i+1], patch_shape[i+1]) for i in range(ndims-1)]

    return itertools.product(*idxs)

    
def extract_patches(volume, patch_shape, extraction_step) :
    patches = sk_extract_patches(
        volume,
        patch_shape=patch_shape,
        extraction_step=extraction_step)

    ndim = len(volume.shape)
    npatches = np.prod(patches.shape[:ndim])
    return patches.reshape((npatches, ) + patch_shape)

# Double check that number of labels is continuous
def get_one_hot(targets, nb_classes):
    #return np.eye(nb_classes)[np.array(targets).reshape(-1)]
    return np.swapaxes(np.eye(nb_classes)[np.array(targets)],0,3) # Jose. To have the same shape as pytorch (batch_size, numclasses,x,y,z)

def build_set(imageData) :
    num_classes = 9
    patch_shape = (27, 27, 27)
    extraction_step=(15, 15, 15)
    label_selector = [slice(None)] + [slice(9, 18) for i in range(3)]

    # Extract patches from input volumes and ground truth
    
    imageData_1 = np.squeeze(imageData[0,:,:,:])
    imageData_2 = np.squeeze(imageData[1,:,:,:])
    imageData_3 = np.squeeze(imageData[2,:,:,:])
    imageData_g = np.squeeze(imageData[3,:,:,:])

    num_classes = len(np.unique(imageData_g))
    x = np.zeros((0, 3, 27, 27, 27))
    #y = np.zeros((0, 9 * 9 * 9, num_classes)) # Karthik
    y = np.zeros((0, num_classes, 9, 9, 9)) # Jose
    
    #for idx in range(len(imageData)) :
    y_length = len(y)

    label_patches = extract_patches(imageData_g, patch_shape, extraction_step)
    label_patches = label_patches[label_selector]
    
    # Select only those who are important for processing
    valid_idxs = np.where(np.sum(label_patches, axis=(1, 2, 3)) != 0)

    # Filtering extracted patches
    label_patches = label_patches[valid_idxs]

    x = np.vstack((x, np.zeros((len(label_patches), 3, 27, 27, 27))))
    #y = np.vstack((y, np.zeros((len(label_patches), 9 * 9 * 9, num_classes)))) # Karthik
    y = np.vstack((y, np.zeros((len(label_patches), num_classes, 9, 9, 9))))  # Jose
    
    for i in range(len(label_patches)) :
        #y[i+y_length, :, :] = get_one_hot(label_patches[i, : ,: ,:].astype('int'), num_classes)  # Karthik
        y[i, :, :, :, :] = get_one_hot(label_patches[i, : ,: ,:].astype('int'), num_classes)  # Jose
    del label_patches

    # Sampling strategy: reject samples which labels are only zeros
    T1_train = extract_patches(imageData_1, patch_shape, extraction_step)
    x[y_length:, 0, :, :, :] = T1_train[valid_idxs]
    del T1_train

    # Sampling strategy: reject samples which labels are only zeros
    T2_train = extract_patches(imageData_2, patch_shape, extraction_step)
    x[y_length:, 1, :, :, :] = T2_train[valid_idxs]
    del T2_train

    # Sampling strategy: reject samples which labels are only zeros
    Fl_train = extract_patches(imageData_3, patch_shape, extraction_step)
    x[y_length:, 2, :, :, :] = Fl_train[valid_idxs]
    del Fl_train

        
    return x, y

def reconstruct_volume(patches, expected_shape) :
    patch_shape = patches.shape

    assert len(patch_shape) - 1 == len(expected_shape)

    reconstructed_img = np.zeros(expected_shape)

    for count, coord in enumerate(generate_indexes(patch_shape, expected_shape)) :
        selection = [slice(coord[i], coord[i] + patch_shape[i+1]) for i in range(len(coord))]
        reconstructed_img[selection] = patches[count]

    return reconstructed_img


def load_data_train(path1, path2, path3, pathg, imgName):
    
    X_train = []
    Y_train = []
    for num in range(len(imgName)):
        # Karthik
        #imageData_1 = nib.load(path1 + '/' + imgName[num]).get_data()
        #imageData_2 = nib.load(path2 + '/' + imgName[num]).get_data()
        #imageData_3 = nib.load(path3 + '/' + imgName[num]).get_data()
        #imageData_g = nib.load(pathg + '/' + imgName[num]).get_data()
        
        # Jose
        imageData_1 = nib.load(path1 + '/' + imgName).get_data()
        imageData_2 = nib.load(path2 + '/' + imgName).get_data()
        imageData_3 = nib.load(path3 + '/' + imgName).get_data()
        imageData_g = nib.load(pathg + '/' + imgName).get_data()
        num_classes = len(np.unique(imageData_g))

        imageData = np.stack((imageData_1, imageData_2, imageData_3, imageData_g))
        img_shape = imageData.shape

        x_train, y_train = build_set(imageData)

        X_train.append(x_train)
        Y_train.append(y_train)

        del x_train
        del y_train


    X_train = np.asarray(X_train)
    Y_train = np.asarray(Y_train)

    X = np.concatenate(X_train, axis=0)
    del X_train
    Y = np.concatenate(Y_train, axis=0)
    del Y_train
    idx = np.arange(X.shape[0])
    np.random.shuffle(idx)
    #pdb.set_trace()
    return X[idx], Y[idx], img_shape


def load_data_test(path1, path2, path3, pathg, imgName):

	imageData_1 = nib.load(path1 + '/' + imgName).get_data()
	imageData_2 = nib.load(path2 + '/' + imgName).get_data()
	imageData_3 = nib.load(path3 + '/' + imgName).get_data()
	imageData_g = nib.load(pathg + '/' + imgName).get_data()

	num_classes = len(np.unique(imageData_g))
	
	imageData = np.stack((imageData_1, imageData_2, imageData_3, imageData_g))
	img_shape = imageData.shape

	patch_1 = extract_patches(imageData_1, patch_shape=(27, 27, 27), extraction_step=(9, 9, 9))
	patch_2 = extract_patches(imageData_2, patch_shape=(27, 27, 27), extraction_step=(9, 9, 9))
	patch_3 = extract_patches(imageData_3, patch_shape=(27, 27, 27), extraction_step=(9, 9, 9))
	patch_g = extract_patches(imageData_g, patch_shape=(27, 27, 27), extraction_step=(9, 9, 9))

	x_train, y_train = build_set(imageData)

	return patch_1, patch_2, patch_3, patch_g, img_shape

