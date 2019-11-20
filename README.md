# Hyperdensenet_Pytorch


This is a Pytorch implementation of Hyperdensenet. For the detailed architecture please refer to https://arxiv.org/abs/1804.02967

This is not the original implementation of the paper. Not to reproduce the resuls


### Dependencies
This code depends on the following libraries:

- Python >= 3.5
- Pytorch 0.3.1 (Testing on more recent versions)
- nibabel
- medpy


### Training

The model can be trained using below command:  
```
python mainHyperDenseNet.py
```

## Preparing your data
- To use your own data, you will have to specify the path to the folder containing this data (--root_dir).
- Images have to be in nifti (.nii) format
- You have to split your data into two folders: Training/Validation. Each folder will contain N sub-folders: N-1 subfolders that will contain each modality and GT, which contain the nifti files for the images and their corresponding ground truths. 
- In the runTraining function, you have to change the name of the subfolders to the names you have in your dataset (lines 128-131 and 144-147).


If you use this code in your research, please consider citing the following paper:

Dolz J, Gopinath K, Yuan J, Lombaert H, Desrosiers C, Ayed IB. HyperDense-Net: A hyper-densely connected CNN for multi-modal image segmentation. IEEE transactions on medical imaging. 2018 Oct 30;38(5):1116-26.

# HyperDenseNet_pytorch
