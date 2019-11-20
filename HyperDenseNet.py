from Blocks import *
import torch.nn.init as init
import torch.nn.functional as F
import pdb
import math
#from layers import *

def croppCenter(tensorToCrop,finalShape):

    org_shape = tensorToCrop.shape
    diff = org_shape[2] - finalShape[2]
    croppBorders = int(diff/2)
    return tensorToCrop[:,
                        :,
                        croppBorders:org_shape[2]-croppBorders,
                        croppBorders:org_shape[3]-croppBorders,
                        croppBorders:org_shape[4]-croppBorders]

def convBlock(nin, nout, kernel_size=3, batchNorm = False, layer=nn.Conv3d, bias=True, dropout_rate = 0.0, dilation = 1):
    
    if batchNorm == False:
        return nn.Sequential(
            nn.PReLU(),
            nn.Dropout(p=dropout_rate),
            layer(nin, nout, kernel_size=kernel_size, bias=bias, dilation=dilation)
        )
    else:
        return nn.Sequential(
            nn.BatchNorm3d(nin),
            nn.PReLU(),
            nn.Dropout(p=dropout_rate),
            layer(nin, nout, kernel_size=kernel_size, bias=bias, dilation=dilation)
        )
        
def convBatch(nin, nout, kernel_size=3, stride=1, padding=1, bias=False, layer=nn.Conv2d, dilation = 1):
    return nn.Sequential(
        layer(nin, nout, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias, dilation=dilation),
        nn.BatchNorm2d(nout),
        #nn.LeakyReLU(0.2)
        nn.PReLU()
    )

class LiviaNet_LateFusion(nn.Module):
    def __init__(self, nClasses):
        super(LiviaNet_LateFusion, self).__init__()
        
        # Path-Top
        #self.conv1_Top = torch.nn.Conv3d(1, 25, kernel_size=3, stride=1, padding=0, dilation=1, groups=1, bias=True)
        self.conv1_Top = convBlock(1, 25)
        self.conv2_Top = convBlock(25, 25, batchNorm = True)
        self.conv3_Top = convBlock(25, 25, batchNorm = True)
        self.conv4_Top = convBlock(25, 50, batchNorm = True)
        self.conv5_Top = convBlock(50, 50, batchNorm = True)
        self.conv6_Top = convBlock(50, 50, batchNorm = True)
        self.conv7_Top = convBlock(50, 75, batchNorm = True)
        self.conv8_Top = convBlock(75, 75, batchNorm = True)
        self.conv9_Top = convBlock(75, 75, batchNorm = True)
        
        
        # Path-Middle
        #self.conv1_Middle = torch.nn.Conv3d(1, 25, kernel_size=3, stride=1, padding=0, dilation=1, groups=1, bias=True)
        self.conv1_Middle = convBlock(1, 25)
        self.conv2_Middle = convBlock(25, 25, batchNorm = True)
        self.conv3_Middle = convBlock(25, 25, batchNorm = True)
        self.conv4_Middle = convBlock(25, 50, batchNorm = True)
        self.conv5_Middle = convBlock(50, 50, batchNorm = True)
        self.conv6_Middle = convBlock(50, 50, batchNorm = True)
        self.conv7_Middle = convBlock(50, 75, batchNorm = True)
        self.conv8_Middle = convBlock(75, 75, batchNorm = True)
        self.conv9_Middle = convBlock(75, 75, batchNorm = True)
        
        # Path-Bottom
        #self.conv1_Bottom = torch.nn.Conv3d(1, 25, kernel_size=3, stride=1, padding=0, dilation=1, groups=1, bias=True)
        self.conv1_Bottom = convBlock(1, 25)
        self.conv2_Bottom = convBlock(25, 25, batchNorm = True)
        self.conv3_Bottom = convBlock(25, 25, batchNorm = True)
        self.conv4_Bottom = convBlock(25, 50, batchNorm = True)
        self.conv5_Bottom = convBlock(50, 50, batchNorm = True)
        self.conv6_Bottom = convBlock(50, 50, batchNorm = True)
        self.conv7_Bottom = convBlock(50, 75, batchNorm = True)
        self.conv8_Bottom = convBlock(75, 75, batchNorm = True)
        self.conv9_Bottom = convBlock(75, 75, batchNorm = True)
        
        
        self.fully_1 = nn.Conv3d(225, 400, kernel_size=1)
        self.fully_2 = nn.Conv3d(400, 100, kernel_size=1)
        self.final = nn.Conv3d(100, nClasses, kernel_size=1)
        
    def forward(self, input):

        # get the 3 channels as 5D tensors
        y_1_top = self.conv1_Top(input[:,0:1,:,:,:])
        y_1_middle = self.conv1_Middle(input[:,1:2,:,:,:])
        y_1_bottom = self.conv1_Bottom(input[:,2:3,:,:,:])
        
        y_2_top = self.conv2_Top(y_1_top)
        y_2_middle = self.conv2_Middle(y_1_middle)
        y_2_bottom = self.conv2_Bottom(y_1_bottom)
        
        y_3_top = self.conv3_Top(y_2_top)
        y_3_middle = self.conv3_Middle(y_2_middle)
        y_3_bottom = self.conv3_Bottom(y_2_bottom)
        
        y_4_top = self.conv4_Top(y_3_top)
        y_4_middle = self.conv4_Middle(y_3_middle)
        y_4_bottom = self.conv4_Bottom(y_3_bottom)
        
        y_5_top = self.conv5_Top(y_4_top)
        y_5_middle = self.conv5_Middle(y_4_middle)
        y_5_bottom = self.conv5_Bottom(y_4_bottom)
        
        y_6_top = self.conv6_Top(y_5_top)
        y_6_middle = self.conv6_Middle(y_5_middle)
        y_6_bottom = self.conv6_Bottom(y_5_bottom)
        
        y_7_top = self.conv7_Top(y_6_top)
        y_7_middle = self.conv7_Middle(y_6_middle)
        y_7_bottom = self.conv7_Bottom(y_6_bottom)
        
        y_8_top = self.conv8_Top(y_7_top)
        y_8_middle = self.conv8_Middle(y_7_middle)
        y_8_bottom = self.conv8_Bottom(y_7_bottom)
        
        y_9_top = self.conv9_Top(y_8_top)
        y_9_middle = self.conv9_Middle(y_8_middle)
        y_9_bottom = self.conv9_Bottom(y_8_bottom)
       
        y = self.fully_1(torch.cat((y_9_top, y_9_middle, y_9_bottom),dim=1))
        y = self.fully_2(y)
        
        return self.final(y)


class HyperDenseNet_2Mod(nn.Module):
    def __init__(self, nClasses):
        super(HyperDenseNet_2Mod, self).__init__()

        # Path-Top
        self.conv1_Top = convBlock(1, 25)
        self.conv2_Top = convBlock(50, 25, batchNorm=True)
        self.conv3_Top = convBlock(100, 25, batchNorm=True)
        self.conv4_Top = convBlock(150, 50, batchNorm=True)
        self.conv5_Top = convBlock(250, 50, batchNorm=True)
        self.conv6_Top = convBlock(350, 50, batchNorm=True)
        self.conv7_Top = convBlock(450, 75, batchNorm=True)
        self.conv8_Top = convBlock(600, 75, batchNorm=True)
        self.conv9_Top = convBlock(750, 75, batchNorm=True)

        # Path-Bottom
        self.conv1_Bottom = convBlock(1, 25)
        self.conv2_Bottom = convBlock(50, 25, batchNorm=True)
        self.conv3_Bottom = convBlock(100, 25, batchNorm=True)
        self.conv4_Bottom = convBlock(150, 50, batchNorm=True)
        self.conv5_Bottom = convBlock(250, 50, batchNorm=True)
        self.conv6_Bottom = convBlock(350, 50, batchNorm=True)
        self.conv7_Bottom = convBlock(450, 75, batchNorm=True)
        self.conv8_Bottom = convBlock(600, 75, batchNorm=True)
        self.conv9_Bottom = convBlock(750, 75, batchNorm=True)

        self.fully_1 = nn.Conv3d(1800, 400, kernel_size=1)
        self.fully_2 = nn.Conv3d(400, 200, kernel_size=1)
        self.fully_3 = nn.Conv3d(200, 150, kernel_size=1)
        self.final = nn.Conv3d(150, nClasses, kernel_size=1)

    def forward(self, input):
        # ----- First layer ------ #
        # get 2 of the channels as 5D tensors
        y1t = self.conv1_Top(input[:, 0:1, :, :, :])
        y1b = self.conv1_Bottom(input[:, 1:2, :, :, :])

        # ----- Second layer ------ #
        # concatenate
        y2t_i = torch.cat((y1t, y1b), dim=1)
        y2b_i = torch.cat((y1b, y1t), dim=1)

        y2t_o = self.conv2_Top(y2t_i)
        y2b_o = self.conv2_Bottom(y2b_i)

        # ----- Third layer ------ #
        y2t_i_cropped = croppCenter(y2t_i, y2t_o.shape)
        y2b_i_cropped = croppCenter(y2b_i, y2t_o.shape)

        # concatenate
        y3t_i = torch.cat((y2t_i_cropped, y2t_o, y2b_o), dim=1)
        y3b_i = torch.cat((y2b_i_cropped, y2b_o, y2t_o), dim=1)

        y3t_o = self.conv3_Top(y3t_i)
        y3b_o = self.conv3_Bottom(y3b_i)

        # ------ Fourth layer ----- #
        y3t_i_cropped = croppCenter(y3t_i, y3t_o.shape)
        y3b_i_cropped = croppCenter(y3b_i, y3t_o.shape)

        # concatenate
        y4t_i = torch.cat((y3t_i_cropped, y3t_o, y3b_o), dim=1)
        y4b_i = torch.cat((y3b_i_cropped, y3b_o, y3t_o), dim=1)

        y4t_o = self.conv4_Top(y4t_i)
        y4b_o = self.conv4_Bottom(y4b_i)

        # ------ Fifth layer ----- #
        y4t_i_cropped = croppCenter(y4t_i, y4t_o.shape)
        y4b_i_cropped = croppCenter(y4b_i, y4t_o.shape)

        # concatenate
        y5t_i = torch.cat((y4t_i_cropped, y4t_o, y4b_o), dim=1)
        y5b_i = torch.cat((y4b_i_cropped, y4b_o, y4t_o), dim=1)

        y5t_o = self.conv5_Top(y5t_i)
        y5b_o = self.conv5_Bottom(y5b_i)

        # ------ Sixth layer ----- #
        y5t_i_cropped = croppCenter(y5t_i, y5t_o.shape)
        y5b_i_cropped = croppCenter(y5b_i, y5t_o.shape)

        # concatenate
        y6t_i = torch.cat((y5t_i_cropped, y5t_o, y5b_o), dim=1)
        y6b_i = torch.cat((y5b_i_cropped, y5b_o, y5t_o), dim=1)

        y6t_o = self.conv6_Top(y6t_i)
        y6b_o = self.conv6_Bottom(y6b_i)

        # ------ Seventh layer ----- #
        y6t_i_cropped = croppCenter(y6t_i, y6t_o.shape)
        y6b_i_cropped = croppCenter(y6b_i, y6t_o.shape)

        # concatenate
        y7t_i = torch.cat((y6t_i_cropped, y6t_o, y6b_o), dim=1)
        y7b_i = torch.cat((y6b_i_cropped, y6b_o, y6t_o), dim=1)

        y7t_o = self.conv7_Top(y7t_i)
        y7b_o = self.conv7_Bottom(y7b_i)

        # ------ Eight layer ----- #
        y7t_i_cropped = croppCenter(y7t_i, y7t_o.shape)
        y7b_i_cropped = croppCenter(y7b_i, y7t_o.shape)

        # concatenate
        y8t_i = torch.cat((y7t_i_cropped, y7t_o, y7b_o), dim=1)
        y8b_i = torch.cat((y7b_i_cropped, y7b_o, y7t_o), dim=1)

        y8t_o = self.conv8_Top(y8t_i)
        y8b_o = self.conv8_Bottom(y8b_i)

        # ------ Ninth layer ----- #
        y8t_i_cropped = croppCenter(y8t_i, y8t_o.shape)
        y8b_i_cropped = croppCenter(y8b_i, y8t_o.shape)

        # concatenate
        y9t_i = torch.cat((y8t_i_cropped, y8t_o, y8b_o), dim=1)
        y9b_i = torch.cat((y8b_i_cropped, y8b_o, y8t_o), dim=1)

        y9t_o = self.conv9_Top(y9t_i)
        y9b_o = self.conv9_Bottom(y9b_i)

        ##### Fully connected layers
        y9t_i_cropped = croppCenter(y9t_i, y9t_o.shape)
        y9b_i_cropped = croppCenter(y9b_i, y9t_o.shape)

        outputPath_top = torch.cat((y9t_i_cropped, y9t_o, y9b_o), dim=1)
        outputPath_bottom = torch.cat((y9b_i_cropped, y9b_o, y9t_o), dim=1)

        inputFully = torch.cat((outputPath_top, outputPath_bottom), dim=1)

        y = self.fully_1(inputFully)
        y = self.fully_2(y)
        y = self.fully_3(y)

        return self.final(y)

class HyperDenseNet(nn.Module):
    def __init__(self, nClasses):
        super(HyperDenseNet, self).__init__()
        
        # Path-Top
        self.conv1_Top = convBlock(1, 25)
        self.conv2_Top = convBlock(75, 25, batchNorm = True)
        self.conv3_Top = convBlock(150, 25, batchNorm = True)
        self.conv4_Top = convBlock(225, 50, batchNorm = True)
        self.conv5_Top = convBlock(375, 50, batchNorm = True)
        self.conv6_Top = convBlock(525, 50, batchNorm = True)
        self.conv7_Top = convBlock(675, 75, batchNorm = True)
        self.conv8_Top = convBlock(900, 75, batchNorm = True)
        self.conv9_Top = convBlock(1125, 75, batchNorm = True)
        
        # Path-Middle
        self.conv1_Middle = convBlock(1, 25)
        self.conv2_Middle = convBlock(75, 25, batchNorm = True)
        self.conv3_Middle = convBlock(150, 25, batchNorm = True)
        self.conv4_Middle = convBlock(225, 50, batchNorm = True)
        self.conv5_Middle = convBlock(375, 50, batchNorm = True)
        self.conv6_Middle = convBlock(525, 50, batchNorm = True)
        self.conv7_Middle = convBlock(675, 75, batchNorm = True)
        self.conv8_Middle = convBlock(900, 75, batchNorm = True)
        self.conv9_Middle = convBlock(1125, 75, batchNorm = True)
        
        # Path-Bottom
        self.conv1_Bottom = convBlock(1, 25)
        self.conv2_Bottom = convBlock(75, 25, batchNorm = True)
        self.conv3_Bottom = convBlock(150, 25, batchNorm = True)
        self.conv4_Bottom = convBlock(225, 50, batchNorm = True)
        self.conv5_Bottom = convBlock(375, 50, batchNorm = True)
        self.conv6_Bottom = convBlock(525, 50, batchNorm = True)
        self.conv7_Bottom = convBlock(675, 75, batchNorm = True)
        self.conv8_Bottom = convBlock(900, 75, batchNorm = True)
        self.conv9_Bottom = convBlock(1125, 75, batchNorm = True)
        
        
        self.fully_1 = nn.Conv3d(4050, 400, kernel_size=1)
        self.fully_2 = nn.Conv3d(400, 200, kernel_size=1)
        self.fully_3 = nn.Conv3d(200, 150, kernel_size=1)
        self.final = nn.Conv3d(150, nClasses, kernel_size=1)
        
    def forward(self, input):

        # ----- First layer ------ #
        # get the 3 channels as 5D tensors
        y1t = self.conv1_Top(input[:,0:1,:,:,:])
        y1m = self.conv1_Middle(input[:,1:2,:,:,:])
        y1b = self.conv1_Bottom(input[:,2:3,:,:,:])
        
        # ----- Second layer ------ #
        # concatenate
        y2t_i = torch.cat((y1t,y1m,y1b),dim=1)
        y2m_i = torch.cat((y1m,y1t,y1b),dim=1)
        y2b_i = torch.cat((y1b,y1t,y1m),dim=1)

        y2t_o = self.conv2_Top(y2t_i)
        y2m_o = self.conv2_Middle(y2m_i)
        y2b_o = self.conv2_Bottom(y2b_i)
        
         # ----- Third layer ------ #
        y2t_i_cropped = croppCenter(y2t_i, y2t_o.shape)
        y2m_i_cropped = croppCenter(y2m_i, y2t_o.shape)
        y2b_i_cropped = croppCenter(y2b_i, y2t_o.shape)
        
        # concatenate
        y3t_i = torch.cat((y2t_i_cropped, y2t_o,y2m_o,y2b_o),dim=1)
        y3m_i = torch.cat((y2m_i_cropped, y2m_o,y2t_o,y2b_o),dim=1)
        y3b_i = torch.cat((y2b_i_cropped, y2b_o,y2t_o,y2m_o),dim=1)
        
        y3t_o = self.conv3_Top(y3t_i)
        y3m_o = self.conv3_Middle(y3m_i)
        y3b_o = self.conv3_Bottom(y3b_i)
        
        # ------ Fourth layer ----- #
        y3t_i_cropped = croppCenter(y3t_i, y3t_o.shape)
        y3m_i_cropped = croppCenter(y3m_i, y3t_o.shape)
        y3b_i_cropped = croppCenter(y3b_i, y3t_o.shape)
        
        # concatenate
        y4t_i = torch.cat((y3t_i_cropped, y3t_o,y3m_o,y3b_o),dim=1)
        y4m_i = torch.cat((y3m_i_cropped, y3m_o,y3t_o,y3b_o),dim=1)
        y4b_i = torch.cat((y3b_i_cropped, y3b_o,y3t_o,y3m_o),dim=1)
        
        y4t_o = self.conv4_Top(y4t_i)
        y4m_o = self.conv4_Middle(y4m_i)
        y4b_o = self.conv4_Bottom(y4b_i)
        
        
        # ------ Fifth layer ----- #
        y4t_i_cropped = croppCenter(y4t_i, y4t_o.shape)
        y4m_i_cropped = croppCenter(y4m_i, y4t_o.shape)
        y4b_i_cropped = croppCenter(y4b_i, y4t_o.shape)
        
        # concatenate
        y5t_i = torch.cat((y4t_i_cropped, y4t_o,y4m_o,y4b_o),dim=1)
        y5m_i = torch.cat((y4m_i_cropped, y4m_o,y4t_o,y4b_o),dim=1)
        y5b_i = torch.cat((y4b_i_cropped, y4b_o,y4t_o,y4m_o),dim=1)
        
        y5t_o = self.conv5_Top(y5t_i)
        y5m_o = self.conv5_Middle(y5m_i)
        y5b_o = self.conv5_Bottom(y5b_i)
        
        # ------ Sixth layer ----- #
        y5t_i_cropped = croppCenter(y5t_i, y5t_o.shape)
        y5m_i_cropped = croppCenter(y5m_i, y5t_o.shape)
        y5b_i_cropped = croppCenter(y5b_i, y5t_o.shape)
        
        # concatenate
        y6t_i = torch.cat((y5t_i_cropped, y5t_o,y5m_o,y5b_o),dim=1)
        y6m_i = torch.cat((y5m_i_cropped, y5m_o,y5t_o,y5b_o),dim=1)
        y6b_i = torch.cat((y5b_i_cropped, y5b_o,y5t_o,y5m_o),dim=1)
        
        y6t_o = self.conv6_Top(y6t_i)
        y6m_o = self.conv6_Middle(y6m_i)
        y6b_o = self.conv6_Bottom(y6b_i)
        
        # ------ Seventh layer ----- #
        y6t_i_cropped = croppCenter(y6t_i, y6t_o.shape)
        y6m_i_cropped = croppCenter(y6m_i, y6t_o.shape)
        y6b_i_cropped = croppCenter(y6b_i, y6t_o.shape)
        
        # concatenate
        y7t_i = torch.cat((y6t_i_cropped, y6t_o,y6m_o,y6b_o),dim=1)
        y7m_i = torch.cat((y6m_i_cropped, y6m_o,y6t_o,y6b_o),dim=1)
        y7b_i = torch.cat((y6b_i_cropped, y6b_o,y6t_o,y6m_o),dim=1)
        
        y7t_o = self.conv7_Top(y7t_i)
        y7m_o = self.conv7_Middle(y7m_i)
        y7b_o = self.conv7_Bottom(y7b_i)
        
        
        # ------ Eight layer ----- #
        y7t_i_cropped = croppCenter(y7t_i, y7t_o.shape)
        y7m_i_cropped = croppCenter(y7m_i, y7t_o.shape)
        y7b_i_cropped = croppCenter(y7b_i, y7t_o.shape)
        
        # concatenate
        y8t_i = torch.cat((y7t_i_cropped, y7t_o,y7m_o,y7b_o),dim=1)
        y8m_i = torch.cat((y7m_i_cropped, y7m_o,y7t_o,y7b_o),dim=1)
        y8b_i = torch.cat((y7b_i_cropped, y7b_o,y7t_o,y7m_o),dim=1)
        
        y8t_o = self.conv8_Top(y8t_i)
        y8m_o = self.conv8_Middle(y8m_i)
        y8b_o = self.conv8_Bottom(y8b_i)
        
        
        # ------ Ninth layer ----- #
        y8t_i_cropped = croppCenter(y8t_i, y8t_o.shape)
        y8m_i_cropped = croppCenter(y8m_i, y8t_o.shape)
        y8b_i_cropped = croppCenter(y8b_i, y8t_o.shape)
        
        # concatenate
        y9t_i = torch.cat((y8t_i_cropped, y8t_o,y8m_o,y8b_o),dim=1)
        y9m_i = torch.cat((y8m_i_cropped, y8m_o,y8t_o,y8b_o),dim=1)
        y9b_i = torch.cat((y8b_i_cropped, y8b_o,y8t_o,y8m_o),dim=1)
        
        y9t_o = self.conv9_Top(y9t_i)
        y9m_o = self.conv9_Middle(y9m_i)
        y9b_o = self.conv9_Bottom(y9b_i)
        
        ##### Fully connected layers
        y9t_i_cropped = croppCenter(y9t_i, y9t_o.shape)
        y9m_i_cropped = croppCenter(y9m_i, y9t_o.shape)
        y9b_i_cropped = croppCenter(y9b_i, y9t_o.shape)
        
        outputPath_top    = torch.cat((y9t_i_cropped, y9t_o, y9m_o, y9b_o), dim=1)
        outputPath_middle = torch.cat((y9m_i_cropped, y9m_o, y9t_o, y9b_o), dim=1)
        outputPath_bottom = torch.cat((y9b_i_cropped, y9b_o, y9t_o, y9m_o), dim=1)
        
        inputFully = torch.cat((outputPath_top, outputPath_middle, outputPath_bottom), dim=1)
        
        y = self.fully_1(inputFully)
        y = self.fully_2(y)
        y = self.fully_3(y)
        
        return self.final(y)


