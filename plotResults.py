import numpy as np
import pdb
import sys
import matplotlib.pyplot as plt

def loadMetrics(folderName):
    # Loss
    loss = np.load(folderName + '/'+folderName+'_loss.npy')
    dice = np.load(folderName + '/'+folderName+'_DSCs.npy')


    # Dice training

    
    return loss,dice

def plot2Models(modelNames):

    model1Name = modelNames[0]
    model2Name = modelNames[1]
    
    [loss1, DSC1] = loadMetrics(model1Name)
    [loss2, DSC2] = loadMetrics(model2Name)
    
    numEpochs1 = len(loss1)
    numEpochs2 = len(loss2)
    
    lim = numEpochs1
    if numEpochs2 < numEpochs1:
        lim = numEpochs2
        

    # Plot features
    #xAxis = np.arange(0, lim, 1)
    xAxis = np.arange(0, 370, 10)

    plt.figure(1)

    # Training Dice
    #plt.subplot(212)

    plt.plot(xAxis, DSC1[0:lim].mean(axis=2), 'r-', label=model1Name,linewidth=2)
    plt.plot(xAxis, DSC2[0:lim].mean(axis=2), 'b-', label=model2Name,linewidth=2)
    legend = plt.legend(loc='lower center', shadow=True, fontsize='large')
    plt.title('DSC Validation)')
    plt.grid(True)
    plt.ylim([0.0, 1])
    plt.xlabel('Number of epochs')
    plt.ylabel('DSC')
    #pdb.set_trace()
    #plt.xlim([0, 10,370])

    plt.show()


def plot(argv):

    modelNames = []
    
    numModels = len(argv)
    
    for i in range(numModels):
        modelNames.append(argv[i])
    
    def oneModel():
        print "-- Ploting one model --"
        plot1Model(modelNames)

    def twoModels():
        print "-- Ploting two models --"
        plot2Models(modelNames)
        
    def threeModels():
        print "-- Ploting three models --"
        plot3Models(modelNames)
        
    def fourModels():
        print "-- Ploting four models --"
        plot4Models(modelNames)
        
    # map the inputs to the function blocks
    options = {1 : oneModel,
               2 : twoModels,
               3: threeModels,
               4 : fourModels
    }
    
    options[numModels]()

    
    
if __name__ == '__main__':
   plot(sys.argv[1:])
