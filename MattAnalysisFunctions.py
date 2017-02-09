import numpy as np
from ROOT import TH1F,TCanvas,TF1
import scipy.interpolate
import sys
from sklearn.metrics import roc_curve, auc
from itertools import compress

def Plot1D(xValues, yValues, plotTitle, xLabel, yLabel, saveName):

    import matplotlib as mpl
    mpl.use('pdf')
    import matplotlib.pyplot as plt

    lw=2
    mpColors=["blue","green","red","cyan","magenta","yellow","black","white"]

    plt.plot(xValues,yValues,color=mpColors[0],
     lw=lw, label=plotTitle)

    plt.xlabel(xLabel)
    plt.ylabel(yLabel)

    plt.legend(loc="lower right")
        
    plt.savefig("TrainedModels/Plots/"+saveName+".pdf")
    plt.close()

def Plot2D(xValues, yValues, zValues, plotTitle, xLabel, yLabel, zLabel, saveName):

    import matplotlib as mpl
    mpl.use('pdf')
    import matplotlib.pyplot as plt
    # from matplotlib import mlab, cm
    from mpl_toolkits.mplot3d import Axes3D

    lw=2

    fig = plt.figure()
    ax = fig.gca(projection='3d')
    cset1 = ax.plot_trisurf(
                xValues, yValues, zValues, linewidth=lw,
                    antialiased=True)

    ax.set_xlabel(xLabel)
    ax.set_ylabel(yLabel)
    ax.set_zlabel(zLabel)

    plt.savefig("TrainedModels/Plots/"+saveName+".pdf")
    plt.close()

def FPR_at90pTPR_SinglePoint(MyModel,Test_X,Test_Y,Test_YT,BatchSize):

    import matplotlib as mpl
    mpl.use('pdf')
    import matplotlib.pyplot as plt
    sys.stdout.flush()

    NClasses=Test_Y.shape[1] # Y is a simple classification vector, where the correct class index is labeled 1

    # Fake rate at 90% efficiency vs. object pT
    max_YT = Test_YT.max()
    pTs = np.linspace(0,max_YT,10) # divide the pT ranges from 0 up to the max pT of an object in our data, into 10 bins

    lw=2
    mpColors=["blue","green","red","cyan","magenta","yellow","black","white"]

    result = MyModel.Model.predict(Test_X, batch_size=BatchSize) # here's what my trained model predicts

    sys.stdout.flush()

    ClassIndex = 0 # electron (I hope)

    fpr, tpr, _ = roc_curve(Test_Y[:,ClassIndex], 
                        result[:,ClassIndex])

    fpr_interp = scipy.interpolate.interp1d(tpr, fpr) # interpolation function that returns fpr, given tpr

    return fpr_interp(0.9)

def FPR_at90pTPR_pTBinned(MyModel,Test_X,Test_Y,Test_YT,BatchSize):

    import matplotlib as mpl
    mpl.use('pdf')
    import matplotlib.pyplot as plt
    sys.stdout.flush()

    NClasses=Test_Y.shape[1] # Y is a simple classification vector, where the correct class index is labeled 1

    # Fake rate at 90% efficiency vs. object pT
    max_YT = Test_YT.max()
    pTs = np.linspace(0,max_YT,10) # divide the pT ranges from 0 up to the max pT of an object in our data, into 10 bins

    lw=2
    mpColors=["blue","green","red","cyan","magenta","yellow","black","white"]

    pT_midranges = []
    fprs = []

    for pT_index in range(len(pTs)-1):

        low_pT = pTs[pT_index] # low end of each bin
        high_pT = pTs[pT_index+1] # high end of each bin

        result = MyModel.Model.predict(Test_X, batch_size=BatchSize) # here's what my trained model predicts

        # now look at X, Y, and YT for events only in our pT range
        sub_Test_YT = [YT if (YT>low_pT and YT<=high_pT) else -1 for YT in Test_YT] # YT is the pT (oh shit actually maybe it's not)
        sub_Test_X = np.array([X for (X, YT) in zip(Test_X, sub_Test_YT) if (YT is not -1)]) # X is the calo data
        sub_Test_Y = np.array([Y for (Y, YT) in zip(Test_Y, sub_Test_YT) if (YT is not -1)]) # Y is whether it's an electron or pion
        sub_result = np.array([r for (r, YT) in zip(result, sub_Test_YT) if (YT is not -1)]) # what the model predicts

        sys.stdout.flush()

        for ClassIndex in xrange(0,NClasses):

            fpr, tpr, _ = roc_curve(sub_Test_Y[:,ClassIndex], 
                                sub_result[:,ClassIndex])

            fpr_interp = scipy.interpolate.interp1d(tpr, fpr) # interpolation function that returns fpr, given tpr

            # print "FPR at TPR of 90% for objects in pT range:", fpr_interp(0.9)

        pT_midranges.append((low_pT+high_pT)*0.5)
        fprs.append(fpr_interp(0.9))

    plt.plot(pT_midranges,fprs,color=mpColors[ClassIndex],
     lw=lw, label='FPR at TPR of 90% vs. pT of Object')

    plt.ylabel('False Positive Rate')
    plt.xlabel('Object pT')

    plt.legend(loc="lower right")
        
    plt.savefig(MyModel.OutDir+"/FPR_at90pTPR_pTBinned.pdf")
    plt.close()

    return result

def PTBinROCCurve(MyModel,Test_X,Test_Y,Test_YT,BatchSize):

    import matplotlib as mpl
    mpl.use('pdf')
    import matplotlib.pyplot as plt
    sys.stdout.flush()

    NClasses=Test_Y.shape[1] # Y is a simple classification vector, where the correct class index is labeled 1

    # Fake rate at 90% efficiency vs. object pT
    max_YT = Test_YT.max()
    pTs = np.linspace(0,max_YT,10) # divide the pT ranges from 0 up to the max pT of an object in our data, into 10 bins

    lw=2
    mpColors=["blue","green","red","cyan","magenta","yellow","black","white"]

    pT_midranges = []
    fprs = []

    for pT_index in range(len(pTs)-1):

        low_pT = pTs[pT_index] # low end of each bin
        high_pT = pTs[pT_index+1] # high end of each bin

        result = MyModel.Model.predict(Test_X, batch_size=BatchSize) # here's what my trained model predicts

        # now look at X, Y, and YT for events only in our pT range
        sub_Test_YT = [YT if (YT>low_pT and YT<=high_pT) else -1 for YT in Test_YT] # YT is the pT (oh shit actually maybe it's not)
        sub_Test_X = np.array([X for (X, YT) in zip(Test_X, sub_Test_YT) if (YT is not -1)]) # X is the calo data
        sub_Test_Y = np.array([Y for (Y, YT) in zip(Test_Y, sub_Test_YT) if (YT is not -1)]) # Y is whether it's an electron or pion
        sub_result = np.array([r for (r, YT) in zip(result, sub_Test_YT) if (YT is not -1)]) # what the model predicts

        sys.stdout.flush()

        for ClassIndex in xrange(0,NClasses):

            fpr, tpr, _ = roc_curve(sub_Test_Y[:,ClassIndex], 
                                sub_result[:,ClassIndex])

            roc_auc = auc(fpr, tpr)    

            plt.plot(fpr,tpr,color=mpColors[ClassIndex],
                     lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)

        plt.plot(pT_midranges,fprs,color=mpColors[ClassIndex],
         lw=lw, label='ROC Curve for PT from %d to %d' % (low_pT, high_pT))

        plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])

        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')

        plt.legend(loc="lower right")
            
        plt.savefig(MyModel.OutDir+"/PTBinROCCurve_%d_%d.pdf" % (low_pT, high_pT))
        plt.close()

    return result
