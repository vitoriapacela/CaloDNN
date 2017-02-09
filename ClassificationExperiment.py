import sys,os,argparse

execfile("/Users/mattzhang/Desktop/DLKit/CaloDNN/ClassificationArguments.py")
from keras.callbacks import EarlyStopping, TensorBoard
from keras.optimizers import RMSprop

# Process the ConfigFile
execfile(ConfigFile)

# Now put config in the current scope. Must find a prettier way.
if "Config" in dir():
    for a in Config:
        exec(a+"="+str(Config[a]))

# Load the Data
from CaloDNN.LoadData import *

if Mode=="Regression":
    Binning=False
if Mode=="Classification":
    Binning=[NBins,M_min,M_max,Sigma]

InputFile="/Users/mattzhang/Desktop/DLKit/data/LCD/LCD-Electrons-Pi0.h5"

if useGenerator:
    NSamples = 90000
    NTestSamples = 10000
    Train_gen = LoadDataGen(InputFile, BatchSize=BatchSize, Max=NSamples)
    Test_gen = LoadDataGen(InputFile, BatchSize=BatchSize, Skip=NSamples)
    Test2_gen = LoadDataGen(InputFile, BatchSize=NTestSamples, Skip=NSamples)

    TXS = BatchSize, 20, 20, 25
    NInputs=TXS[1]*TXS[2]*TXS[3]
    print "NInputs is %i" % NInputs

else:
    (Train_X, Train_Y, Train_YT), (Test_X, Test_Y, Test_YT) = LoadData(InputFile, FractionTest, MaxEvents, BatchSize)
    #(Train_X, Train_Y),(Test_X, Test_Y) = LoadData(InputFile,FractionTest,MaxEvents=MaxEvents)

    # Normalize the Data... seems to be critical!
    Norm = np.max(Train_X)
    Train_X = Train_X/Norm
    Test_X = Test_X/Norm
    NSamples = len(Train_X)
    NTestSamples = len(Test_X)
    print "Norm is %g" % Norm
    print "NSamples is %g" % NSamples
    print "NTestSamples is %g" % NTestSamples

    TXS = BatchSize, 20, 20, 25
    NInputs=TXS[1]*TXS[2]*TXS[3]
    print "NInputs is %i" % NInputs

# Build/Load the Model
from DLTools.ModelWrapper import ModelWrapper
from CaloDNN.Classification import *

if not LoadModels: # single input model (or no model)

    if LoadModel:
        print "Loading Model From:",LoadModel
        if LoadModel[-1]=="/":
            LoadModel=LoadModel[:-1]
        Name=os.path.basename(LoadModel)
        MyModel=ModelWrapper(Name, True)
        MyModel.InDir=os.path.dirname(LoadModel)
        MyModel.InDir=LoadModel
        MyModel.Load()

    else:
        print "Building Model...",
        sys.stdout.flush()
        Hyperparameters = {"width":Width, "depth":Depth, "dropout_rate":DropoutRate}
        MyModel=Fully3DImageClassification(Name, TXS, Hyperparameters, BatchSize, NClasses, WeightInitialization)

        # Build it
        MyModel.Build()
        print " Done."

    # Print out the Model Summary
    MyModel.Model.summary()

    # Compile The Model
    print "Compiling Model."
    optimizer = RMSprop(lr=LearningRate, rho=0.9, epsilon=1e-08, decay=0.0)
    MyModel.Compile(Loss=loss,Optimizer=optimizer) 

    # Train
    if Train:
        if useGenerator:
            print "Training."
            callbacks=[EarlyStopping(monitor='val_loss', patience=2, verbose=1, mode='min') ]
            callbacks=[]

            MyModel.Model.fit_generator(Train_gen,
                    nb_epoch=Epochs,
                    nb_worker=1,
                    verbose=2,
                    samples_per_epoch=90000, #HARDCODED
                    callbacks=callbacks,
                    pickle_safe=True)
            score = MyModel.Model.evaluate_generator(Test_gen,
                    val_samples=10000, #HARDCODED
                    max_q_size=10,
                    nb_worker=1,
                    pickle_safe=True)

        else:
            print "Training. Using TensorBoard."
            #callbacks=[]
            callbacks=[EarlyStopping(monitor='loss', patience=2, verbose=1, mode='min'), TensorBoard(log_dir='./TensorBoardLogs', histogram_freq=0, write_graph=True, write_images=False)]
            #callbacks=[TensorBoard(log_dir='./TensorBoardLogs', histogram_freq=0, write_graph=True, write_images=False)]

            MyModel.Train(Train_X, Train_Y, Epochs, BatchSize, Callbacks=callbacks)
            score = MyModel.Model.evaluate(Test_X, Test_Y, batch_size=BatchSize)

        print "Final Score:", score

        # Save Model
        MyModel.Save()

    # Analysis
    if Analyze:
        # ROC curve... not useful here:
        #from CaloDNN.Analysis import MultiClassificationAnalysis
        #result=MultiClassificationAnalysis(MyModel,Test_X,Test_Y,BatchSize )

        if useGenerator:
            Test_X, Test_Y = Test2_gen.next()

        from CaloDNN.Analysis import MultiClassificationAnalysis
        MultiClassificationAnalysis(MyModel,Test_X,Test_Y,BatchSize) # ROC curve

        from CaloDNN.MattAnalysisFunctions import *
        FPR_at90pTPR_pTBinned(MyModel,Test_X,Test_Y,Test_YT,BatchSize) # Fake rate at 90% accuracy by pT bin
        PTBinROCCurve(MyModel,Test_X,Test_Y,Test_YT,BatchSize) # ROC curve by pT bin

else: # hyperparameter plots

    # Analysis
    if Analyze:

        from CaloDNN.MattAnalysisFunctions import *

        # hyperparameters for each model
        epoch = []
        learningRate = []
        FPRs_at90pTPR = [] # hyperparameters for each model

        print "Loading Models From:",LoadModels
        with open(LoadModels) as myModels:
            for model in myModels:
                if model[-1]=="/":
                    model=model[:-1]
                Name=os.path.basename(model).strip('\n')

                epoch.append(float(Name.split("_")[2]))
                learningRate.append(float(Name.split("_")[6]))

                MyModel=ModelWrapper(Name, True)
                # MyModel.InDir=os.path.dirname(model)
                # MyModel.InDir=model
                MyModel.InDir="TrainedModels/"+Name
                MyModel.Load()

                print "Compiling Model."
                optimizer = RMSprop(lr=LearningRate, rho=0.9, epsilon=1e-08, decay=0.0)
                MyModel.Compile(Loss=loss,Optimizer=optimizer) 

                FPRs_at90pTPR.append(FPR_at90pTPR_SinglePoint(MyModel,Test_X,Test_Y,Test_YT,BatchSize))

        # Plot1D(epoch, FPRs_at90pTPR, "FPR when TPR is 90% for Various Epochs", "Epochs", "FPR at 90% TPR", "FPRs_at90TPR_scan")
        Plot2D(epoch, learningRate, FPRs_at90pTPR, "FPR when TPR is 90% for Various Epochs and Learning Rates", "Epochs", "Learning Rate", "FPR at 90% TPR", "FPRs_at90TPR_2Dscan")
