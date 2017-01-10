import sys,os,argparse

execfile("CaloDNN/ClassificationArguments.py")
from keras.callbacks import EarlyStopping, TensorBoard

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

InputFile="/scratch/data-backup/afarbin/LCD/LCD-Electrons-Pi0.h5"

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


if LoadModel:
    print "Loading Model From:",LoadModel
    if LoadModel[-1]=="/":
        LoadModel=LoadModel[:-1]
    Name=os.path.basename(LoadModel)
    MyModel=ModelWrapper(Name)
    MyModel.InDir=os.path.dirname(LoadModel)
    MyModel.InDir=LoadModel
    MyModel.Load()

else:
    print "Building Model...",
    sys.stdout.flush()
    MyModel=Fully3DImageClassification(Name, TXS, Width, Depth, BatchSize, NClasses, WeightInitialization)

    # Build it
    MyModel.Build()
    print " Done."

# Print out the Model Summary
MyModel.Model.summary()

# Compile The Model
print "Compiling Model."
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
    # turn off for now because it breaks TensorFlow somehow
    result=MultiClassificationAnalysis(MyModel,Test_X,Test_Y,BatchSize)


