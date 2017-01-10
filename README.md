# CaloNN

Get the packages:

    git clone https://bitbucket.org/anomalousai/DLKit
    cd DLKit
    git clone https://github.com/UTA-HEP-Computing/CaloNN


Work from DLKit Directory:

    cd DLKit

Check out the arguments:

    python -m CaloNN.Experiment --help


Run an experiment:

    python -m CaloNN.Experiment


Look at the results in `TrainedModels` directory.

# Running a hyperparameter scan

qsub -q gpu_queue -t 0-5 CaloDNN/ScanJob.sh

# Loading a model to check results interactively in python

python -im CaloDNN.ClassificationExperiment --NoTrain --NoAnalysis -L TrainedModels/CaloDNN_32_1
MyModel.MetaData.keys()
