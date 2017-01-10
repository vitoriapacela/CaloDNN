#PBS -V

cd ~/DLKit/
source setup.sh

mkdir -p ScanLogs
output=ScanLogs/$PBS_ARRAYID.log

echo $output > $output

python -m CaloDNN.ClassificationExperiment &> $output



