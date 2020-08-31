#$ -l tmem=16G
#$ -l h_rt=9:0:0
#$ -l gpu=true
#$ -S /bin/bash
#$ -j y
#$ -N gpu_worker50
#$ -t 1-50
#$ -tc 4

#$ -o /home/vauvelle/doctor_signature/jobs/logs
hostname
date
PROJECT_DIR='/home/vauvelle/doctor_signature/'
export PYTHONPATH=$PYTHONPATH:$PROJECT_DIR
cd $PROJECT_DIR || exit
source /share/apps/source_files/python/python-3.7.0.source
source /share/apps/source_files/cuda/cuda-10.1.source
source .env
source ./.myenv/bin/activate
echo "Pulling any jobs with status 0"
hyperopt-mongo-worker --mongo=bigtop:27017/hyperopt --poll-interval=0.1 --max-consecutive-failures=5
date
