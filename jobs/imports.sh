hostname
date
PROJECT_DIR='/home/vauvelle/doctor_signature/'
export PYTHONPATH=$PYTHONPATH:$PROJECT_DIR
cd $PROJECT_DIR || exit
source /share/apps/source_files/python/python-3.7.0.source
source /share/apps/source_files/cuda/cuda-10.1.source
source .env
source ./.myenv/bin/activate
