#!/usr/bin/env bash

node=Pose
numGPU=8
allGPU=8

CONFIG=configs/TopDown/resnet/coco/res50_coco_256x192.py
WORK_DIR='work_dirs/res50_coco_256x192/'

PY_ARGS=${@:5}
PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \

OMPI_MCA_mpi_warn_on_fork=0 \
srun -p $node \
    --gres=gpu:$numGPU \
    -n$allGPU \
    --ntasks-per-node=$numGPU \
    --job-name=$expID \
    --kill-on-bad-exit=1 \
    python -u tools/train.py ${CONFIG} --work-dir=${WORK_DIR} --launcher="slurm" ${PY_ARGS}
