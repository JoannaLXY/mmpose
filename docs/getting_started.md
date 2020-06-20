# Getting Started

This page provides basic tutorials about the usage of MMPose.
For installation instructions, please see [install.md](install.md).

## Prepare datasets

It is recommended to symlink the dataset root to `$MMPOSE/data`.
If your folder structure is different, you may need to change the corresponding paths in config files.

```
mmpose
├── mmpose
├── docs
├── tests
├── tools
├── configs
`── data
    │── coco
        │-- annotations
        │   │-- person_keypoints_train2017.json
        │   `-- person_keypoints_val2017.json
        │-- train2017
        │   │-- 000000000009.jpg
        │   │-- 000000000025.jpg
        │   │-- 000000000030.jpg
        │   │-- ...
        `-- val2017
            │-- 000000000139.jpg
            │-- 000000000285.jpg
            │-- 000000000632.jpg
            │-- ...

```

For using custom datasets, please refer to [Tutorial 2: Adding New Dataset](tutorials/new_dataset.md)

## Inference with pretrained models

We provide testing scripts to evaluate a whole dataset (COCO, etc.),
and provide some high-level apis for easier integration to other projects.

### Test a dataset

- [x] single GPU
- [x] single node multiple GPUs
- [x] multiple node

You can use the following commands to test a dataset.

```shell
# single-gpu testing
python tools/test.py ${CONFIG_FILE} ${CHECKPOINT_FILE} [--out ${RESULT_FILE}] [--eval ${EVAL_METRICS}] \
    [--proc_per_gpu ${NUM_PROC_PER_GPU}] [--gpu_collect] [--tmpdir ${TMPDIR}] [--average_clips ${AVG_TYPE}] \
    [--launcher ${JOB_LAUNCHER}] [--local_rank ${LOCAL_RANK}]

python tools/test.py ${CONFIG_FILE} ${CHECKPOINT_FILE} ${GPU_NUM} [--out ${RESULT_FILE}] [--eval ${EVAL_METRICS}] \
    [--proc_per_gpu ${NUM_PROC_PER_GPU}] [--gpu_collect] [--tmpdir ${TMPDIR}] [--average_clips ${AVG_TYPE}] \
    [--launcher ${JOB_LAUNCHER}] [--local_rank ${LOCAL_RANK}]
```

Optional arguments:
- `RESULT_FILE`: Filename of the output results. If not specified, the results will not be saved to a file.
- `EVAL_METRICS`: Items to be evaluated on the results. Allowed values depend on the dataset, e.g., `top_k_accuracy`, `mean_class_accuracy` are available for all datasets in recognition, `mean_average_precision` for Multi-Moments in Time, `AR@AN` for ActivityNet, etc.
- `NUM_PROC_PER_GPU`: Number of processes per GPU. If not specified, only one process will be assigned for a single gpu.
- `--gpu_collect`: If specified, recognition results will be collected using gpu communication. Otherwise, it will save the results on different gpus to `TMPDIR` and collect them by the rank 0 worker.
- `TMPDIR`: Temporary directory used for collecting results from multiple workers, available when `--gpu_collect` is not specified.
- `AVG_TYPE`: Items to average the test clips. If set to `prob`, it will apply softmax before averaging the clip scores. Otherwise, it will directly average the clip scores.
- `JOB_LAUNCHER`: Items for distributed job initialization launcher. Allowed choices are `none`, `pytorch`, `slurm`, `mpi`. Especially, if set to none, it will test in a non-distributed mode.
- `LOCAL_RANK`: ID for local rank. If not specified, it will be set to 0.

Examples:

Assume that you have already downloaded the checkpoints to the directory `checkpoints/`.

1. Test SBL on COCO (without saving the test results) and evaluate the mAP.

```shell
python tools/test.py config/i3d_rgb_32x2x1_r50_3d_kinetics400_100e.py \
    checkpoints/SOME_CHECKPOINT.pth \
```

2. Test SBL on COCO with 8 GPUS, and evaluate the mAP.

```shell
python tools/test.py config/tsn_rgb_1x1x8_r50_2d_sthv1_50e.py \
    checkpoints/SOME_CHECKPOINT.pth \
    8 --out results.pkl --eval top_k_accuracy
```

3. Test SBL on COCO in slurm environment and evaluate the mAP.

```shell
python tools/test.py config/i3d_rgb_32x2x1_r50_3d_kinetics400_100e.py \
    checkpoints/SOME_CHECKPOINT.pth \
    --launcher slurm --eval top_k_accuracy
```

### Video demo

We provide a demo script to predict the recognition result using a single video.

```shell
python demo/demo.py ${CONFIG_FILE} ${CHECKPOINT_FILE} ${VIDEO_FILE} [--device ${GPU_ID}]
```

Examples:

```shell
python demo/demo.py config/tsn_rgb_1x1x3_r50_2d_kinetics400_video_100e.py checkpoints/tsn.pth demo/demo.mp4
```

### High-level APIs for testing a video.

Here is an example of building the model and test a given video.

```python
from mmpose.core import init_pose_detector, inference_pose_detector

config_file = 'configs/TopDown/resnet/coco/res18_coco_256x192.py'
# download the checkpoint from model zoo and put it in `checkpoints/`
checkpoint_file = 'checkpoints/tsn.pth'

 # build the model from a config file and a checkpoint file
model = init_pose_detector(config_file, checkpoint_file, device='cpu')

# test a single video and show the result:
video = 'demo/demo.mp4'
labels = 'demo/label_map.txt'
result = inference_pose_detector(model, video, labels)

# show the results
for key in result:
    print(f'{key}: ', result[key])
```

A notebook demo can be found in [demo/demo.ipynb](../demo/demo.ipynb)


## Train a model

MMPose implements distributed training and non-distributed training,
which uses `MMDistributedDataParallel` and `MMDataParallel` respectively.

All outputs (log files and checkpoints) will be saved to the working directory,
which is specified by `work_dir` in the config file.

By default we evaluate the model on the validation set after each epoch, you can change the evaluation interval by modifying the interval argument in the training config
```python
evaluation = dict(interval=5)  # This evaluate the model per 5 epoch.
```

According to the [Linear Scaling Rule](https://arxiv.org/abs/1706.02677), you need to set the learning rate proportional to the batch size if you use different GPUs or videos per GPU, e.g., lr=0.01 for 4 GPUs * 2 video/gpu and lr=0.08 for 16 GPUs * 4 video/gpu.

### Train with a single GPU

```shell
python tools/train.py ${CONFIG_FILE} [optional arguments]
```

If you want to specify the working directory in the command, you can add an argument `--work_dir ${YOUR_WORK_DIR}`.

### Train with multiple GPUs

```shell
./tools/dist_train.sh ${CONFIG_FILE} ${GPU_NUM} [optional arguments]
```

Optional arguments are:

- `--validate` (**strongly recommended**): Perform evaluation at every k (default value is 5 epochs during the training.
- `--work-dir ${WORK_DIR}`: Override the working directory specified in the config file.
- `--resume-from ${CHECKPOINT_FILE}`: Resume from a previous checkpoint file.
- `--gpus ${GPU_NUM}`: Number of gpus to use, which is only applicable to non-distributed training.
- `--seed ${SEED}`: Seed id for random state in python, numpy and pytorch to generate random numbers.
- `--deterministic`: If specified, it will set deterministic options for CUDNN backend.
- `JOB_LAUNCHER`: Items for distributed job initialization launcher. Allowed choices are `none`, `pytorch`, `slurm`, `mpi`. Especially, if set to none, it will test in a non-distributed mode.
- `LOCAL_RANK`: ID for local rank. If not specified, it will be set to 0.
- `--autoscale-lr`: If specified, it will automatically scale lr with the number of gpus by [Linear Scaling Rule](https://arxiv.org/abs/1706.02677).

Difference between `resume-from` and `load-from`:
`resume-from` loads both the model weights and optimizer status, and the epoch is also inherited from the specified checkpoint. It is usually used for resuming the training process that is interrupted accidentally.
`load-from` only loads the model weights and the training epoch starts from 0. It is usually used for finetuning.

Here is an example of using 8 GPUs to load SBL checkpoint.

```shell
./tools/dist_train.sh config/tsn_rgb_1x1x3_r50_2d_kinetics400_100e.py 8 --resume_from work_dirs/tsn_rgb_1x1x3_r50_2d_kinetics400_100e/latest.pth
```

### Train with multiple machines

If you can run MMPose on a cluster managed with [slurm](https://slurm.schedmd.com/), you can use the script `slurm_train.sh`. (This script also supports single machine training.)

```shell
./tools/slurm_train.sh ${PARTITION} ${JOB_NAME} ${CONFIG_FILE} ${WORK_DIR} [${GPUS}]
```

Here is an example of using 16 GPUs to train SBL on the dev partition in a slurm cluster. (use `GPUS_PER_NODE=8` to specify a single slurm cluster node with 8 GPUs.)

```shell
GPUS_PER_NODE=8 ./tools/slurm_train.sh dev tsn_r50_k400 config/tsn_rgb_1x1x3_r50_2d_kinetics400_100e.py work_dirs/tsn_rgb_1x1x3_r50_2d_kinetics400_100e 16
```

You can check [slurm_train.sh](../tools/slurm_train.sh) for full arguments and environment variables.

If you have just multiple machines connected with ethernet, you can refer to
pytorch [launch utility](https://pytorch.org/docs/stable/distributed_deprecated.html#launch-utility).
Usually it is slow if you do not have high speed networking like infiniband.

### Launch multiple jobs on a single machine

If you launch multiple jobs on a single machine, e.g., 2 jobs of 4-GPU training on a machine with 8 GPUs,
you need to specify different ports (29500 by default) for each job to avoid communication conflict.

If you use `dist_train.sh` to launch training jobs, you can set the port in commands.

```shell
CUDA_VISIBLE_DEVICES=0,1,2,3 PORT=29500 ./tools/dist_train.sh ${CONFIG_FILE} 4
CUDA_VISIBLE_DEVICES=4,5,6,7 PORT=29501 ./tools/dist_train.sh ${CONFIG_FILE} 4
```

If you use launch training jobs with slurm, you need to modify the config files (usually the 6th line from the bottom in config files) to set different communication ports.

In `config1.py`,
```python
dist_params = dict(backend='nccl', port=29500)
```

In `config2.py`,
```python
dist_params = dict(backend='nccl', port=29501)
```

Then you can launch two jobs with `config1.py` ang `config2.py`.

```shell
CUDA_VISIBLE_DEVICES=0,1,2,3 ./tools/slurm_train.sh ${PARTITION} ${JOB_NAME} config1.py ${WORK_DIR} 4
CUDA_VISIBLE_DEVICES=4,5,6,7 ./tools/slurm_train.sh ${PARTITION} ${JOB_NAME} config2.py ${WORK_DIR} 4
```

## Useful tools

We provide lots of useful tools under `tools/` directory.

### Analyze logs

You can plot loss/top-k acc curves given a training log file. Run `pip install seaborn` first to install the dependency.

![acc_curve_image](./acc_curve.png)

```shell
python tools/analyze_logs.py plot_curve ${JSON_LOGS} [--keys ${KEYS}] [--title ${TITLE}] [--legend ${LEGEND}] [--backend ${BACKEND}] [--style ${STYLE}] [--out ${OUT_FILE}]
```

Examples:

- Plot the classification loss of some run.

```shell
python tools/analyze_logs.py plot_curve log.json --keys loss_cls --legend loss_cls
```

- Plot the top-1 acc and top-5 acc of some run, and save the figure to a pdf.

```shell
python tools/analyze_logs.py plot_curve log.json --keys top1_acc top5_acc --out results.pdf
```

- Compare the top-1 acc of two runs in the same figure.

```shell
python tools/analyze_logs.py plot_curve log1.json log2.json --keys top1_acc --legend run1 run2
```

You can also compute the average training speed.

```shell
python tools/analyze_logs.py cal_train_time ${JSON_LOGS} [--include-outliers]
```

- Compute the average training speed for a config file

```shell
python tools/analyze_logs.py cal_train_time work_dirs/some_exp/20200422_153324.log.json
```

The output is expected to be like the following.

```
-----Analyze train time of work_dirs/some_exp/20200422_153324.log.json-----
slowest epoch 60, average time is 0.9736
fastest epoch 18, average time is 0.9001
time std over epochs is 0.0177
average iter time: 0.9330 s/iter

```

### Get the FLOPs and params (experimental)

We provide a script adapted from [flops-counter.pytorch](https://github.com/sovrasov/flops-counter.pytorch) to compute the FLOPs and params of a given model.

```shell
python tools/get_flops.py ${CONFIG_FILE} [--shape ${INPUT_SHAPE}]
```

We will get the result like this

```
Input shape: (1, 3, 32, 340, 256)
Flops: 37.1 GMac
Params: 28.04 M
```

**Note**: This tool is still experimental and we do not guarantee that the number is correct.
You may well use the result for simple comparisons, but double check it before you adopt it in technical reports or papers.

(1) FLOPs are related to the input shape while parameters are not. The default input shape is (1, 3, 340, 256) for 2D recognizer, (1, 3, 32, 340, 256) for 3D recognizer.
(2) Some custom operators are not counted into FLOPs.
You can add support for new operators by modifying [`mmaction/utils/flops_counter.py`](../mmaction/utils/file_client.py).

### Publish a model

Before you upload a model to AWS, you may want to
(1) convert model weights to CPU tensors,
(2) delete the optimizer states and
(3) compute the hash of the checkpoint file and append the hash id to the filename.

```shell
python tools/publish_model.py ${INPUT_FILENAME} ${OUTPUT_FILENAME}
```

E.g.,

```shell
python tools/publish_model.py work_dirs/tsn_rgb_1x1x3_r50_2d_kinetics400_100e/latest.pth tsn_rgb_1x1x3_r50_2d_kinetics400_100e.pth
```

The final output filename will be `tsn_rgb_1x1x3_r50_2d_kinetics400_100e-{hash id}.pth`


## Tutorials

Currently, we provide some tutorials for users to [finetune model](tutorials/finetune.md),
[add new dataset](tutorials/new_dataset.md), [add new modules](tutorials/new_modules.md).
