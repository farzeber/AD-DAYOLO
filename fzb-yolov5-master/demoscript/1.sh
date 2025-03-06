#!/bin/bash
#SBATCH -J zero-dce-regularall
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 5
#SBATCH --gres=gpu:1
#SBATCH -o out/instance_domain_detect_%J.out
#SBATCH -e err/instance_domain_detect_%J.out
env=py39cu116
# note: dir <slurm_logs> should be created manually

echo "job start time : $(date "+%Y-%m-%d %H:%M:%S")"
if [ $SLURM_JOB_ID ];then
	echo "job id : ${SLURM_JOB_ID}"
	echo "job run on node : ${SLURM_JOB_NODELIST}"
fi

# conda env
if [ $env ];then
	source activate $env
fi

# gpu for normal machine ( not slurm )
if [ $gpu ];then
	export CUDA_VISIBLE_DEVICES=$gpu
fi

# gpu for slurm
if [ $SLURM_JOB_GPUS ];then
	CUDA_VISIBLE_DEVICES=$SLURM_JOB_GPUS
fi


# job main shell
while [ 1 ]
do
	bash 2.sh
	python 1.py
done

echo "job end time : $(date "+%Y-%m-%d %H:%M:%S")"