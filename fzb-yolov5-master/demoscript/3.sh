echo "job start time : $(date "+%Y-%m-%d %H:%M:%S")"
if [ $SLURM_JOB_ID ];then
	echo "job id : ${SLURM_JOB_ID}"
	echo "job run on node : ${SLURM_JOB_NODELIST}"
fi

# conda env
if [ $env ];then
	source activate $env
	echo "job env : ${env}"
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
#python ../val.py --verbose --weights ../runs/train/exp3-voc-exdark/weights/best.pt --data /public/home/jd_fzb/workspace/yinmengyuVOC.yaml
python  ../DomainTrain.py --cfg ../models/fzbyolov5s.yaml --data /public/home/jd_fzb/workspace/CitySpaceImageDa.yaml --epochs 300 --batch-size 8

#python  ../ganInstTrain.py --cfg ../models/ganHrDAyolov5s.yaml --data /public/home/jd_xbh/jd_fzb/vocDInNgan1.yaml --epochs 600 --batch-size 9
#python  ../ganInstTrain.py --cfg ../models/ganHrDAyolov5s.yaml --data /public/home/jd_xbh/jd_fzb/bddDInN2.yaml --epochs 600 --batch-size 9
#python  ../train.py --dataroot ./datasets/maps --name maps_cyclegan --model cycle_gan
#python  ../DomainHRTrain.py --cfg ../models/deDAyolov5sHr.yaml --data /public/home/jd_xbh/jd_fzb/bddDToN.yaml --epochs 600 --batch-size 8
# python  ../DomainHRTrain.py --cfg ../models/hrDAyolov5s.yaml --data /public/home/jd_xbh/jd_fzb/bddDToN.yaml --epochs 600 --batch-size 64
echo "job end time : $(date "+%Y-%m-%d %H:%M:%S")"