# AD-DAYOLO
 
Training Steps:

# get Yolovs models trained on Source
1.python fzb-yolov5-master\train.py --cfg ./models/yolov5s.yaml --epochs 600  --data ./data/remoteData/bddAdomain.yaml --batch-size 12


# move YOLOv5 model to contrastive-unpaired-translation\models,train CUT with yolo detectloss
2.python contrastive-unpaired-translation\train.py --save_epoch_freq 1 --datacfg remoteDataConfig1.yaml --dataroot ./datasets/grumpifycat --name BDDToNight_CUT_YOLO_6_10 --CUT_mode CUT

# get auxilliary-domain Images
3.python contrastive-unpaired-translation\test.py --datacfg remoteDataConfig1.yaml --dataroot ./datasets/maps --name BDDToNight_CUT_YOLO10_6_10 --no_dropout --preprocess notuse --epoch 15

# train on ad-dayolo
4.python fzb-yolov5-master\DomainTrain_Inst.py --cfg ./models/ms-dayolov5sInst.yaml --epochs 600 --data ./data/remoteData/bddDToN_threeDomain.yaml --batch-size 12
