@echo off
title Custom Tensorflow-GPU Model Build!
echo This make take a few hours. Press CTRL-C to cancel processes.
pause
@ echo on

cd C:\
call activate tensorflow1
set PYTHONPATH=C:\tensorflow1\models;C:\tensorflow1\models\research;C:\tensorflow1\models\research\slim
cd C:\tensorflow1\models\research\object_detection
python train.py --logtostderr --train_dir=training/ --pipeline_config_path=training/faster_rcnn_inception_v2_pets.config

@echo off
pause