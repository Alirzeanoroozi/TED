config_name="config_1"
nohup python train.py --config configs/${config_name}.yaml > ${config_name}.txt 2>&1 &
