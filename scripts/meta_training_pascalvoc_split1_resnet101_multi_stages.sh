python3 fsod_train_net.py --num-gpus 1 --dist-url auto \
	--config-file configs/fsod/meta_training_pascalvoc_split1_resnet101_stage_1.yaml 2>&1 | tee log/meta_training_pascalvoc_split1_resnet101_stage_1.txt

python3 fsod_train_net.py --num-gpus 1 --dist-url auto \
        --config-file configs/fsod/meta_training_pascalvoc_split1_resnet101_stage_2.yaml 2>&1 | tee log/meta_training_pascalvoc_split1_resnet101_stage_2.txt

python3 fsod_train_net.py --num-gpus 1 --dist-url auto \
        --config-file configs/fsod/meta_training_pascalvoc_split1_resnet101_stage_3.yaml 2>&1 | tee log/meta_training_pascalvoc_split1_resnet101_stage_3.txt
