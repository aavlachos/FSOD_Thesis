python3 fsod_train_net.py --num-gpus 1 --dist-url auto \
        --config-file configs/fsod/10shot_finetune_pascalvoc_split1_resnet101.yaml --resume SOLVER.IMS_PER_BATCH 2 2>&1 | tee log/10shot_finetune_pascalvoc_split1_resnet101.txt
python3 fsod_train_net.py --num-gpus 1 --dist-url auto \
        --config-file configs/fsod/5shot_finetune_pascalvoc_split1_resnet101.yaml --resume SOLVER.IMS_PER_BATCH 2 2>&1 | tee log/5shot_finetune_pascalvoc_split1_resnet101.txt
python3 fsod_train_net.py --num-gpus 1 --dist-url auto \
        --config-file configs/fsod/3shot_finetune_pascalvoc_split1_resnet101.yaml --resume SOLVER.IMS_PER_BATCH 2 2>&1 | tee log/3shot_finetune_pascalvoc_split1_resnet101.txt
python3 fsod_train_net.py --num-gpus 1 --dist-url auto \
        --config-file configs/fsod/2shot_finetune_pascalvoc_split1_resnet101.yaml --resume SOLVER.IMS_PER_BATCH 2 2>&1 | tee log/2shot_finetune_pascalvoc_split1_resnet101.txt
python3 fsod_train_net.py --num-gpus 1 --dist-url auto \
        --config-file configs/fsod/1shot_finetune_pascalvoc_split1_resnet101.yaml --resume SOLVER.IMS_PER_BATCH 2 2>&1 | tee log/1shot_finetune_pascalvoc_split1_resnet101.txt
