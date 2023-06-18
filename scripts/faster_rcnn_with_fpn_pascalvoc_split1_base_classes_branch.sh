python update_params_resnet101_fpn.py output/fsod/meta_training_pascalvoc_split1_resnet101_stage_1/model_final.pth

python3 faster_rcnn_train_net.py --num-gpus 1 --dist-url auto \
	--config-file configs/fsod/faster_rcnn_with_fpn_pascalvoc_split1_base_classes_branch.yaml 2>&1 | tee log/faster_rcnn_with_fpn_pascalvoc_split1_base_classes_branch.txt
