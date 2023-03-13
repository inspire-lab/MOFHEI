# usage: main_mlsurgery [-h] [-MA MODEL_AVAILABLE] [-HS HEFRIENDLY_STAT] [-PS PRUNING_STAT] [-AD {0,2,3,4}] [-EO EPOCHS_ORIGINAL] [-ET EPOCHS_TRANSFER] [-EF EPOCHS_FINETUNE] [-LR LR] [-LT LR_TRANSFER]
#                       [-LF LR_FINETUNE] [-BS BATCH_SIZE] [-PE PATIENCE] [-OE EPOCHS_PRUNING] [-TS TARGET_SPARSITY] [-TF TARGET_FREQUENCY]
#                       {0,1,2,3,4,5,6,7,8,9,10,11,12} {0,1}

# ** Version 202303011200 ** | Run MLSurgery Paper Experiment of Interest

# positional arguments:
#   {0,1,2,3,4,5,6,7,8,9,10,11,12}
#                         Select one of the following experiments (number) from MLSurgery paper: (0) CUSTOM Model by User, (1) ELECTRICAL-STABILITY-FCNet, (2) MNIST-LeNet, (3) CIFAR10-AlexNet, (4)
#                         X-RAY-AlexNet, (5) CIFAR10-VGG16, (6) X-RAY-VGG16, (7) MNIST-HEPEX-AE1, (8) MNIST-HEPEX-AE2, (9) MNIST-HEPEX-AE3, (10) CIFAR10-HEPEX-AE1, (11) CIFAR10-HEPEX-AE2, (12)
#                         CIFAR10-HEPEX-AE3 ||| If the custom model option (i.e., 0) is selected, the user requires to have it saved at ./experiment_custom/original/model.h5, and include a dictionary
#                         dataset at ./experiment_custom/original/data.npy with the following keys: 'datain_tr': A numpy array training inputs, 'dataou_tr': A numpy array training outputs (for
#                         classification: [0, 1, 2, ... ]), 'datain_vl': A numpy array validation inputs, 'dataou_vl': A numpy array validation outputs (for classification: [0, 1, 2, ... ]),
#                         'datain_te': A numpy array testing inputs, 'dataou_te': A numpy array testing outputs (for classification: [0, 1, 2, ... ])
#   {0,1}                 Select one of the following problems: (0) for classification and (1) for regression

# optional arguments:
#   -h, --help            show this help message and exit
#   -MA MODEL_AVAILABLE, --model_available MODEL_AVAILABLE
#                         True if a ./$EXPERIMENT$/original/model.h5 is available (in cloud or on device); otherwise we develop the model | should always be True for 'CUSTOM' models
#   -HS HEFRIENDLY_STAT, --hefriendly_stat HEFRIENDLY_STAT
#                         True to make the model HE-Friendly
#   -PS PRUNING_STAT, --pruning_stat PRUNING_STAT
#                         True to make the model pruned
#   -AD {0,2,3,4}, --polynomial_activation_degree {0,2,3,4}
#                         Select one of the following degrees: 0 (for square), 2, 3, and 4
#   -EO EPOCHS_ORIGINAL, --epochs_original EPOCHS_ORIGINAL
#                         Transfer learning maximum number of epochs
#   -ET EPOCHS_TRANSFER, --epochs_transfer EPOCHS_TRANSFER
#                         Transfer learning maximum number of epochs
#   -EF EPOCHS_FINETUNE, --epochs_finetune EPOCHS_FINETUNE
#                         Fine-tuning maximum number of epochs
#   -LR LR, --lr LR       Transfer learning's learning rate
#   -LT LR_TRANSFER, --lr_transfer LR_TRANSFER
#                         Transfer learning's learning rate
#   -LF LR_FINETUNE, --lr_finetune LR_FINETUNE
#                         Fine-tuning's learning rate
#   -BS BATCH_SIZE, --batch_size BATCH_SIZE
#                         Training batch size
#   -PE PATIENCE, --patience PATIENCE
#                         Training call back patience
#   -OE EPOCHS_PRUNING, --epochs_pruning EPOCHS_PRUNING
#                         Training epochs for pruning
#   -TS TARGET_SPARSITY, --target_sparsity TARGET_SPARSITY
#                         Prunable layers (i.e., conv and dense layers) target sparsity | does not necessary imply the final model sparsity
#   -TF TARGET_FREQUENCY, --target_frequency TARGET_FREQUENCY
#                         Epoch frequency of pruning


python3 main_mlsurgery.py 5 1 -MA False -AD 0 -EO 20 -ET 50 -EF 50 -PE 5 -OE 5 -TS 0.50
python3 main_mlsurgery.py 5 1 -MA True  -AD 0 -EO 20 -ET 50 -EF 50 -PE 5 -OE 5 -TS 0.55
python3 main_mlsurgery.py 5 1 -MA True  -AD 0 -EO 20 -ET 50 -EF 50 -PE 5 -OE 5 -TS 0.60
python3 main_mlsurgery.py 5 1 -MA True  -AD 0 -EO 20 -ET 50 -EF 50 -PE 5 -OE 5 -TS 0.65
python3 main_mlsurgery.py 5 1 -MA True  -AD 0 -EO 20 -ET 50 -EF 50 -PE 5 -OE 5 -TS 0.70
python3 main_mlsurgery.py 5 1 -MA True  -AD 0 -EO 20 -ET 50 -EF 50 -PE 5 -OE 5 -TS 0.75
python3 main_mlsurgery.py 5 1 -MA True  -AD 0 -EO 20 -ET 50 -EF 50 -PE 5 -OE 5 -TS 0.80
python3 main_mlsurgery.py 5 1 -MA True  -AD 0 -EO 20 -ET 50 -EF 50 -PE 5 -OE 5 -TS 0.85
python3 main_mlsurgery.py 5 1 -MA True  -AD 0 -EO 20 -ET 50 -EF 50 -PE 5 -OE 5 -TS 0.90
python3 main_mlsurgery.py 5 1 -MA True  -AD 0 -EO 20 -ET 50 -EF 50 -PE 5 -OE 5 -TS 0.95