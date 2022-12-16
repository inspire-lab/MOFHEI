<img src="docs/logo.png" width="200">

## Requirements
- TensorFlow 2+
- Tensorflow_model_optimization 0.7.3+
- Numpy 1.19+
- Pandas 1.2.4+
- Matplotlib 3.4.1+
- Sklearn 0.24.1+

## Overview

This Python 3+ library converts TensorFlow models into a HE-friendly pruned model. Only models with Conv2d & Dense layers are supported (as of Dec 12, 2022). It requires a trained plain-text model and plain-text training and validation dataset in addition to a dictionary of options ```opt```.

```Python
from mlsurgery import *

opt                                = {}
opt['he_friendly_stat']            = True     # Making a model HE-Friendly if True
opt['he_polynomial_Degree']        = 2        # Currently supports polynomial degrees 2, 3, & 4
opt['pruning_stat']                = True     # Prune a model if True
opt['packing_stat']                = True     # Packing-Aware prune a model if True
opt['num_slots']                   = 2**10    # Number of slots 
opt['nonzero_tiles_rate']          = 0.50     # Percentage of tiles with all-zero values
opt['minimum_acceptable_accuracy'] = 0.90     # Minimum acceptable accuracy
opt['lr_transfer']                 = 0.00001  # Transfer learning learnign rate 
opt['lr_finetune']                 = 0.000001 # Fine-tuning learning rate
opt['epochs']                      = 1        # Number of epochs
opt['pruning_epochs']              = 1        # Number of pruning epochs
opt['epochs_transfer']             = 1        # Number of transfer learning epochs
opt['epochs_finetune']             = 1        # Number of fine-tuning epochs
opt['batch_size']                  = 128      # Batch size
opt['initial_sparsity']            = 0.50     # only if opt['pruning_stat'] == True & opt['packing_stat'] == False
opt['final_sparsity']              = 0.85     # only if opt['pruning_stat'] == True & opt['packing_stat'] == False
opt['pruning_patience']            = 5        # only if opt['pruning_stat'] == True & opt['packing_stat'] == False

my_obj                             = MLSurgery(data_tr, data_vl, model, opt)
model_pruned, acc                  = my_obj.run()

```

```data_tr``` & ```data_vl``` are tuples of numpy inputs and outputs for training and validation data, respectively. 

## Example

Step 0: Clone the library with an access token ```git clone https://YOURTOKEN@github.com/mhrgroup/MLSurgery.git``` 

Step 1: Run ```mlsurgery_example.py``` (default is set for ```'mnist'```)

Step 2: Verify ```model_pruned.h5``` and ```tile_shape.npy```
