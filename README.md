## MOFHEI

__MOFHEI__ is a model optimization framework that optimizes pre-trained ML models for faster and more efficient non-interactive __private inference (PI)__ under __Homomorphic Encryption (HE)__. It effectively transforms an ML model into an HE-friendly version using our learning-based method, then applies our iterative block pruning method to prune the model with respect to the HE packing method. Therefore, while maintaining accuracy, it reduces the number of HE operations, thereby PI latency and memory usage. Our pruning technique works on 2D convolutional and fully connected layers, along with batch packing.


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
from mofhei import *

opt                                = {}

opt['he_friendly_stat' ]           = True # set True if we need to make model HE-friendly
opt['pruning_stat']                = True # set True if we want to prune the model
opt['culling_stat']                = True # set True if we want to cull the model

# only if opt['he_friendly_stat'] == True
opt['he_polynomial_Degree']        = 2  #Currently supports polynomial degrees 0, 2, 3, & 4 for making a model HE-friendly

# if  opt['he_polynomial_Degree'] == 0, then set transfer and finetune epochs to 1
opt['epochs_transfer']             = 25
opt['epochs_finetune']             = 25
opt['lr_transfer']                 = 0.00001
opt['lr_finetune']                 = 0.000001
opt['batch_size']                  = 128
opt['patience']                    = 3 


# only if opt['he_friendly_stat' ] == True 
opt['epochs_pruning']              = 25
opt['epochs_culling']              = 25
opt['minimum_acceptable_accuracy'] = 0.50
opt['target_sparsity']             = 0.75
opt['begin_step']                  = 0 
opt['frequency']                   = 100


my_obj                             = mofhei(data_tr, data_vl, model, opt)
model_culled, acc                  = my_obj.run()

```

```data_tr``` & ```data_vl``` are tuples of numpy inputs and outputs for training and validation data, respectively. 

## Example

Step 0: Clone the library with an access token ```git clone https://YOURTOKEN@github.com/mhrgroup/mofhei.git``` 

Step 1: Run ```mofhei_example.py``` (default is set for ```'mnist'```)

Step 2: Verify ```model_culled.h5```
