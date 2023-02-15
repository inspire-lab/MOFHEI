from mlsurgery import *

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

# wget https://raw.githubusercontent.com/mhrafiei/data/main/electrical_grid_stability_simulated_data.csv
# wget https://storage.googleapis.com/mplus/repo/ray/xray128.npy 

data_name       = 'cifar10' #'xray16', 'xray32', 'xray64', 'xray128', 'xray256', 'xray512', #'electric_grid_stability' # cifar10 & electric_grid_stability -> (for electric_grid_stability make sure the csv file is in the directory using the above wget command)
model_available = True

if not model_available:
    print("TRAIN AN ORIGINAL MODEL FOR $$ {} $$ DATASET".format(data_name))
    # the model is saved as model.h5 in the directory
    fun_model_example(name = data_name)

datain_tr, dataou_tr, datain_vl, dataou_vl = fun_data(name = data_name, calibrate = True)
data_tr, data_vl                           = (datain_tr, dataou_tr), (datain_vl, dataou_vl)

model_original     = fun_load_model("model.h5")

my_obj             = MLSurgery(data_tr, data_vl, model_original, opt)

model_culled, acc  = my_obj.run()

fun_save_model(model_culled, "./model_culled_{}.h5".format(data_name))

