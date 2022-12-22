from mlsurgery import *

opt = {}
opt['he_friendly_stat']            = True
opt['he_polynomial_Degree']        = 2 # Currently supports polynomial degrees 2, 3, & 4 for making a model HE-friendly
opt['pruning_stat']                = True
opt['packing_stat']                = True
opt['num_slots']                   = 2**8
opt['nonzero_tiles_rate']          = 0.50 
opt['minimum_acceptable_accuracy'] = 0.90
opt['lr_transfer']                 = 0.00001
opt['lr_finetune']                 = 0.000001
opt['epochs']                      = 1
opt['pruning_epochs']              = 25
opt['epochs_transfer']             = 2
opt['epochs_finetune']             = 2
opt['batch_size']                  = 128
opt['initial_sparsity']            = 0.50 # only if opt['pruning_stat'] == True and opt['packing_stat'] == False
opt['final_sparsity']              = 0.85 # only if opt['pruning_stat'] == True and opt['packing_stat'] == False
opt['pruning_patience']            = 5    # only if opt['pruning_stat'] == True and opt['packing_stat'] == False

# wget !wget https://raw.githubusercontent.com/mhrafiei/data/main/electrical_grid_stability_simulated_data.csv

data_name       = 'mnist' # cifar10 & electric_grid_stability -> (for electric_grid_stability make sure the csv file is in the directory using the above wget command)
model_available = False

if not model_available:
    print("TRAIN AN ORIGINAL MODEL FOR $$ {} $$ DATASET".format(data_name))
    # the model is saved as model.h5 in the directory
    fun_model_example(name = data_name)

datain_tr, dataou_tr, datain_vl, dataou_vl = fun_data(name = data_name, calibrate = True)
data_tr, data_vl                           = (datain_tr, dataou_tr), (datain_vl, dataou_vl)

model_original     = fun_load_model("model.h5")

my_obj             = MLSurgery(data_tr, data_vl, model_original, opt)

model_pruned, acc  = my_obj.run()

fun_save_model(model_pruned, 'model_pruned.h5')

