from mlsurgery import *
import argparse

parser = argparse.ArgumentParser(description = 'model_observation')
parser.add_argument("problem",     help = "either 'classification' or 'regression'")
parser.add_argument("data_name",   help = "it supportss 'mnist', 'cifar10', 'electric_grid_stability'")


args = parser.parse_args()

data_name = args.data_name
problem   = args.problem

print(data_name)
print(problem)

custom_objects = {'DynamicPolyReLU_D2':DynamicPolyReLU_D2, 'DynamicPolyReLU_D3':DynamicPolyReLU_D3, 'DynamicPolyReLU_D4':DynamicPolyReLU_D4, 'Square':Square, 'CustomModel': CustomModel}

model_original   = tf.keras.models.load_model("./model_original_{}.h5".format(data_name), custom_objects = custom_objects)
model_culled     = tf.keras.models.load_model("./model_culled_{}.h5".format(data_name), custom_objects = custom_objects)

MLSurgery.fun_clear()

model_original.summary()

model_culled.summary()

# model_culled_clone = tf.keras.models.clone_model(model_culled)
# model_culled_clone.set_weights(model_culled.get_weights())

if problem == 'classification':

    loss    = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    metrics = [tf.keras.metrics.SparseCategoricalAccuracy()]

else:

    loss    = 'mse'
    metrics = [tf.keras.metrics.MeanSquaredError(name='mean_squared_error', dtype=None)]

model_culled.compile(optimizer = tf.keras.optimizers.Adam(0.001), 
                     loss      = loss, 
                     metrics   = metrics )

datain_tr, dataou_tr, datain_vl, dataou_vl = fun_data(name = data_name, calibrate = True)
data_tr, data_vl                           = (datain_tr, dataou_tr), (datain_vl, dataou_vl)

_, msm = model_culled.evaluate(data_vl[0], data_vl[1], verbose = 0, batch_size = 32)

print("======================")
print("======================")

print("Measurement (e.g., MSE or ACC) of the culled {} model is {}".format(data_name, msm))

print("======================")
print("======================")