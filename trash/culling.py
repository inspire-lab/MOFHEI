from mlsurgery import *

data_name = 'cifar10'#'electric_grid_stability'#'cifar10'#'mnist' #'electric_grid_stability'

#custom_objects = {'DynamicPolyReLU_D2':DynamicPolyReLU_D2}
#custom_objects = {'Square':Square}
custom_objects = {'DynamicPolyReLU_D2':DynamicPolyReLU_D2, 'Square':Square}

model_pruned     = tf.keras.models.load_model("./model_pruned_{}.h5".format(data_name), custom_objects = custom_objects)

model_pruned.summary()

model_pruned_clone = tf.keras.models.clone_model(model_pruned)
model_pruned_clone.set_weights(model_pruned.get_weights())

model_pruned_clone.compile(optimizer = tf.keras.optimizers.Adam(0.0001), 
                           loss      = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), 
                           metrics   = [tf.keras.metrics.SparseCategoricalAccuracy()] )

info = {}
for layer in model_pruned_clone.layers:
    #print(layer.name)
    info_keys         = list(info.keys())
    if 'conv2d' in layer.name:

        weight             = layer.weights[0].numpy()
        bias               = layer.weights[1].numpy()
        
        ind_good4          = [i0 for i0 in range(weight.shape[-1]) if weight[:,:,:,i0].sum() != 0]
        bias_good          = bias[ind_good4]

        if len(info_keys)   == 0:
            weight_good     = weight[:,:,:,ind_good4]
            #ind_good_row    =  np.ndarray.flatten(np.array(range(np.prod(weight.shape))).reshape(weight.shape)[:,:,:,ind_good4])
        else:
            ind_good3          = deepcopy(info[info_keys[-1]]['ind_good_channel'])

            # print(weight.shape)
            # print(len(ind_good3))
            # print(len(ind_good4))
            
            weight_good        = weight[:,:,ind_good3, :][:,:,:,ind_good4]
            #ind_good_row    = np.ndarray.flatten(np.array(range(np.prod(weight.shape))).reshape(weight.shape)[:,:,ind_good3,ind_good4])

        ind_good_row = np.ndarray.flatten(np.array(range(np.prod(layer.output_shape[1:]))).reshape(layer.output_shape[1:][::-1])[ind_good4,:,:])
        #np.ndarray.flatten(np.array(range(np.prod(layer.output_shape[1:]))).reshape(layer.output_shape[1:])[:,:,ind_good4])
        
        weight_initializer = tf.initializers.constant(weight_good)
        bias_initializer   = tf.initializers.constant(bias_good)

        # gather information
        info[layer.name]                          = {}
        info[layer.name]['ind_good_channel']      = ind_good4
        info[layer.name]['ind_good_row']          = ind_good_row
        #info[layer.name]['ind_fm_good']           = ind_fm_good 
        info[layer.name]['weight_initializer']    = weight_initializer
        info[layer.name]['bias_initializer']      = bias_initializer
        info[layer.name]['filters']               = weight_good.shape[-1]
        info[layer.name]['kernel_size']           = layer.kernel_size
        info[layer.name]['strides']               = layer.strides
        info[layer.name]['padding']               = layer.padding


    elif 'batch_normalization' in layer.name:

        if len(info_keys)   == 0:
            gamma             = layer.weights[0].numpy()
            beta              = layer.weights[1].numpy()
            moving_mean       = layer.weights[2].numpy()
            moving_variance   = layer.weights[3].numpy()
            
            ind_good_channel  = list(range(gamma.shape[0]))
            ind_good_row      = np.ndarray.flatten(np.array(range(np.prod(layer.output_shape[1:]))).reshape(layer.output_shape[1:][::-1])) #np.ndarray.flatten(np.array(range(np.prod(layer.output_shape[1:]))).reshape(layer.output_shape[1:]))
            

        else:
            ind_good_channel  = info[info_keys[-1]]['ind_good_channel']
            ind_good_row      = np.ndarray.flatten(np.array(range(np.prod(layer.output_shape[1:]))).reshape(layer.output_shape[1:][::-1])[ind_good_channel,:,:])#np.ndarray.flatten(np.array(range(np.prod(layer.output_shape[1:]))).reshape(layer.output_shape[1:])[:,:,ind_good_channel]) #info[info_keys[-1]]['ind_good_row']

            gamma             = layer.weights[0].numpy()[ind_good_channel]
            beta              = layer.weights[1].numpy()[ind_good_channel]
            moving_mean       = layer.weights[2].numpy()[ind_good_channel]
            moving_variance   = layer.weights[3].numpy()[ind_good_channel]

        gamma_initializer             = tf.initializers.constant(gamma)
        beta_initializer              = tf.initializers.constant(beta)
        moving_mean_initializer       = tf.initializers.constant(moving_mean)
        moving_variance_initializer   = tf.initializers.constant(moving_variance)

        # gather information
        info[layer.name]                                   = {}
        info[layer.name]['ind_good_channel']               = ind_good_channel
        info[layer.name]['ind_good_row']                   = ind_good_row
        info[layer.name]['gamma_initializer_initializer']  = gamma_initializer
        info[layer.name]['beta_initializer']               = beta_initializer   
        info[layer.name]['moving_mean_initializer']        = moving_mean_initializer 
        info[layer.name]['moving_variance_initializer']    = beta_initializer 

    elif 'pooling2d' in layer.name:
        if (len(info_keys)   == 0): #or ('conv2d' in info_keys[-1]) or ('batch_normalization' in info_keys[-1]):
            ind_good_channel  = list(range(layer.output_shape[-1]))
            ind_good_row      = np.ndarray.flatten(np.array(range(np.prod(layer.output_shape[1:]))).reshape(layer.output_shape[1:][::-1]))#np.ndarray.flatten(np.array(range(np.prod(layer.output_shape[1:]))).reshape(layer.output_shape[1:]))
        else:
            ind_good_channel  = info[info_keys[-1]]['ind_good_channel']
            ind_good_row      = np.ndarray.flatten(np.array(range(np.prod(layer.output_shape[1:]))).reshape(layer.output_shape[1:][::-1])[ind_good_channel,:,:])#np.ndarray.flatten(np.array(range(np.prod(layer.output_shape[1:]))).reshape(layer.output_shape[1:])[:,:,ind_good_channel])
        
        # gather information 
        info[layer.name]                                   = {}
        info[layer.name]['ind_good_channel']               = ind_good_channel
        info[layer.name]['ind_good_row']                   = ind_good_row

    elif 'dense' in layer.name:
        #info_keys         = list(info.keys())

        weight             = layer.weights[0].numpy()
        bias               = layer.weights[1].numpy()
        
        ind_good_column    = [i0 for i0 in range(weight.shape[-1]) if weight[:,i0].sum() != 0]
        bias_good          = bias[ind_good_column]  

        #print(len(ind_good_row))

        if (len(info_keys)   == 0): #or ('conv2d' in info_keys[-1]) or ('batch_normalization' in info_keys[-1]):
            weight_good         = weight[:,ind_good_column]
            #print(len(ind_good_row))
            #ind_good_row        = deepcopy(ind_good_column)#np.ndarray.flatten(np.array(range(np.prod(weight.shape))).reshape(weight.shape)[:,ind_good_row])
            #print(len(ind_good_row))
        else:
            ind_good_row        = deepcopy(info[info_keys[-1]]['ind_good_row'])


            # print(weight.shape)
            # print(len(ind_good_row))
            # print(len(ind_good_row))
            # print(info[info_keys[-1]]['ind_good_channel'])
            #print(np.array(ind_good_row).reshape((12,12,17)))
            weight_good         = weight[ind_good_row, :][:,ind_good_column]
            # print(weight_good.shape)
            
        ind_good_row        = deepcopy(ind_good_column) #np.ndarray.flatten(np.array(range(np.prod(weight.shape))).reshape(weight.shape)[ind_good_row, :][:,ind_good_column])
          
        weight_initializer = tf.initializers.constant(weight_good)
        bias_initializer   = tf.initializers.constant(bias_good)

        # gather information
        info[layer.name]                          = {}
        # info[layer.name]['ind_good']              = ind_good_row
        info[layer.name]['ind_good_row']          = ind_good_row
        info[layer.name]['weight_initializer']    = weight_initializer
        info[layer.name]['bias_initializer']      = bias_initializer
        info[layer.name]['units']                 = weight_good.shape[-1]


#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$


input_shape = model_pruned_clone.layers[0].get_input_shape_at(0)[1:]
inputs      = tf.keras.Input(input_shape)

print('===========')
for i0, layer in enumerate(model_pruned_clone.layers):
    #print(layer.name)
    
    if i0 == 0:
        x = inputs

    if 'conv2d' in layer.name:

        x = tf.keras.layers.Conv2D(filters            = info[layer.name]['filters'],
                                   kernel_size        = info[layer.name]['kernel_size'], 
                                   strides            = info[layer.name]['strides'], 
                                   padding            = info[layer.name]['padding'], 
                                   kernel_initializer = info[layer.name]['weight_initializer'],
                                   bias_initializer   = info[layer.name]['bias_initializer']) (x)
        #print(x.shape)

    elif 'batch_normalization' in layer.name:
        x = tf.keras.layers.BatchNormalization(gamma_initializer           = info[layer.name]['gamma_initializer_initializer'],
                                               beta_initializer            = info[layer.name]['beta_initializer'],
                                               moving_mean_initializer     = info[layer.name]['moving_mean_initializer'],
                                               moving_variance_initializer = info[layer.name]['moving_variance_initializer']) (x)

    elif 'dense' in layer.name:
        # print(x.shape)
        # print(info[layer.name]['units'], x.shape,info[layer.name]['weight_initializer'].value.shape)
        x = tf.keras.layers.Dense(units              = info[layer.name]['units'],
                                  kernel_initializer = info[layer.name]['weight_initializer'],
                                  bias_initializer   = info[layer.name]['bias_initializer']) (x)
        
        #print(x.shape)

    else:
        if i0 != 0:
            x = layer (x)

    print(layer.name, x.shape)



outputs      = x #model_pruned_clone.layers[-1]

#================


class CustomModel(tf.keras.Model):
    def train_step(self, data):
        x, y = data

        print(cond_transfer_learning)

        with tf.GradientTape() as tape:
            y_pred = self(x, training=True)  # Forward pass
            loss = self.compiled_loss(y, y_pred, regularization_losses=self.losses)
        
        # Compute gradients
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)

        #print(type(gradients[0]))

        gradient_mask = []
        for weight in trainable_vars:
            if ('bias' in weight.name) or ('batch_normalization' in weight.name):
                gradient_mask.append(tf.ones((weight.shape), dtype = tf.float32))
            else:
                gradient_mask.append(tf.zeros((weight.shape), dtype = tf.float32))


        if cond_transfer_learning:
            gradients = [g * m for g,m in zip(gradients, gradient_mask)]

        # Update weights
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        # Update metrics (includes the metric that tracks the loss)
        self.compiled_metrics.update_state(y, y_pred)
        # Return a dict mapping metric names to current value
        return {m.name: m.result() for m in self.metrics}
    


# global gradient_mask

# gradient_mask = [] 
# for weight in model_pruned_clone.trainable_variables:
#     if 'bias' in weight.name:
#         gradient_mask.append(np.zeros((weight.shape)) == np.ones((weight.shape)))
#     else:
#         gradient_mask.append(np.zeros((weight.shape)) == np.zeros((weight.shape)))


model_culled = CustomModel(inputs, outputs)
# model_culled.compile(optimizer = model_pruned_clone.optimizer,
#                      loss      = model_pruned_clone.loss,
#                      metrics   = model_pruned_clone.metrics[-1:])


model_culled.compile(optimizer = tf.keras.optimizers.Adam(0.01),
                     loss      = model_pruned_clone.loss,
                     metrics   = [tf.keras.metrics.SparseCategoricalAccuracy()])

model_culled.summary()

# asdasd =asdasdasd
# model_culled.compile(optimizer = tf.keras.optimizers.Adam(0.0005), 
#                            loss      = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), 
#                            metrics   = [tf.keras.metrics.SparseCategoricalAccuracy()] )

#=============

datain_tr, dataou_tr, datain_vl, dataou_vl = fun_data(name = data_name, calibrate = True)
data_tr, data_vl                           = (datain_tr, dataou_tr), (datain_vl, dataou_vl)

model_culled

_, acc = model_culled.evaluate(data_vl[0], data_vl[1], verbose = 0, batch_size = 32)

print(acc)

_, acc = model_pruned_clone.evaluate(data_vl[0], data_vl[1], verbose = 0, batch_size = 32)

print(acc)





# for layer in model_culled.layers:
#     if ('conv2d' in layer.name):
#         weight = layer.weights[0].numpy()

#         for i0 in range(weight.shape[-1]):
#             print(layer.name, i0, weight[:,:,:,i0].sum())
    
#     elif ('dense' in layer.name):
#         weight = layer.weights[0].numpy()

#         for i0 in range(weight.shape[-1]):
#             print(layer.name, i0, weight[:,i0].sum())

#     else:
#         pass
        
callback = tf.keras.callbacks.EarlyStopping(
    monitor='val_sparse_categorical_accuracy',
    patience=3,
    restore_best_weights=True)


global cond_transfer_learning

cond_transfer_learning = True

model_culled.fit(
    data_tr[0],
    data_tr[1],
    epochs=100,
    verbose = 1,
    validation_data=data_vl,
    validation_freq=1,
    callbacks=[callback],   
    )

cond_transfer_learning = False

model_culled.compile(optimizer = tf.keras.optimizers.Adam(0.001),
                     loss      = model_pruned_clone.loss,
                     metrics   = [tf.keras.metrics.SparseCategoricalAccuracy()])

model_culled.fit(
    data_tr[0],
    data_tr[1],
    epochs=100,
    verbose = 1,
    validation_data=data_vl,
    validation_freq=1,
    callbacks=[callback],   
    )

_, acc = model_culled.evaluate(data_vl[0], data_vl[1], verbose = 0, batch_size = 32)

print(acc)

fun_save_model(model_culled, "./model_culled_{}.h5".format(data_name))