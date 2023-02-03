import numpy                         as np
import pandas                        as pd
import tensorflow                    as tf
import tensorflow_model_optimization as tfmot

import warnings
import shutil
import os

from copy       import deepcopy
from matplotlib import pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing   import MinMaxScaler

warnings.filterwarnings("ignore")

@tf.function
def reshape(input, shape): # just for inputs 
    '''
    x = reshape(input, shape)
    '''
    return tf.keras.layers.Reshape(shape)(input)

class Square(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(Square, self).__init__()

    def call(self, inputs):
        return tf.square(inputs)

class DynamicPolyReLU_D2(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(DynamicPolyReLU_D2, self).__init__()

    def build(self, input_shape):
        initializer = tf.constant_initializer(value=0.00001)
        self.kernel = self.add_weight("kernel", shape=[3], trainable = True, initializer=initializer)

    def call(self, inputs):
        return self.kernel[2] * tf.square(inputs) + self.kernel[1] * inputs + self.kernel[0]

class DynamicPolyReLU_D3(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(DynamicPolyReLU_D3, self).__init__()

    def build(self, input_shape):
        initializer = tf.constant_initializer(value=0.00001)
        self.kernel = self.add_weight("kernel", shape=[4], trainable = True, initializer=initializer)

    def call(self, inputs):
        return self.kernel[3]*tf.math.pow(inputs, 3) +  self.kernel[2] * tf.square(inputs) + self.kernel[1] * inputs + self.kernel[0]

class DynamicPolyReLU_D4(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(DynamicPolyReLU_D4, self).__init__()

    def build(self, input_shape):
        initializer = tf.constant_initializer(value=0.00001)
        self.kernel = self.add_weight("kernel", shape=[5], trainable = True, initializer=initializer)

    def call(self, inputs):
        return self.kernel[4]*tf.math.pow(inputs, 4) + self.kernel[3]*tf.math.pow(inputs, 3) +  self.kernel[2] * tf.square(inputs) + self.kernel[1] * inputs + self.kernel[0]

class PDynamicPolyReLU_D2(DynamicPolyReLU_D2, tfmot.sparsity.keras.PrunableLayer):
    def get_prunable_weights(self):
        return [self.kernel]

class PDynamicPolyReLU_D3(DynamicPolyReLU_D3, tfmot.sparsity.keras.PrunableLayer):
    def get_prunable_weights(self):
        return [self.kernel]

class PDynamicPolyReLU_D4(DynamicPolyReLU_D3, tfmot.sparsity.keras.PrunableLayer):
    def get_prunable_weights(self):
        return [self.kernel]

class MyThresholdCallback(tf.keras.callbacks.Callback):
    def __init__(self, threshold):
        super(MyThresholdCallback, self).__init__()
        self.threshold = threshold

    def on_epoch_end(self, epoch, logs=None): 
        #print(logs.keys())
        val_acc = logs["val_sparse_categorical_accuracy"]
        if val_acc >= self.threshold:
            self.model.stop_training = True

class CustomModel(tf.keras.Model):



    '''
    Custom training with gradient tape
    '''



    def train_step(self, data):
        x, y = data

        with tf.GradientTape() as tape:
            y_pred = self(x, training=True)  # Forward pass
            loss   = self.compiled_loss(y, y_pred, regularization_losses=self.losses)

        trainable_vars = self.trainable_variables
        gradients      = tape.gradient(loss, trainable_vars)

        gradients      = [g * m for g, m in zip(gradients, gradient_mask)]

        # Update weights
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        # Update metrics (includes the metric that tracks the loss)

        self.compiled_metrics.update_state(y, y_pred)
        # Return a dict mapping metric names to current value

        return {m.name: m.result() for m in self.metrics}

class Features_4D_To_2D(tf.keras.layers.Layer):
    '''

    x = Features_4D_To_2D(kernel_size, strides) (x)

    To convert 4D feature map (including the batch_size) into a 2D map to be later
    employed in a block-based prunable dense operaion
    '''
    def __init__(self, kernel_size, strides, **kwargs):
        super(Features_4D_To_2D, self).__init__()
        self.kernel_size = kernel_size
        self.strides     = strides

    def call(self, inputs):
        batch_size, width, hight, depth = inputs.shape

        # num_strides_horizontal  = int(tf.math.ceil((width-self.kernel_size[0])/self.strides[0]) + 1)
        # num_strides_vertical    = int(tf.math.ceil((hight-self.kernel_size[1])/self.strides[1]) + 1)

        num_strides_horizontal  = int(-((width-self.kernel_size[0])//-self.strides[0]) + 1)
        num_strides_vertical    = int(-((hight-self.kernel_size[1])//-self.strides[1]) + 1)


        output_rank             = int(num_strides_vertical * num_strides_horizontal)

        ind_horizontal_start    = tf.expand_dims(tf.range(0,int(num_strides_horizontal * self.strides[0]), self.strides[0]), axis = 0)
        ind_horizontal_end      = ind_horizontal_start + self.kernel_size[0]
        ind_horizontal_end      = tf.concat([ind_horizontal_end[:,:-1], tf.expand_dims(tf.expand_dims(tf.constant(width), axis = 0), axis = 0)], axis = 1)  

        ind_vertical_start      = tf.expand_dims(tf.range(0,int(num_strides_vertical * self.strides[1]), self.strides[1]), axis = 0)
        ind_vertical_end        = ind_vertical_start + self.kernel_size[1]
        ind_vertical_end        = tf.concat([ind_vertical_end[:,:-1], tf.expand_dims(tf.expand_dims(tf.constant(hight), axis = 0), axis = 0)], axis = 1) 

        ind_horizontal_start    = tf.squeeze(ind_horizontal_start)
        ind_horizontal_end      = tf.squeeze(ind_horizontal_end)
        ind_vertical_start      = tf.squeeze(ind_vertical_start)
        ind_vertical_end        = tf.squeeze(ind_vertical_end)

        ind            = tf.repeat(tf.expand_dims(tf.cast(tf.linspace(0,self.kernel_size[0]- 1,self.kernel_size[0]), dtype = tf.int32), axis = 0), int(num_strides_horizontal), axis = 0)
        val_sum        = tf.transpose(tf.repeat(tf.expand_dims(ind_horizontal_start, axis = 0), self.kernel_size[0], axis = 0))
        ind_horizontal = tf.reshape(ind + val_sum, (int(val_sum.shape[0] * val_sum.shape[1]),))

        ind            = tf.repeat(tf.expand_dims(tf.cast(tf.linspace(0,self.kernel_size[1]- 1,self.kernel_size[1]), dtype = tf.int32), axis = 0), int(num_strides_vertical), axis = 0)
        val_sum        = tf.transpose(tf.repeat(tf.expand_dims(ind_vertical_start, axis = 0), self.kernel_size[1], axis = 0))
        ind_vertical   = tf.reshape(ind + val_sum, (int(val_sum.shape[0] * val_sum.shape[1]),))

        input_reshaped = tf.gather(tf.gather(inputs, ind_horizontal, axis = 1), ind_vertical, axis = 2)

        input_reshaped = reshape(tf.transpose(reshape(reshape(input_reshaped, (num_strides_horizontal,self.kernel_size[0],ind_vertical.shape[0],inputs.shape[-1])), (num_strides_horizontal,self.kernel_size[0],num_strides_vertical,self.kernel_size[1],inputs.shape[-1])), perm=[0,1,3,2,4,5]),(output_rank,self.kernel_size[0],self.kernel_size[1],inputs.shape[-1]))
        input_reshaped = reshape(input_reshaped , (output_rank , int(depth * self.kernel_size[0] * self.kernel_size[1])))



        return input_reshaped

class Features_2D_To_4D(tf.keras.layers.Layer):
    
    '''
    x = Features_2D_To_4D(width,
                          height,
                          kernel_size,
                          strides) (x)

    To convert 2D feature map (no batch_size) into a 4D map (including batch_size)
    '''

    def __init__(self,                   
                 width,
                 height,
                 kernel_size,
                 strides, 
                 **kwargs):
      
        super(Features_2D_To_4D, self).__init__()
        self.width       = width
        self.height      = height
        self.strides     = strides
        self.kernel_size = kernel_size

    def call(self, inputs):
        num_strides_horizontal  = int(-((self.width-self.kernel_size[0])//-self.strides[0]) + 1)
        num_strides_vertical    = int(-((self.height-self.kernel_size[1])//-self.strides[1]) + 1)

        # print(self.width, self.kernel_size[0], self.strides[0])
        # print(inputs.shape, (num_strides_horizontal, num_strides_vertical, inputs.shape[-1]))
        return reshape(inputs, (num_strides_horizontal, num_strides_vertical, inputs.shape[-1]))

class MLSurgery():
    
    ''' 
    Update 20221207
    Requires the following libraries:
        -- Tensorflow 2.0+
        -- Tensorflow_model_optimization
        -- Numpy
        -- Pandas
        -- Matpplotlib
        --
    Currently supports classification models with Dense and/or Conv2D layers with no embeded models/layers
    Currently supports polynomial degrees 0, 2, 3, & 4 for making a model HE-friendly
    '''


    '''
    opt                                = {}

    opt['he_friendly_stat' ]           = True # set True if we need to make model HE-friendly
    opt['pruning_stat']                = True # set True if we want to prune and cul the model

    # only if opt['he_friendly_stat'] == True
    opt['he_polynomial_Degree']        = 0  #Currently supports polynomial degrees 0, 2, 3, & 4 for making a model HE-friendly

    # if  opt['he_polynomial_Degree'] == 0, then set transfer and finetune epochs to 1
    opt['epochs_transfer']             = 5
    opt['epochs_finetune']             = 5
    opt['lr_transfer']                 = 0.00001
    opt['lr_finetune']                 = 0.000001
    opt['batch_size']                  = 128


    # only if opt['he_friendly_stat' ] == True 
    opt['epochs_pruning']              = 5
    opt['minimum_acceptable_accuracy'] = 0.90
    opt['target_sparsity']             = 0.5 
    opt['begin_step']                  = 0 
    opt['frequency']                   = 100
    '''

    # Class Object Attributes



    def __init__(self, data_tr, data_te, model_original, opt):

        self.data_tr                  = data_tr
        self.data_te                  = data_te
        self.model_original           = model_original
        self.opt                      = opt
        self.lr                       = model_original.optimizer.lr.numpy()
        self.optimizer                = model_original.optimizer
        self.loss                     = model_original.loss
        self.metrics                  = [tf.keras.metrics.SparseCategoricalAccuracy()] #model_original.metrics
        self.path                     = os.getcwd()
        self.path_temp                = os.path.join(self.path, 'temp')

        _, self.model_original_acc_te = model_original.evaluate(data_te[0], data_te[1], verbose = 0, batch_size = 32)
        callback_threshold            = MyThresholdCallback(threshold = self.model_original_acc_te)
        self.callbacks                = [callback_threshold]


    def fun_clear():

        '''
        Clear the terminal
        '''

        # for windows
        if os.name == "nt":
            _ = os.system("cls")

        # for mac and linux(here, os.name is 'posix')
        else:
            _ = os.system("clear")

    def fun_model_clone(self, model):

        '''
        Create a clone version of the model
        '''

        model_clone = tf.keras.models.clone_model(model)
        model_clone.set_weights(model.get_weights())

        # metrics = model.metrics[-1:]
        # if type(metrics) != list:
        #     metrics = [metrics]

        model_clone.compile(self.optimizer, loss = self.loss, metrics = self.metrics)
        
        return model_clone

    def fun_model_info(model):

        '''
        Create a dataframe version of the model's layers
        '''
        
        df                 = {}
        df['layer']        = []
        df['name']         = []
        df['type']         = []
        df['input_name']   = []
        df['input_shape']  = []
        df['output_shape'] = []
        df['trainable']    = []
        df['num_param']    = []
        df['relu']         = []
        df['maxpool']      = []

        c = 0
        for layer in model.layers:
            df['layer'].append(layer)
            df['name'].append(layer.name)
            df['type'].append(str(layer).split()[0].split('.')[-1])

            if c == 0:
                df['input_name'].append([])
                df['input_shape'].append(())
            else:
                if type(layer.input) == list:
                    df['input_name'].append([str(input).split('created by layer')[-1][2:-3] for input in layer.input])
                    df['input_shape'].append([tuple(input.shape[1:]) for input in layer.input])
                else:
                    df['input_name'].append([str(layer.input).split('created by layer')[-1][2:-3]])
                    df['input_shape'].append([tuple(layer.input.shape[1:])])

            df['output_shape'].append(tuple(layer.output.shape[1:]))
            df['trainable'].append(layer.trainable)
            df['num_param'].append(int(sum([tf.reduce_prod(weight.shape) for weight in layer.weights if weight.trainable])))

            if ('relu' in df['type'][-1].lower()) or ('activation' in df['type'][-1].lower()):
                df['relu'].append(True)
            else:
                df['relu'].append(False)

            if ('max' in df['type'][-1].lower()):
                df['maxpool'].append(True)
            else:
                df['maxpool'].append(False)


            c += 1

        df = pd.DataFrame(df)

        for i0 in range(1,len(df)):
            df['input_name'].values[i0] = [df['name'].values[i0-1]]

        return df

    def fun_model_stack(df):



        '''
        Stack a data frame of model layers into a list of tesnsor x
        '''



        x = [ tf.keras.Input(shape = df['output_shape'].values[0]) ]

        for i0 in range(1, len(df)):
            l = len(df['input_name'].values[i0])
            if l > 1:
                ind = [ list(df['name']).index(df['input_name'].values[i0][i1]) for i1 in range(l) ]
                x.append( df['layer'].values[i0] ( [x[i1] for i1 in ind] ) )
            else:
                ind = list(df['name']).index(df['input_name'].values[i0][0])
                x.append( df['layer'].values[i0] (x[ind]) )

        return x
    
    def fun_model_create(self, x, trainability = [], lr = None):

        '''
        Create a model from a list of tensors 
        '''

        model_case = tf.keras.Model(inputs = x[0], outputs = x[-1])
        
        try:
            for layer, cond in zip(model_case.layers, trainability):
                layer.trainable = cond
        except:
            pass

        if lr != None:
            optimizer = deepcopy(self.optimizer)
            optimizer.lr = lr
        else:
            optimizer = deepcopy(self.optimizer)

        model_case.compile(optimizer = optimizer, loss = self.loss, metrics = self.metrics)

        return model_case 
        
    def fun_max2ave(self, model):




        '''
        Trun every max pooling layer into average pooling layer
        '''




        # we first clone the original model to make sure we dont mess with the model_original
        model_clone     = MLSurgery.fun_model_clone(self, model)
        df_clone        = MLSurgery.fun_model_info(model_clone)

        # find indeices of all max pooling layers
        ind_pool        = np.where(df_clone['maxpool'].values)[0]

        # we start from the latest max pooling layer and move towards the first one
        ind_pool        = ind_pool[::-1]

        # go through each max pooling layer and convert it to average pooling layer
        for ind in ind_pool:
            
            df_clone = MLSurgery.fun_model_info(model_clone)

            # we first freez all layers
            for i0 in range(len(df_clone)):
                df_clone['layer'][i0].trainable = False
                df_clone['trainable'][i0]       = False

            pool_size = df_clone['layer'][ind].pool_size
            strides   = df_clone['layer'][ind].strides
            padding   = df_clone['layer'][ind].padding
        
            df_clone['layer'][ind]     = tf.keras.layers.AveragePooling2D(pool_size = pool_size, strides = strides, padding = padding)
            df_clone['type'][ind]      = 'AveragePooling2D'
            df_clone['trainable'][ind] = True
            df_clone['maxpool'][ind]   = False

            # here we only unfreez layers after ind
            for i0 in range(ind,len(df_clone)):
                df_clone['layer'][i0].trainable = True
                df_clone['trainable'][i0]       = True

            c           = 1
            lr_transfer = deepcopy(self.opt['lr_transfer'])
            cond        = True
            
            while cond:
                model_clone = MLSurgery.fun_model_create(self, MLSurgery.fun_model_stack(df_clone), lr = lr_transfer)

                model_clone.fit(self.data_tr[0], self.data_tr[1], epochs = 1, verbose = 0, shuffle = True, batch_size = self.opt['batch_size'], validation_data = self.data_te, callbacks = self.callbacks)
                _ , acc_case = model_clone.evaluate(self.data_te[0], self.data_te[1],  verbose = 0, batch_size = 32)
                

                if (acc_case >= self.model_original_acc_te) or (c >= self.opt['epochs_transfer']):
                    cond = False

                    for layer in model_clone.layers:
                        layer.trainable = True

                    optimizer = deepcopy(self.optimizer)
                    optimizer.lr = self.opt['lr_finetune']
                    
                    # one round of fine-tuning
                    model_clone.compile(optimizer = self.optimizer, loss = self.loss, metrics = self.metrics)
                    model_clone.fit(self.data_tr[0],self.data_tr[1], epochs = self.opt['epochs_finetune'], shuffle = True, batch_size = self.opt['batch_size'], verbose = 0, validation_data = self.data_te, callbacks = self.callbacks)
                    
                else:
                    c = c + 1
                    lr_transfer = lr_transfer *0.5


        _, acc_case = model_clone.evaluate(self.data_te[0], self.data_te[1], verbose = 0, batch_size = 32)

        return model_clone, acc_case 

    def fun_relu2poly(self, model):



        '''
        Trun every relu layer into average pooling layer
        '''



        # we first clone the original model to make sure we dont mess with the model_original
        model_clone  = MLSurgery.fun_model_clone(self, model)
        df_clone     = MLSurgery.fun_model_info(model_clone)
        
        # find indeices of all max pooling layers
        ind_relu       = np.where(df_clone['relu'].values)[0]

        # we start from the latest relu layer and move towards the first one
        ind_relu       = ind_relu[::-1]


        for ind in ind_relu:
            df_clone  = MLSurgery.fun_model_info(model_clone)

            for i0 in range(len(df_clone)):
                df_clone['layer'][i0].trainable = False
                df_clone['trainable']           = False


            if self.opt['he_polynomial_Degree'] == 0:
                df_clone['layer'][ind] = Square()
            
            elif self.opt['he_polynomial_Degree'] == 2:
                df_clone['layer'][ind] =  DynamicPolyReLU_D2()

            elif self.opt['he_polynomial_Degree'] == 3:
                df_clone['layer'][ind] =  DynamicPolyReLU_D3()

            elif self.opt['he_polynomial_Degree'] == 4:
                df_clone['layer'][ind] =  DynamicPolyReLU_D4()
            
            else:
                assert True, "MLSurgery ERROR: Only polynomials of degree 0, 2, 3, & 4 are supported"
            
            #df_clone['layer'][ind]          = self.poly

            df_clone['type'][ind]      = 'PolyReLU'
            df_clone['trainable'][ind] = True
            df_clone['relu'][ind]      = False

            c           = 1
            lr_transfer = deepcopy(self.opt['lr_transfer'])
            cond        = True
            while cond:
                model_clone = MLSurgery.fun_model_create(self, MLSurgery.fun_model_stack(df_clone), lr = lr_transfer)
                model_clone.fit(self.data_tr[0], self.data_tr[1], epochs = 1, verbose = 0, shuffle = True, batch_size = self.opt['batch_size'], validation_data = self.data_te, callbacks = self.callbacks)
                
                _, acc_case = model_clone.evaluate(self.data_te[0], self.data_te[1], verbose = 0, batch_size = 32)

                if (acc_case >= self.model_original_acc_te) or (c >= self.opt['epochs_transfer']):
                    cond = False

                    for layer in model_clone.layers:
                        layer.trainable = True
                    
                    # one round of fine-tuning
                    model_clone.compile(optimizer = self.optimizer, loss = self.loss, metrics = self.metrics)
                    model_clone.fit(self.data_tr[0],self.data_tr[1], epochs = self.opt['epochs_finetune'], shuffle = True, batch_size = self.opt['batch_size'], verbose = 0, validation_data = self.data_te, callbacks = self.callbacks)
                    
                else:
                    c = c + 1
                    lr_transfer = lr_transfer *0.5

        _, acc_case = model_clone.evaluate(self.data_te[0], self.data_te[1], verbose = 0, batch_size = 32)

        return model_clone, acc_case

    # def fun_prun_nopacking(self, model):


    #     '''
    #     Random Prning of the model using Tensorflow available tools
    #     '''

    #     warnings.filterwarnings("ignore")

    #     tf.keras.utils.get_custom_objects().update(self.custom_objects)

    #     model_clone = MLSurgery.fun_model_clone(self, model)


    #     prune_low_magnitude = tfmot.sparsity.keras.prune_low_magnitude

    #     num_images = self.data_tr[0].shape[0]

    #     end_step = np.ceil(num_images / self.opt['batch_size']).astype(np.int32) * self.opt['epochs']

    #     pruning_params = {"pruning_schedule": tfmot.sparsity.keras.PolynomialDecay(initial_sparsity = self.opt['initial_sparsity'], 
    #                                                                                final_sparsity   = self.opt['final_sparsity'], 
    #                                                                                begin_step       = 0, 
    #                                                                                end_step         = end_step)}
    #     model_for_pruning = prune_low_magnitude(model_clone, **pruning_params)

    #     # `prune_low_magnitude` requires a recompile.
    #     model_for_pruning.compile(optimizer=self.optimizer, loss=self.loss, metrics = self.metrics)


    #     if not os.path.exists(self.path_temp):
    #         os.makedirs(self.path_temp)

    #     callbacks = [tf.keras.callbacks.EarlyStopping(monitor="val_sparse_categorical_accuracy", restore_best_weights=True, patience = self.opt['pruning_patience']),
    #                  tfmot.sparsity.keras.UpdatePruningStep(),
    #                  tfmot.sparsity.keras.PruningSummaries(log_dir = self.path_temp)]


    #     model_for_pruning.fit(self.data_tr[0],self.data_tr[1], epochs = self.opt['epochs'], verbose = 0, validation_data = self.data_te, callbacks = callbacks)

    #     model_pruned  = tfmot.sparsity.keras.strip_pruning(model_for_pruning)

    #     model_pruned.compile(optimizer=self.optimizer, loss=self.loss, metrics = self.metrics)

    #     _, acc_case = model_pruned.evaluate(self.data_te[0], self.data_te[1], verbose = 0, batch_size = 32)
        
    #     return model_pruned, acc_case

    # def fun_best_tile_configuration_2d(self, mat_shape):


    #     """
    #     Configures packing-aware tiles for dense layers
    #     """


    #     c = 0
    #     while True:
    #         if 2**c >= self.opt['num_slots']:
    #             if 2**c > self.opt['num_slots']:
    #                 c = c - 1
    #             break
    #         c = c + 1

    #     tile_shape = []
    #     num_tiles = []
    #     for i0 in range(c + 1):
    #         if self.opt['num_slots'] % (2**i0) == 0:
    #             tile_shape.append((2**i0, int(self.opt['num_slots'] / (2**i0))))
    #             num_row = -(-mat_shape[0] // tile_shape[-1][0])
    #             num_col = -(-mat_shape[1] // tile_shape[-1][1])
    #             num_tiles.append(int(num_row * num_col))

    #     num_best_tile   = min(num_tiles)
    #     ind_best_tile   = num_tiles.index(min(num_tiles))
    #     shape_best_tile = tile_shape[ind_best_tile]

    #     return shape_best_tile, num_best_tile

    # def fun_tiling_2d(mat, tile_shape):
        
        
        
    #     """
    #     Trun a 2d mat into tiles given number of slots and tile shape
    #     """



    #     mat = np.array(mat)

    #     m, n = mat.shape
    #     ind = np.arange(0, m * n).reshape((m, n))

    #     num_row = -(-ind.shape[0] // tile_shape[0])
    #     num_col = -(-ind.shape[1] // tile_shape[1])

    #     num_tiles = num_row * num_col

    #     c_row     = 0
    #     tiles_ind = []
    #     tiles_mat = []

    #     for row in range(num_row):
    #         c_col = 0
    #         for col in range(num_col):

    #             tile_ind_case = [[None for _ in range(tile_shape[1])] for _ in range(tile_shape[0])]
    #             tile_mat_case = [[0 for _ in range(tile_shape[1])] for _ in range(tile_shape[0])]

    #             if row == num_row - 1 and col == num_col - 1:
    #                 ind_case = ind[c_row:, c_col:]
    #                 mat_case = mat[c_row:, c_col:]

    #             elif row != num_row - 1 and col == num_col - 1:
    #                 ind_case = ind[c_row : c_row + tile_shape[0], c_col:]
    #                 mat_case = mat[c_row : c_row + tile_shape[0], c_col:]

    #             elif row == num_row - 1 and col != num_col - 1:
    #                 ind_case = ind[c_row:, c_col : c_col + tile_shape[1]]
    #                 mat_case = mat[c_row:, c_col : c_col + tile_shape[1]]

    #             else:
    #                 ind_case = ind[c_row : c_row + tile_shape[0], c_col : c_col + tile_shape[1]]
    #                 mat_case = mat[c_row : c_row + tile_shape[0], c_col : c_col + tile_shape[1]]

    #             for i0 in range(ind_case.shape[0]):
    #                 for i1 in range(ind_case.shape[1]):
    #                     tile_ind_case[i0][i1] = ind_case[i0, i1]
    #                     tile_mat_case[i0][i1] = mat_case[i0, i1]

    #             c_col += tile_shape[1]

    #             tiles_ind.append(tile_ind_case)
    #             tiles_mat.append(tile_mat_case)

    #         c_row += tile_shape[0]

    #     return tiles_mat, tiles_ind, num_tiles

    # def fun_tiling_conv2d(mat, self):



    #     """
    #     Trun a Con2d weights into tiles given number of slots and tile shape
    #     Condition: products of kernel size, (e.g. (3,3) = 9) cannot be larger than the num_slots (e.g., 2**7)
    #     """



    #     mat = np.array(mat)
    #     mat_shape = list(mat.shape)

    #     ind = np.arange(0, MLSurgery.fun_prod_list(mat_shape)).reshape(mat_shape)

    #     if MLSurgery.fun_prod_list(mat_shape) <= self.opt['num_slots']:
    #         tiles_ind, num_tiles, tile_shape = [ind], 1, ind.shape
    #     else:
    #         # find which dimension would fullfill the slots!
    #         r = 1
    #         for dim in range(len(mat_shape)):
    #             r = r * mat_shape[dim]
    #             if self.opt['num_slots'] / r < 1:
    #                 break
            
    #         if dim <2:
    #             print("ERROR | num_slots must be larer than the product of conv2d kernel size in each conv2d layer")


    #         num_mat_shape_rest = len(mat_shape[dim + 1 :])  # number of reamining dimensions if any
    #         tile_base_shape    = mat_shape[:dim]  # early dimensions
    #         tile_prime_shape   = [self.opt['num_slots'] // (MLSurgery.fun_prod_list(tile_base_shape))]  # primary dimesnion of the tiles such best fit in the slots
    #         tile_shape         = tile_base_shape + tile_prime_shape + [1] * num_mat_shape_rest

    #         tiles_ind = []
    #         if dim == 2:
    #             num_tiles = 0
    #             for i0 in range(mat_shape[-1]):
    #                 for i1 in range(0, mat_shape[dim], tile_prime_shape[0]):
    #                     if i1 == list(range(0, mat_shape[dim], tile_prime_shape[0]))[-1]:
    #                         ind_case = ind[:, :, i1:, i0]
    #                     else:
    #                         ind_case = ind[:, :, i1 : i1 + tile_prime_shape[0], i0]

    #                     tiles_ind.append(ind_case)
    #                     num_tiles = num_tiles + 1

    #         if dim == 3:
    #             num_tiles = 0
    #             for i0 in range(0, mat_shape[dim], tile_prime_shape[0]):
    #                 if i0 == list(range(0, mat_shape[dim], tile_prime_shape[0]))[-1]:
    #                     ind_case = ind[:, :, :, i0]
    #                 else:
    #                     ind_case = ind[:, :, :, i0 : i0 + tile_prime_shape[0]]

    #                 tiles_ind.append(ind_case)
    #                 num_tiles = num_tiles + 1

    #     return tiles_ind, num_tiles, tile_shape

    # def fun_prod_list(x):
    #     r = 1
    #     for i0 in range(len(x)):
    #         r = r * x[i0]

    #     return r

    # def fun_gradient_mask(model, self):


    #     '''
    #     generates gradient mask
    #     '''


    #     gradient_mask = []
    #     tile_info     = []

    #     for var in model.trainable_variables:
    #         var = np.array(var)
    #         if len(var.shape) == 1:
    #             var = np.expand_dims(var, axis=0)

    #         if len(var.shape) == 2:
    #             tile_shape, _           = MLSurgery.fun_best_tile_configuration_2d(self, var.shape)
    #             _, tiles_ind, num_tiles = MLSurgery.fun_tiling_2d(var, tile_shape)

    #         if len(var.shape) == 4:
    #             tiles_ind, num_tiles, tile_shape = MLSurgery.fun_tiling_conv2d(var, self)

    #         num_learning_tiles   = int(num_tiles * self.opt['nonzero_tiles_rate']) + 1

    #         ind_learning_tiles   = np.random.permutation(num_tiles)[:num_learning_tiles]
    #         ind_learning_weights = np.concatenate([np.array(tiles_ind[i0]).ravel() for i0 in ind_learning_tiles])
    #         ind_learning_weights = ind_learning_weights[ind_learning_weights != np.array(None)]
    #         ind_learning_weights = np.array(ind_learning_weights, dtype = np.int32)

    #         grad_case            = np.zeros(var.shape, dtype=np.float32).ravel()

    #         grad_case[ind_learning_weights] = 1
    #         grad_case                       = np.reshape(grad_case, var.shape)

    #         if grad_case.shape[0] == 1:
    #             grad_case = grad_case[0, :]

    #         gradient_mask.append(grad_case)

    #         tile_info_case                    = {}
    #         tile_info_case["tile_shape"]      = tile_shape
    #         tile_info_case["num_tiles"]       = num_tiles
    #         tile_info_case["tiles_ind"]       = tiles_ind
    #         tile_info_case["tiles_ind_ravel"] = ind_learning_weights

    #         tile_info.append(tile_info_case)

    #     return gradient_mask, tile_info

    # def fun_weight_initialization_with_tiles(model, tile_info):



    #     '''
    #     Set weights all zero except the tiles
    #     '''

        
        
    #     for i0 in range(len(model.trainable_variables)):
    #         weight_case      = np.array(model.trainable_variables[i0])
    #         layer_case_name  = model.trainable_variables[i0].name
    #         layer_name       = layer_case_name.split('/')[0]
    #         weight_ravel     = weight_case.ravel()

    #         ind_case         = tile_info[i0]["tiles_ind_ravel"]
    #         zmat             = np.zeros(weight_ravel.shape)
    #         zmat[ind_case]   = weight_ravel[ind_case]
    #         weight_case      = zmat.reshape(weight_case.shape)

    #         for layer in model.layers:
    #             if layer.name == layer_name:
    #                 w_case = []
    #                 for w in layer.weights:
    #                     if w.name == layer_case_name:
    #                         w_case.append(weight_case)
    #                     else:
    #                         w_case.append(np.array(w))

    #                 layer.set_weights(w_case)

    #     return model

    def fun_plot_tiles(model):



        '''
        ploting packs of dense and conv2d weight of a model
        
        '''



        for weight in model.trainable_variables:
            name = weight.name 
            name = '_'.join([i0 if ':' not in i0 else ''.join(i0.split(':')) for i0 in name.split('/')]) + '.png'
            weight  = np.array(weight)
            max_val = np.max(np.abs(weight.ravel()))
            weight  = np.abs(weight / max_val)
            weight  = weight > 0

            if len(weight.shape) != 1:
                if len(weight.shape) == 4:
                    plt.figure()
                    plt.imshow(weight.reshape(np.prod(weight.shape[:-1]), weight.shape[-1]), cmap = "gray")
                    plt.show()
                    

                else:
                    plt.figure()
                    plt.imshow(weight, cmap = "gray")
                    plt.show()

                plt.savefig(name)

    def fun_layerwise_inou(model, cond = True):




        '''
        Turning a model into input putput tensors
        '''




        layers = [layer for layer in model.layers]
        layers_name = [layer.name for layer in model.layers]

        if type(model.layers[0].input_shape) == list:
            inp_shape = model.layers[0].input_shape[0][1:]
        else:
            inp_shape = model.layers[0].input_shape[1:]

        if cond:
            layers_inpt = []
            for layer in model.layers:
                if type(layer.input) == list:
                    layers_inpt.append(
                        [
                            str(var).split("created by layer ")[-1].split("'")[1]
                            for var in layer.input
                        ]
                    )
                else:
                    layers_inpt.append(
                        [str(layer.input).split("created by layer ")[-1].split("'")[1]]
                    )

            x = tf.keras.Input(inp_shape)
            x_hist = [x]

            for i0 in range(1, len(layers)):

                if len(layers_inpt[i0]) == 1:
                    ind_layer = layers_name.index(layers_inpt[i0][0])
                    layer_case = layers[i0]
                    print(i0, ind_layer)
                    print(x_hist)
                    print(x_hist[ind_layer])
                    x = layer_case(x_hist[ind_layer])
                else:
                    layer_case = layers[i0]
                    ind_layer = [
                        layers_name.index(layers_inpt[i0][i1])
                        for i1 in range(len(layers_inpt[i0]))
                    ]
                    x = layer_case([x_hist[ind] for ind in ind_layer])

                x_hist.append(x)

        else:
            x = tf.keras.Input(inp_shape)
            x_hist = [x]

            for i0 in range(1, len(layers)):
                x = layers[i0](x)
                x_hist.append(x)

        out = [x_hist[0], x]

        return out

    def fun_custom_training(self, model):


        '''
        Function custom training with gradient tape
        '''


        out          = MLSurgery.fun_layerwise_inou(model, cond=False)
        model_custom = CustomModel(out[0], out[-1])

        optimizer    = deepcopy(self.optimizer)
        optimizer.lr = optimizer.lr/10

        model_custom.compile(optimizer = optimizer, loss=self.loss, metrics = self.metrics)

        model_custom.fit(self.data_tr[0],self.data_tr[1], batch_size = self.opt['batch_size'], epochs = self.opt['pruning_epochs'], shuffle = True, verbose = 0, validation_data = self.data_te)

        return model_custom

    # def fun_tile_save(tile_info, model):
        
    #     tiles_shape = {}

    #     for i0, weight in enumerate(model.trainable_weights):
    #         name             = weight.name
    #         key              = '_'.join([i0 if ':' not in i0 else ''.join(i0.split(':')) for i0 in name.split('/')])
    #         tiles_shape[key] = tile_info[i0]['tile_shape']
        
    #     np.save('tiles_shape.npy', tiles_shape)

    # def fun_prun_withpacking(self, model):



    #     '''
    #     Packing-aware pruning a model 
    #     '''



    #     global gradient_mask
    #     global tile_info

    #     for layer in model.layers:
    #         layer.trainable = True

    #     model.compile(optimizer = self.optimizer, loss = self.loss, metrics = self.metrics)

    #     model_clone = MLSurgery.fun_model_clone(self, model) #tf.keras.models.clone_model(model)

    #     """
    #     For every weight matrix in the model, Identify the best tile shape such minimizes
    #     the total number of tiles. Next, for every weight matrix in the model,
    #     randomly selects nonzero_tile_percentage of the tiles to be trained by creating
    #     a list of gradient_mask. The list tile_info includes tile information and indices
    #     for every weight matrix 
    #     """

    #     gradient_mask, tile_info = MLSurgery.fun_gradient_mask(model_clone, self)

    #     MLSurgery.fun_tile_save(tile_info, model_clone)


    #     """
    #     Based on the index information in tile_info, for every weight, non-zero tiles are
    #     modified to have random uniform weight initiations beween -0.1 and +0.1 and the 
    #     otehr tiles to have zero.
    #     """

    #     model_clone = MLSurgery.fun_weight_initialization_with_tiles(model_clone, tile_info)

    #     model_custom = MLSurgery.fun_custom_training(self, model_clone)

    #     _, acc_case = model_custom.evaluate(self.data_te[0], self.data_te[1], verbose = 0, batch_size = 32)

    #     return model_custom, acc_case

    def fun_evaluate(self, model):
        _, acc = model.evaluate(self.data_tr[0], self.data_tr[1], verbose = 0, batch_size = 32)
        return acc
    
    def conv2d_information_extractor(self, layer, info = {}):
        '''
        info = conv_information_extractor(self, layer, info = {})
        
        layer is a Conv2D layer of a model_original                                                                                                                                        

        '''

        initial_weights = layer.weights[0].numpy()
        weights_shape   = initial_weights.shape
        initial_biases  = layer.weights[1].numpy()
        num_filters     = layer.weights[0].numpy().shape[-1]

        initial_weights = tf.reshape(initial_weights, (tf.math.reduce_prod(initial_weights.shape[:-1]), num_filters)).numpy()
        
        bias_initializer   = tf.constant_initializer(initial_biases)
        weight_initializer = tf.constant_initializer(initial_weights)

        tf.constant_initializer(initial_weights)

        kernel_size     = layer.kernel_size
        strides         = layer.strides

        width           = layer.input_shape[1]
        height          = layer.input_shape[2]
        depth           = layer.input_shape[-1]

    
        pruning_params                     = {}
        pruning_params['block_size']       = (initial_weights.shape[0]-1,1)
        pruning_params['pruning_schedule'] = tfmot.sparsity.keras.ConstantSparsity(target_sparsity = self.opt['target_sparsity'],
                                                                                   begin_step      = self.opt['begin_step'],
                                                                                   frequency       = self.opt['frequency']) 
        
        # gather info

        name                             = layer.name
        info[name]                       = {}

        info[name]['pruning_params']     = pruning_params
        info[name]['weight_initializer'] = weight_initializer
        info[name]['bias_initializer']   = bias_initializer
        info[name]['num_filters']        = num_filters
        info[name]['kernel_size']        = kernel_size
        info[name]['strides']            = strides
        info[name]['width']              = width
        info[name]['height']             = height
        info[name]['depth']              = depth
        info[name]['weight_shape']       = weights_shape

        return info

    def dense_infomation_extractor(self, layer, info = {}):
    
        '''
        info = dense_information_extractor(self, layer, info = {})

        layer is a dense layer of a model_original
        '''
    

        pruning_params                     = {}
        pruning_params['block_size']       = (layer.input_shape[1]-1,1)
        pruning_params['pruning_schedule'] = tfmot.sparsity.keras.ConstantSparsity(target_sparsity = self.opt['target_sparsity'],
                                                                                   begin_step      = self.opt['begin_step'],
                                                                                   frequency       = self.opt['frequency']) 
        initial_biases     = layer.weights[1].numpy()
        initial_weights    = layer.weights[0].numpy()

        bias_initializer   = tf.constant_initializer(initial_biases)
        weight_initializer = tf.constant_initializer(initial_weights)

        # gather info

        name                             = layer.name
        info[name]                       = {}

        info[name]['pruning_params']     = pruning_params
        info[name]['weight_initializer'] = weight_initializer
        info[name]['bias_initializer']   = bias_initializer
        
        return info

    def conv2d_plugback_initializers(layer, info, weight_shape):

        '''
        info = conv2d_plugback_initializers(layer, info, weight_shape)
        '''

        initial_weights = layer.weights[0].numpy()
        initial_biases  = layer.weights[1].numpy()

        initial_weights = initial_weights.reshape(weight_shape)
        
        bias_initializer   = tf.constant_initializer(initial_biases)
        weight_initializer = tf.constant_initializer(initial_weights)

        # gather info

        name                             = layer.name
        info[name]                       = {}
        info[name]['weight_initializer'] = weight_initializer
        info[name]['bias_initializer']   = bias_initializer

        return info

    def dense_plugback_initializers(layer, info):

        '''
        info = dense_plugback_initializers(layer, info)
        '''

        initial_weights = layer.weights[0].numpy()
        initial_biases  = layer.weights[1].numpy()
        
        bias_initializer   = tf.constant_initializer(initial_biases)
        weight_initializer = tf.constant_initializer(initial_weights)

        # gather info

        name                             = layer.name
        info[name]                       = {}
        info[name]['weight_initializer'] = weight_initializer
        info[name]['bias_initializer']   = bias_initializer

        return info

    def fun_generate_model_pruning(self, model):
        
        '''
        model_pruning = fun_generate_model_pruning(self, model)
        '''
    
        model_clone = MLSurgery.fun_model_clone(self, model)

        input_shape = model_clone.layers[0].input_shape[0][1:]
        inputs      = tf.keras.Input(input_shape)

        info        = {}

        for i0 , layer in enumerate(model_clone.layers[1:-1]):

            if i0 == 0:
                x = inputs

            name  = layer.name

            if 'conv2d' in name:

                info = MLSurgery.conv2d_information_extractor(self, layer, info = info)
                
                x = Features_4D_To_2D(info[name]['kernel_size'], info[name]['strides']) (x)
                x = tfmot.sparsity.keras.prune_low_magnitude(tf.keras.layers.Dense(units              = info[name]['num_filters'], 
                                                                                   kernel_initializer = info[name]['weight_initializer'], 
                                                                                   bias_initializer   = info[name]['bias_initializer']), **info[name]['pruning_params'])(x)
                
                x = Features_2D_To_4D(info[name]['width'], info[name]['height'], info[name]['kernel_size'], info[name]['strides']) (x)

            elif 'dense' in name:
                info = MLSurgery.dense_infomation_extractor(self, layer, info = info)

                x = tfmot.sparsity.keras.prune_low_magnitude(tf.keras.layers.Dense(units              = layer.units,
                                                                                   kernel_initializer = info[name]['weight_initializer'],
                                                                                   bias_initializer   = info[name]['bias_initializer']),  **info[name]['pruning_params']) (x)

            else:
                x = layer(x)

        outputs = model_clone.layers[-1] (x) # you may need to clone it first

        model_pruning = tf.keras.Model(inputs, outputs)
        model_pruning.compile(optimizer = model.optimizer, 
                              loss      = model.loss,
                              metrics   = model.metrics[-1:])
        
        #model_pruning.summary()

        return model_pruning, info

    def fun_weight_observation(model, range_limit = True):
        '''
        weight_observation(model, range_limit = True)
        '''
        for layer in model.layers:
            if 'dense' in layer.name:
                weight = layer.weights[0].numpy()
                name   = layer.weights[0].name 
                name   = '_'.join([i0 if ':' not in i0 else ''.join(i0.split(':')) for i0 in name.split('/')]) + '.png'
                if range_limit:
                    if weight.shape[0] > 100:
                        weight = weight[:100,:]

                    if weight.shape[1] > 100:
                        weight = weight[:,:100]
                
                plt.figure(figsize = [5,5])
                plt.imshow(np.abs(weight)==0, cmap = 'gray');
                plt.show();
                plt.title(layer.name)
                plt.savefig(name)

    def fun_generate_model_plugbacked(model, info):

        '''
        model_plugbacked = fun_generate_model_plugbacked(model, info)
        '''
        
        keys        = list(info.keys())

        input_shape = model.layers[0].get_input_shape_at(0)[1:]
        inputs      = tf.keras.Input(input_shape)

        layer_counter = 0
        info_counter  = 0
        while True:
            layer = model.layers[layer_counter]
            name  = layer.name

            if layer_counter == 0:
                x = inputs 

            if 'features_4d' in name:
                layer         = model.layers[layer_counter + 1]
                name          = layer.name
                weight_shape  = info[keys[info_counter]]['weight_shape']
                info          = MLSurgery.conv2d_plugback_initializers(layer, info, weight_shape)

                x = tf.keras.layers.Conv2D(filters            = info[keys[info_counter]]['num_filters'],
                                           kernel_size        = info[keys[info_counter]]['kernel_size'], 
                                           strides            = info[keys[info_counter]]['strides'], 
                                           kernel_initializer = info[name]['weight_initializer'],
                                           bias_initializer   = info[name]['bias_initializer'])(x)
                
                info_counter  = info_counter  + 1
                layer_counter = layer_counter + 3

            elif 'dense' in name:
                info  = MLSurgery.dense_plugback_initializers(layer, info)

                x = tf.keras.layers.Dense(units              = layer.weights[0].shape[1],
                                          kernel_initializer = info[name]['weight_initializer'],
                                          bias_initializer   = info[name]['bias_initializer']) (x)

                info_counter  = info_counter  + 1
                layer_counter = layer_counter + 1
            else:
                if layer_counter !=0:
                    x = layer(x)
                
                layer_counter = layer_counter + 1

            if layer_counter == len(model.layers) - 1:
                break


        outputs  = model.layers[-1] (x)

        model_plugbacked = tf.keras.Model(inputs, outputs)
        model_plugbacked.compile(optimizer = model.optimizer,
                                 loss      = model.loss,
                                 metrics   = model.metrics[-1:])

        #model_plugbacked.summary()

        return model_plugbacked

    def fun_tfmot_prune(self, model):
        '''
        model, acc = fun_tfmot_prune(self, model)
        '''
        model_pruning, info = MLSurgery.fun_generate_model_pruning(self, model)
        callbacks           = [MyThresholdCallback(threshold = self.model_original_acc_te), tfmot.sparsity.keras.UpdatePruningStep()]
        
        model_pruning.fit(self.data_tr[0],self.data_tr[1], epochs = self.opt['epochs_pruning'], verbose = 0, shuffle = True, batch_size = self.opt['batch_size'], validation_data = self.data_te, callbacks = callbacks)

        #_, acc              = model_pruning.evaluate(self.data_te[0], self.data_te[1], verbose=0, batch_size = 32)

        #MLSurgery.weight_observation(model_pruning)
        
        model_plugbacked    = MLSurgery.fun_generate_model_plugbacked(model_pruning, info)

        _, acc              = model_plugbacked.evaluate(self.data_te[0], self.data_te[1], verbose=0, batch_size = 32)  

        return model_plugbacked, acc

    def run(self):

        model = MLSurgery.fun_model_clone(self, self.model_original)

        MLSurgery.fun_clear()

        print('Accuracy of The Original Model: {}'.format(self.model_original_acc_te))
        print('-------------------------------------------------------------------------------------------------------------------------')

        if self.opt['he_friendly_stat']:
            
            print('Make The Model HE-Friendly | Converting MaxPoolings into AvergePoolings | Start |')

            model, acc = MLSurgery.fun_max2ave(self, model)

            print('Make The Model HE-Friendly | Converting MaxPoolings into AvergePoolings | End   | Validation Accuracy: {}'.format(acc))

            print('Make The Model HE-Friendly | Converting ReLUs into Polynomials          | Start |')

            model, acc = MLSurgery.fun_relu2poly(self, model)

            print('Make The Model HE-Friendly | Converting ReLUs into Polynomials          | End   | Validation Accuracy: {}'.format(acc))


        if self.opt['pruning_stat']:
            print('Packing-Aware Pruning      | TF-Optimization                            | Start |')

            model, acc = MLSurgery.fun_tfmot_prune(self, model)

            print('Packing-Aware Pruning      | TF-Optimization                            | End   | Validation Accuracy: {}'.format(acc))



        #MLSurgery.fun_plot_tiles(model)
        MLSurgery.fun_weight_observation(model)

        shutil.rmtree(self.path_temp,            ignore_errors=True)
        shutil.rmtree(self.path + '__pycache__', ignore_errors=True)

        return model, acc

def fun_calibrate(datain_tr, datain_te, feature_range = (0, 1)):
    scalerin = MinMaxScaler(feature_range = feature_range)
    scalerin.fit(datain_tr)
    datain_tr_calibrated = scalerin.transform(datain_tr)
    datain_te_calibrated = scalerin.transform(datain_te)

    return datain_tr_calibrated, datain_te_calibrated

def fun_data(name='mnist', calibrate = True):
    if name == 'cifar10':
        (datain_tr, dataou_tr), (datain_te, dataou_te) = tf.keras.datasets.cifar10.load_data()
    elif name =='cifar100':
        (datain_tr, dataou_tr), (datain_te, dataou_te) = tf.keras.datasets.cifar100.load_data()
    elif name == 'mnist':
        (datain_tr, dataou_tr), (datain_te, dataou_te) = tf.keras.datasets.mnist.load_data()
    elif name == 'electric_grid_stability':
        data = pd.read_csv('electrical_grid_stability_simulated_data.csv') 
 
        datain = data.iloc[:,0:12] 
        dataou = data.iloc[:,-1:] 
        
        dataou[dataou == 'stable'] = 1 
        dataou[dataou == 'unstable'] = 0 
        
        datain = datain.values  
        dataou = dataou.values 
        
        # dataype consideration: object to float data 
        mat = np.empty(dataou.shape,dtype=float) 
        for i0 in range(dataou.shape[0]): mat[i0,0] = dataou[i0] 
        dataou = mat 

        datain_tr, datain_te, dataou_tr, dataou_te = train_test_split(datain, dataou, test_size=0.1, random_state = np.random.randint(1000))

        datain_tr, datain_te = fun_calibrate(datain_tr, datain_te, feature_range = (0, 1))

    else:
        print('ERROR')
        (datain_tr, dataou_tr), (datain_te, dataou_te) = (None, None), (None, None)
    
    if len(datain_tr.shape) == 3:
        datain_tr = np.expand_dims(datain_tr, axis = 3)
        datain_te = np.expand_dims(datain_te, axis = 3)

    if calibrate:
        datain_tr, dataou_tr, datain_te, dataou_te = datain_tr/255, dataou_tr, datain_te/255, dataou_te

    return datain_tr, dataou_tr, datain_te, dataou_te

def fun_process_image(image,label):
    image=tf.image.per_image_standardization(image)
    image=tf.image.resize(image,(224,224))
    
    return image,label

def fun_load_model(name):

    model_original = tf.keras.models.load_model(name)
    model_original.compile(
    optimizer=tf.keras.optimizers.Adam(0.001),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=[tf.keras.metrics.SparseCategoricalAccuracy()],)
    
    return model_original

def fun_save_model(model, filepath):
    tf.keras.models.save_model(model,
    filepath,
    overwrite=True,
    include_optimizer=False)

def fun_model_example(name = 'mnist'):
    
    datain_tr, dataou_tr, datain_te, dataou_te = fun_data(name = name, calibrate = True)
    out_size = np.unique(dataou_tr).shape[0]

    x = [tf.keras.Input(datain_tr.shape[1:])]

    if name == 'mnist':

        epochs = 2


        '''

        x.append(tf.keras.layers.Conv2D(filters=32, kernel_size=(6,6), strides=(4,4)) (x[-1]))
        x.append(tf.keras.layers.ReLU()  (x[-1]))
        #x.append(tf.keras.layers.BatchNormalization()  (x[-1]))
        x.append(tf.keras.layers.MaxPooling2D(pool_size=(2,2), strides = (2,2))  (x[-1]))

        x.append(tf.keras.layers.Flatten()  (x[-1]))

        x.append(tf.keras.layers.Dense(16) (x[-1]))
        x.append(tf.keras.layers.ReLU()  (x[-1]))

        '''

        #inputs = tf.keras.layers.Input((28,28,1))
        x.append(tf.keras.layers.Conv2D(filters = 32,kernel_size = (5,5), strides = (1,1))( x[-1]))
        x.append(tf.keras.layers.AveragePooling2D(pool_size = (2,2), strides = (2,2)) (x[-1]))
        x.append(tf.keras.layers.BatchNormalization() (x[-1]))
        x.append(tf.keras.layers.ReLU()(x[-1]))
        x.append(tf.keras.layers.Flatten()(x[-1]))
        x.append(tf.keras.layers.Dense(units = 1024) (x[-1]))
        x.append(tf.keras.layers.ReLU() (x[-1]))
        x.append(tf.keras.layers.Dropout(0.4) (x[-1]))
        # outputs = tf.keras.layers.Dense(10) (x)

        # model_original = tf.keras.Model(inputs, outputs)
        # model_original.compile(optimizer = 'adam',
        #                     loss      = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        #                     metrics   = ['accuracy'])

        # model_original.summary()


    elif name == 'cifar10':

        epochs = 25

        x.append(tf.keras.layers.Conv2D(filters=256, kernel_size=(3,3), strides=(1,1)) (x[-1]))
        x.append(tf.keras.layers.ReLU()  (x[-1]))
        x.append(tf.keras.layers.BatchNormalization()  (x[-1]))
        x.append(tf.keras.layers.MaxPooling2D(pool_size=(2,2), strides = (2,2))  (x[-1]))

        x.append(tf.keras.layers.Conv2D(filters=256, kernel_size=(3,3), strides=(1,1))  (x[-1]))
        x.append(tf.keras.layers.ReLU()  (x[-1]))
        x.append(tf.keras.layers.BatchNormalization()  (x[-1]))
        x.append(tf.keras.layers.MaxPooling2D(pool_size=(2,2), strides = (2,2))  (x[-1]))

        x.append(tf.keras.layers.Conv2D(filters=256, kernel_size=(3,3), strides=(1,1))  (x[-1]))
        x.append(tf.keras.layers.ReLU()  (x[-1]))
        x.append(tf.keras.layers.BatchNormalization()  (x[-1]))
        x.append(tf.keras.layers.MaxPooling2D(pool_size=(2,2), strides = (2,2))  (x[-1]))

        x.append(tf.keras.layers.Flatten()  (x[-1]))
 
        x.append(tf.keras.layers.Dense(256) (x[-1]))
        x.append(tf.keras.layers.ReLU()  (x[-1]))
        x.append(tf.keras.layers.Dropout(0.25)  (x[-1]))

        x.append(tf.keras.layers.Dense(128) (x[-1]))
        x.append(tf.keras.layers.ReLU()  (x[-1]))
        x.append(tf.keras.layers.Dropout(0.25)  (x[-1]))

    elif name == 'electric_grid_stability':

        epochs = 10

        x.append(tf.keras.layers.Dense(64) (x[-1]))
        x.append(tf.keras.layers.ReLU()  (x[-1]))
        x.append(tf.keras.layers.Dropout(0.25)  (x[-1]))

        x.append(tf.keras.layers.Dense(128) (x[-1]))
        x.append(tf.keras.layers.ReLU()  (x[-1]))
        x.append(tf.keras.layers.Dropout(0.25)  (x[-1]))

        x.append(tf.keras.layers.Dense(256) (x[-1]))
        x.append(tf.keras.layers.ReLU()  (x[-1]))
        x.append(tf.keras.layers.Dropout(0.25)  (x[-1]))

    else:
        print("ERROR | EXAMPLE IS NOT SUPPORTED YET!")

    x.append(tf.keras.layers.Dense(out_size)  (x[-1]))

    model = tf.keras.Model(x[0], x[-1])


    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss',
                                                    factor=0.5,
                                                    patience=5,
                                                    verbose=0,
                                                    mode='auto',
                                                    min_delta=0.0001,
                                                    cooldown=0,
                                                    min_lr=0.0000001)

    model.compile(optimizer = tf.keras.optimizers.Adam(0.001), 
        loss      = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics   = [tf.keras.metrics.SparseCategoricalAccuracy()])

    model.summary()

    model.fit(
        datain_tr,
        dataou_tr,
        epochs=epochs,
        verbose = 1,
        validation_data=(datain_te, dataou_te),
        validation_freq=1,
        callbacks = [reduce_lr]    
    )

    model.save('model.h5')

