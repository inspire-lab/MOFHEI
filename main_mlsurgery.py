'''
LIBRARIES
'''

import argparse
import requests
import warnings
import shutil
import os

import numpy                         as np
import pandas                        as pd
import tensorflow                    as tf
import tensorflow_model_optimization as tfmot


from copy                     import deepcopy
from sklearn.model_selection  import train_test_split
from sklearn.preprocessing    import MinMaxScaler
from sklearn.metrics          import mean_squared_error
from sklearn.model_selection  import train_test_split

from matplotlib               import pyplot             as plt

warnings.filterwarnings("ignore")

'''
CUSTOM LAYERS, CLASSES, AND OBJECTS
'''


class ReconstructionMeanSquaredError(tf.keras.losses.Loss):

  def call(self, y_true, y_pred, **kwargs):
    return tf.reduce_mean(tf.square(tf.subtract(y_pred, y_true)))

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
    def __init__(self, threshold, problem):
        super(MyThresholdCallback, self).__init__()
        self.threshold = threshold
        self.problem = problem

    def on_epoch_end(self, epoch, logs=None): 
        #print(logs.keys())
        if self.problem == 'classification':
            val_msm = logs["val_sparse_categorical_accuracy"]
        else:
            val_msm = logs["val_loss"]
            
        if val_msm >= self.threshold:
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

    def call(self, inputs, padding = 'valid'):
        
        if padding == 'same':
            inputs = tf.keras.layers.ZeroPadding2D(padding=1)(inputs)
            #print('here', inputs.shape)
        
        batch_size, width, height, depth = inputs.shape

        # num_strides_horizontal  = int(tf.math.ceil((width-self.kernel_size[0])/self.strides[0]) + 1)
        # num_strides_vertical    = int(tf.math.ceil((hight-self.kernel_size[1])/self.strides[1]) + 1)

        # num_strides_horizontal  = int(-((width-self.kernel_size[0])//-self.strides[0]) + 1)
        # num_strides_vertical    = int(-((hight-self.kernel_size[1])//-self.strides[1]) + 1)

        # num_strides_horizontal  = int(-((self.width-self.kernel_size[0])//-self.strides[0]) + 1)
        # num_strides_vertical    = int(-((self.height-self.kernel_size[1])//-self.strides[1]) + 1)

        num_strides_horizontal  = int(((width-self.kernel_size[0])//self.strides[0]) + 1)
        num_strides_vertical    = int(((height-self.kernel_size[1])//self.strides[1]) + 1)

        # print(self.kernel_size[0])
        # print(self.strides[0])
        # print(width)

        # print(num_strides_horizontal)

        # sdfdsf =sdsdf


        output_rank             = int(num_strides_vertical * num_strides_horizontal)

        ind_horizontal_start    = tf.expand_dims(tf.range(0,int(num_strides_horizontal * self.strides[0]), self.strides[0]), axis = 0)
        ind_horizontal_end      = ind_horizontal_start + self.kernel_size[0]
        ind_horizontal_end      = tf.concat([ind_horizontal_end[:,:-1], tf.expand_dims(tf.expand_dims(tf.constant(width), axis = 0), axis = 0)], axis = 1)  

        ind_vertical_start      = tf.expand_dims(tf.range(0,int(num_strides_vertical * self.strides[1]), self.strides[1]), axis = 0)
        ind_vertical_end        = ind_vertical_start + self.kernel_size[1]
        ind_vertical_end        = tf.concat([ind_vertical_end[:,:-1], tf.expand_dims(tf.expand_dims(tf.constant(height), axis = 0), axis = 0)], axis = 1) 

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
                 padding,
                 **kwargs):
      
        super(Features_2D_To_4D, self).__init__()

        if padding == 'same':           
            self.width       = width  + 2
            self.height      = height + 2
        else:
            self.width       = width
            self.height      = height 

        self.strides     = strides
        self.kernel_size = kernel_size

        #print(width, height, strides, kernel_size)

    def call(self, inputs):

        
        # num_strides_horizontal  = int(-((self.width-self.kernel_size[0])//-self.strides[0]) + 1)
        # num_strides_vertical    = int(-((self.height-self.kernel_size[1])//-self.strides[1]) + 1)

        num_strides_horizontal  = int(((self.width-self.kernel_size[0])//self.strides[0]) + 1)
        num_strides_vertical    = int(((self.height-self.kernel_size[1])//self.strides[1]) + 1)

        # print(self.width, self.kernel_size[0], self.strides[0])
        # print(inputs.shape, (num_strides_horizontal, num_strides_vertical, inputs.shape[-1]))
        return reshape(inputs, (num_strides_horizontal, num_strides_vertical, inputs.shape[-1]))

class CustomModel(tf.keras.Model):
    def train_step(self, data):
        x, y = data

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


'''
$$$$$$$$$$$$$$$$$$$$$
CLASS MLSURGERY
$$$$$$$$$$$$$$$$$$$$$
'''

class MLSurgery():
    
    # Class Object Attributes
    def __init__(self, opt):
        self.opt = opt

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

    def fun_model_saving(self, name):
        self.opt['models'][name].save(self.opt['files'][name], 
                                      overwrite         = True, 
                                      include_optimizer = False)
        
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

    def fun_model_clone(self, name):
        '''
        Create a clone version of the model
        '''

        model_clone = tf.keras.models.clone_model(self.opt['models'][name])
        model_clone.set_weights(self.opt['models'][name].get_weights())

        model_clone.compile(optimizer = self.opt['config']['optimizer'][name], 
                            loss      = self.opt['config']['loss'], 
                            metrics   = self.opt['config']['metrics'])
        
        return model_clone

    def fun_model_info(self, name):
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
        for layer in self.opt['models'][name].layers:
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

    def fun_model_create(self, name, x):

        '''
        Create a model from a list of tensors 
        '''

        model_case = tf.keras.Model(inputs = x[0], outputs = x[-1])

        model_case.compile(optimizer = self.opt['config']['optimizer'][name], 
                           loss      = self.opt['config']['loss'], 
                           metrics   = self.opt['config']['metrics'])

        return model_case 

    def fun_model_validation(self, name):
        _, msm = self.opt['models'][name].evaluate(self.opt['data']['datain_vl'], 
                                                   self.opt['data']['dataou_vl'], 
                                                   verbose    = 0, 
                                                   batch_size = 32) 
        return msm   
    
    def fun_model_testing(self, name):
        _, msm = self.opt['models'][name].evaluate(self.opt['data']['datain_te'], 
                                                   self.opt['data']['dataou_te'], 
                                                   verbose    = 0, 
                                                   batch_size = 32) 
        return msm  

    def fun_max2ave(self):
        '''
        Trun every max pooling layer into average pooling layer
        '''

        # we first clone the original model to make sure we dont mess with the model_original
        self.opt['models']['hefriendly'] = MLSurgery.fun_model_clone(self, name = 'original')
        df_clone        = MLSurgery.fun_model_info(self, name = 'hefriendly')

        # find indeices of all max pooling layers
        ind_pool        = np.where(df_clone['maxpool'].values)[0]

        # we start from the latest max pooling layer and move towards the first one
        ind_pool        = ind_pool[::-1]

        # go through each max pooling layer and convert it to average pooling layer
        for ind in ind_pool:
            df_clone = MLSurgery.fun_model_info(self, name = 'hefriendly')

            # we first freez all layers
            for i0 in range(len(df_clone)):
                df_clone['layer'][i0].trainable = False
                df_clone['trainable'][i0]       = False

            pool_size = df_clone['layer'][ind].pool_size
            strides   = df_clone['layer'][ind].strides
            padding   = df_clone['layer'][ind].padding
        
            df_clone['layer'][ind]     = tf.keras.layers.AveragePooling2D(pool_size = pool_size, 
                                                                          strides   = strides, 
                                                                          padding   = padding)
            df_clone['type'][ind]      = 'AveragePooling2D'
            df_clone['trainable'][ind] = True
            df_clone['maxpool'][ind]   = False

            # here we only unfreez layers after ind
            for i0 in range(ind,len(df_clone)):
                df_clone['layer'][i0].trainable = True
                df_clone['trainable'][i0]       = True
            
            x           = MLSurgery.fun_model_stack(df_clone)
            self.opt['models']['hefriendly'] = MLSurgery.fun_model_create(self, 'transfer', x)

            self.opt['models']['hefriendly'].fit(self.opt['data']['datain_tr'], 
                                                 self.opt['data']['dataou_tr'], 
                                                 epochs          = self.opt['config']['epochs']['transfer'], 
                                                 verbose         = 0, 
                                                 shuffle         = True, 
                                                 batch_size      = self.opt['batch_size'], 
                                                 validation_data = (self.opt['data']['datain_vl'], self.opt['data']['dataou_vl']), 
                                                 callbacks       = self.opt['config']['callbacks'])
            
            # unfreez all
            for layer in self.opt['models']['hefriendly'].layers:
                layer.trainable = True

            # one round of fine-tuning
            self.opt['models']['hefriendly'].compile(optimizer = self.opt['config']['optimizer']['finetune'], 
                                                     loss      = self.opt['config']['loss'], 
                                                     metrics   = self.opt['config']['metrics'])
            
            #model_clone.compile(optimizer = self.optimizer, loss = self.loss)
            self.opt['models']['hefriendly'].fit(self.opt['data']['datain_tr'], 
                                                 self.opt['data']['dataou_tr'], 
                                                 epochs          = self.opt['config']['epochs']['finetune'], 
                                                 verbose         = 0, 
                                                 shuffle         = True, 
                                                 batch_size      = self.opt['batch_size'], 
                                                 validation_data = (self.opt['data']['datain_vl'], self.opt['data']['dataou_vl']), 
                                                 callbacks       = self.opt['config']['callbacks'])

            
        msm_case = MLSurgery.fun_model_testing(self, name = 'hefriendly')    

        return msm_case 

    def fun_relu2poly(self):
        '''
        Trun every relu layer into average pooling layer (after max2ave)
        '''

        self.opt['models']['hefriendly'] = MLSurgery.fun_model_clone(self, name = 'hefriendly')
        df_clone        = MLSurgery.fun_model_info(self, name = 'hefriendly')

        # find indeices of all relu layers
        ind_relu        = np.where(df_clone['relu'].values)[0]

        # we start from the latest max pooling layer and move towards the first one
        ind_relu        = ind_relu[::-1]

        for ind in ind_relu:
            df_clone = MLSurgery.fun_model_info(self, name = 'hefriendly')

            for i0 in range(len(df_clone)):
                df_clone['layer'][i0].trainable = False
                df_clone['trainable']           = False


            if self.opt['polynomial_activation_degree'] == 0:
                df_clone['layer'][ind] = Square()
            
            elif self.opt['polynomial_activation_degree'] == 2:
                df_clone['layer'][ind] =  DynamicPolyReLU_D2()

            elif self.opt['polynomial_activation_degree'] == 3:
                df_clone['layer'][ind] =  DynamicPolyReLU_D3()

            elif self.opt['polynomial_activation_degree'] == 4:
                df_clone['layer'][ind] =  DynamicPolyReLU_D4()
            
            else:
                assert True, "MLSurgery ERROR: Only polynomials of degree 0, 2, 3, & 4 are supported"

            df_clone['type'][ind]      = 'PolyReLU'
            df_clone['trainable'][ind] = True
            df_clone['relu'][ind]      = False


            # here we only unfreez layers after ind
            for i0 in range(ind,len(df_clone)):
                df_clone['layer'][i0].trainable = True
                df_clone['trainable'][i0]       = True
            
            x           = MLSurgery.fun_model_stack(df_clone)
            self.opt['models']['hefriendly'] = MLSurgery.fun_model_create(self, 'transfer', x)

            self.opt['models']['hefriendly'].fit(self.opt['data']['datain_tr'], 
                                                 self.opt['data']['dataou_tr'], 
                                                 epochs          = self.opt['config']['epochs']['transfer'], 
                                                 verbose         = 0, 
                                                 shuffle         = True, 
                                                 batch_size      = self.opt['batch_size'], 
                                                 validation_data = (self.opt['data']['datain_vl'], self.opt['data']['dataou_vl']), 
                                                 callbacks       = self.opt['config']['callbacks'])
            
            # unfreez all
            for layer in self.opt['models']['hefriendly'].layers:
                layer.trainable = True

            # one round of fine-tuning
            self.opt['models']['hefriendly'].compile(optimizer = self.opt['config']['optimizer']['finetune'], 
                                                     loss      = self.opt['config']['loss'], 
                                                     metrics   = self.opt['config']['metrics'])
            
            #model_clone.compile(optimizer = self.optimizer, loss = self.loss)
            self.opt['models']['hefriendly'].fit(self.opt['data']['datain_tr'], 
                                                 self.opt['data']['dataou_tr'], 
                                                 epochs          = self.opt['config']['epochs']['finetune'], 
                                                 verbose         = 0, 
                                                 shuffle         = True, 
                                                 batch_size      = self.opt['batch_size'], 
                                                 validation_data = (self.opt['data']['datain_vl'], self.opt['data']['dataou_vl']), 
                                                 callbacks       = self.opt['config']['callbacks'])

            
        msm_case = MLSurgery.fun_model_testing(self, name = 'hefriendly')   

        return msm_case
    
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
                                                                                   frequency       = self.opt['target_frequency'],
                                                                                   begin_step      = 0) 
        
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
        info[name]['padding']            = layer.padding

        return info

    def dense_infomation_extractor(self, layer, info = {}):
    
        '''
        info = dense_information_extractor(self, layer, info = {})

        layer is a dense layer of a model_original
        '''
    

        pruning_params                     = {}
        pruning_params['block_size']       = (layer.input_shape[1]-1,1)
        pruning_params['pruning_schedule'] = tfmot.sparsity.keras.ConstantSparsity(target_sparsity = self.opt['target_sparsity'],
                                                                                   frequency       = self.opt['target_frequency'],
                                                                                   begin_step      = 0) 
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

    def fun_generate_model_pruning(self, name):
        
        '''
        model_pruning = fun_generate_model_pruning(self, model)
        '''
    
        self.opt['models']['pruned'] = MLSurgery.fun_model_clone(self, name=name)

        input_shape = self.opt['models']['pruned'].layers[0].input_shape[0][1:]
        inputs      = tf.keras.Input(input_shape)

        info        = {}

        dense_names = [layer.name for layer in self.opt['models']['pruned'].layers if 'dense' in layer.name ]

        # find latest dense layer

        for i0 , layer in enumerate(self.opt['models']['pruned'].layers[1:-1]):

            if i0 == 0:
                x = inputs

            name  = layer.name

            if 'conv2d' in name:

                info = MLSurgery.conv2d_information_extractor(self, layer, info = info)
                
                x = Features_4D_To_2D(info[name]['kernel_size'], info[name]['strides']) (x, padding = info[name]['padding'])

                x = tfmot.sparsity.keras.prune_low_magnitude(tf.keras.layers.Dense(units              = info[name]['num_filters'], 
                                                                                   kernel_initializer = info[name]['weight_initializer'], 
                                                                                   bias_initializer   = info[name]['bias_initializer']), **info[name]['pruning_params'])(x)
                
                x = Features_2D_To_4D(info[name]['width'], info[name]['height'], info[name]['kernel_size'], info[name]['strides'], info[name]['padding']) (x)

            elif ('dense' in name) and (name !=  dense_names[-1]):
        
                info = MLSurgery.dense_infomation_extractor(self, layer, info = info)
                
                x = tfmot.sparsity.keras.prune_low_magnitude(tf.keras.layers.Dense(units              = layer.units,
                                                                                   kernel_initializer = info[name]['weight_initializer'],
                                                                                   bias_initializer   = info[name]['bias_initializer']),  **info[name]['pruning_params']) (x)

            else:
                x = layer(x)

        outputs = self.opt['models']['pruned'].layers[-1] (x) # you may need to clone it first

        self.opt['models']['pruned'] = tf.keras.Model(inputs, outputs)

        self.opt['models']['pruned'].compile(optimizer = self.opt['config']['optimizer']['pruned'], 
                                             loss      = self.opt['config']['loss'],
                                             metrics   = self.opt['config']['metrics'])
        
        return info

    def fun_weight_observation(self, range_limit = True):
        '''
        weight_observation(model, range_limit = True)
        '''
        
        for layer in self.opt['models']['pruned'].layers:
            if 'dense' in layer.name:
                weight = layer.weights[0].numpy()
                name_case   = layer.weights[0].name 
                name_case   = '_'.join([i0 if ':' not in i0 else ''.join(i0.split(':')) for i0 in name_case.split('/')]) + '.png'
                name_case   = os.path.join(self.opt['path']['results'] , name_case) 
                if range_limit:
                    if weight.shape[0] > 100:
                        weight = weight[:100,:]

                    if weight.shape[1] > 100:
                        weight = weight[:,:100]
                
                plt.figure(figsize = [5,5])
                plt.imshow(np.abs(weight)==0, cmap = 'gray');
                #plt.grid()
                plt.show();
                plt.title(layer.name, fontsize = 20)
                plt.savefig(name_case, dpi = 600)

    def fun_generate_model_plugbacked(self, info):

        '''
        model_plugbacked = fun_generate_model_plugbacked(self, name, info)
        '''

        self.opt['models']['pruned'] = MLSurgery.fun_model_clone(self, 'pruned')
        
        keys        = list(info.keys())

        input_shape = self.opt['models']['pruned'].layers[0].get_input_shape_at(0)[1:]
        inputs      = tf.keras.Input(input_shape)

        layer_counter = 0
        info_counter  = 0
        while True:
            layer = self.opt['models']['pruned'].layers[layer_counter]
            name  = layer.name
            #print(name)

            if layer_counter == 0:
                x = inputs 

            if 'features_4d' in name:
                layer         = self.opt['models']['pruned'].layers[layer_counter + 1]
                name          = layer.name
                weight_shape  = info[keys[info_counter]]['weight_shape']
                info          = MLSurgery.conv2d_plugback_initializers(layer, info, weight_shape)

                x = tf.keras.layers.Conv2D(filters            = info[keys[info_counter]]['num_filters'],
                                           kernel_size        = info[keys[info_counter]]['kernel_size'], 
                                           strides            = info[keys[info_counter]]['strides'],
                                           padding            = info[keys[info_counter]]['padding'],  
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

            if layer_counter == len(self.opt['models']['pruned'].layers) - 1:
                break

        outputs  = self.opt['models']['pruned'].layers[-1] (x)

        self.opt['models']['pruned'].layers[-1]._name = 'dense_output'

        self.opt['models']['pruned'] = tf.keras.Model(inputs, outputs)
        self.opt['models']['pruned'].compile(optimizer = self.opt['config']['optimizer']['pruned'], 
                                             loss      = self.opt['config']['loss'],
                                             metrics   = self.opt['config']['metrics'])
        
    def fun_tfmot_prune(self, name):
        '''
        model, msm = fun_tfmot_prune(self, model)
        '''

        self.opt['models']['pruned'] = MLSurgery.fun_model_clone(self, name = name)

        info = MLSurgery.fun_generate_model_pruning(self, name = 'pruned')

        self.opt['models']['pruned'].fit(self.opt['data']['datain_tr'], 
                                         self.opt['data']['dataou_tr'], 
                                         epochs          = self.opt['config']['epochs']['pruned'], 
                                         verbose         = 0, 
                                         shuffle         = True, 
                                         batch_size      = self.opt['batch_size'], 
                                         validation_data = (self.opt['data']['datain_vl'], self.opt['data']['dataou_vl']), 
                                         callbacks       = self.opt['config']['callbacks_tfmot'])

        MLSurgery.fun_weight_observation(self)
        
        MLSurgery.fun_generate_model_plugbacked(self, info)

        msm_case = MLSurgery.fun_model_testing(self, name = 'pruned')   

        return msm_case
        
    def fun_culling(self):

        self.opt['models']['pruned'] = MLSurgery.fun_model_clone(self, name = 'pruned')
 
        info = {}
        for layer in self.opt['models']['pruned'].layers:

            info_keys         = list(info.keys())
            if 'conv2d' in layer.name:
                weight             = layer.weights[0].numpy()
                bias               = layer.weights[1].numpy()
                
                ind_good4          = [i0 for i0 in range(weight.shape[-1]) if weight[:,:,:,i0].sum() != 0]
                bias_good          = bias[ind_good4]

                if len(info_keys)   == 0:
                    weight_good     = weight[:,:,:,ind_good4]
                else:
                    ind_good3          = deepcopy(info[info_keys[-1]]['ind_good_channel'])
                    weight_good        = weight[:,:,ind_good3, :][:,:,:,ind_good4]

                ind_good_row = np.ndarray.flatten(np.array(range(np.prod(layer.output_shape[1:]))).reshape(layer.output_shape[1:][::-1])[ind_good4,:,:])
                
                weight_initializer = tf.initializers.constant(weight_good)
                bias_initializer   = tf.initializers.constant(bias_good)

                # gather information
                info[layer.name]                          = {}
                info[layer.name]['ind_good_channel']      = ind_good4
                info[layer.name]['ind_good_row']          = ind_good_row
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
                info[layer.name]['moving_variance_initializer']    = moving_variance_initializer 

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
                else:
                    ind_good_row        = deepcopy(info[info_keys[-1]]['ind_good_row'])
                    weight_good         = weight[ind_good_row, :][:,ind_good_column]

                    
                ind_good_row        = deepcopy(ind_good_column) #np.ndarray.flatten(np.array(range(np.prod(weight.shape))).reshape(weight.shape)[ind_good_row, :][:,ind_good_column])
                
                weight_initializer = tf.initializers.constant(weight_good)
                bias_initializer   = tf.initializers.constant(bias_good)

                # gather information
                info[layer.name]                          = {}
                info[layer.name]['ind_good_row']          = ind_good_row
                info[layer.name]['weight_initializer']    = weight_initializer
                info[layer.name]['bias_initializer']      = bias_initializer
                info[layer.name]['units']                 = weight_good.shape[-1]


        #$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$



        input_shape = self.opt['models']['pruned'].layers[0].get_input_shape_at(0)[1:]
        inputs      = tf.keras.Input(input_shape)

        for i0, layer in enumerate(self.opt['models']['pruned'].layers):
            
            if i0 == 0:
                x = inputs

            if 'conv2d' in layer.name:

                x = tf.keras.layers.Conv2D(filters            = info[layer.name]['filters'],
                                           kernel_size        = info[layer.name]['kernel_size'], 
                                           strides            = info[layer.name]['strides'], 
                                           padding            = info[layer.name]['padding'], 
                                           kernel_initializer = info[layer.name]['weight_initializer'],
                                           bias_initializer   = info[layer.name]['bias_initializer']) (x)

            elif 'batch_normalization' in layer.name:
                x = tf.keras.layers.BatchNormalization(gamma_initializer           = info[layer.name]['gamma_initializer_initializer'],
                                                      beta_initializer            = info[layer.name]['beta_initializer'],
                                                      moving_mean_initializer     = info[layer.name]['moving_mean_initializer'],
                                                      moving_variance_initializer = info[layer.name]['moving_variance_initializer']) (x)

            elif 'dense' in layer.name:
                x = tf.keras.layers.Dense(units              = info[layer.name]['units'],
                                          kernel_initializer = info[layer.name]['weight_initializer'],
                                          bias_initializer   = info[layer.name]['bias_initializer']) (x)

            else:
                if i0 != 0:
                    x = layer (x)

        outputs      = x 

        #$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$       

        self.opt['models']['pruned'] = CustomModel(inputs, outputs)

        global cond_transfer_learning

        # transfer learning
        cond_transfer_learning = True

        self.opt['models']['pruned'].compile(optimizer  = self.opt['config']['optimizer']['pruned'], 
                                             loss       = self.opt['config']['loss'], 
                                             metrics    = self.opt['config']['metrics'])

        
        self.opt['models']['pruned'].fit(self.opt['data']['datain_tr'], 
                                         self.opt['data']['dataou_tr'], 
                                         epochs          = self.opt['config']['epochs']['pruned'], 
                                         verbose         = 0, 
                                         shuffle         = True, 
                                         batch_size      = self.opt['batch_size'], 
                                         validation_data = (self.opt['data']['datain_vl'], self.opt['data']['dataou_vl']), 
                                         callbacks       = self.opt['config']['callbacks'])
        
        # fine-tuning
        cond_transfer_learning = False

        self.opt['models']['pruned'].compile(optimizer = self.opt['config']['optimizer']['pruned'], 
                                             loss      = self.opt['config']['loss'], 
                                             metrics   = self.opt['config']['metrics'])

        # model_culled.compile(optimizer = optimizer,
        #                     loss       = self.loss)
        
        self.opt['models']['pruned'].fit(self.opt['data']['datain_tr'], 
                                         self.opt['data']['dataou_tr'], 
                                         epochs          = self.opt['config']['epochs']['pruned'], 
                                         verbose         = 0, 
                                         shuffle         = True, 
                                         batch_size      = self.opt['batch_size'], 
                                         validation_data = (self.opt['data']['datain_vl'], self.opt['data']['dataou_vl']), 
                                         callbacks       = self.opt['config']['callbacks'])
        
        msm_case = MLSurgery.fun_model_testing(self, name = 'pruned')   
        
        return msm_case

    def run(self):

        MLSurgery.fun_clear()

        if self.opt['problem'] == 0 :
            msm_name = 'ACC'
        else:
            msm_name = 'MSE'

        print("{} of The Original Model Testing Accuracy: {}".format(msm_name, self.opt['results']['original']['te']))
        print("-------------------------------------------------------------------------------------------------------------------------")

        if self.opt['hefriendly_stat']:
            
            print("Make The Model HE-Friendly | Converting MaxPoolings into AvergePoolings | Start |")

            msm = MLSurgery.fun_max2ave(self)

            print("Make The Model HE-Friendly | Converting MaxPoolings into AvergePoolings | End   | Testing {}: {}".format(msm_name, msm))

            print("Make The Model HE-Friendly | Converting ReLUs       into Polynomials    | Start |")

            msm = MLSurgery.fun_relu2poly(self)

            print("Make The Model HE-Friendly | Converting ReLUs       into Polynomials    | End   | Testing {}: {}".format(msm_name, msm))

            MLSurgery.fun_model_saving(self, name = 'hefriendly')

            
        if self.opt['pruning_stat']:

            if self.opt['hefriendly_stat']:
                name = 'hefriendly'
            else:
                name = 'original'

            print("Packing-Aware Pruning      | TF-Optimization                            | Start |")

            msm = MLSurgery.fun_tfmot_prune(self, name)

            #print("Packing-Aware Pruning      | TF-Optimization                            | End   | Validation {}: {}".format(msm_name, msm))

            #print("Culling                    | Removing Filters and Weights               | Start |")

            msm = MLSurgery.fun_culling(self)

            #print("Culling                    | Removing Filters and Weights               | End   | Validation {}: {}".format(msm_name, msm))

            #MLSurgery.fun_model_save(self, model, model_type = 'prunedandculled')

            print("Packing-Aware Pruning      | TF-Optimization                            | End   | Testing {}: {}".format(msm_name, msm))
            
            MLSurgery.fun_model_saving(self, name = 'pruned')


        return self.opt




'''
CUSTOM FUNCTIONS
'''

@tf.function
def reshape(input, shape): # just for inputs 
    '''
    x = reshape(input, shape)
    '''
    return tf.keras.layers.Reshape(shape)(input)

def fun_download(url, file_path):
    response = requests.get(url)
    with open(file_path, "wb") as f:
        f.write(response.content);

def fun_create_path(path):
    if not os.path.exists(path):
        os.makedirs(path)

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


'''
-CALIBRATION
'''

def fun_image_calibration(datain_tr, 
                          datain_vl,
                          datain_te,
                          cond = False):
    if cond:
        # converts between -1 and 1
        return datain_tr/127.5 - 1, datain_vl/127.5 - 1, datain_te/127.5 - 1
    else:
        # converts between 0 and 1
        return datain_tr/255, datain_vl/255, datain_te/255

def fun_nonimage_callibration(datain_tr, 
                              datain_vl,
                              datain_te,
                              cond = False):
    
    if cond:
        feature_range = (-1, 1)
    else:
        feature_range = (0,1)

    scalerin = MinMaxScaler(feature_range = feature_range)
    scalerin.fit(datain_tr)
    datain_tr = scalerin.transform(datain_tr)
    datain_vl = scalerin.transform(datain_vl)
    datain_te = scalerin.transform(datain_te)

    return datain_tr, datain_vl, datain_te

'''
-DATA
'''

def fun_data_mnist(file_path, validation_split = 0.05):
    (datain_tr, dataou_tr), (datain_te, dataou_te) = tf.keras.datasets.mnist.load_data()
    datain_tr, datain_vl, dataou_tr, dataou_vl = train_test_split(datain_tr, 
                                                                  dataou_tr, 
                                                                  test_size    = validation_split,
                                                                  random_state = np.random.randint(1000))
    
    datain_tr, datain_vl, datain_te = fun_image_calibration(datain_tr, 
                                                            datain_vl,
                                                            datain_te,
                                                            cond = False)
    
    data = {}
    data['datain_tr'] = datain_tr
    data['dataou_tr'] = dataou_tr
    data['datain_vl'] = datain_vl
    data['dataou_vl'] = dataou_vl
    data['datain_te'] = datain_te
    data['dataou_te'] = dataou_te

    np.save(file_path, data)

    return data
    
def fun_data_cifar10(file_path, validation_split = 0.05):
    (datain_tr, dataou_tr), (datain_te, dataou_te) = tf.keras.datasets.cifar10.load_data()
    datain_tr, datain_vl, dataou_tr, dataou_vl = train_test_split(datain_tr, 
                                                                  dataou_tr, 
                                                                  test_size    = validation_split,
                                                                  random_state = np.random.randint(1000))
    
    datain_tr, datain_vl, datain_te = fun_image_calibration(datain_tr, 
                                                            datain_vl,
                                                            datain_te,
                                                            cond = False)
    
    data = {}
    data['datain_tr'] = datain_tr
    data['dataou_tr'] = dataou_tr
    data['datain_vl'] = datain_vl
    data['dataou_vl'] = dataou_vl
    data['datain_te'] = datain_te
    data['dataou_te'] = dataou_te

    np.save(file_path, data)

    return data

def fun_data_electrical_stability(file_path, validation_split = 0.05):
    url = 'https://github.com/mhrafiei/data/raw/main/electrical_grid_stability_simulated_data.npy'

    if not os.path.exists(file_path):
        fun_download(url, file_path)

    data = np.load(file_path, allow_pickle=True).item()

    datain_tr, datain_vl, datain_te = fun_nonimage_callibration(data['datain_tr'], 
                                                                data['datain_vl'],
                                                                data['datain_te'],
                                                                cond = False)

    data['datain_tr'] = datain_tr
    data['datain_vl'] = datain_vl
    data['datain_te'] = datain_te


    return data

def fun_data_xray64(file_path):
    url = 'https://github.com/mhrafiei/data/raw/main/xray.64.npy'

    fun_download(url, file_path)

    data = np.load(file_path, allow_pickle=True).item()

    datain_tr, datain_vl, datain_te = fun_nonimage_callibration(data['datain_tr'], 
                                                                data['datain_vl'],
                                                                data['datain_te'],
                                                                cond = False)

    data['datain_tr'] = datain_tr
    data['datain_vl'] = datain_vl
    data['datain_te'] = datain_te

    return data

def fun_data_custom(file_path):
    return np.load(file_path, allow_pickle=True).item()

def fun_data_autoencoder(data):
    data['dataou_tr'] = data['datain_tr']
    data['dataou_vl'] = data['datain_vl']
    data['dataou_te'] = data['datain_te']

    return data


'''
-CONFIG
'''

def fun_regression_config(args):
    config              = {}
    config['optimizer'] = {} 
    
    config['optimizer']['original']            = tf.keras.optimizers.Adam(float(args.lr))
    config['optimizer']['hefriendly']          = tf.keras.optimizers.Adam(float(args.lr))
    config['optimizer']['transfer']            = tf.keras.optimizers.Adam(float(args.lr_transfer))
    config['optimizer']['finetune']            = tf.keras.optimizers.Adam(float(args.lr_finetune)) 
    config['optimizer']['pruned']              = tf.keras.optimizers.Adam(float(args.lr)) 

    config['loss']                             = 'mse'
    config['metrics']                          = [tf.keras.metrics.MeanSquaredError(name='mean_squared_error', dtype=None)]
    config['monitor']                          = 'val_loss'

    callback_patience                          = tf.keras.callbacks.EarlyStopping(monitor              = config['monitor'],
                                                                                  patience             = int(args.patience),
                                                                                  restore_best_weights = True)
    
    callback_lr                                =    tf.keras.callbacks.ReduceLROnPlateau(monitor   = config['monitor'],
                                                                                         factor    = 0.5,
                                                                                         patience  = 5,
                                                                                         verbose   = 0,
                                                                                         mode      = 'auto',
                                                                                         min_delta = 0.0001,
                                                                                         cooldown  = 0,
                                                                                         min_lr    = 0.0000001)

    config['callbacks']                        = [callback_patience, callback_lr]
    config['callbacks_tfmot']                  = [callback_patience, callback_lr, tfmot.sparsity.keras.UpdatePruningStep()]
    
    config['custom_objects']                   = {'DynamicPolyReLU_D2':DynamicPolyReLU_D2, 
                                                  'DynamicPolyReLU_D3':DynamicPolyReLU_D3, 
                                                  'DynamicPolyReLU_D4':DynamicPolyReLU_D4, 
                                                  'Square':Square, 
                                                  'CustomModel': CustomModel}
    
    config['epochs'] = {}
    config['epochs']['original'] = int(args.epochs_original)
    config['epochs']['transfer'] = int(args.epochs_transfer)      
    config['epochs']['finetune'] = int(args.epochs_finetune)  
    config['epochs']['pruned']  = int(args.epochs_pruning)   

    config['batch_size']         = int(args.batch_size) 

    return config

def fun_classification_config(args):
    config              = {}
    config['optimizer'] = {} 
    
    config['optimizer']['original']            = tf.keras.optimizers.Adam(float(args.lr))
    config['optimizer']['hefriendly']          = tf.keras.optimizers.Adam(float(args.lr))
    config['optimizer']['transfer']            = tf.keras.optimizers.Adam(float(args.lr_transfer))
    config['optimizer']['finetune']            = tf.keras.optimizers.Adam(float(args.lr_finetune)) 
    config['optimizer']['pruned']             = tf.keras.optimizers.Adam(float(args.lr)) 

    config['loss']                             = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    config['metrics']                          = [tf.keras.metrics.SparseCategoricalAccuracy()]
    config['monitor']                          = 'val_sparse_categorical_accuracy'

    callback_patience                          = tf.keras.callbacks.EarlyStopping(monitor              = config['monitor'],
                                                                                  patience             = int(args.patience),
                                                                                  restore_best_weights = True)
    
    callback_lr                                = tf.keras.callbacks.ReduceLROnPlateau(monitor   = config['monitor'],
                                                                                      factor    = 0.5,
                                                                                      patience  = 5,
                                                                                      verbose   = 0,
                                                                                      mode      = 'auto',
                                                                                      min_delta = 0.0001,
                                                                                      cooldown  = 0,
                                                                                      min_lr    = 0.0000001)

    config['callbacks']                        = [callback_patience, callback_lr]
    config['callbacks_tfmot']                  = [callback_patience, callback_lr, tfmot.sparsity.keras.UpdatePruningStep()]
    
    config['custom_objects']                   = {'DynamicPolyReLU_D2':DynamicPolyReLU_D2, 
                                                  'DynamicPolyReLU_D3':DynamicPolyReLU_D3, 
                                                  'DynamicPolyReLU_D4':DynamicPolyReLU_D4, 
                                                  'Square':Square, 
                                                  'CustomModel': CustomModel}

    config['epochs'] = {}
    config['epochs']['original'] = int(args.epochs_original)
    config['epochs']['transfer'] = int(args.epochs_transfer)      
    config['epochs']['finetune'] = int(args.epochs_finetune)  
    config['epochs']['pruned']  = int(args.epochs_pruning)   

    config['batch_size']         = int(args.batch_size)

    return config

'''
-MODELS
'''

def fun_model_loading(opt):

    if ('custom' not in opt['experiment'].lower()) and (not os.path.exists(opt['files']['original'])):
        fun_download(opt['urls'][opt['experiment']], opt['files']['original'])

    model = tf.keras.models.load_model(opt['files']['original'], custom_objects = opt['config']['custom_objects'] )
    model.compile(optimizer = opt['config']['optimizer']['original'], 
                  loss        = opt['config']['loss'], 
                  metrics     = opt['config']['metrics'])

    return model

def fun_model_result(opt, model_type = 'original'):
    results = {}
    _ , results['tr'] = opt['models'][model_type].evaluate(opt['data']['datain_tr'], 
                                                           opt['data']['dataou_tr'], 
                                                           verbose    = 0, 
                                                           batch_size = 32)
    
    _ , results['vl'] = opt['models'][model_type].evaluate(opt['data']['datain_vl'], 
                                                           opt['data']['dataou_vl'], 
                                                           verbose    = 0, 
                                                           batch_size = 32)

    _ , results['te'] = opt['models'][model_type].evaluate(opt['data']['datain_te'], 
                                                           opt['data']['dataou_te'], 
                                                           verbose    = 0, 
                                                           batch_size = 32)
    
    return results

def fun_model_final_sparsity(model_original, model_pruned):
    num_original = sum([np.product(weight.shape) for layer in model_original.layers for weight in layer.weights if weight.trainable])
    num_pruned   = sum([np.product(weight.shape) for layer in model_pruned.layers   for weight in layer.weights if weight.trainable])

    return 1 - num_pruned/num_original

def fun_model_development(opt):
    return eval("fun_model_{}(opt)".format(opt['experiment'].replace('-', '_').lower()))

def fun_model_electrical_stability_fcnet(opt):
    # build, fit, and save the model
    shapein = opt['data']['datain_tr'].shape[1:]
    shapeou = np.unique(opt['data']['dataou_tr']).shape[0]

    inputs  = tf.keras.Input(shapein)

    x       = tf.keras.layers.Dense(64)      (inputs)
    x       = tf.keras.layers.ReLU()         (x)
    x       = tf.keras.layers.Dropout(0.25)  (x)

    x       = tf.keras.layers.Dense(128)     (x)
    x       = tf.keras.layers.ReLU()         (x)
    x       = tf.keras.layers.Dropout(0.25)  (x)

    x       = tf.keras.layers.Dense(256)     (x)
    x       = tf.keras.layers.ReLU()         (x)
    x       = tf.keras.layers.Dropout(0.25)  (x)

    outputs = tf.keras.layers.Dense(shapeou) (x)

    
    model = tf.keras.Model(inputs, outputs)
    model.compile(optimizer = opt['config']['optimizer']['original'], 
                  loss      = opt['config']['loss'],
                  metrics   = opt['config']['metrics'])
    
    model.summary() # can be depreciated

    print(opt['config']['optimizer']['original'])
    print(opt['config']['loss'])
    print(opt['config']['metrics'])
    print(opt['config']['optimizer']['original'].lr)

    model.fit(opt['data']['datain_tr'],
              opt['data']['dataou_tr'],
              epochs          = opt['config']['epochs']['original'],
              callbacks       = opt['config']['callbacks'],
              validation_data = (opt['data']['datain_vl'], opt['data']['dataou_vl']),
              batch_size      = opt['config']['batch_size'],
              verbose         = 1,
              validation_freq = 1)
    
    model.save(opt['files']['original'], 
               overwrite         = True, 
               include_optimizer = False)

    return model

# def fun_mnist_lenet(opt):
#     return model

# def fun_cifar10_alexnet(opt):
#     return model

# def fun_cifar10_vgg16(opt):
#     return model

# def fun_x_ray_vgg16(opt):
#     return model

# def fun_mnist_hepex_ae1(opt):
#     return model

# def fun_mnist_hepex_ae2(opt):
#     return model

# def fun_mnist_hepex_ae3(opt):
#     return model

# def fun_cifar10_hepex_ae1(opt):
#     return model

# def fun_cifar10_hepex_ae2(opt):
#     return model

# def fun_cifar10_hepex_ae3(opt):
#     return model

'''
INITIATE
'''

def fun_initiate(args):
    opt = {}

    experiments = ['CUSTOM', 
                   'ELECTRICAL-STABILITY-FCNet', 
                   'MNIST-LeNet', 
                   'CIFAR10-AlexNet',
                   'CIFAR10-VGG16', 
                   'X-RAY-VGG16', 
                   'MNIST-HEPEX-AE1', 
                   'MNIST-HEPEX-AE2', 
                   'MNIST-HEPEX-AE3', 
                   'CIFAR10-HEPEX-AE1', 
                   'CIFAR10-HEPEX-AE2',
                   'CIFAR10-HEPEX-AE3']
    
    # capture the experiment 
    experiment                          = experiments[int(args.experiment)]

    opt['experiment']                   = experiment
    opt['problem']                      = int(args.problem_type)
    opt['polynomial_activation_degree'] = int(args.polynomial_activation_degree)
    opt['target_sparsity']              = float(args.target_sparsity)
    opt['target_frequency']             = int(args.target_frequency)
    opt['batch_size']                   = int(args.batch_size)

    if args.hefriendly_stat == 'True':
        opt['hefriendly_stat'] = True
    else:
        opt['hefriendly_stat'] = False

    if args.pruning_stat == 'True':
        opt['pruning_stat'] = True
    else:
        opt['pruning_stat'] = False

    # folder and file management
    opt['path']               = {}
    opt['path']['current']    = os.getcwd()
    opt['path']['experiment'] = os.path.join(opt['path']['current'], experiment.lower())
    opt['path']['original']   = os.path.join(opt['path']['current'], experiment.lower(), 'original')
    opt['path']['hefriendly'] = os.path.join(opt['path']['current'], experiment.lower(), 'hefriendly')
    opt['path']['pruned']     = os.path.join(opt['path']['current'], experiment.lower(), 'pruned')
    opt['path']['data']       = os.path.join(opt['path']['current'], experiment.lower(), 'data')
    opt['path']['results']    = os.path.join(opt['path']['current'], experiment.lower(), 'results')
    
    opt['files']                = {}
    opt['files']['original']    = os.path.join(opt['path']['original'],   'model.h5')  
    opt['files']['hefriendly']  = os.path.join(opt['path']['hefriendly'], 'model.h5')  
    opt['files']['pruned']      = os.path.join(opt['path']['pruned'],     'model.h5')  
    opt['files']['data']        = os.path.join(opt['path']['data'],       'data.npy')  
    opt['files']['results']     = os.path.join(opt['path']['results'],    'results.txt') 

    fun_create_path(opt['path']['experiment'])
    fun_create_path(opt['path']['original'])
    fun_create_path(opt['path']['hefriendly'])
    fun_create_path(opt['path']['pruned'])
    fun_create_path(opt['path']['data'])
    fun_create_path(opt['path']['results'])

    # get the data and config
    if 'mnist' in experiment.lower():
       opt['data']  = fun_data_mnist()
    elif 'cifar10' in experiment.lower():
        opt['data'] = fun_data_mnist()
    elif 'electrical' in experiment.lower():
        opt['data'] = fun_data_electrical_stability(opt['files']['data'])
    elif 'custom' in experiment.lower():
        opt['data'] = fun_data_custom()
    else:
        assert False, 'Error! Read the guideline and double check the experiment information'


    if args.problem_type == '1':
        opt['data']   = fun_data_autoencoder(opt['data'])
        opt['config'] = fun_regression_config(args)
    else:
        opt['config'] = fun_classification_config(args)

    # get/create the original model
    opt['models'] = {}    
    if args.model_available == 'True':
        # get the original model from the cloud/device
        opt['models']['original'] = fun_model_loading(opt)
    else:
        # develop and train the original model
        opt['models']['original'] = fun_model_development(opt)

    # get the results 
    opt['results']             = {}
    opt['results']['original'] = fun_model_result(opt, model_type = 'original')

    #update callbacks 
    opt['config']['callbacks']       = opt['config']['callbacks']       + [MyThresholdCallback(threshold = opt['results']['original']['vl'], problem = opt['experiment'])]
    opt['config']['callbacks_tfmot'] = opt['config']['callbacks_tfmot'] + [MyThresholdCallback(threshold = opt['results']['original']['vl'], problem = opt['experiment'])]

    return opt

def fun_conclude(opt):

    print("-------------------------------------------------------------------------------------------------------------------------")
    print('Wait ... ')
    print("-------------------------------------------------------------------------------------------------------------------------")

    opt['results']['hefriendly'] = fun_model_result(opt, model_type = 'hefriendly')
    opt['results']['pruned']     = fun_model_result(opt, model_type = 'pruned')

    final_sparsity = fun_model_final_sparsity(opt['models']['original'], 
                                              opt['models']['pruned']) 

    if opt['problem'] == 0 :
        msm_name = 'ACC'
    else:
        msm_name = 'MSE'

    if os.path.exists(opt['files']['results']):
        os.remove(opt['files']['results'])

    result = '''Original Model:    \n\n{} Train: {} \n{} Val:   {} \n{} Test:  {}\n\nHE-Friendly Model: \n\n{} Train: {} \n{} Val:   {} \n{} Test:  {}\n\nPruned Model: \n\n{} Train: {} \n{}  Val:  {} \n{} Test:  {} \n\nFinal Sparsity: \n{}
             '''.format(msm_name, opt['results']['original']['tr'],   msm_name, opt['results']['original']['vl'],   msm_name, opt['results']['original']['te'],
                        msm_name, opt['results']['hefriendly']['tr'], msm_name, opt['results']['hefriendly']['vl'], msm_name, opt['results']['hefriendly']['te'],
                        msm_name, opt['results']['pruned']['tr'],     msm_name, opt['results']['pruned']['vl'],     msm_name, opt['results']['pruned']['te'],
                        final_sparsity)
    
    f = open(opt['files']['results'], "a")
    f.write(result)
    f.close()

    print("Find the Original model at:     ./{}/original/   model.h5".format(opt['experiment'].lower()))
    print("Find the HE-Friendly model at:  ./{}/hefriendly/ model.h5".format(opt['experiment'].lower()))
    print("Find the Pruned model at:       ./{}/pruned/     model.h5".format(opt['experiment'].lower()))
    print("Find the data at:               ./{}/data/       data.npy".format(opt['experiment'].lower()))   
    print("Find a summary of results:      ./{}/results/    results.txt".format(opt['experiment'].lower()))

    print("-------------------------------------------------------------------------------------------------------------------------")
    print("SUMMARY OF THE ORIGINAL MODEL")
    print("-------------------------------------------------------------------------------------------------------------------------")
    print(" ")

    opt['models']['original'].summary()
    
    print(" ")
    print("-------------------------------------------------------------------------------------------------------------------------")
    print("SUMMARY OF THE HE-FRIENDLY MODEL")
    print("-------------------------------------------------------------------------------------------------------------------------")
    print(" ")

    opt['models']['hefriendly'].summary()

    print(" ")
    print("-------------------------------------------------------------------------------------------------------------------------")
    print("SUMMARY OF THE PRUNED MODEL")
    print("-------------------------------------------------------------------------------------------------------------------------")
    print(" ")

    opt['models']['pruned'].summary()

    print(" ")
    print("-------------------------------------------------------------------------------------------------------------------------")
    print(" ")



    



def main():
    parser = argparse.ArgumentParser(prog        = 'main_mlsurgery',
                                    description = '** Version 202303011200 ** | Run MLSurgery Paper Experiment of Interest')

    parser.add_argument('experiment',
                        help    = '''Select one of the following experiments (number) from MLSurgery paper: 
                                    
                                    (0) CUSTOM Model by User, 
                                    (1) ELECTRICAL-STABILITY-FCNet, 
                                    (2) MNIST-LeNet, 
                                    (3) CIFAR10-AlexNet,
                                    (4) X-RAY-AlexNet, 
                                    (5) CIFAR10-VGG16, 
                                    (6) X-RAY-VGG16, 
                                    (7) MNIST-HEPEX-AE1, 
                                    (8) MNIST-HEPEX-AE2, 
                                    (9) MNIST-HEPEX-AE3, 
                                    (10) CIFAR10-HEPEX-AE1, 
                                    (11) CIFAR10-HEPEX-AE2,
                                    (12) CIFAR10-HEPEX-AE3 ||| 
                                    If the custom model option (i.e., 0) is selected, the user 
                                    requires to have it saved at ./experiment_custom/original/model.h5, and 
                                    include a dictionary dataset at ./experiment_custom/original/data.npy
                                    with the following keys: 
                                        'datain_tr': A numpy array training inputs,  
                                        'dataou_tr': A numpy array training outputs (for classification: [0, 1, 2, ... ]),
                                        'datain_vl': A numpy array validation inputs,  
                                        'dataou_vl': A numpy array validation outputs (for classification: [0, 1, 2, ... ]), 
                                        'datain_te': A numpy array testing inputs,  
                                        'dataou_te': A numpy array testing outputs  (for classification: [0, 1, 2, ... ])
                                    ''',
                        choices = [str(i0) for i0 in range(0,13,1)])
    
    parser.add_argument('problem_type',
                        help    = "Select one of the following problems: (0) for classification and (1) for regression",
                        default = '1',
                        choices = ['0', '1']) 
                                    
                        
    parser.add_argument('-MA',
                        '--model_available',
                        default = 'False', 
                        help    = "True if a ./$EXPERIMENT$/original/model.h5 is available (in cloud or on device); otherwise we develop the model | should always be True for 'CUSTOM' models")

    parser.add_argument('-HS',
                        '--hefriendly_stat',
                        default = 'True', 
                        help    = 'True to make the model HE-Friendly')


    parser.add_argument('-PS',
                        '--pruning_stat',
                        default = 'True', 
                        help    = 'True to make the model pruned')

    parser.add_argument('-AD',
                        '--polynomial_activation_degree',
                        default = '0',
                        help    = 'Select one of the following degrees: 0 (for square), 2, 3, and 4',
                        choices = ['0', '2', '3', '4'])

    parser.add_argument('-EO',
                        '--epochs_original',
                        default = '100',
                        help    = 'Transfer learning maximum number of epochs')

    parser.add_argument('-ET',
                        '--epochs_transfer',
                        default = '100',
                        help    = 'Transfer learning maximum number of epochs')

    parser.add_argument('-EF',
                        '--epochs_finetune',
                        default = '100',
                        help    = 'Fine-tuning maximum number of epochs')

    parser.add_argument('-LR',
                        '--lr',
                        default = '0.001',
                        help    = "Transfer learning's learning rate")

    parser.add_argument('-LT',
                        '--lr_transfer',
                        default = '0.001',
                        help    = "Transfer learning's learning rate")

    parser.add_argument('-LF',
                        '--lr_finetune',
                        default = '0.0001',
                        help    = "Fine-tuning's learning rate")

    parser.add_argument('-BS',
                        '--batch_size',
                        default = '128',
                        help    = "Training batch size")

    parser.add_argument('-PE',
                        '--patience',
                        default = '25',
                        help    = "Training call back patience")

    parser.add_argument('-OE',
                        '--epochs_pruning',
                        default = '100',
                        help    = "Training epochs for pruning")

    parser.add_argument('-TS',
                        '--target_sparsity',
                        default = '0.50',
                        help    = "Prunable layers (i.e., conv and dense layers) target sparsity | does not necessary imply the final model sparsity")

    parser.add_argument('-TF',
                        '--target_frequency',
                        default = '100',
                        help    = "Epoch frequency of pruning")

    args = parser.parse_args()

    if args.experiment == '0':
        args.model_available='True'

    opt    = fun_initiate(args)

    my_obj = MLSurgery(opt)

    opt    = my_obj.run()

    fun_conclude(opt)


if __name__ == '__main__':
    main()