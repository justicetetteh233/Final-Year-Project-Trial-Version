#%%
# print("lets_start")
#import all the necessary libraries and tell why 
from utils.Configs import * #has the parameters for the various models
from tensorflow.python.keras import backend as K
#provides low-level operations for Keras models, such as: Tensor manipulations Mathematical operations Session management Device configuration (e.g., CPU/GPU selection) Memory handling

from cnn.inputpipeline import InputProcessor
from cnn.mammocnn import MammoCNN
from utils.SavedResult import ProcessResult

from cnn.binaryoptimzers import feature_optimizer

#%%
"""
we train a model  and store it:
"""
print('medical image classification project started')
print('we are going to train our model')
# Taking the model to handle our mammogram images 

#we need to define the various hyperparameters and optimizers for our model which has to deduce the feature set from an image.
# opt, lr = optimizers[1], learning_rates[6], #we obtain this from our configuration file  = 1:"Adam",  6:1e-06


# this is the optimization algorithm and learning rates we want to use for the model traning process  Adam , 1e-06

#start data reading process

#define an object to show the mammogramme data 
    #Load mammo mias datasets
print("this is the dataset features and the split we will like to use ")

data_params={
        'num_classes':len(mammo_mias_classes), 'class_names':mammo_mias_classes, 
        'input_dataset':mias_input_dataset, 'testing_dataset':mias_input_dataset,
        'img_size':mammo_img_size,  'num_channels':mammo_num_channels,
        'train_split':train_split, 'test_split':test_split, 'eval_split':eval_split,
        'train_using':train_using_mammo_mias, 'K':K
        }

print('this is the data parameters we will use for the model training process ',data_params)

#this is the data parameters we will use for the model training process  {'num_classes': 5, 'class_names': {0: 'N', 1: 'BC', 2: 'BM', 3: 'CALC', 4: 'M'}, 'input_dataset': './data/mias/', 'testing_dataset': './data/mias/', 'img_size': {'width': 299, 'height': 299}, 'num_channels': 1 (meaning one grayscale), 'train_split': 0.75, 'test_split': 0.15, 'eval_split': 0.1, 'train_using': 'mias', 'K': <module 'tensorflow.python.keras.backend' from '/home/justice/anaconda3/lib/python3.12/site-packages/tensorflow/python/keras/backend.py'>}

print('now that we we have our images and splits lets process our input data')
mammo_input_mias=InputProcessor(data_params=data_params) #this is actually a class which requires the given structure to construct what it takes to construct the model 

print('lets get our training data and splits from the image in the form of array images')
mammo_input_mias.get_train_input() #why: this function is used to fetch the image data, convert it to arrays, convert it  GRAYSCALE, standarize it to fall with the interval of 0 to 1 and also use percentages define to construct the train, eval, and test data. but does not return anything it just does an internal job. thus a void method. to update the properties of the class Object made by the InputProcessor.

#Todo 1:i have commented this out because am just using mias dataset.
# repeat same for ddsm data
#Load mammo ddsm datasets
# data_params['num_classes']=len(mammo_ddsm_classes)
# data_params['class_names']=mammo_ddsm_classes 
# data_params['input_dataset']=ddsm_input_dataset 
# data_params['testing_dataset']=ddsm_input_dataset            
# data_params['train_using']=train_using_mammo_ddsm
# mammo_input_ddsm=InputProcessor(data_params=data_params)
# #mammo_input_ddsm.get_train_input()

# if isCombineMammoDatasets:
#     x, y=mammo_input_mias.get_train_data()
#     mammo_input_ddsm.set_training_data(x, y)
#     x, y=mammo_input_mias.get_eval_data()
#     mammo_input_ddsm.set_eval_data(x, y)
#     x, y=mammo_input_mias.get_test_data()
#     mammo_input_ddsm.set_test_data(x, y)



#End of data reading and processing

    #Train MammoCNN
'''
Run1:optimizers[6], learning_rates[5]
Run2:optimizers[5], learning_rates[5]
Run3:optimizers[1], learning_rates[5]
Run4:optimizers[1], learning_rates[4] 
'''
#%%

opt, lr=optimizers[1], learning_rates[4]
print('this is the optimization algorithm and learning rates we want to use to fetch the features of the images ',opt,lr)

mamocnn_paras={
    'learning_rates':lr,  'optimizers':opt, 'activations':activations[1], 
    'pooling':pooling[1], 'regularizers':regularizers[1], 'fcactivations':fcactivations[0], 
    'lossfunc':lossfunc[0], 'cnn_epoch':cnn_epoch, 'batch_size':batch_size,        
    'log_mode':log_mode, 'models_path':models_path, 'model_filename':mammo_model_filename,
    'save_results_dir':save_results_dir, 'show':show, 'K':K, 
    'checkpoint_epoch':40, 'fromCheckpoint':False, 'train_model': True,
    
    'num_classes':len(mammo_mias_classes), 'class_names':mammo_mias_classes, 
    'img_size':mammo_img_size, 'num_channels':mammo_num_channels, 
    'input_source':mammo_input_mias,
    "experiment":"mammocnn_eph{}_optz{}_lr{}".format(cnn_epoch,opt,lr),
    "filename":"mammocnn_eph{}_optz{}_lr{}".format(cnn_epoch,opt,lr),
    'optimzed_result_file':"optimized_mammo_eph{}_{}_lr{}".format(cnn_epoch,opt,lr),
    
    'checkpoint_path':mammo_checkpoint_path, "save_multimodal_results_dir":save_mammo_results_dir,         
}

print('this is the parameters we will use for by the cnn to fetch features of various images from the image_array data associated with each  split ',mamocnn_paras)
#this is the parameters we will use for the model training process  {'learning_rates': 0.0001, 'optimizers': 'Adam', 'activations': 'leakyrelu', 'pooling': 'Avg', 'regularizers': 'L2', 'fcactivations': 'softmax', 'lossfunc': 'categorical_crossentropy', 'cnn_epoch': 40, 'batch_size': 64, 'log_mode': 1, 'models_path': './outputs/models/', 'model_filename': 'trainedmodelmammo', 'save_results_dir': './outputs/results/', 'show': 1, 'K': <module 'tensorflow.python.keras.backend' from '/home/justice/anaconda3/lib/python3.12/site-packages/tensorflow/python/keras/backend.py'>, 'checkpoint_epoch': 40, 'fromCheckpoint': True, 'train_model': True, 'num_classes': 5, 'class_names': {0: 'N', 1: 'BC', 2: 'BM', 3: 'CALC', 4: 'M'}, 'img_size': {'width': 299, 'height': 299}, 'num_channels': 1, 'input_source': <cnn.inputpipeline.InputProcessor object at 0x79d3feb745c0>, 'experiment': 'mammocnn_eph40_optzAdam_lr0.0001', 'filename': 'mammocnn_eph40_optzAdam_lr0.0001', 'optimzed_result_file': 'optimized_mammo_eph40_Adam_lr0.0001', 'checkpoint_path': './outputs/checkpoints/mammo/', 'save_multimodal_results_dir': './outputs/results/training/mammo/'}


mcnn=MammoCNN(mamocnn_paras=mamocnn_paras)
mcnn.build_architecture()
xtrain, ytrain=mammo_input_mias.get_training_data()
xeval, yeval=mammo_input_mias.get_eval_data()
x_test, y_test=mammo_input_mias.get_test_data()
pr=ProcessResult(params=mamocnn_paras)


if mamocnn_paras['train_model']:
    if mamocnn_paras['fromCheckpoint']:
        training_outcome=mcnn.load_trained_model(xtrain, ytrain, xeval, yeval)
    else:
        training_outcome=mcnn.trainmodel(mcnn.model, xtrain, ytrain, xeval, yeval)
    
        
    #Predict with HistoCNN model
    # prediction=mcnn.predict()
    # pr.save_results(training_outcome, prediction)
    
    #Extract and optimize features from HistoCNN model
    mcnn.extract_vector_features(mcnn.model, xtrain, ytrain, xeval, yeval, 'pool_4')


# import numpy as np
# import matplotlib.pyplot as plt
# train_features = np.load("./outputs/checkpoints/mammo/train_features.npy")
# print(train_features)
# print("Shape:", train_features.shape)  # e.g., (1000, 8192)
# plt.figure(figsize=(10, 4))
# plt.plot(train_features[0])  # Plot first sample
# plt.title("Feature Vector")
# plt.show()


#%%

lb, ub, x, y, tx, ty=mcnn.optimize_vector_features(xtrain, ytrain, xeval, yeval)
#Send features to binary optimizer for optimization process
save_optimized_path=mamocnn_paras['checkpoint_path']
#We are using the BEOSA method for the optimization, hence the selection of first binaryalgorithms[0]
method = binaryalgorithms[0] #for method in binaryalgorithms:
new_x_train, y_train=feature_optimizer(save_optimized_path, method, mcnn.model, x, y, tx, ty, modelrates, lb, ub, pr, mamocnn_paras['num_classes'], 'mammo', metrics_dir)   

# %%
