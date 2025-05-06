import tensorflow as tf
from tensorflow import keras
from keras import backend as K  #from keras import backend as K
from numpy import array
# from cnn.root.rootcnn import RootCNN
#from keras.utils import to_categorical
from keras.utils import to_categorical
from time import time
import os
import numpy as np
import cv2
import random

class InputProcessor(object):
    REGULARIZER_RATES=0.0002

    # this is the constructor function 
    def __init__(self, data_params=None):
        self.num_classes=data_params["num_classes"]
        self.classes=data_params["class_names"]
        self.input_dataset=data_params['input_dataset'] #has localtion to the files 
        self.testing_dataset=data_params['testing_dataset']
        self.img_width=data_params["img_size"]["width"] 
        self.img_height=data_params["img_size"]["height"]
        self.num_channels=data_params["num_channels"]
        self.train_using=data_params["train_using"]
        self.train_split=data_params["train_split"]
        self.test_split=data_params["test_split"]
        self.eval_split=data_params["eval_split"]
        self.K=data_params["K"]
        self.x_train, self.y_train=[], []
        self.x_eval, self.y_eval=[], [],
        self.x_test, self.y_test=[], [],
        self.img_ids_train, self.img_ids_eval, self.img_ids_test=[], [], []

    #get train input
    #this function is called  by the get train_input function 
    def init_imgs(self, nFiles):
        dim=(self.img_width, self.img_height, self.num_channels) #gives us the diminitions of the various various images in the data set(like width,height,channel if 1 or 3)
        img_data_array = np.empty((nFiles, self.img_width, self.img_height, self.num_channels)) #this is the array that will hold the images(array of arrays that represent the various images)
        self.x_train, self.y_train=np.empty((0, self.img_width, self.img_height, self.num_channels)), np.empty((0, )) # this sets  array that will hold the training images(array of arrays that represent the various images)
        self.x_eval, self.y_eval=np.empty((0, self.img_width, self.img_height, self.num_channels)), np.empty((0, )) # this sets array that will hold the evaluation images(array of arrays that represent the various images)
        self.x_test, self.y_test=np.empty((0, self.img_width, self.img_height, self.num_channels)), np.empty((0, )) # this sets  array that will hold the testing images(array of arrays that represent the various images)
        return img_data_array, dim
    

    def get_train_input(self):
        nFiles=len(os.listdir(self.input_dataset)) #show the number of files in the database
        img_ids=[]
        num_imgs_per_label = {}

        img_data_array, dim=self.init_imgs(nFiles)   # this is what will execute  we use this function to 
        # print(img_data_array, dim) 
        class_name = np.empty((nFiles, ))            
        num_imgs_per_label={'N': 0, 'BC': 0, 'BM': 0, 'CALC': 0, 'M': 0}
        n =0
        prev_label=''
        keys=list(self.classes.keys()) #this is the list of keys in the dictionary that contains the class names 
        values=list(self.classes.values())  #this is the list of values in the dictionary that contains the class names
        # print(keys,values)
        
        for file in os.listdir(self.input_dataset):
            if file.endswith(('jpg', 'png')):
                image_path= os.path.join(self.input_dataset, file)
                image= cv2.imread( image_path, cv2.COLOR_BGR2GRAY) #image= np.array(Image.open(image_path)) COLOR_BGR2GRAY
                image=cv2.resize(image, (self.img_width, self.img_height),interpolation = cv2.INTER_AREA) #resize the image to the specified width and height

                zrows, zcols= image.shape[0], image.shape[1]
                image=np.array(image) # collect the image to an array 
                image = image.astype('float32') # convert the image to a float32
                image=image.reshape(dim) # reshape the image to the specified dimention
                image /= 255 # reduce the normalize the image to range form 0 to 1

                label=(file.split('_')[-1]).split('.')[0]   
                tmpf=label #we try to extract a label form the file name. by splitting the first part of ht file name by _ select the first part and spit by . and pick the first part.

                #am doing this because i dont have the right data set here.
                rand_value = random.random() * 100

                if rand_value < 20:
                    tmpf = 'N'
                elif rand_value < 40:
                    tmpf = 'BC'
                elif rand_value < 60:
                    tmpf = 'BM'
                elif rand_value < 80:
                    tmpf = 'CALC'
                else:
                    tmpf = 'M'
                #i am ending my random values here 
                
                value_at_index = values.index(tmpf) #keys[values.index(label)]
                label = value_at_index                    
                img_data_array[n, :, :, :] = image # add this to the image data 
                class_name[n] = label # store its corresponding label for it 
                num_imgs_per_label[tmpf]=num_imgs_per_label[tmpf] + 1  # we try to get statistics for the various images labels 
                n=n+1
        
        class_name=array(class_name)
        class_name = to_categorical(class_name, self.num_classes)
        
        # we set the various arrays to contain  our spit data and set the various labels to contain the labels of the images while me make this to be a categorical value
        self.y_train=array(self.y_train)
        self.y_train = to_categorical(self.y_train, self.num_classes)
        
        self.y_eval=array(self.y_eval) 
        self.y_eval = to_categorical(self.y_eval, self.num_classes)
        
        self.y_test=array(self.y_test)
        self.y_test = to_categorical(self.y_test, self.num_classes)
        
       
        clabel_count=0
        
        # based on each label we add the various training, test, and eval data to the total train, test, and eval data set 
        for clabel, count in num_imgs_per_label.items():
            train_split = int(self.train_split * count) 
            test_split = int(self.test_split * count) 
            eval_split = int(self.eval_split * count) 
            
            print(clabel, ":", count, ":", train_split, ":", eval_split, ":", test_split)#show the various categories and the number of images in it and the number of images in the train test and eval set

            train_stop=(clabel_count+train_split)
            self.x_train=np.concatenate((self.x_train,img_data_array[clabel_count:train_stop]))
            self.y_train=np.concatenate((self.y_train,class_name[clabel_count:train_stop]))
            
            eval_start=train_stop+1
            eval_end=(eval_start+eval_split)
            self.x_eval=np.concatenate((self.x_eval, img_data_array[eval_start:eval_end])) 
            self.y_eval=np.concatenate((self.y_eval, class_name[eval_start:eval_end]))       
            
            test_start=eval_end+1
            test_end=(test_start+test_split)
            self.x_test=np.concatenate((self.x_test, img_data_array[test_start:test_end]))  
            self.y_test=np.concatenate((self.y_test, class_name[test_start:test_end]))
            
            # print(clabel_count, "-", train_stop, ",", eval_start, "-", eval_end, ",", test_start, "-", test_end)
            # clabel_count=clabel_count+count

        print('finally this is what we have as our training, eval and test')
        print(len(self.x_train))
        print(len(self.x_eval))
        print(len(self.x_test))

    
    #this functions are used to get train , eval, test data
    def get_training_data(self):
        return self.x_train, self.y_train
    
    def get_eval_data(self):
        return self.x_eval, self.y_eval
    
    def get_test_data(self):
        return self.x_test, self.y_test
    
    #this is used to set or add on to the training, test or eval data
    def set_training_data(self, x, y):
         self.x_train, self.y_train=np.concatenate(self.x_train,x), np.concatenate(self.y_train,x)
         return None
    
    def set_eval_data(self, x, y):
        self.x_eval, self.y_eval=np.concatenate(self.x_eval,x), np.concatenate(self.y_eval,y)
        return None
    
    def set_test_data(self, x, y):
        self.x_test, self.y_test=np.concatenate(self.x_test,x), np.concatenate(self.y_test,y)
        return None

