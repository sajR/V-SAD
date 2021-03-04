#3 layer(3 CONVs) CNN for Visual-Speech Activity Detection (SAD) from literature. Adapted with different kernal and filter sizes, dropout and BN 
#model tested to compare perfomance and time against smaller VGG

from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Flatten, Dense,Dropout
from keras import backend as K
import matplotlib.pyplot as plt  
import os
from keras.preprocessing.image import image
from keras.preprocessing.image import img_to_array
import numpy as np




img_width=100
img_height =100
train_data_dir = '/train'
validation_data_dir = '/val'
test_data_dir='/test'

    
class CNN():
    # Initalising variables passed
    def __init__(self, conv_size_1,conv_size_2,conv_size_3, filter_size,epochs, batch_size, dropout_rate_1, dropout_rate_2):
        self.conv_size_1=conv_size_1
        self.conv_size_2=conv_size_2
        self.conv_size_3=conv_size_3
        self.filter_size=filter_size
        self.epochs=epochs
        self.batch_size=batch_size
        self.dropout_rate_1=dropout_rate_1
        self.dropout_rate_2=dropout_rate_2
        #Sets the order of the channels depending on keras version
        if K.image_data_format() == 'channels_first':
            self.input_shape = (1, img_width, img_height)
        else:
            self.input_shape = (img_width, img_height, 1) 
        
        
        train_generator=CNN1.preProcessing(train_data_dir)
        validation_generator=self.preProcessing(validation_data_dir)
        test_generator=self.preProcessing(test_data_dir)
        model=self.defineModel()
        history=self.executeModel(model, train_generator, validation_generator,test_generator)
        self.plotFigure(history)    
        self.predict(model, test_data_dir)
    def preProcessing(self, data_dir):
        # Pre-processing of images with Image Data Generator https://keras.io/api/preprocessing/image/
        data_rescale = ImageDataGenerator(rescale=1./ 255)
        data_generator = data_rescale.flow_from_directory(
            data_dir,
            target_size=(img_height,img_width),
            color_mode='grayscale',
            batch_size=self.batch_size,
            class_mode='categorical',
            shuffle=True)
        return data_generator
        
    def defineModel(self):
        # Model definition. Uses 2 dropout rates - this showed to be more effective in perfomance.
        
        model = Sequential()
        model.add(Conv2D(self.conv_size_1,self.filter_size,padding='same',activation='relu',input_shape=self.input_shape))
        model.add(MaxPooling2D(pool_size=(2, 2),strides=2))
        model.add(Dropout(self.dropout_rate_1))
        
        model.add(Conv2D(self.conv_size_2, self.filter_size,padding='same',activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2),strides=2))
        model.add(Dropout(self.dropout_rate_1))
        
        model.add(Conv2D(self.conv_size_3, self.filter_size, padding='same',activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2),strides=2))
        model.add(Dropout(self.dropout_rate_1))
        
        model.add(Flatten())
        model.add(Dense(1024,activation='relu'))
        model.add(Dropout(self.dropout_rate_2))
        model.add(Dense(2))
        model.add(Activation('softmax'))
        model.compile(loss='categorical_crossentropy',
                      optimizer='adam',
                      metrics=['accuracy'])
        model.summary()
        return model
    
    def executeModel(self,model,train_generator,validation_generator, test_generator):
        # Compiles and runs the model. model is trained on train and val sets with test set used to evaluate the model.
        history=model.fit_generator(
            train_generator,
            steps_per_epoch=len(train_generator),
            epochs=self.epochs,
            validation_data=validation_generator,
            validation_steps=len(validation_generator))
        scores=model.evaluate_generator(test_generator,len(test_generator))
        print ("scores",scores)
        # model.save('CNN.h5')
        return history
    
    def plotFigure(self,history):
        # Plots a graph of the model's perfomance.
        plt.figure()
        plt.plot(history.history['acc'])  
        plt.plot(history.history['val_acc'])
        plt.plot(history.history['loss']) 
        plt.plot(history.history['val_loss'])  
        plt.title('Training Loss and Accuracy')  
        plt.ylabel('Loss/Accuracy')  
        plt.xlabel('epoch')  
        plt.legend(['t acc', 'v acc','t loss','v loss'], loc='upper left')  
        plt.show()  
    def getPredictions(self,model,test_dir):
        # Obtains predictions from a test directory
        count=0
        imageFormat='.png'
        fileList=[os.path.join(test_dir,f) for f in os.listdir(test_dir) if f.endswith(imageFormat)]
        for imagename in fileList:
            img = image.load_img(imagename, target_size=(img_width, img_height),color_mode="grayscale")
            img = image.img_to_array(img)
            img=img/255
            img = np.expand_dims(img, axis=0)
            classes=model.predict_classes(img)
            count=count+1
            print (classes)
        
        
if __name__ == '__main__':
    # Creats CNN model with defined hyperparameters
    CNN1=CNN(conv_size_1=0,conv_size_2=0,conv_size_3=0,filter_size=(0,0), epochs=0,batch_size=0, dropout_rate_1=0, dropout_rate_2=0)
   