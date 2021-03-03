# Smaller VGG (6 conv, 2 dense) for detection of speech activity with still images. utilises the VGG model but adapted to run within
# the computional resources available. Uses dropout and BN which has proven to improve perfomance
# https://arxiv.org/abs/1409.1556

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.layers import Activation, Flatten, Dense,BatchNormalization,Dropout
from tensorflow.keras import backend as K
import matplotlib.pyplot as plt  

img_width=100
img_height =100
train_data_dir = '/train'
val_data_dir = '/val'
test_data_dir='/test'


class VGGCNN():
    # Initalising variables passed
    def __init__(self, conv_size_1, conv_size_2, conv_size_3,conv_size_4, filter_size, epochs, batch_size, dropout_rate):
        self.conv_size_1=conv_size_1
        self.conv_size_2=conv_size_2
        self.conv_size_3=conv_size_3
        self.conv_size_4=conv_size_4
        self.filter_size=filter_size
        self.epochs=epochs
        self.batch_size=batch_size
        self.dropout_rate=dropout_rate
        # Setting the order of channels
        if K.image_data_format() == 'channels_first':
            self.input_shape = (1, img_width, img_height)
        else:
            self.input_shape = (img_width, img_height, 1)
        #Flow of program   
        train_generator=self.preProcessing(train_data_dir)
        validation_generator=self.preProcessing(val_data_dir)
        test_generator=self.preProcessing(test_data_dir)
        model=self.defineModel()
        history=self.executeModel(model, train_generator, validation_generator,test_generator)
        self.plotFigure(history)  
        
    def preProcessing(self, data_dir):
        # Pre-processing of images with Image Data Generator https://keras.io/api/preprocessing/image/
        data_rescale=ImageDataGenerator(rescale=1./ 255)
        data_generator = data_rescale.flow_from_directory(
            data_dir,
            target_size=(img_height,img_width),
            color_mode='grayscale',
            batch_size=self.batch_size,
            class_mode='categorical',
            shuffle=True)
        return data_generator
            
    def defineModel(self):
        # Model definition. Uses BN before activiation as recommended in the BN paper. Addition of dropout showed better perfomance regarding loss
        # Dropout marginally affects accuracy
        model = Sequential()
        model.add(Conv2D(self.conv_size_1,self.filter_size,strides=2,padding='same',input_shape=self.input_shape))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2),strides=2))
        model.add(Dropout(self.dropout_rate))
        
        model.add(Conv2D(self.conv_size_2, self.filter_size,padding='same'))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2),strides=2))
        model.add(Dropout(self.dropout_rate))
        
        model.add(Conv2D(self.conv_size_3,self.filter_size, padding='same'))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(Conv2D(self.conv_size_3,self.filter_size, padding='same'))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2),strides=2))
        model.add(Dropout(self.dropout_rate))
        
        model.add(Conv2D(self.conv_size_4, self.filter_size, padding='same'))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(Conv2D(self.conv_size_4, self.filter_size, padding='same'))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2),strides=2))
        model.add(Dropout(self.dropout_rate))
        
        model.add(Flatten())
        model.add(Dense(1024))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(Dropout(self.dropout_rate))
        
        model.add(Dense(1024))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(Dropout(self.dropout_rate))
        
        model.add(Dense(2))
        model.add(Activation('softmax'))
        model.compile(loss='categorical_crossentropy',
                      optimizer='adam',
                      metrics=['accuracy'])
        model.summary()
        
        return model
    
    def executeModel(self, model, train_generator,validation_generator, test_generator):
        # Compiles and runs the model. model is trained on train and val sets with test set used to evaluate the model.
        history=model.fit_generator(
            train_generator,
            steps_per_epoch=len(train_generator),
            epochs=self.epochs,
            validation_data=validation_generator,
            validation_steps=len(validation_generator))
        scores=model.evaluate_generator(test_generator,len(test_generator))
        print ("scores",scores)
            #model.save('VGGCNN.h5')
        return history
            
    def plotFigure(self, history):
        # Plots a graph of the model's perfomance.
        plt.figure()  
        plt.plot(history.history['acc'])  
        plt.plot(history.history['val_acc'])
        plt.plot(history.history['loss']) 
        plt.plot(history.history['val_loss'])  
        plt.title('Training Loss and Accuracy')  
        plt.ylabel('Loss/Accuracy')  
        plt.xlabel('epoch')  
        plt.legend(['t acc', 'v acc','t loss','v loss'], loc='best')  
        plt.show() 
if __name__ == '__main__':
    # Creats VGG model with defined hyperparameters
    VGGCNN1=VGGCNN(conv_size_1=0,conv_size_2=0,conv_size_3=0,conv_size_4=0,filter_size=(0,0),epochs=0,batch_size=0, dropout_rate=0)  
        