# Transfer learning with Xception. Uses the Xception model to classify SAD (Speech Activity Detection)
# The model in this case is tweaked to match the problem, this involves changing second to last FC layer to 1024 and classification layer to 2 
# The model expects images in sets (train, val, test). Each set includes folders of classes (speech, non-speech)
from keras import applications
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model 
from keras.layers import Flatten, Dense
import matplotlib.pyplot as plt
import numpy as np
import os
from keras.preprocessing.image import image



img_width, img_height = 100, 100
train_data_dir = "/Train"
validation_data_dir = "/validation"
test_data_dir="/Test"

class transferLearning():
    # Initalising variables passed
    def __init__(self, epochs, batch_size):
        self.epochs=epochs
        self.batch_size=batch_size
        
        
        train_generator=self.preProcessing(train_data_dir)
        validation_generator=self.preProcessing(validation_data_dir)
        test_generator=self.preProcessing(validation_data_dir)
        model=self.getModel()
        history=self.executeModel(model, train_generator, validation_generator,test_generator)
        self.plotFigure(history)
        self.getPredictions(model,test_data_dir)
        
        
        
    def getModel(self):
        # Gets Xception model. The classification layer of the model is sliced off by setting include_top=False. New classification layer is added as the model has typically 1000 classes
        base_model = applications.Xception(weights = "imagenet", include_top=False,input_shape = (img_width, img_height, 3))
        x = base_model.output
        x = Flatten()(x)
        x = Dense(1024, activation="relu")(x)
        predictions = Dense(2, activation="softmax")(x)
        for layer in base_model.layers:
            layer.trainable=False
        model = Model(input = base_model.input, output = predictions)
        model.summary()   
        return model
    
        
    def preProcessing(self, data_dir):
        # Pre-processing of images with Image Data Generator https://keras.io/api/preprocessing/image/
        data_rescale=ImageDataGenerator(rescale=1./255)
        data_generator=data_rescale.flow_from_directory(
            data_dir,
            target_size=(img_height,img_width),
            batch_size=self.batch_size,
            class_mode='categorical',
            shuffle=True)
        return data_generator
    
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
        plt.legend(['t ac', 'v ac','t loss','v loss'], loc='upper left')  
        plt.show()  
        
    def executeModel(self,model, train_generator,validation_generator,test_generator):
        # Compiles and runs the model. model is trained on train and val with test used to evaluate the model.
        model.compile(loss = "categorical_crossentropy", optimizer = "adam", metrics=["accuracy"])
        history=model.fit_generator(
            train_generator,
            steps_per_epoch=len(train_generator),
            epochs=self.epochs,
            validation_data=validation_generator,
            validation_steps=len(validation_generator))
        scores=model.evaluate_generator(validation_generator,len(validation_generator))
        print ("scores",scores)
        #model.save('transferLearning.h5')
        return history
    
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
    # Creats model with defined hyperparameters
    transferLearning1=transferLearning(epochs=0,batch_size=0)
        
        
        
    










