# The code below allows to choose between encoder/encoder-decoder and epochs, batch size, sequence size and number of GRU Units
# See Embedding.py - obtains the features from Xception and saves it as CSV utilised as input in this case.
#https://towardsdatascience.com/step-by-step-understanding-lstm-autoencoder-layers-ffab055b6352

import numpy as np
import os
import csv
from tensorflow.keras.models import Sequential
from keras.layers import Dense,  BatchNormalization,CuDNNGRU,TimeDistributed,RepeatVector
from keras.utils import to_categorical
from sklearn.utils import shuffle
import matplotlib.pyplot as plt 
from keras.applications.xception import preprocess_input,Xception
from keras.preprocessing import image

# Directories for train, val and test CSVs and image path - used for predictions
train_dir="/Xception/TrainXception.csv"
val_dir="/Xception/ValXception.csv"
test_dir="/Xception/TestXception.csv"
image_dir="/test images"

class CNNRNN(): 
    #Xception embedding is at 2048 by default, labels in this case is 2 - speech/non-speech
    embedding_length=2048
    label_length=1
    
    def __init__(self,epochs,batch_size,seq_len, model_type, GRU_units):
        # Initlisasing variables passed
        self.epoch=epochs
        self.batch_size=batch_size
        self.seq_len=seq_len
        self.input_shape=(seq_len,self.embedding_length)
        self.model_type=model_type
        self.GRU_units=GRU_units
        
        print (self.model_type)  
          
        #preprocess and generates labels (y) and data (x) for train, val and test sets
        x_train,y_train=self.preprocessing(train_dir)
        x_val,y_val=self.preprocessing(val_dir)
        x_test,y_test=self.preprocessing(test_dir)
        if self.model_type=='Encoder':
            model=self.encoder()
        else:
            model=self.encoderDecoder()
        history=self.executeModel(model, x_train,y_train,x_test,y_test,x_val,y_val)
        self.plotFigure(history) 
        self.getPredictions(model)
                  
    def preprocessing(self,path):
        # Pre-processing required for the network. reads the images and labels into numpy array. once read, each array is placed into a sequence based on Sequence size set.
        # The sequences are shuffled avoiding the network memorise the data. 
        # Labels are one-hot coded - 0,1
        label,data=self.getData(path)
        x=self.generateSeq(data,self.embedding_length)
        y=self.generateSeq(label,self.label_length)
        if self.model_type=='Encoder':
            y=y[:,-1]
        x,y=shuffle(x,y)
        y=to_categorical(y)
        return x,y
    
    def getData(self,path):
        #reads the images into an arrays - data and labels.
        counter=0
        data=[]
        label=[]
        with open(path) as file:
            reader = csv.reader(file)
            for i in reader:
                data.append(i[1:])
                label.append(i[0])
                counter=counter+1
                if counter>500:
                    break
        return label,data
    
    def generateSeq(self,data,length): 
        # Generates sequences of images based on sequence size. adds (padding) zeroes at the begining so the data is introduced as 0,0,0,0,1 | 0,0,0,1,1| 0,0,0,1,1,1 where 0=non-speech 1=speech
        temp_data = data
        num_of_seqs = len(temp_data)
        data = np.zeros((num_of_seqs,self.seq_len,length), dtype='float32')
        count = 0
        for j in range(0,len(temp_data)):
            if j+1 < self.seq_len :
                for x in range(0,j+1):
                    # data[count][x] = arr[x] # padding zeros at the end
                    data[count][self.seq_len+x-j-1] = temp_data[x] # padding zeros at the beginning
            if j+1 >= self.seq_len:
                for x in range(0,self.seq_len):
                    data[count][x] = temp_data[j+x+1-self.seq_len]
            count += 1
        return data
    def encoder(self):
        # Defines the encoder model with single layer of GRU and defined number of GRU units. BN is added for improved training perfomance and quickness
        # Cuddn GRU provides much faster perfomance but does not provide dropout within.
        model = Sequential()
        model.add(CuDNNGRU(self.GRU_units,input_shape=self.input_shape))
        model.add(BatchNormalization())
        model.add(Dense(2,activation='softmax'))
        model.summary()
        return model
    def encoderDecoder(self):
        # Defines the encoder decoder with single layers encoder and decoder. Defined number of GRUs are repeated based on the sequence size
        model=Sequential()
        #encoder
        model.add(CuDNNGRU(self.GRU_units,return_sequences=False,input_shape=self.input_shape))
        model.add(BatchNormalization())
        # Repeat vecotr allows to repeat the input based on sequence size at each timestep
        model.add(RepeatVector(self.seq_len))
        # Decoder
        model.add(CuDNNGRU(self.GRU_units,return_sequences=True))
        model.add(BatchNormalization())
        # Time distrubuted allows for classifcation at each timestep
        model.add(TimeDistributed(Dense(2,activation="softmax")))
        model.summary()
        return model
        
    def executeModel(self,model,x_train,y_train,x_test,y_test,x_val,y_val):
        # Compiles and runs the model. model is trained on train and val sets with test set used to evaluate the model.
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        history=model.fit(x_train,y_train, epochs=self.epochs, batch_size=self.batch_size,validation_data=(x_val,y_val),verbose=1)
        testScores=model.evaluate(x_test,y_test)  
        print ("test scores", testScores)
        #model.save('CNNRNNS.h5')
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
        plt.legend(['t acc', 'v acc','t loss','v loss'], loc='best')  
        plt.show()  
        #plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)
        
    def getPredictions(self,model):
        # Obtains predictions from a seperate image directory for the model just trained.
        # Embeddings from Xception for the images are obtained, processesed, and used to predict with the model trained.     
        Xception_model=Xception(weights='imagenet', include_top=False, pooling='avg')
        Xception_features=self.getFeature(Xception_model)
        seq_features=self.generateSeq(Xception_features,self.embedding_length)
        outputs=model.predict_classes(seq_features)
        print (outputs)
               
    def getFeature(self,Xception_model):
        #Obtains features for every image in the directory and returns a array of image features.
        features=[]
        image_format=".png"
        fileList=[os.path.join(image_dir,f) for f in os.listdir(image_dir) if f.endswith(image_format)]
        for imageName in fileList:
            img = image.load_img(imageName,target_size=(100,100))
            if img is not None:
                # formats the image as array and pre-processes the array based on the model.     
                image_array=image.img_to_array(img)
                # converts image array to numpy array and alters the shape of the array, expected by the model
                image_array=np.expand_dims(image_array,axis=0)
                # pre-proces the image array based on the model
                image_array=preprocess_input(image_array)
                #processed array is then used to predict against Xception
                img_feature=Xception_model.predict(image_array)
                features.append(img_feature)
        return features
        
if __name__ == '__main__':
    # Creates CNNRNN with defined hyperparameters. model_type = encoder/encoder-decoder
    CNNRNN1=CNNRNN(epochs=0, batch_size=0, seq_len=0, model_type='Encoder Decoder', GRU_units= 0)
    
