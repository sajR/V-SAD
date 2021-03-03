# Code below creates a CRNN - Convolutional Recurrent Neural Network with defined hyperparamters and choice of encoder/encoder-decoder
# Encoder has a single layer of RNN and classifies last image of sequence
# Encoder-decoder has single layer of RNN for encoding and decoding. Classifies all images in the sequence Seq2Seq
#https://towardsdatascience.com/step-by-step-understanding-lstm-autoencoder-layers-ffab055b6352

import os 
import cv2
import numpy as np
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Flatten, Dense,BatchNormalization,Dropout,CuDNNGRU,RepeatVector
from keras.layers.wrappers import TimeDistributed
from keras import backend as K
import matplotlib.pyplot as plt 
from keras.utils import to_categorical
from sklearn.utils import shuffle
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix

# Directories for train, val and test 
train_dir="/data/train"
val_dir="/data/val"
test_dir="/data/test"


class CRNN():
    #initalises the model with hyperparamters below
    def __init__(self, seq_len,epochs,model_type, conv_1, conv_2,conv_3, filter_size, dropout_rate, GRU_units):
        self.model_type=model_type
        print (self.model_type)
        self.seq_len=seq_len
        self.epochs=epochs
        self.conv_1=conv_1
        self.conv_2=conv_2
        self.conv_3=conv_3
        self.filter_size=filter_size
        self.dropout_rate=dropout_rate
        self.GRU_units=GRU_units
      
        #Flow of the program
        x_train,y_train=self.getData(train_dir)
        x_val,y_val=self.getData(val_dir)
        x_test,y_test=self.getData(test_dir)
        x_train,x_val,x_test=self.sortChannels(x_train, x_val, x_test)
        x_train,y_train=self.preProcess(x_train,y_train)
        x_val,y_val=self.preProcess(x_val, y_val)
        x_test,y_test=self.preProcess(x_test,y_test)
        model=self.defineBaseModel()
        if self.model_type=='Encoder':
            model=self.encoder(model)
        else:
            model=self.encoderDecoder(model)
        history=self.executeModel(model,x_train,y_train,x_val,y_val,x_test,y_test)
        self.plotFigure(history)
        self.getPredictions(model, x_test, y_test)
        
    def getData(self,pathName):
        #reads images to arrays - data and labels
        num_samples=len(os.listdir(pathName))
        img_rows=100
        img_cols=100
        data=np.zeros((num_samples,img_rows,img_cols))
        labels=[]
        counter=0
        for fileName in os.listdir(pathName):
            image=cv2.imread(pathName+"/"+fileName)
            data[counter]=image[:,:,0]
            if "nospeech" in fileName:
                labels.append("0")
            elif "speech" in fileName:
                labels.append("1")
            else:
                print ("problem")
            counter+=1
          
        return data,np.array(labels)
    def sortChannels(self, x_train,x_val,x_test):
        # Sets the order of the channels depending on keras version
        if K.image_data_format() == 'channels_first':
            x_train=x_train.reshape(x_train.shape[0],1,100,100)
            x_test=x_test.reshape(x_test.shape[0],1,100,100)
            x_val=x_val.reshape(x_val.shape[0],1,100,100)
            
        else:
            x_train=x_train.reshape(x_train.shape[0],100,100,1)
            x_test=x_test.reshape(x_test.shape[0],100,100,1)
            x_val=x_val.reshape(x_val.shape[0],100,100,1)
        return x_train,x_val,x_test    
    def preProcess(self,x,y):
        #Processes x (data) and y(labels) and formats it for the model. 
        # Converts the array to float
        x=x.astype('float32')
        # Normalizes the value to 0,1
        x/=255
        # Creates one hot encoding i.e. 0,1 for labels
        y=to_categorical(y)
        # Data is sorted into sequences
        x=self.generateSeq(x,"x")
        y=self.generateSeq(y,"y")
        
        if self.model_type=='Encoder':
        # Encoder classifies the last image of the sequence thus, only the last label of the sequence is kept
            y=y[:,-1]
        x,y=shuffle(x,y)
        return x,y
    def generateSeq(self,data,dataType):
        # Generates sequences of images based on sequence size. adds (padding) zeroes at the begining so the data is introduced as 0,0,0,0,1 | 0,0,0,1,1| 0,0,0,1,1,1 where 0=non-speech 1=speech
        temp_data = data
        num_of_seqs = len(temp_data)
        if dataType=="x":
            data = np.zeros((num_of_seqs,self.seq_len,100,100,1), dtype='float32')
        elif dataType=="y":
            data = np.zeros((num_of_seqs,self.seq_len,2), dtype='float32')
        count = 0
        for j in range(0,len(temp_data)):
            if j+1 < self.seq_len :
                for x in range(0,j+1):
                    # data[count][x] = arr[x] # padding zeros at the end
                    data[count][self.seq_len+x-j-1] = np.array(temp_data[x]) # padding zeros at the beginning
            if j+1 >= self.seq_len:
                for x in range(0,self.seq_len):
                    data[count][x] = np.array(temp_data[j+x+1-self.seq_len])
            count += 1
        return data
    def defineBaseModel(self):
        # Defines the base model (CNN). Uses time distrubted to allow obtaining features for each timestep
        model = Sequential()
        model.add(TimeDistributed(Conv2D(self.conv_1, self.filter_size,strides=(2,2),padding='same',activation='relu'),input_shape=(self.seq_len,100,100,1)))
        model.add(TimeDistributed(MaxPooling2D((2, 2), strides=(2, 2))))
        model.add(TimeDistributed(Dropout(self.dropout_rate)))
        
        model.add(TimeDistributed(Conv2D(self.conv_2, self.filter_size,padding='same',activation='relu')))
        model.add(TimeDistributed(MaxPooling2D((2, 2), strides=(2, 2))))
        model.add(TimeDistributed(Dropout(self.dropout_rate)))
        
        model.add(TimeDistributed(Conv2D(self.conv_3, self.filter_size,padding='same',activation='relu')))
        model.add(TimeDistributed(MaxPooling2D((2, 2), strides=(2, 2))))
        model.add(TimeDistributed(Dropout(self.dropout_rate)))
        
        model.add(TimeDistributed(Flatten()))
        
        return model
    def encoder(self,model):
        # Defines the encoder model with single layer of GRU and defined number of GRU units. BN is added for improved training perfomance and quickness
        # Cuddn GRU provides much faster perfomance but does not provide dropout within.
        model.add(CuDNNGRU(self.GRU_units, return_sequences=False))
        model.add(Dense(2, activation='softmax'))
        model.summary()
        return model

    def encoderDecoder(self,model):
        # Defines the encoder decoder with single layers encoder and decoder. Defined number of GRUs are repeated based on the sequence size
        model.add(CuDNNGRU(self.GRU_units, return_sequences=False))
        model.add(BatchNormalization())
        # Repeat vecotr allows to repeat the input based on sequence size at each timestep
        model.add(RepeatVector(self.seq_len))
        # Decoder
        model.add(CuDNNGRU(self.GRU_units,return_sequences=True))
        model.add(BatchNormalization())
        # Time distrubuted allows for classifcation at each timestep
        model.add(TimeDistributed(Dense(2,activation='softmax')))
        model.summary()
        return model
    def executeModel(self,model,x_train,y_train,x_val,y_val,x_test,y_test):
        # Compiles and runs the model. model is trained on train and val sets with test set used to evaluate the model.
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        print(model.summary())
        history=model.fit(x_train, y_train, epochs=self.epochs, batch_size=32,validation_data=(x_test,y_test),verbose=1)
        # model.save('CRNN.h5') 
        testScore=model.evaluate(x_test,y_test,verbose=1)
        print ("test scores",testScore)
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
        plt.legend(['t ac', 'v ac','t loss','v loss'], loc='best') 
        plt.show() 
        
    def getPredictions(self,model, x_test, y_test):
        # Evaluates the model based on the evaluation metrics
        classes=model.predict(x_test)
        # The evaluation metrics expect single labels, encoder-decoder has multiple labels in a sequence.
        if self.model_type!='Encoder':
            classes = classes[:, 0]
            y_test=y_test[:, 0]
        # obtains the highest probability for the class (in every row of the sequences) and passes it to the relevant metric
        print("precision",precision_score(y_test.argmax(axis=1), classes.argmax(axis=1)))
        print("recall",recall_score(y_test.argmax(axis=1), classes.argmax(axis=1)))
        print("f1",f1_score(y_test.argmax(axis=1), classes.argmax(axis=1)))
        matrix = confusion_matrix(y_test.argmax(axis=1), classes.argmax(axis=1))
        print('Confusion Matrix : \n')
        print(matrix)
if __name__ == '__main__':
    # Creates CRNN with defined hyperparameters. model_type = encoder/encoder-decoder
    CRNN1=CRNN(seq_len=0, epochs=0, model_type='EncoderDecoder',conv_1=0,conv_2=0,conv_3=0,filter_size=(0,0,),dropout_rate=0.0,GRU_units=0)
    
