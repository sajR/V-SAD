# The code below obtains embeddings/features from existing CNN models.
# Due to a high number of images the pre-processing (getting features from the existing models) can take time and power. 
# Thus, the pre-processing is done beforehand whereby features are stored as CSV. Avoids generating features each time a model is experimented.
# Images are read in the order of non-speech to speech. Provides control and order in which the features are saved and the abilty to generate sequences for the images/features (as the features/images are in order) 
# By default the embedding length is 2048 for Xception,  
 

import os
from keras import applications
from keras.preprocessing import image
import numpy as np

#images directories
train_speech_dir="/Train/speach"
train_nonspeech_dir="/Train/nonspeech"
test_speech_dir="/Test/speach"
test_nonspeech_dir="/Test/nonspeech"
val_speech_dir="/Validation/speach"
val_nonspeech_dir="/Validation/nonspeech"


class Embeddings():
    def __init__(self, model_name):
    # Initialising key variables and obtaining relevant models for pre-processing and obtaining features with a pretrained model 
    # include_top=False removes the fully connected layer at the end/top of the network
    # This allows to get the feature vector as opposed to a classification
        self.model_name=model_name
        if model_name == 'Xception':
            self.model=applications.xception
            self.pretrained_model=applications.xception.Xception(weights='imagenet', include_top=False, pooling='avg')
            
        if model_name == 'VGG16':
            self.model.vgg16
            self.pretrained_model=applications.vgg16.VGG16(weights='imagenet', include_top=False, pooling='avg')

        if model_name == 'InceptionV3':
            self.model=applications.inception_v3
            self.pretrained_model=applications.inception_v3.InceptionV3(weights='imagenet', include_top=False, pooling='avg')
    def getEmbeddings(self,imagePath,filename):
        #Obtains features for every image in a given folder and stores as train,val and test.
        print (filename)
        imageFormat=".png"
        label=""
        fileList=[os.path.join(imagePath,f) for f in os.listdir(imagePath) if f.endswith(imageFormat)]
        for imageName in fileList:
            img = image.load_img(imageName, target_size=(100, 100))
            if img is not None:
                processed_array=self.formatEmbedding(img)
                # obtains features by predicting the processed/formatted image array to the existing model
                features=self.pretrained_model.predict(processed_array)
                # Adds a "1" or "0" depending on speech/non-speech to establish feature as a speech/non-speech
                if "nospeaking" in imageName:
                    label="0,"
                else:
                    label="1,"
                with open(filename,"a") as file:
                    file.write(label)
                    # Each line is written with a label followed by feature
                    np.savetxt(file,features,delimiter=",")
                    
            else:
                print ("error with obtaining image")
    def formatEmbedding(self,img):
        # formats the image as array and pre-processes the array based on the model.     
        image_array=image.img_to_array(img)
        # converts image array to numpy array and alters the shape of the array, expected by the model
        image_array=np.expand_dims(image_array,axis=0)
        # pre-proces the image array based on the model, either 0-255 or -1 to +1 etc
        processed_array=self.model.preprocess_input(image_array)
        return processed_array
        
if __name__ == '__main__':
    # flow of the program. model name is defined, embeddings for each image directory, names for the CSV files.
    XceptionEmbeddings=Embeddings('Xception')
    model=XceptionEmbeddings.pretrained_model
    model.summary()
    XceptionEmbeddings.getEmbeddings(train_speech_dir,("train"+XceptionEmbeddings.model_name+'.csv'))
    XceptionEmbeddings.getEmbeddings(train_nonspeech_dir,("train"+XceptionEmbeddings.model_name+'.csv'))

    XceptionEmbeddings.getEmbeddings(val_speech_dir,("val"+XceptionEmbeddings.model_name+'.csv'))
    XceptionEmbeddings.getEmbeddings(val_nonspeech_dir,("val"+XceptionEmbeddings.model_name+'.csv'))

    XceptionEmbeddings.getEmbeddings(test_speech_dir,("test"+XceptionEmbeddings.model_name+'.csv'))
    XceptionEmbeddings.getEmbeddings(test_nonspeech_dir,("test"+XceptionEmbeddings.model_name+'.csv'))

