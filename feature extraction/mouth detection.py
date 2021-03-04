# Code below extracts features (in this case, images of mouth)
# This is done by utilising the haar cascade which provides an image of the face.
# Mouth-region is calculated based on the face-image as noted below
# This method provides a simple and quick way to extract features. Other methods such as using YOLO may provide better results.

import os,random
import cv2
import os

os.getcwd()
dataset_Path = "/datasets1"
path_To_Save="/data1"
face_cascade=cv2.CascadeClassifier('/haarcascade_frontalface_default.xml')

file_count=0
face_Not_Found=0
face_Found=0
for i, person_Folder in enumerate(os.listdir(dataset_Path)):
    video_Path=dataset_Path+"//"+person_Folder+"//video"
    for i,folder_Type in enumerate(os.listdir(video_Path)):
        folder_Type_Path=video_Path+"//"+folder_Type
        for i,speech_Type in enumerate(os.listdir(folder_Type_Path)):
            speech_Type_Path=folder_Type_Path+"//"+speech_Type
            for i,filename in enumerate(os.listdir(speech_Type_Path)): 
                #print (filename)
                file_count=file_count+1
                png_Image=cv2.imread(speech_Type_Path+"//"+filename)
                gray_Image=cv2.cvtColor(png_Image,cv2.COLOR_BGR2GRAY)
                faces=face_cascade.detectMultiScale(gray_Image, 1.25,5) #1.05,3 #1.1,3 #1.1,4-pretty accurate #1.3,4
                #2732 filecount
                #1.1,4 - 546 not found, - 8 errors , 2135 images
                #1.25-4 - 662 not found - 3 errors - 2061 images
                #1.2,5 - 687 not found - 2 errors 2037 images
                 #1.25-5 - 707 not found - 1 error 2021 images
                #1.3,5 - 751 not found 1980 images 
               
                if len(faces) == 0:
                    face_Not_Found=face_Not_Found+1
                else:
                    face_Found=face_Found+1
                    for (x,y,w,h) in faces:       
                        # cv2.rectangle(savedImage,(x,y),(x+w,y+h),(255,0,0),2)
                        ew=w//4
                        eh=h//3
                        ex=x+ew
                        ey=y+(eh*2)
                        mouth=gray_Image[ey:ey+eh, ex:ex+(ew*2)]
                        mouth_resized=cv2.resize(mouth,(100,100))
                        cv2.imwrite(os.path.join(path_To_Save, speech_Type+"_"+person_Folder+"_"+folder_Type+"_"+filename),mouth_resized)
                
print (str(file_count)+' filecount')                   
print (str(face_Not_Found)+' not found')
print (str(face_Found)+' found')

                    
    
                
        
       
             
