# Code below enhances images. As after feature extraction some images can appear dark/inconsisten etc due to race/colour. 
# Enhancing images allows for images to be consistent thus reducing noise in the network. 
# Enhancement of the image is done by the pixel median of the image.
# If the image median is above a threshold, brightness increased/decreased and contrast is added to add colour/definition
# Arguebly there are better methods out there, but this method is simple, quick and effective.
from PIL import Image, ImageEnhance,ImageStat
import os

imageFormat='.png'
path='\\images'
count=0
fileList=[os.path.join(path,f) for f in os.listdir(path) if f.endswith(imageFormat)]

for imageName in fileList:
    count+=1
    image = Image.open(imageName)
    stat=ImageStat.Stat(image)
    median=stat.median[0]
    factor=0
    print (median)
    if median<=170:
        factor=170/(median*1.0)
        print (170/(median*1.0))
        if factor<=1:
            factor=1
    elif median>=171 and median<=180:
        factor=1
    elif median>= 182:
        factor=(1/(median*1.0))
    if factor>=0.0 and factor<1.0:
        factor=1
    print (factor)
    brightness=ImageEnhance.Brightness(image)
    bright_image=brightness.enhance(factor)
    bright_image.save(imageName)
    image2=Image.open(imageName)
    contrast=ImageEnhance.Contrast(image2)
    contrast_image=contrast.enhance(2)
    contrast_image.save(contrast_image) 
print (count)
    
