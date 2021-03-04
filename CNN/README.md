# CNN - Convolutional Neural Network
# Detecting speech acivity with still images (single image - whether its speech or non-speech) using CNN.  
CNN - 3 CONV layer CNN with single Fully Connected (FC) layer and a classification layer.  Adapted from literature with different kernal and filter sizes, dropout and BN. Model tested to compare perfomance and time against smaller VGG

VGGCNN - Adapted VGG from the paper https://arxiv.org/abs/1409.1556. Smaller VGG (6 conv, 2 FC, 1 classification) utilises the VGG model but adapted to run within the computional resources available. Uses dropout and BN which has proven to improve perfomance 

Transfer Learning - using an exisisting model (Xception) to classifiy SAD. 

Images are stored in relevant folder train, val, test.
However, each folder the images need to be stored based on names of classes as this is how the image data genrator identifies the classes.
