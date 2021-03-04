# V-SAD - Visual-Speech Activity Detection.

Detecting speech activity with neural networks via facial images. The purpose of the network is to identify if the image belongs to speech or non-speech

The networks are compiled with Anaconda and Spyder utilising python - 3.6.12, Keras (GPU) - 2.2.4, TensorFlow (GPU) - 1.12 

The networks are trained with Train, Validation, and Test sets with a ratio of 70:15:15 whereby Train and Validation sets are used to train the network whilst Test set is used to evaluate the networks. The data is split on speaker-dependent i.e. data from an individual is occurs in each set, as opposed to speaker-independent where data is split based on individuals
