# CRNN
# Convolutional Reccurent Neural Network for SAD (Speech Activity Detection)
Network that creates its own features and learns the features across the images within a sequence. 

3 Layer CNN learns the features of the images which are then passed to RNN to learn contextual dependencies across the images.

The RNN are in form of Encoder/Encoder-Decoder.

Encoder - single layer of RNN classifying last image of sequence.
Encoder-Decoder- single layer for encoding/decoding classifying all images in a sequence (Seq2Seq)
