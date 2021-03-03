# CNN-RNN - Convolutional Neural Network - Recurrent Neural Network
Based on https://arxiv.org/abs/1604.04573 for visual-Speech Activity Detection (SAD).

Uilises existing CNN (Xception) for embeddings (in this case as input) followed by an RNN - Encoder/Encoder-Decoder.

Enocoder classifies based on previous images i.e. classification occurs on last image of the sequence.
Encoder-Decoder classfies the whole sequence i.e. classification on every image of the sequence.
