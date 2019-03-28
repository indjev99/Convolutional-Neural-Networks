# Convolutional-Neural-Networks

A work in progress C++ machine learning library. Focused on supervised training.

### The neural network can consist of:
- Convultional layers with optional padding to preserve spatial information
- Fully connected layers
- Polling layers
- Activation layers with arbitrary different activation functions

### Built in activation functions are:
- RELU
- Leaky RELU
- Logistic curve

### Planned features:
- Ability to chain NNs one after another during training for Auto-encoders
- Reccurent NNs such as LSTMs
- Porting the code to work on the GPU and other optimizations

### Planned fixes:
- Deallocation of memory when deleting a network.

### Use:
- Construct the network layer by layer
- Couple it with a trainer and give it the training dataset
- Train it for however many epochs potentialy with different learning rates
- Discard the trainer and use the NN itself

See the example for an implementation of a digit recognizer.