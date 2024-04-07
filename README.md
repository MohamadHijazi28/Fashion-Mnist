# Fashion-Mnist

Fashion Mnist is a dataset with 60K 28x28 greyscale images. The images
belong to 10 categories of clothing articles. In this exercise you will design
and train a network on this dataset.
In this exercise you will design 4 networks, showing the difference
between Feed Forward and CNN and the importance of network depth.

Assignment (Code):
1. Net_1: Fully Connected, 3 classes, 2 weighted layers.
For this network, use only classes {0,1,2} of the dataset. Design and
train a 2 layer fully connected network. The network should reach
about 80% accuracy and use at most 50K parameters.

2. Net_2: Fully Connected, 7 classes, 2 weighted layers.
Same as (1), but use classes {0,…6}. Reach as high accuracy as you
can. Use no more than 50K parameters.

4. Net_3: Fully Connected, 7 classes, 4 weighted layers.
Same as (2), with 4 fully connected layers instead of 2. Reach as high
accuracy as you can. Use no more than 50K parameters.

5. Net_4: CNN, 7 classes:
Design a CNN for the dataset with the 7 labels {0,…6}. Train to reach
better performance than Net_3. The network’s weighted layers
should be: 3 convolutional layers, and one fully connected layer in
the end to flatten the tensors. Use no more than 50k parameters.


Note: the architecture limitations apply only to weighted (trainable)
layers. Regularization, normalization, activation, and max-pooling layers
are by definition not trainable layers, and you can use them freely.
