# Image-Classification-Using-CNN

Image classification using Convolutional Neural Networks (CNNs) with visualization is a common task in computer vision.
CNNs are a powerful deep learning technique for image classification, and visualization can help you understand and interpret the model's behavior. 
The model is a Convolutional Neural Network (CNN), which is a type of deep learning model commonly used for image-related tasks.

Architecture:

Input Layer: The model takes as input images with dimensions of 32x32 pixels and 3 color channels (RGB).

First Convolutional Layer: This is the initial layer that performs convolutions on the input images. It consists of 32 filters (also called kernels) each with a size of 3x3 pixels. ReLU (Rectified Linear Unit) activation is applied to the output of this layer, introducing non-linearity. After this, a max-pooling layer with a 2x2 pooling window reduces the spatial dimensions of the feature maps.

Second Convolutional Layer: Similar to the first layer, this layer consists of 64 filters of size 3x3 pixels. It also uses ReLU activation and is followed by max-pooling.

Flatten Layer: This layer flattens the output of the previous convolutional layers into a 1D vector. This is necessary to connect the convolutional layers to the fully connected layers.

Fully Connected Layers: There are two fully connected (dense) layers in the model. The first has 64 neurons with ReLU activation, and the second has 10 neurons with softmax activation. The final layer with softmax activation provides probabilities for classifying the input image into one of 10 classes (e.g., in the case of CIFAR-10 dataset).

Compilation and Training:

The model is compiled with the Adam optimizer, which is a popular choice for training deep neural networks.
The loss function used is "sparse categorical cross-entropy," which is appropriate for multi-class classification tasks.
Training is performed using the training data with a batch size of 64 and for 10 epochs (you can adjust these hyperparameters based on your specific problem).
Visualization:

The code also includes visualization components, such as plotting the training loss and accuracy over epochs.
It can visualize the learned filters (kernels) in the first convolutional layer.
It can also visualize feature maps (activations) produced by the convolutional layers.
Overall, this model is designed for image classification tasks, and it's a relatively simple CNN architecture with two convolutional layers followed by fully connected layers. It can be customized and extended to suit specific image classification problems.
