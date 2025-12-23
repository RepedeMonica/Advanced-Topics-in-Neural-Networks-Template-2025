In this project, I trained a neural network to approximate a fixed image transformation pipeline consisting of resizing, grayscale conversion, horizontal and vertical flipping. The model takes RGB images of size 3×32×32 as input and outputs grayscale images of size 1×28×28.

The model is a simple fully connected network with a single linear layer that maps the flattened input image directly to the flattened output image. The model is trained as an image regression task using Mean Squared Error (MSE) loss and early stopping based on validation loss.

After training, comparisons between the input images, ground truth transformations, and model outputs are provided. Inference time is benchmarked against sequential CPU-based transformations for different batch sizes and devices, showing that the model can achieve faster inference when batching is used.
