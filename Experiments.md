Experiments
1. Can Controller adapt to different datasets
   - CIFAR has 3000 features, while MNIST has 700ish.
   - Providing image size, number of classes, etc. as input to the Controller
2. Can Controller predict intermediate convolution layers of existing models?
   - AlexNet Layers to predict (2,3),VGG16 Layers to predict (3,4,5)
   - Make modification to controller to accept previous layers as input
3. Can Controller use training time as an additional metric to learn optimal architecture?
4. Can Controller trained on a different dataset converge faster towards a feasible model on a similar dataset?
   - MNIST,Fashion MNIST