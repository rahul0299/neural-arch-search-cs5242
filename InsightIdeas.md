1. 1 vocab for all vs bounded for each (what we are doing) (CNN)
2. Does NAS collapse to using RELU as we always do for standard CNN?
3. Does the MLP model learn to increase the number of features to reduce data loss from compression in small layers?
4. Does the controller adapt to different datasets. e.g. CIFAR has 3000 features, while MNIST has 700ish, does a model which has trained on CIFAR know that it could reduce the complexity and still work
5. Transfer learning: Does a model trained on a CIFAR dataset able to reach a threshold accuracy faster on MNIST
6. Transfer learning: Also if we provide training time as a metric, does it reduce the number of layers realizing that with MNIST it doesnt need as many layers