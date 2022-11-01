# Max-Affine Spline Insights Into Deep Network Pruning

## For ConvNet experiments

* How to visualize ConvNet

Here we demonstrate code that can visualize three different types of Convolutional Neural Network: simplecnn, alexnet, and preresnet. The canvas we use to draw is a grid of linear combination of two images, from datasets like, mnist, cifar10, or cifar100. We can then draw the decision boundaries of all the layers on that grid.


* How to run the code? For any of the CNN type and dataset, you can train the network with early-bird ticket detection by:

````shell
python experiments.py --network alexnet --dataset cifar10 --epochs 10
````

After the checkpoints for the training epochs are saved, you can run the individual visualization script to visualize the decision boundaries

````shell
python vis_alexnet.py
````