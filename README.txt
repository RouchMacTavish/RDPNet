This is the code group of RDPNet.

Environment:
torch  1.8
torchvision 0.6
cuda 

The convolutional block of RDPNet contains two 3x3 convolutional layers. It available to adapt the number by manually changing the convolutonal layers on function convX_(...). 

We achieve classification accuracy of 93.720% on CIFAR-10, 74.510% on CIFAR-100. 
RDPNet has a good performance compared with some classical networks (VGGs, Resnets, DenseNet-201) and some state-of-the-art lightweight networks (MobileNetV2, GhostNet). 

