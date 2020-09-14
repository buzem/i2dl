"""SegmentationNN"""
import torch
import torch.nn as nn
import pytorch_lightning as pl


class SegmentationNN(pl.LightningModule):

    def __init__(self, num_classes=23, hparams=None):
        super().__init__()
        self.hparams = hparams
        #######################################################################
        #                             YOUR CODE                               #
        #######################################################################

        self.conv_pool_ups = nn.Sequential(
            
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=7),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.01, inplace=True),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=5),
            nn.LeakyReLU(0.01, inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            #nn.Dropout2d(p=0.1),
            
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.01, inplace=True),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3),
            nn.LeakyReLU(0.01, inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            
            nn.Upsample([240,240]),
            nn.Conv2d(in_channels=64, out_channels=23, kernel_size=1)
        )
        #output of size N*23*240*240
        #######################################################################
        #                           END OF YOUR CODE                          #
        #######################################################################

    def forward(self, x):
        """
        Forward pass of the convolutional neural network. Should not be called
        manually but by calling a model instance directly.

        Inputs:
        - x: PyTorch input Variable
        """
        #######################################################################
        #                             YOUR CODE                               #
        #######################################################################

        x = self.conv_pool_ups(x)

        #######################################################################
        #                           END OF YOUR CODE                          #
        #######################################################################

        return x

    @property
    def is_cuda(self):
        """
        Check if model parameters are allocated on the GPU.
        """
        return next(self.parameters()).is_cuda

    def save(self, path):
        """
        Save model with its parameters to the given path. Conventionally the
        path should end with "*.model".

        Inputs:
        - path: path string
        """
        print('Saving model... %s' % path)
        torch.save(self, path)

        
class DummySegmentationModel(pl.LightningModule):

    def __init__(self, target_image):
        super().__init__()
        def _to_one_hot(y, num_classes):
            scatter_dim = len(y.size())
            y_tensor = y.view(*y.size(), -1)
            zeros = torch.zeros(*y.size(), num_classes, dtype=y.dtype)

            return zeros.scatter(scatter_dim, y_tensor, 1)

        target_image[target_image == -1] = 1

        self.prediction = _to_one_hot(target_image, 23).permute(2, 0, 1).unsqueeze(0)

    def forward(self, x):
        return self.prediction.float()
