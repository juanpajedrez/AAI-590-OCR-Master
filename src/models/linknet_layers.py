'''
Authors: Juan Pablo Triana Martinez
Date: 2026-03-18
The following contains all of the PyTorch linknet necessary layers.
'''
import torch.nn as nn
import torch

# Let's import all necessary modules for linknet architecture
class LinknetStem(nn.Module):
    '''
    Class that will define the stem of the linknet architecture

    Args:
        m (int): Number of input channels from linknet paper.
        n (int): Number of output channels from linknet paper.
    '''

    def __init__(self, m:int = 3, n:int = 64) -> None:
        super().__init__()
        # Let's define the nn.Sequential module
        self.linknet_stem = nn.Sequential(
            nn.Conv2d(in_channels=m, out_channels=n, kernel_size=(7, 7), stride= (2, 2), padding=(3, 3), bias=False),
            nn.BatchNorm2d(num_features=n, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
        )
    
    def forward(self, x) -> torch.Tensor:
        return self.linknet_stem(x)

class LinknetEncoderBlock(nn.Module):
    '''
    Class that will define a encoder block of the linknet architecture

    Args:
        m (int): number of input channels from linknet paper.
        n (int): number of output channels from linknet paper.
    
    Returns:
        nn.Module: encoder block of linknet architecture.
    '''
    def __init__(self, m:int, n:int) -> None:
        super().__init__()
        # Let's define the first set of convolutional blocks
        self.convs_blocks_1 = nn.Sequential(
            nn.Conv2d(in_channels = m, out_channels= n, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False),
            nn.BatchNorm2d(num_features=n, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(),
            nn.Conv2d(in_channels = n, out_channels=n, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
            nn.BatchNorm2d(num_features=n, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU()
            )
        
        # Let's define the skip connection projection
        self.skip_conn = nn.Sequential(
            nn.Conv2d(in_channels=m, out_channels=n, kernel_size=(1, 1), stride=(2, 2), padding=(0, 0), bias=False),
            nn.BatchNorm2d(num_features=n, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )

        # Let's define the second set of convolutional blocks.
        self.convs_block_2 = nn.Sequential(
        nn.Conv2d(in_channels = n, out_channels= n, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
        nn.BatchNorm2d(num_features=n, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
        nn.ReLU(),
        nn.Conv2d(in_channels = n, out_channels=n, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
        nn.BatchNorm2d(num_features=n, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
        nn.ReLU())

    def forward(self, x) -> torch.tensor:
        # Pass the first convolutional blocks
        x1 = self.convs_blocks_1(x)

        # Add the residual connection
        x2 = x1 + self.skip_conn(x)

        # Pass the second convolutional blocks
        x3 = self.convs_block_2(x2)

        # Add the final residual connection and output it
        return x3 + x2

class LinknetDecoderBlock(nn.Module):
    '''
    Class that will simulate the Linknet Decoder blocks of linknnet architecture

    Args:
        m (int): number of input channels from linknet paper.
        n (int): number of output channels from linknet paper.
    '''

    def __init__(self, m:int, n:int) -> None:
        super().__init__()

        # Let's define the convolutional 1st block
        self.conv_block_1 = nn.Sequential(
        nn.Conv2d(in_channels=m, out_channels = int(m/4), kernel_size=(1, 1), stride = (1, 1), padding = (0, 0), bias = False),
        nn.BatchNorm2d(num_features=int(m/4), eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
        nn.ReLU())

        # Let's define the full convolutional transpose block
        # NOTE: In linknet paper they use "full" convolution, which are ConvTranspose2d in Pytorch
        # However, to avoid checkerboard artifacts, we will use nn.Upsample followed by nn.Conv2d.
        self.upsample_block = nn.Sequential(
            nn.Upsample(scale_factor=2, mode = 'bilinear', align_corners=True),
            nn.Conv2d(in_channels=int(m/4), out_channels=int(m/4), kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
            nn.BatchNorm2d(num_features=int(m/4), eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU()
        )

        # Let's define the convolutional 2nd block now
        self.conv_block_2 = nn.Sequential(
        nn.Conv2d(in_channels=int(m/4), out_channels = n, kernel_size=(1, 1), stride = (1, 1), padding = (0, 0), bias = False),
        nn.BatchNorm2d(num_features=n, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
        nn.ReLU())

    def forward(self, x) -> torch.Tensor:
        x = self.conv_block_1(x)
        x = self.upsample_block(x)
        x = self.conv_block_2(x)
        return x

class LinknetReconstructer(nn.Module):
    '''
    Class that will resemble the last Linknet reconstructer layer to upsample the desired mask

    Args:
        N (int): The number of desired out channels from the output
        m (int): number of input channels from linknet paper.
        n (int): number of output channels from linknet paper.

    '''
    def __init__(self, N:int = 1024, m:int = 64, n:int = 32):
        super().__init__()

        # Let's define the first upsample block
        self.upsample_block_1 = nn.Sequential(
        nn.Upsample(scale_factor=2, mode = 'bilinear', align_corners=True),
        nn.Conv2d(in_channels=m, out_channels=n, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
        nn.BatchNorm2d(num_features=n, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
        nn.ReLU())

        # Let's define the convolutional final block
        self.conv_block = nn.Sequential(
            nn.Conv2d(in_channels=n, out_channels=n, kernel_size=(3, 3), stride = (1, 1), padding=(1, 1), bias=False),
            nn.BatchNorm2d(num_features=n, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU())
        
        # Let's define the final upsample layer
        # NOTE: The last Conv2d will output desired N channels, the paper says (2 x 2); but close inspection shows
        # that using (2 x 2) with padding (1, 1) will increase the output size by 1 pixel in height and width.
        # Therefore, we will use (3 x 3) kernel with padding (1, 1) to keep the same size after upsampling.

        self.upsample_block_2 = nn.Sequential(
        nn.Upsample(scale_factor=2, mode = 'bilinear', align_corners=True),
        nn.Conv2d(in_channels=n, out_channels=N, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False))
    
    def forward(self, x) -> torch.Tensor:
        x = self.upsample_block_1(x)
        x = self.conv_block(x)
        x = self.upsample_block_2(x)
        return x