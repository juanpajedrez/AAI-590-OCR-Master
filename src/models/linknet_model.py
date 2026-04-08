'''
Author: Juan Pablo Triana Martinez
Date: 2026-03-18
Contains the linknet_model architecture for text detection tasks.
'''

import torch
import torch.nn as nn
from .linknet_layers import LinknetStem, LinknetReconstructer, LinknetDecoderBlock, LinknetEncoderBlock

class LinknetModel(nn.Module):
    '''
    Class that simulates the full linknet architecture, using
    the rest of the blocks define before

    Args:
        Cin (int): number of input channels for the stem layer.
        N (int): number of output channels for the final reconstructer layer.
    '''

    def __init__(self, Cin:int = 3, N:int = 3) -> None:
        super().__init__()

        # Let's define the stem part
        self.stem = LinknetStem(m = Cin, n = 64)

        # Let's define each of the encoder blocks
        self.encoder_block_1 = LinknetEncoderBlock(m = 64, n = 64)
        self.encoder_block_2 = LinknetEncoderBlock(m = 64, n = 128)
        self.encoder_block_3 = LinknetEncoderBlock(m = 128, n = 256)
        self.encoder_block_4 = LinknetEncoderBlock(m = 256, n = 512)

        # Let's define each of the decoder blocls
        self.decoder_block_4 = LinknetDecoderBlock(m = 512, n = 256)
        self.decoder_block_3 = LinknetDecoderBlock(m = 256, n = 128)
        self.decoder_block_2 = LinknetDecoderBlock(m = 128, n = 64)
        self.decoder_block_1 = LinknetDecoderBlock(m = 64, n = 64)

        # Let's define the reconstructer part
        self.reconstructer = LinknetReconstructer(N=N, m=64, n=32)

    def forward(self, x) -> torch.Tensor:
        # Let's get the stem output
        x = self.stem(x)

        # Let's pass through each of the encoder blocks and store the outputs
        x1 = self.encoder_block_1(x)
        x2 = self.encoder_block_2(x1)
        x3 = self.encoder_block_3(x2)
        x4 = self.encoder_block_4(x3)

        # Let's pass through each of decoder blocks, while adding the skip connections
        x = self.decoder_block_4(x4) + x3
        x = self.decoder_block_3(x) + x2
        x = self.decoder_block_2(x) + x1

        # Finall, pass through the last decoder block and reconstructer
        x = self.decoder_block_1(x)
        x = self.reconstructer(x)
        return x