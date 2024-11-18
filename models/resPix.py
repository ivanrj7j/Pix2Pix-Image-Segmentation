from models.generator import Generator
import torch.nn as nn
import torch

class ResBlock(nn.Module):
    def __init__(self, channels:int, dropout:bool=True, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.conv1 = nn.Conv2d(channels, channels, 3, 1, 1)
        self.leaky1 = nn.LeakyReLU(0.2)
        self.conv2 = nn.Conv2d(channels, channels, 3, 1, 1)
        self.leaky2 = nn.LeakyReLU(0.2)
        self.dropout = nn.Dropout(0.5) if dropout else nn.Identity()

    def forward(self, x):
        out = self.leaky1(self.conv1(x))
        out = self.leaky2(self.conv2(out))
        out = self.dropout(out)
        return out + x
    

class Res2Pix(Generator):
    """
    # Res2Pix

    Modified pix2pix generator architecture with extra Residual blocks in the end
    """
    def __init__(self, inChannels: int, outChannels: int = None, resBlocks:int=5, features: int = 64, *args, **kwargs) -> None:
        
        """
        Generator model generates the desired output image.
        The generator takes in a latent vector and produces the corresponding image
        
        Keyword arguments:
        inChannels -- number of channels of the input latent vector
        outChannels -- number of channels of the output image (default: inChannels)
        resBlocks -- number of residual blocks to be added in the end
        features -- the (base) number of features in the middle layers
        Return: None
        """
        super().__init__(inChannels, outChannels, features, *args, **kwargs)

        resBlock = []
        for i in range(resBlocks):
            resBlock.append(ResBlock(features, i==resBlocks-1))

        self.final = nn.ConvTranspose2d(features*2, features, 4, 2, 1)
        # this is the final block of the pix2pix generator model, and not the final block of the model 

        self.resBlocks = nn.Sequential(*resBlock)

        self.finalBlock = nn.Sequential(
            nn.Conv2d(features, outChannels, 3, 1, 1),
            nn.Tanh()
        )


    def forward(self, x):
        out = super().forward(x)
        out = self.resBlocks.forward(out)
        return self.finalBlock.forward(out)
    
    def loadFromParent(self, path:str):
        """
        Load the pretrained model from a given path.
        This method loads the pretrained weights from the parent model and initializes the weights of the current model.
        
        Keyword arguments:
        path -- path to the pretrained model
        Return: None
        """
        weights = torch.load(path, weights_only=True)
        self.load_state_dict(weights, strict=False)