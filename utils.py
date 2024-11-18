import torch
import os
from torchvision.utils import save_image
from models.discriminator import Discriminator
from models.resPix import Res2Pix

def saveModel(model:torch.nn.Module, path:str, epoch:int|str):
    """
    Save the model to the given path.
    """

    torch.save(model.state_dict(), os.path.join(path, f"model_{epoch}.pth"))

def saveImage(image:torch.Tensor, path:str, epoch:int|str):
    """
    Save the image to the given path.
    """
    image = (image + 1) / 2
    imagePath = os.path.join(path, f"preview_{epoch}.png")

    save_image(image, imagePath)

def loadModels(generatorPath:str, discriminatorPath:str, device:str, generatorFromParent:bool=False):
    generator = Res2Pix(3, 1)
    discriminator = Discriminator(3, 1)
    # initializing generator and discriminator 

    if generatorPath != "":
        print(f"Loading {generatorPath}")
        if generatorFromParent:
            generator.loadFromParent(generatorPath)
        else:
            generator.load_state_dict(torch.load(generatorPath, weights_only=True))
    if discriminatorPath != "":
        print(f"Loading {discriminatorPath}")
        discriminator.load_state_dict(torch.load(discriminatorPath, weights_only=True))

    return generator.to(device), discriminator.to(device)