import torch
from models.generator import Generator
from torchvision.transforms import Compose, Resize, ToTensor, Normalize

device = "cuda"
# configuration 

modelPath ="saves/fcf88d77-c866-46e8-a33c-9b328392e877/model_gen_final.pth"
model = Generator(3, 1)
model.load_state_dict(torch.load(modelPath, weights_only=True))
model = model.to(device)
# loading the model 

model.eval()

transforms = Compose([
            Resize((256, 256)),
            ToTensor(),
            Normalize(0.5, 0.5, 0.5)
    ])
# transforms for image


def addMask(image:torch.Tensor, mask:torch.Tensor) -> torch.Tensor:
    """
    Add a mask to the image by blending the input image and mask tensors.

    Parameters:
    image (torch.Tensor): The input image tensor with shape (N, 3, H, W) or (3, H, W), where N is the batch size, H is the height, and W is the width. The tensor should have pixel values normalized to [-1, 1].
    mask (torch.Tensor): The mask tensor with shape (N, 1, H, W) or (1, H, W), where N is the batch size, H is the height, and W is the width. The tensor should have pixel values normalized to [-1, 1].

    Returns:
    torch.Tensor: The masked image tensor, with values blended by applying the mask to the image tensor, maintaining the original shape of the input image.
    """

    image = (image + 1) / 2
    mask = (mask + 1) / 2

    mask = torch.round(mask)
    return mask * image

def getMask(img) -> torch.Tensor:
    """
    Generate a mask for the input image using a pre-trained model.

    Parameters:
    img (PIL.Image): The input image loaded as a PIL image. It will be resized, normalized, and converted to a tensor.

    Returns:
    torch.Tensor: The generated mask tensor with shape (1, H, W), where H and W match the dimensions of the input image. The mask is resized to the input image dimensions using bicubic interpolation if needed.
    """

    image = transforms(img).to(device)
    dimensions = tuple(reversed(img.size))

    with torch.no_grad():
        mask = model(image.unsqueeze(0))
    
    if tuple(mask.shape[2:]) != dimensions:
        mask = torch.nn.functional.interpolate(mask, dimensions, mode="bicubic")

    return mask