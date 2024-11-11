from torch.utils.data import Dataset, DataLoader
import os
import torch
from PIL import Image
import torchvision.transforms as transforms

class SegmentationDataset(Dataset):
    def __init__(self, featurePath:str, targetPath:str, resolution:tuple[int, int]=(256, 256)) -> None:
        super().__init__()
        self.featurePath = featurePath
        self.targetPath = targetPath
        # initializing the paths 

        self.featureImages = os.listdir(self.featurePath)
        self.targetImages = os.listdir(self.targetPath)

        assert len(self.featureImages) == len(self.targetImages), "There should be equal amount of features and target"
        # asserting that the number of feature and target images are equal

        self.resolution = resolution
        # initializing the resolution

        self.trans = transforms.Compose([
            transforms.Resize(self.resolution),
            transforms.ToTensor(),
            transforms.Normalize(0.5, 0.5, 0.5)
        ])
        # transformations to the image 

    def __len__(self) -> int:
        return len(self.featureImages)
    
    def __getitem__(self, idx:int) -> tuple[torch.Tensor, torch.Tensor]:
        featureImagePath = os.path.join(self.featurePath, self.featureImages[idx])
        targetImagePath = os.path.join(self.targetPath, self.targetImages[idx])
        # getting the path to the images 

        featureImage = Image.open(featureImagePath).convert("RGB")
        targetImage = Image.open(targetImagePath).convert("L")

        featureImage = self.trans(featureImage)
        targetImage = self.trans(targetImage)
        # applying the transformations to the images

        return featureImage, targetImage
    

def getDataLoader(featurePath:str, targetPath:str, resolution:tuple[int, int]=(256, 256), batchSize:int=32):
    dataset = SegmentationDataset(featurePath, targetPath, resolution)
    dataloader = DataLoader(dataset, batch_size=batchSize, shuffle=True)
    return dataloader

    
if __name__ == "__main__":
    from torchvision.utils import save_image

    targetPath = "Human-Segmentation-Dataset/Ground_Truth"
    featurePath = "Human-Segmentation-Dataset/Training_Images"

    loader = getDataLoader(featurePath, targetPath)

    for x, y in loader:
        print(f"Image shape: {x.shape}, Target shape: {y.shape}")

        x = (x+1) / 2
        y = (y+1) / 2

        save_image(x, "testFeature.png")
        save_image(y, "testTarget.png")
        pass