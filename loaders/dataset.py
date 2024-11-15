from torch.utils.data import Dataset, DataLoader
import os
import torch
from PIL import Image
import torchvision.transforms as transforms
import albumentations as A
import numpy as np

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

        self.trans1 = transforms.Compose([
            transforms.Resize(self.resolution),
            np.array
        ])
        # transformations to the image 

        self.trans2 = A.Compose([
            # Geometric transformations (applied to both image and mask)
            A.HorizontalFlip(p=0.2),
            A.VerticalFlip(p=0.2),
            A.Rotate(limit=25, p=0.65),
            A.RandomBrightnessContrast(),
            A.RandomFog(p=0.3),
            A.MotionBlur(p=0.1),
            A.GaussianBlur(p=0.1),
            A.GaussNoise((0, 0.3))
        ], additional_targets={'mask': 'mask'})  

        self.trans3 = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(0.5, 0.5, 0.5)
        ])

    def __len__(self) -> int:
        return len(self.featureImages)
    
    def __getitem__(self, idx:int) -> tuple[torch.Tensor, torch.Tensor]:
        featureImagePath = os.path.join(self.featurePath, self.featureImages[idx])
        targetImagePath = os.path.join(self.targetPath, self.targetImages[idx])
        # getting the path to the images 

        featureImage = Image.open(featureImagePath).convert("RGB")
        targetImage = Image.open(targetImagePath).convert("L")

        featureImage = self.trans1(featureImage)
        targetImage = self.trans1(targetImage)
        # applying the transformations to the images

        augmented = self.trans2(image=featureImage, mask=targetImage)
        featureImage = augmented['image']
        targetImage = augmented['mask']

        featureImage = self.trans3(featureImage)
        targetImage = self.trans3(targetImage)

        return featureImage, targetImage
    

def getDataLoader(featurePath:str, targetPath:str, resolution:tuple[int, int]=(256, 256), batchSize:int=32):
    dataset = SegmentationDataset(featurePath, targetPath, resolution)
    dataloader = DataLoader(dataset, batch_size=batchSize, shuffle=True)
    return dataloader

    
if __name__ == "__main__":
    from torchvision.utils import save_image

    featurePath = "testData/images"
    targetPath = "testData/masks"

    loader = getDataLoader(featurePath, targetPath)

    for x, y in loader:
        print(f"Image shape: {x.shape}, Target shape: {y.shape}")

        x = (x+1) / 2
        y = (y+1) / 2

        save_image(x, "testFeature.png")
        save_image(y, "testTarget.png")
        break