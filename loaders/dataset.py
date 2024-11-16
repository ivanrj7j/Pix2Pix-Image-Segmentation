from torch.utils.data import Dataset, DataLoader
import os
import torch
from PIL import Image
import torchvision.transforms as transforms
from albumentations.pytorch.transforms import ToTensorV2
import albumentations as A
import numpy as np
import cv2

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
            transforms.Resize(self.resolution)
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
            A.GaussNoise((0, 0.3)),
            A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5), max_pixel_value=1),
            ToTensorV2(),
        ], additional_targets={'mask': 'mask'})  

    def __len__(self) -> int:
        return len(self.featureImages)
    
    def __getitem__(self, idx:int) -> tuple[torch.Tensor, torch.Tensor]:
        featureImagePath = os.path.join(self.featurePath, self.featureImages[idx])
        targetImagePath = os.path.join(self.targetPath, self.targetImages[idx])
        # getting the path to the images 

        featureImage = cv2.resize(cv2.imread(featureImagePath), self.resolution)
        targetImage = cv2.resize(cv2.imread(targetImagePath), self.resolution)

        featureImage = cv2.cvtColor(featureImage, cv2.COLOR_BGR2RGB)
        targetImage = cv2.cvtColor(targetImage, cv2.COLOR_BGR2GRAY)
        # applying the transformations to the images

        augmented = self.trans2(image=featureImage, mask=targetImage)
        featureImage = augmented['image']
        targetImage = augmented['mask']

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