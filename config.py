from uuid import uuid4
import os

trainingID = str(uuid4())
training = False
# training conigurations 

savePath = os.path.join("saves", trainingID)
previewPath = os.path.join("preview", trainingID)
# path configuration 

os.makedirs(savePath, exist_ok=True)
os.makedirs(previewPath, exist_ok=True)
# Create directories if they don't exist

savedDiscriminator = ""
savedGenerator = ""
# save path of the discriminator and generator to load from 

epochs = 100
lr = 2e-4
batchSize = 32
betas = (0.5, 0.999)
l1Lambda = 100
generatorLRDecay = 0.999
discriminatorLRDecay = 0.99
# training loop configuration 

device = "cuda"

trainingFeaturePath = "Human-Segmentation-Dataset/Training_Images"
trainingTargetPath = "Human-Segmentation-Dataset/Ground_Truth"
testFeaturePath = "test dataset/inputImage"
testTargetPath = "test dataset/outputImage"
# defining paths to data 

resolution = (128, 128) 
# image resolution 

decayEvery = 2
saveEvery = 5

if training:
    print(f"Training ID: {trainingID}")