from uuid import uuid4
import os

trainingID = str(uuid4())
training = True
# training conigurations 

savePath = os.path.join("saves", trainingID)
previewPath = os.path.join("preview", trainingID)
# path configuration 

os.makedirs(savePath, exist_ok=True)
os.makedirs(previewPath, exist_ok=True)
# Create directories if they don't exist

savedDiscriminator = ""
savedGenerator = ""
# this defines if the laoded weights is from Generator Model or it's child Res2Pix

epochs = 100
lr = 2e-4
batchSize = 8
betas = (0.5, 0.999)
l1Lambda = 100
generatorLRDecay = 0.999
discriminatorLRDecay = 0.99
# training loop configuration 

device = "cuda"

trainingFeaturePath = "data/images"
trainingTargetPath = "data/masks"
testFeaturePath = "testData/images"
testTargetPath = "testData/masks"
# defining paths to data 

resolution = (256, 256) 
# image resolution 

decayEvery = 2
saveEvery = 4

edgeThreshold = 0.3

if training:
    print(f"Training ID: {trainingID}")