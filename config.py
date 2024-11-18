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

savedDiscriminator = "saves/4bb3f5d0-29a5-4841-96d7-8fe58a3c3bdd/model_disc_36.pth"
savedGenerator = "saves/4bb3f5d0-29a5-4841-96d7-8fe58a3c3bdd/model_gen_36.pth"
# this defines if the laoded weights is from Generator Model or it's child Res2Pix

epochs = 100
lr = 2e-4
batchSize = 10
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

if training:
    print(f"Training ID: {trainingID}")