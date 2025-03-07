from utils import saveImage, saveModel, loadModels
from loaders.dataset import getDataLoader
# importing custom code 

import torch
from torch.optim.adam import Adam
from torch.optim.lr_scheduler import ExponentialLR
from torch.nn import L1Loss, BCEWithLogitsLoss
# importing torch related stuff

import config
# importing configurations for training

from tqdm import tqdm
# importing non torch stuff 

generator, discriminator = loadModels(config.savedGenerator, config.savedDiscriminator, config.device, config.edgeThreshold)
# loading models 

generatorOptim = Adam(generator.parameters(), config.lr, config.betas)
discOptim = Adam(discriminator.parameters(), config.lr, config.betas)
# initalizing the optimizers 
  
schedulerG = ExponentialLR(generatorOptim, gamma=config.generatorLRDecay)
schedulerD = ExponentialLR(discOptim, gamma=config.discriminatorLRDecay)
# initialzing schedulers 

l1 = L1Loss()
bce = BCEWithLogitsLoss()
# initializing losses 

trainLoader = getDataLoader(config.trainingFeaturePath, config.trainingTargetPath, config.resolution, config.batchSize)
testLoader = getDataLoader(config.testFeaturePath, config.testTargetPath, config.resolution, config.batchSize)
# initializing data loaders 

genScaler = torch.GradScaler(config.device)
discScaler = torch.GradScaler(config.device)


def step(x:torch.Tensor, y:torch.Tensor) -> tuple[float, float]:
    with torch.autocast(config.device, torch.float16):
        generatedImage = generator.forward(x)

        discReal = discriminator.forward(x, y)
        discFake = discriminator.forward(x, generatedImage.detach())

        zeroNoise = torch.zeros_like(discFake) + (torch.rand_like(discFake) * 0.05)
        oneNoise = torch.ones_like(discReal) - (torch.rand_like(discReal) * 0.05)

        discLoss = (bce(discReal, oneNoise) + bce(discFake, zeroNoise))/2
    # calculating the loss of discriminator 

    discriminator.zero_grad()
    discOptim.zero_grad()
    discScaler.scale(discLoss).backward()
    discScaler.step(discOptim)
    discScaler.update()

    with torch.autocast(config.device, torch.float16):
        criticLoss = discriminator.forward(x, generatedImage)
        criticLoss = bce(criticLoss, oneNoise)
        l1Loss = l1(generatedImage, y)
        genLoss = criticLoss + (config.l1Lambda * l1Loss)
        # calculating generator loss 

    generator.zero_grad()
    generatorOptim.zero_grad()
    genScaler.scale(genLoss).backward()
    genScaler.step(generatorOptim)
    genScaler.update()

    return genLoss.item(), discLoss.item()

def train():
    if not config.training:
        print("Training mode is disabled")
        return
    
    for epoch in range(1, config.epochs+1):
        generator.train()
        discriminator.train()

        loader = tqdm(trainLoader, f"[{epoch}/{config.epochs}]", len(trainLoader))
        genLoss, discLoss = 0, 0
        for x, y in loader:
            x, y = x.to(config.device), y.unsqueeze(1).to(config.device)
            genLoopLoss, discLoopLoss = step(x, y)
            genLoss += genLoopLoss / len(trainLoader)
            discLoss += discLoopLoss / len(trainLoader)

            loader.set_postfix({"genLoss": genLoss, "discLoss": discLoss, "lrG":schedulerG.get_last_lr()[-1], "lrD": schedulerD.get_last_lr()[-1]})

        if epoch % config.decayEvery == 0:
            schedulerG.step()
            schedulerD.step()
            # decaying the learning rates 

        if epoch % config.saveEvery == 0:
            saveModel(generator, config.savePath, f"gen_{epoch}")
            saveModel(discriminator, config.savePath, f"disc_{epoch}")
            # saving models every saveEvery epochs

        for x, _ in testLoader:
            generator.eval()
            discriminator.eval()
            x = x.to(config.device)
            image = generator.forward(x)
            saveImage(x, config.previewPath, f"inp_{epoch}")
            saveImage(image, config.previewPath, epoch)
            break
            # testing the models on the test set

    saveModel(generator, config.savePath, "gen_final")
    saveModel(discriminator, config.savePath, "dis_final")


if __name__ == "__main__":
    train()