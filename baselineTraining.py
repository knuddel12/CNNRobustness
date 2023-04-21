import matplotlib
matplotlib.use("TkAgg")
import torch
import numpy as np
from training import init_dataloaders, init_new_resnet, paths, init_datasets, reset_randomness
from Utilities.Datasets.transforms import Load, ToTensor, cache_preprocessing
import neptune.new as neptune
from Dev.LarsDev.transforms import FaceAlignAndCrop, Grayscale
import math
from torch import nn, optim
import tqdm
from neptune.new.types import File
from kornia.augmentation.container import AugmentationSequential
from kornia.augmentation import (RandomHorizontalFlip,
                                 RandomRotation)


def train_loop():
    """ Trains a given NN on given parameters and saves NN in the end to given Path.
    """
    # Run neptune logging process
    run = neptune.init(
        project="dermalog-face/FaceRecognitionWithBlur",
        api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiI3OWYyNGIwYi03MGEyLTRjOTAtYTVmMi01NGQxOTZjYzJiMmMifQ==",
        description=description,
        tags=platform,
    )
    print("Training Parameters:")
    print("Batch size =", batchsize)
    batchesperepoch = math.floor(len(dataset_train) / batchsize)
    print("Batches per epoch: ", batchesperepoch)
    print("Number of Classes =", num_classes)
    print("Number of instances =", len(dataset_train.data))
    print("Commencing Training")
    # Define loss and optimizer
    min_valid_loss = np.inf
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(net.parameters(), lr=lr, amsgrad=True, eps=1e-6)
    # Define parameters for neptune log
    params = {
        "lr": lr,
        "batch size": batchsize,
        "input size": input_size,
        "n_classes": num_classes,
        "optimizer": type(optimizer).__name__,
        "NN": "resnet101",
        "Dataset": csv_name,
        "Model_Path": MODEL_PATH,
        "platform": platform,
        "epochs": epochs,
    }
    run["parameters"] = params
    # Training, loop over the dataset
    for epoch in tqdm.tqdm(range(epochs), total=epochs):
        running_loss = 0.0
        net.train()
        for i, data in enumerate(train_dataloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs = torch.squeeze(data[0]).to(device)
            labels = torch.squeeze(data[1]).to(device)
            inputs = inputs - 0.5
            # zero the parameter gradients
            optimizer.zero_grad()
            # forward + backward + optimize
            outputs = net(inputs.float())
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            # Logging loss batch-wise
            run["train/batch/train_loss_per_batch"].log(loss.item())
            # Logging the very first 90 queried images
            if epoch == 0 and i == 0:
                for j in range(90):
                    run[f'train/images_sigma1'].log(File.as_image(((inputs[j] + 0.5).permute(1, 2, 0).cpu().numpy())))
        valid_loss = 0.0
        # Validate training progress after every epoch
        net.eval()
        for i, data in enumerate(validation_dataloader, 0):
            inputsVal = torch.squeeze(data[0]).to(device)
            labelsVal = torch.squeeze(data[1]).to(device)
            inputsVal = inputsVal - 0.5
            outputsVal = net(inputsVal.float())
            loss = criterion(outputsVal, labelsVal)
            valid_loss += loss.item()
            run["train/batch/valid_loss_per_batch"].log(loss.item())
        run["train/batch/train_loss"].log((running_loss / len(train_dataloader)))
        run["train/batch/valid_loss"].log((valid_loss / len(validation_dataloader)))
        print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / batchesperepoch:.6f}')
        print(
            f'Epoch {epoch + 1} \t\t Training Loss: {running_loss / len(train_dataloader):.6f} \t\t Validation Loss: {valid_loss / len(validation_dataloader):.6f}')
        # If validation loss decreased, save network state
        if min_valid_loss > valid_loss:
            print(f'Validation Loss Decreased({min_valid_loss:.6f}--->{valid_loss:.6f}) \t Saving The Model')
            min_valid_loss = valid_loss
            # Saving State Dict
            torch.save(net.state_dict(), MODEL_PATH)
    print('Finished Training, testing Accuracy...')
    params["Accuracy"] = f'{test_loop(MODEL_PATH, net)}%'
    run["parameters"] = params

# Test loop for testing the accuracy of the network
def test_loop(MODEL_PATH,
              net):
    net.load_state_dict(torch.load(MODEL_PATH))
    net.eval()
    print("Last saved Model loaded.")
    correct = 0
    total = 0
    correct_pred = np.zeros(num_classes)
    pred_counter = np.zeros(num_classes)
    with torch.no_grad():
        for data in test_dataloader:
            imagesTest = torch.squeeze(data[0]).to(device)
            labelsTest = torch.squeeze(data[1]).to(device)
            # calculate outputs by running images through the network
            imagesTest = imagesTest - 0.5
            outputsTest = net(imagesTest.float())
            # the class with the highest energy is what we choose as prediction
            _, predicted = torch.max(outputsTest.data, 1)
            for j in range(len(imagesTest)):
                pred_counter[labelsTest[j]] += 1
                if labelsTest[j] == predicted[j]:
                    correct_pred[labelsTest[j]] += 1
            total += labelsTest.size(0)
            correct += (predicted == labelsTest).sum().item()
    print(f'Accuracy of the network on the test images: {100 * correct // total} %')
    return (100 * correct // total)


# Reset all randomization variables
reset_randomness()
""" Set training parameters"""
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
input_size = 112
batchsize = 512
epochs = 400
lr = 5e-3
csv_name = 'facescrub_cleanup.csv'
CSV_PATH, CACHE_PATH, platform = paths(csv_name)
MODEL_PATH = 'baseline.pth'
description = "baseline"
""" define transforms """
transforms_once = [Load(cache_images=True, cache_folder=CACHE_PATH),
                   FaceAlignAndCrop(),
                   Grayscale(),
                   ToTensor(),
                   cache_preprocessing(CACHE_PATH)]
kornia_augs = AugmentationSequential(
    RandomRotation(degrees=20, p=0.25),
    RandomHorizontalFlip(p=0.2))

""" create dataset """
dataset_train, dataset_test, dataset_validation = init_datasets(CSV_PATH, transforms_once,
                                                                image_only_transforms=kornia_augs)

train_dataloader, test_dataloader, validation_dataloader = init_dataloaders(dataset_train,
                                                                            dataset_test,
                                                                            dataset_validation,
                                                                            batchsize,
                                                                            shuffle=True)
num_classes = max(max(dataset_train.df.iloc[:, 3]), max(dataset_test.df.iloc[:, 3]),
                  max(dataset_validation.df.iloc[:, 3])) + 1
net = init_new_resnet(num_classes, device, pretrained=False)
train_loop()
