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
                                 RandomRotation,
                                 RandomGaussianBlur)

def iterative_blur_train_loop(sigma1):
    """ Trains a given NN on given parameters and saves NN in the end to given Path.
    params:
    sigma: blur std dev
    """
    MODEL_PATH = f'iterative_blur_{sigma1}_lr{lr}_epochs{epochs}_new.pth'
    description = f"iterative blur Training, GaussianBlur: {sigma1}"
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    run = neptune.init(
        project="dermalog-face/FaceRecognitionWithBlur",
        api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiI3OWYyNGIwYi03MGEyLTRjOTAtYTVmMi01NGQxOTZjYzJiMmMifQ==",
        description=description,
        tags=platform)
    print("Training Parameters:")
    print("Batch size =", batchsize)
    print("Batches per epoch: ", math.floor(len(dataset_train) / batchsize))
    print("Number of Classes =", num_classes)
    print("Number of instances =", len(dataset_train.data))
    print("Sigma1 =", sigma1)
    print("Commencing Training")
    min_valid_loss = np.inf
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(net.parameters(), lr=lr, amsgrad=True, eps=1e-6)
    curr_sigma = sigma1
    blur_img = RandomGaussianBlur([9, 9], sigma=[sigma1, sigma1], p=1)
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
        "sigma1": sigma1}
    run["parameters"] = params
    epoch_counter = 0
    for epoch in tqdm.tqdm(range(epochs), total=epochs):  # loop over the dataset multiple times
        running_loss = 0.0
        epoch_counter += 1
        if epoch >= 60:
            if curr_sigma > 0:
                curr_sigma -= 0.025
                blur_img = RandomGaussianBlur([9, 9], sigma=[curr_sigma, curr_sigma], p=1)
        net.train()
        for i, data in enumerate(train_dataloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs = torch.squeeze(data[0]).to(device)
            labels = torch.squeeze(data[1]).to(device)
            if curr_sigma != 0:
                inputs = blur_img(inputs)
            inputs = inputs - 0.5
            # zero the parameter gradients
            optimizer.zero_grad()
            # forward + backward + optimize
            outputs = net(inputs.float())
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            run["train/batch/train_loss_per_batch"].log(loss.item())
            if epoch == 0 and i == 0:
                for j in range(90):
                    run[f'train/images_sigma1'].log(File.as_image(((inputs[j] + 0.5).permute(1, 2, 0).cpu().numpy())))
            if epoch == (epochs/2) and i == 0:
                for j in range(90):
                    run[f'train/images_sigma2'].log(File.as_image(((inputs[j] + 0.5).permute(1, 2, 0).cpu().numpy())))
        valid_loss = 0.0
        net.eval()
        for i, data in enumerate(validation_dataloader, 0):
            inputsVal = torch.squeeze(data[0]).to(device)
            labelsVal = torch.squeeze(data[1]).to(device)
            if curr_sigma != 0:
                inputsVal = blur_img(inputsVal)
            inputsVal = inputsVal - 0.5
            outputsVal = net(inputsVal.float())
            loss = criterion(outputsVal, labelsVal)
            valid_loss += loss.item()
            run["train/batch/valid_loss_per_batch"].log(loss.item())
        run["train/batch/train_loss"].log((running_loss / len(train_dataloader)))
        run["train/batch/valid_loss"].log((valid_loss / len(validation_dataloader)))
        run["train/batch/blur_stddev_sigma"].log(curr_sigma)
        # print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / batchesperepoch:.6f}')
        print(
            f'Epoch {epoch + 1} \t\t Training Loss: {running_loss / len(train_dataloader):.6f} \t\t Validation Loss: {valid_loss / len(validation_dataloader):.6f}')
        MODEL_PATH_EXTENDED = MODEL_PATH + f"_curr_{epoch+1}_iterative.pth"
        if min_valid_loss > valid_loss:
            print(f'Validation Loss Decreased({min_valid_loss:.6f}--->{valid_loss:.6f}) \t Saving The Model')
            min_valid_loss = valid_loss
            # Saving State Dict
            #torch.save(net.state_dict(), MODEL_PATH)
        if epoch_counter == 10:
            torch.save(net.state_dict(), MODEL_PATH_EXTENDED)
            epoch_counter = 0
    print('Finished Training, testing Accuracy...')
    params["Accuracy"] = f'{test_loop(MODEL_PATH, net)}%'
    run["parameters"] = params

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
            imagesTest = imagesTest - 0.5
            # calculate outputs by running images through the network
            outputsTest = net(imagesTest.float())
            # the class with the highest energy is what we choose as prediction
            _, predicted = torch.max(outputsTest.data, 1)
            for j in range(len(imagesTest)):
                pred_counter[labelsTest[j]] +=1
                if labelsTest[j] == predicted[j]:
                    correct_pred[labelsTest[j]] += 1
            total += labelsTest.size(0)
            correct += (predicted == labelsTest).sum().item()
    print(f'Accuracy of the network on the test images: {100 * correct // total} %')
    return (100 * correct // total)

reset_randomness()
""" Set training parameters"""
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
input_size = 112
batchsize = 512
epochs = 400
lr = 5e-3
sigma1 = 3
csv_name = 'facescrub_cleanup.csv'
CSV_PATH, CACHE_PATH, platform = paths(csv_name)

""" define transforms """
transforms_once = [Load(cache_images=True, cache_folder=CACHE_PATH),
                   FaceAlignAndCrop(),
                   Grayscale(),
                   ToTensor(),
                   cache_preprocessing(CACHE_PATH)]
kornia_augs = AugmentationSequential(
    RandomRotation(degrees=20, p=0.25),
    RandomHorizontalFlip(p=0.2)
    )
""" create datasets and dataloaders """
dataset_train, dataset_test, dataset_validation = init_datasets(CSV_PATH, transforms_once, image_only_transforms=kornia_augs)

train_dataloader, test_dataloader, validation_dataloader = init_dataloaders(dataset_train,
                                                                            dataset_test,
                                                                            dataset_validation,
                                                                            batchsize,
                                                                            shuffle=True)
num_classes = max(max(dataset_train.df.iloc[:, 3]), max(dataset_test.df.iloc[:, 3]), max(dataset_validation.df.iloc[:, 3])) + 1

net = init_new_resnet(num_classes, device, pretrained=False)
""" Run training """
iterative_blur_train_loop(sigma1=sigma1)