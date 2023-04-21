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
from kornia import filters
from kornia.augmentation import (RandomHorizontalFlip,
                                 RandomRotation,
                                 RandomGaussianBlur,
                                 RandomMotionBlur)


def twosplit_gaussian_blur_train_loop(input_size, batchsize, epochs, lr, sigma1, sigma2, kernel_size):
    """ Trains a given NN on given parameters and saves NN in the end to given Path.
    params:
    sigma: blur std dev
    """
    MODEL_PATH = f'two_split_gaussian_blur_{sigma1}_to_{sigma2}_lr{lr}_epochs{epochs}.pth'
    description = f"two-split Training, GaussianBlur: {sigma1} to {sigma2}"
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
    print("Sigma2 =", sigma2)
    print("Commencing Training")
    min_valid_loss = np.inf
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(net.parameters(), lr=lr, amsgrad=True, eps=1e-6)
    curr_sigma = sigma1
    blur_img = RandomGaussianBlur(kernel_size, sigma=[sigma1, sigma1], p=1)
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
        "sigma1": sigma1,
        "sigma2": sigma2}
    run["parameters"] = params
    for epoch in tqdm.tqdm(range(epochs), total=epochs):  # loop over the dataset multiple times
        running_loss = 0.0
        if epoch == (epochs / 2):
            blur_img = RandomGaussianBlur([9, 9], sigma=[sigma2, sigma2], p=1)
            curr_sigma = sigma2
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
            # Log very first 90 images
            if epoch == 0 and i == 0:
                for j in range(90):
                    run[f'train/images_sigma1'].log(File.as_image(((inputs[j] + 0.5).permute(1, 2, 0).cpu().numpy())))
            # Log first 90 images after blur change
            if epoch == (epochs/2) and i == 0:
                for j in range(90):
                    run[f'train/images_sigma2'].log(File.as_image(((inputs[j] + 0.5).permute(1, 2, 0).cpu().numpy())))
        valid_loss = 0.0
        # Validation
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
        # print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / batchesperepoch:.6f}')
        print(
            f'Epoch {epoch + 1} \t\t Training Loss: {running_loss / len(train_dataloader):.6f} \t\t Validation Loss: {valid_loss / len(validation_dataloader):.6f}')
        if min_valid_loss > valid_loss:
            print(f'Validation Loss Decreased({min_valid_loss:.6f}--->{valid_loss:.6f}) \t Saving The Model')
            min_valid_loss = valid_loss
            # Saving State Dict
            torch.save(net.state_dict(), MODEL_PATH)
    print('Finished Training, testing Accuracy...')
    params["Accuracy"] = f'{test_loop(MODEL_PATH, net)}%'
    run["parameters"] = params


def twosplit_median_blur_train_loop(ks1, ks2):
    """ Trains a given NN on given parameters and saves NN in the end to given Path.
    params:
    sigma: blur std dev
    """
    MODEL_PATH = f'2split_median_blur_{ks1}_to_{ks2}_lr{lr}_epochs{epochs}.pth'
    description = f"2-split Training, Median Blur: kernelsize {ks1} to {ks1}"
    run = neptune.init(
        project="dermalog-face/FaceRecognitionWithBlur",
        api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiI3OWYyNGIwYi03MGEyLTRjOTAtYTVmMi01NGQxOTZjYzJiMmMifQ==",
        description=description,
        tags=platform)
    print("Training Parameters:")
    print("Batch size =", batchsize)
    batchesperepoch = math.floor(len(dataset_train) / batchsize)
    print("Batches per epoch: ", batchesperepoch)
    print("Number of Classes =", num_classes)
    print("Number of instances =", len(dataset_train.data))
    print("ks1 =", ks1)
    print("ks2 =", ks2)
    print("Commencing Training")
    min_valid_loss = np.inf
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(net.parameters(), lr=lr, amsgrad=True, eps=1e-6)
    blur_img = filters.MedianBlur([ks1,ks1])
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
        "sigma1": sigma1,
        "sigma2": sigma2}
    run["parameters"] = params
    for epoch in tqdm.tqdm(range(epochs), total=epochs):  # loop over the dataset multiple times
        running_loss = 0.0
        if epoch == (epochs/2):
            blur_img = filters.MedianBlur([ks2, ks2])
        net.train()
        for i, data in enumerate(train_dataloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs = torch.squeeze(data[0]).to(device)
            labels = torch.squeeze(data[1]).to(device)
            if blur_img.kernel_size != [0,0]:
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
            if blur_img.kernel_size != [0,0]:
                inputsVal = blur_img(inputsVal)
            inputsVal = inputsVal - 0.5
            outputsVal = net(inputsVal.float())
            loss = criterion(outputsVal, labelsVal)
            valid_loss += loss.item()
            run["train/batch/valid_loss_per_batch"].log(loss.item())
        run["train/batch/train_loss"].log((running_loss / len(train_dataloader)))
        run["train/batch/valid_loss"].log((valid_loss / len(validation_dataloader)))
        # print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / batchesperepoch:.6f}')
        print(
            f'Epoch {epoch + 1} \t\t Training Loss: {running_loss / len(train_dataloader):.6f} \t\t Validation Loss: {valid_loss / len(validation_dataloader):.6f}')
        if min_valid_loss > valid_loss:
            print(f'Validation Loss Decreased({min_valid_loss:.6f}--->{valid_loss:.6f}) \t Saving The Model')
            min_valid_loss = valid_loss
            # Saving State Dict
            torch.save(net.state_dict(), MODEL_PATH)
    print('Finished Training, testing Accuracy...')
    params["Accuracy"] = f'{test_loop(MODEL_PATH, net)}%'
    run["parameters"] = params


def twosplit_box_blur_train_loop(ks1, ks2):
    """ Trains a given NN on given parameters and saves NN in the end to given Path.
    params:
    sigma: blur std dev
    """
    MODEL_PATH = f'2split_box_blur_{ks1}_to_{ks2}_lr{lr}_epochs{epochs}.pth'
    description = f"2-split Training, Box Blur: kernelsize {ks1} to {ks2}"
    run = neptune.init(
        project="dermalog-face/FaceRecognitionWithBlur",
        api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiI3OWYyNGIwYi03MGEyLTRjOTAtYTVmMi01NGQxOTZjYzJiMmMifQ==",
        description=description,
        tags=platform)
    print("Training Parameters:")
    print("Batch size =", batchsize)
    batchesperepoch = math.floor(len(dataset_train) / batchsize)
    print("Batches per epoch: ", batchesperepoch)
    print("Number of Classes =", num_classes)
    print("Number of instances =", len(dataset_train.data))
    print("ks1 =", ks1)
    print("ks2 =", ks2)
    print("Commencing Training")
    min_valid_loss = np.inf
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(net.parameters(), lr=lr, amsgrad=True, eps=1e-6)
    blur_img = filters.BoxBlur(kernel_size=[ks1,ks1])
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
        "ks1": ks1,
        "ks2": ks2}
    run["parameters"] = params
    for epoch in tqdm.tqdm(range(epochs), total=epochs):  # loop over the dataset multiple times
        running_loss = 0.0
        if epoch == (epochs/2):
            blur_img = filters.BoxBlur(kernel_size=[ks2,ks2])
        net.train()
        for i, data in enumerate(train_dataloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs = torch.squeeze(data[0]).to(device)
            labels = torch.squeeze(data[1]).to(device)
            if blur_img.kernel_size != [0,0]:
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
            if blur_img.kernel_size != [0,0]:
                inputsVal = blur_img(inputsVal)
            inputsVal = inputsVal - 0.5
            outputsVal = net(inputsVal.float())
            loss = criterion(outputsVal, labelsVal)
            valid_loss += loss.item()
            run["train/batch/valid_loss_per_batch"].log(loss.item())
        run["train/batch/train_loss"].log((running_loss / len(train_dataloader)))
        run["train/batch/valid_loss"].log((valid_loss / len(validation_dataloader)))
        # print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / batchesperepoch:.6f}')
        print(
            f'Epoch {epoch + 1} \t\t Training Loss: {running_loss / len(train_dataloader):.6f} \t\t Validation Loss: {valid_loss / len(validation_dataloader):.6f}')
        if min_valid_loss > valid_loss:
            print(f'Validation Loss Decreased({min_valid_loss:.6f}--->{valid_loss:.6f}) \t Saving The Model')
            min_valid_loss = valid_loss
            # Saving State Dict
            torch.save(net.state_dict(), MODEL_PATH)
    print('Finished Training, testing Accuracy...')
    params["Accuracy"] = f'{test_loop(MODEL_PATH, net)}%'
    run["parameters"] = params


def twosplit_motion_blur_train_loop(ks1, ks2, angle):
    """ Trains a given NN on given parameters and saves NN in the end to given Path.
    params:
    sigma: blur std dev
    """
    MODEL_PATH = f'2split_motion_blur_{ks1}_to_{ks2}_angle{angle}_lr{lr}_epochs{epochs}.pth'
    description = f"2-split Training, Motion Blur: kernelsize {ks1} to {ks2}, angle {angle}"
    run = neptune.init(
        project="dermalog-face/FaceRecognitionWithBlur",
        api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiI3OWYyNGIwYi03MGEyLTRjOTAtYTVmMi01NGQxOTZjYzJiMmMifQ==",
        description=description,
        tags=platform)
    print("Training Parameters:")
    print("Batch size =", batchsize)
    batchesperepoch = math.floor(len(dataset_train) / batchsize)
    print("Batches per epoch: ", batchesperepoch)
    print("Number of Classes =", num_classes)
    print("Number of instances =", len(dataset_train.data))
    print("ks1 =", ks1)
    print("ks2 =", ks2)
    print("Commencing Training")
    min_valid_loss = np.inf
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(net.parameters(), lr=lr, amsgrad=True, eps=1e-6)
    if ks1 != 0:
        blur_img = RandomMotionBlur(kernel_size=int(ks1), angle=360, direction=0, p=1)
    current_ks = ks1
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
        "ks1": ks1,
        "ks2": ks2}
    run["parameters"] = params
    for epoch in tqdm.tqdm(range(epochs), total=epochs):  # loop over the dataset multiple times
        running_loss = 0.0
        if epoch == (epochs/2):
            current_ks = ks2
            if ks2 != 0:
                blur_img = RandomMotionBlur(kernel_size=int(ks2), angle=360, direction=0, p=1)
        net.train()
        for i, data in enumerate(train_dataloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs = torch.squeeze(data[0]).to(device)
            labels = torch.squeeze(data[1]).to(device)
            if current_ks != 0:
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
            if current_ks != 0:
                inputsVal = blur_img(inputsVal)
            inputsVal = inputsVal - 0.5
            outputsVal = net(inputsVal.float())
            loss = criterion(outputsVal, labelsVal)
            valid_loss += loss.item()
            run["train/batch/valid_loss_per_batch"].log(loss.item())
        run["train/batch/train_loss"].log((running_loss / len(train_dataloader)))
        run["train/batch/valid_loss"].log((valid_loss / len(validation_dataloader)))
        # print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / batchesperepoch:.6f}')
        print(
            f'Epoch {epoch + 1} \t\t Training Loss: {running_loss / len(train_dataloader):.6f} \t\t Validation Loss: {valid_loss / len(validation_dataloader):.6f}')
        if min_valid_loss > valid_loss:
            print(f'Validation Loss Decreased({min_valid_loss:.6f}--->{valid_loss:.6f}) \t Saving The Model')
            min_valid_loss = valid_loss
            # Saving State Dict
            torch.save(net.state_dict(), MODEL_PATH)
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
    write_list = []
    correct_pred = np.zeros(num_classes)
    pred_counter = np.zeros(num_classes)
    with torch.no_grad():
        for data in test_dataloader:
            imagesTest = torch.squeeze(data[0]).to(device)
            labelsTest = torch.squeeze(data[1]).to(device)
            imagesTest = imagesTest - 0.5
            # calculate outputs by running images through the network
            outputsTest = net(imagesTest.float())
            # for k in range(batchsize):
            #     if not labelsTest[k] in write_list:
            #         write_list.append((labelsTest[k],data[2]['path']))
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
sigma2 = 0
ks1 = 9
ks2 = 0
angle = 360
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
""" Choose training loop """
twosplit_gaussian_blur_train_loop(sigma1=sigma1, sigma2=sigma2)