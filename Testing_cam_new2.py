import torch
import torch.nn as nn
from torchvision import transforms
from efficientnet_pytorch import EfficientNet
from collections import OrderedDict
from torch.autograd import Variable
from PIL import Image
import os
import cv2
import time

# Define the device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Running on {device}...")

# Define the classes
classes = ('Basalt', 'Conglomerate', 'Coprolite', 'Dolomite', 'Fossil', 'Granite', 'Limestone', 'Obsidian', 'Sandstone', 'Shale')

# Load the EfficientNet model
efficientnet_model = EfficientNet.from_pretrained('efficientnet-b0', num_classes=10)
efficientnet_model.to(device)
# Modify the classifier part for the EfficientNet mode
classifier = nn.Sequential(OrderedDict([
        ('fc1', nn.Linear(efficientnet_model._fc.in_features, 512)),
        ('relu1', nn.LeakyReLU()),
        ('fc2', nn.Linear(512, 10))
    ]))

efficientnet_model._fc = classifier
efficientnet_model.to(device)

# Load the EfficientNet model's state dictionary from a saved checkpoint
checkpoint_path = 'C:/Users/opvv1/Downloads/epoch_300_acc_0.7553571462631226.pth'
pota=torch.load(checkpoint_path, map_location=device)
efficientnet_model.load_state_dict(pota)
efficientnet_model.eval()

# Function to preprocess and classify an image using the EfficientNet model
def pre_image(image_path, model):
    img = Image.open(image_path)
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    transform_norm = transforms.Compose([transforms.ToTensor(),
                                         transforms.Resize((224, 224)), transforms.Normalize(mean, std)])
    # Get normalized image
    img_normalized = transform_norm(img).unsqueeze(0).to(device)

    with torch.no_grad():
        model.eval()
        outputs = model(img_normalized)
        ps = torch.softmax(outputs, dim=1)
        index = ps.argmax().item()
        class_name = classes[index]
        return class_name, ps

# Function to count the number of images in a directory
def num_img(site_num):
    dir_path = f'C:/Users/opvv1/Downloads/kratosw/site_{site_num}'
    count = 0
    for path in os.listdir(dir_path):
        if os.path.isfile(os.path.join(dir_path, path)):
            count += 1
    return count

# Get user input for site number
site_num = input("Enter Site Number: ")

# OpenCV code for capturing images
cap = cv2.VideoCapture(0)
i = 0
n = 0
while True:
    _, frame = cap.read()
    cv2.imshow('Frame', frame)
    if n == 0:
        n += 1
        begin = time.time()
        end = time.time() + 20
    if time.time() > begin + 10:
        i += 1
        if i % 6 == 0:
            frame_6 = frame
            cv2.imshow('6th frame', frame_6)
            img_path = f'C:/Users/opvv1/Downloads/kratosw/site_{site_num}/{int(i/6)}.jpg'
            cv2.imwrite(img_path, frame_6)
            print(f"Image {int(i/6)} captured and saved at {img_path}")

    k = cv2.waitKey(5) & 0xFF
    if k == 27:
        break

    if time.time() > end:
        print("\nImage capture over\n")
        print(f"Time taken: {time.time()-begin} secs\n")
        break

cv2.destroyAllWindows()
cap.release()

# Passing all images through the model
sum_preds = torch.zeros(len(classes))
total_imgs = num_img(site_num)
print(f"Total Number of images for site {site_num} = {total_imgs}")
for img_num in range(1, total_imgs+1):
    img_path = f'C:/Users/opvv1/Downloads/kratosw/site_{site_num}/{img_num}.jpg'
    class_name, predicted_vals = pre_image(img_path, efficientnet_model)
    sum_preds += predicted_vals[0]

average_preds = sum_preds / total_imgs
index = average_preds.argmax().item()

print(f"\nTotal breakdown for Site {site_num} :")
for i, class_name in enumerate(classes):
    print(f"{class_name}: {average_preds[i]}")

print(f"\nFinal Prediction: {classes[index]}\n")