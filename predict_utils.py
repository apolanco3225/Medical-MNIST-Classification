import torchvision as tv
import torch
from PIL import Image
import io
import base64

from medNet import MedNet

model = torch.load("saved_model", map_location="cpu")
classNames = ['AbdomenCT', 'BreastMRI', 'ChestCT', 'CXR', 'Hand', 'HeadCT']

def scale_image(x):          # Pass a PIL image, return a tensor
    y = tv.transforms.ToTensor()(x)
    if(y.min() < y.max()):  # Assuming the image isn't empty, rescale so its values run from 0 to 1
        y = (y - y.min())/(y.max() - y.min()) 
    z = y - y.mean()        # Subtract the mean value of the image
    return z

def predict_image(image: Image):
    imageScaled = scale_image(image)
    imageScaled = imageScaled[None,:]
    image_out = model(imageScaled)
    pred = int(image_out.max(1, keepdim=True)[1][0][0])
    return classNames[pred]

def encode_image(file):
    image = Image.open(file)
    data = io.BytesIO()
    image.save(data, "JPEG")
    encoded_img_data = base64.b64encode(data.getvalue())
    return encoded_img_data.decode("UTF-8")

def image_transform(file):
    image_dict = {"image_data": None, "image_name": None, "image_pred": None}
    image = Image.open(file)
    image_dict["image_name"] = file.filename
    image_dict["image_pred"] = predict_image(image)
    image_dict["image_data"] = encode_image(file)
    return image_dict
    