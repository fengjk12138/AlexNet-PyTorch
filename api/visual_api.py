import os
import json
import torch
from PIL import Image
from django.shortcuts import render
from rest_framework.views import APIView
import urllib

import torchvision.transforms as transforms

from alexnet import AlexNet

# model
def index(request):
  """Get the image based on the base64 encoding or url address
          and do the pencil style conversion
  Args:
      request: Post request in url.
      - image_code: 64-bit encoding of images.
      - url:        The URL of the image.
  Returns:
      Base64 bit encoding of the image.
  Notes:
      Later versions will not return an image's address,
      but instead a base64-bit encoded address
  """
  return render(request, "index.html")


class Classifier(APIView):
  """ use dcgan generate animel sister
  """

  @staticmethod
  def get(request):
    """ Get the image based on the base64 encoding or url address
        and do the pencil style conversion
    Args:
        request: Post request in url.
        - image_code: 64-bit encoding of images.
        - url:        The URL of the image.
    Returns:
        Base64 bit encoding of the image.
    Notes:
        Later versions will not return an image's address,
        but instead a base64-bit encoded address
    """
    # Get the url for the image
    image_size = 224

    # Open image
    url = request.POST.get("url")
    image = urllib.request.urlopen(url)
    with open("./static/img.png", "wb") as v:
        v.write(image.read())
    # Preprocess image
    tfms = transforms.Compose([transforms.Resize(image_size), transforms.CenterCrop(image_size), 
                            transforms.ToTensor(),
                            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),])
    image = tfms(image).unsqueeze(0)

    # Load class names
    labels_map = json.load(open('labels_map.txt'))
    labels_map = [labels_map[str(i)] for i in range(1000)]

    # Classify with EfficientNet
    model = AlexNet().from_pretrained('alexnet')
    print('Loaded pretrained weights for alexnet.')
    model.eval()
    with torch.no_grad():
        logits = model(image)
    preds = torch.topk(logits, k=5).indices.squeeze(0).tolist()

    print('-----')
    for idx in preds:
        label = labels_map[idx]
        prob = torch.softmax(logits, dim=1)[0, idx].item()
        message = {"label": label, "prob": prob * 100}
        # print('{:<75} ({:.2f}%)'.format(label, prob*100))

    ret = {
      "status_code": 20000,
      "message": message
    }
    return render(request, "alexnet.html", ret)

  @staticmethod
  def post(request):
    """ Get the image based on the base64 encoding or url address
        and do the pencil style conversion
    Args:
        request: Post request in url.
        - image_code: 64-bit encoding of images.
        - url:        The URL of the image.
    Returns:
        Base64 bit encoding of the image.
    Notes:
        Later versions will not return an image's address,
        but instead a base64-bit encoded address
    """
    # Get the url for the image
    image_size = 224

    # Open image
    url = request.POST.get("url")
    image = urllib.request.urlopen(url)
    with open("./static/img.png", "wb") as v:
        v.write(image.read())
    # Preprocess image
    tfms = transforms.Compose([transforms.Resize(image_size), transforms.CenterCrop(image_size), 
                            transforms.ToTensor(),
                            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),])
    image = tfms(image).unsqueeze(0)

    # Load class names
    labels_map = json.load(open('labels_map.txt'))
    labels_map = [labels_map[str(i)] for i in range(1000)]

    # Classify with EfficientNet
    model = AlexNet().from_pretrained('alexnet')
    print('Loaded pretrained weights for alexnet.')
    model.eval()
    with torch.no_grad():
        logits = model(image)
    preds = torch.topk(logits, k=5).indices.squeeze(0).tolist()

    print('-----')
    for idx in preds:
        label = labels_map[idx]
        prob = torch.softmax(logits, dim=1)[0, idx].item()
        message = {"label": label, "prob": prob * 100}
        # print('{:<75} ({:.2f}%)'.format(label, prob*100))

    ret = {
      "status_code": 20000,
      "message": message
    }
    return render(request, "alexnet.html", ret)
