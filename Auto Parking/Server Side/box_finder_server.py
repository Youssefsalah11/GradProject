from slots_data_finder import slots_data_finder
import torch
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import torchvision.transforms.functional as F
import matplotlib.pyplot as plt
from PIL import Image
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as T
from torchvision.utils import draw_bounding_boxes
import matplotlib.patches as patches
from flask import Flask, request, jsonify
import io

app = Flask(__name__)

def load_image(image):
    transforms = T.Compose([
        T.ToTensor(),
    ])
    # image = Image.open(image_path).convert("L")
    image = transforms(image)
    return image

def show_image_with_boxes2(img, target, scores,ax=None):
    """Utility function to display an image with its bounding boxes."""
    if ax is None:
        _, ax = plt.subplots(1, figsize=(12, 9))
    
    ax.imshow(img, cmap='gray')  # Grayscale display
    
    for idx,box in enumerate(target):
        if scores[idx] < 0.5:
            continue
            
        xmin, ymin, xmax, ymax = box
        width = xmax - xmin
        height = ymax - ymin

        rect = patches.Rectangle((box[0].item(), box[1].item()), box[2].item() - box[0].item(), box[3].item() - box[1].item(), linewidth=2, edgecolor='r', facecolor='none')

        ax.add_patch(rect)


        


class ToTensor(object):
    """Convert PIL image to tensor and adjust target format."""
    def __call__(self, image, target):
        image = F.to_tensor(image)
        return image, target


def get_transform(train):
    transforms = []
    transforms.append(ToTensor())
#     if train:
#         transforms.append(RandomHorizontalFlip())
    return Compose(transforms)

class Compose(object):
    """Composes several transforms together for image and target."""
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target
def get_model_instance_segmentation(num_classes):
    # Load a pre-trained model for classification and return only the features
    model = fasterrcnn_resnet50_fpn(pretrained=True)
    # Replace the classifier with a new one, that has num_classes which is user-defined
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model


def box_finder(image):

    # Assuming you have 2 classes (1 parking slot + background)
    model = get_model_instance_segmentation(2)
    model.load_state_dict(torch.load('../parking_slot_detector.pth',map_location=torch.device('cpu')))
    model.eval()
    model.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    # image_path = '/Users/youssefsalah/Documents/Final Project (siemens)/Parking slot dataset/batch3/New Project (36).jpg'
    image = image
    # Prepare image for model
    image = image.unsqueeze(0)  # Add batch dimension
    image = image.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

    # Get predictions
    with torch.no_grad():
        prediction = model(image)

    prediction.append(image.shape) 
    # show_image_with_boxes2(image.squeeze(0).squeeze(0).cpu(), prediction[0]['boxes'], prediction[0]['scores'])
    return prediction
    


@app.route('/call_function', methods=['POST'])
def call_function():
    file = request.files['image']
    image = Image.open(io.BytesIO(file.read())).convert("L")
    image = load_image(image)

    # Call your function with the parameters
    result = box_finder(image)
    poses_data = slots_data_finder(result)

    # result[0]['boxes'] = result[0]['boxes'].numpy().tolist()
    # result[0]['labels'] = result[0]['labels'].numpy().tolist()
    # result[0]['scores'] = result[0]['scores'].numpy().tolist()
    print(poses_data)

    poses_data[0]['start'][0] = poses_data[0]['start'][0].item()
    poses_data[0]['start'][1] = poses_data[0]['start'][1].item()

    poses_data[0]['mid'][0] = poses_data[0]['mid'][0].item()
    poses_data[0]['mid'][1] = poses_data[0]['mid'][1].item()

    poses_data[0]['end'][0] = poses_data[0]['end'][0].item()
    poses_data[0]['end'][1] = poses_data[0]['end'][1].item()
    
    print(poses_data)
    # Return the result as JSON
    return jsonify({'result': poses_data})



if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5005)
