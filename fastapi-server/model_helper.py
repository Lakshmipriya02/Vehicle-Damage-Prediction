from PIL import Image
import torch
from torchvision import transforms, models
from torch import nn
trained_model = None
class_names = ['F_Breakage', 'F_Crushed', 'F_Normal', 'R_Breakage', 'R_Crushed', 'R_Normal']
# load the pre-trained ResNet model
class CarClassifierResNet(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.model = models.resnet50(weights='DEFAULT')
        # Freeze all Layers except the final fully connected layer
        for param in self.model.parameters():
            param.requires_grad = False
        #Unfreeze Layer 4 and fc layers
        for param in self.model.layer4.parameters():
            param.requires_grad = True
        #Replace the final fully connected layer
        self.model.fc = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(self.model.fc.in_features, num_classes)
        )
    def forward(self, x):
        x = self.model(x)
        return x

def predict(image_path):
    image = Image.open(image_path).convert("RGB")
    #convert to 224 , 224 as our model expects
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    image_tensor = transform(image).unsqueeze(0)
    # (32, 3,224,224) 32 images per batch -> unsqeeze make it like this (1,3,224,224)
    global trained_model
    if trained_model is None:
        trained_model = CarClassifierResNet(num_classes=len(class_names))
        trained_model.load_state_dict(torch.load("model/saved_model.pth"))
        trained_model.eval()
    with torch.no_grad():
        output = trained_model(image_tensor)  # value of o/p = [12,14,9,15,8,20]
        _, predicted_class = torch.max(output, 1)  # 20, 5
        return class_names[predicted_class.item()]