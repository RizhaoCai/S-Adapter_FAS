import torch
import timm
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms

# Define the model
model = timm.create_model('vit_base_patch16_224', pretrained=True)

# Load the image and preprocess it
img_path = 'path/to/your/image.jpg'
img = Image.open(img_path)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
img_tensor = transform(img)
img_tensor = img_tensor.unsqueeze(0) # Add batch dimension

# Get the attention maps
outputs = model(img_tensor)
attention_maps = outputs[-1] # Last output of the model is the attention maps

# Plot the attention maps
plt.figure(figsize=(16, 16))
for i, attention_map in enumerate(attention_maps):
    plt.subplot(8, 8, i+1)
    plt.imshow(attention_map.squeeze().detach().cpu(), cmap='gray')
    plt.axis('off')
plt.tight_layout()
plt.show()