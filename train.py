from ultralytics import YOLO
import torch

# paths
log_dir = 'datasets/runs/detect'
yaml_path = 'datasets/safety.yaml'

# Select the device where we are going to train the model
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)

# Define the base model
model = YOLO('yolov8n.pt') 

# train the model
_ = model.train(data=yaml_path, epochs = 50, batch = 32)  

# evaluate model performance on the validation set
_ = model.val(data=yaml_path)