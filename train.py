from ultralytics import YOLO
import torch

# paths
log_dir = '/home/sredondo/Projects/YOLOV8/datasets'
yaml_path = '/home/sredondo/Projects/YOLOV8/datasets/safety_full.yaml'

# Select the device where we are going to train the model
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)

# Define the base model
model = YOLO('yolov8n.pt') 

# train the model
_ = model.train(data=yaml_path, epochs = 100, batch = 16)  

# evaluate model performance on the validation set
_ = model.val(data=yaml_path)