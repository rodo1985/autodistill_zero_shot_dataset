# %%
from GroundingDINO.groundingdino.util.inference import load_model, load_image, predict, annotate
import cv2
from matplotlib import pyplot as plt
import os
import shutil
import torch
import yaml
import numpy as np
from tqdm import tqdm

# check if cuda is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# %%
# load model
model = load_model("GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py", "GroundingDINO/weights/groundingdino_swint_ogc.pth")

# model to gpu
model.to(device)

IMAGE_PATH = "simple_images/safety glasses/safety glasses_47.jpg"
TEXT_PROMPT = 'a person with a hard hat and safety glasses'
BOX_TRESHOLD = 0.5
TEXT_TRESHOLD = 0.25

# load image
image_source, image = load_image(IMAGE_PATH)

# move image to gpu
image.to(device)

boxes, logits, phrases = predict(
    model=model,
    image=image,
    caption=TEXT_PROMPT,
    box_threshold=BOX_TRESHOLD,
    text_threshold=TEXT_TRESHOLD
)

annotated_frame = annotate(image_source=image_source, boxes=boxes, logits=logits, phrases=phrases)

# show the image using opencv and matplotlib
plt.imshow(cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB))
plt.show()


# %%
# parameters
CLEAN_FOLDER = True
PLOT_IMAGES = False
BREAK_AFTER = 0 # if > 0, break after this many images
BOX_TRESHOLD = 0.5
TEXT_TRESHOLD = 0.25

TEXT_PROMPT = 'a person with a hard hat and safety glasses'
classes = ['hard hat', 'safety glasses', 'person']

input_folder = 'simple_images'
output_folder = '/media/datasets/safety'
image_folder = os.path.join(output_folder, 'images')
annotation_folder = os.path.join(output_folder, 'labels')

# if clean folder is true, delete the output folder
if CLEAN_FOLDER and os.path.exists(output_folder):
    shutil.rmtree(output_folder)

# if output folder does not exist, create it
if not os.path.exists(output_folder):
    os.makedirs(output_folder)
    os.makedirs(image_folder)
    os.makedirs(annotation_folder)

# TODO create yaml file

# # create the dictionary to be converted to YAML
# data = {
#     'path': output_folder,
#     'train': 'images',
#     'val': 'images',
#     'nc': len(classes),
#     'names': '['
# }

# # convert the dictionary to YAML
# yaml_data = yaml.dump(data)

# # write the YAML data to a file
# with open(output_folder + '.yaml', 'w') as f:
#     f.write(yaml_data)


# get all directories in input folder
directories = os.listdir(input_folder)

# iterate over all directories
for directory in directories:

    print('Processing directory:', directory)
    print('-'*10)

    # get all images in the input folder
    images = os.listdir(os.path.join(input_folder, directory))

    if BREAK_AFTER and len(images) > BREAK_AFTER:
        images = images[:BREAK_AFTER]

    # for each image in the input folder
    for idx, file in enumerate(tqdm(images)):

        # create the full input path and read the file
        input_path = os.path.join(input_folder, directory, file)
        image_source, image = load_image(input_path)

        # move image to gpu
        image.to(device)

        # for each class
        for class_name in classes:
            with torch.no_grad():
                
                # predict the bounding boxes, logits and phrases
                boxes, logits, phrases = predict(
                    model=model,
                    image=image,
                    caption=TEXT_PROMPT,
                    box_threshold=BOX_TRESHOLD,
                    text_threshold=TEXT_TRESHOLD
                )

                # modify phrases with classes names
                for c in classes:
                    for idx, p in enumerate(phrases):
                        if c in p:
                            phrases[idx] = c
        
        
        # if the model finds any bounding boxes
        if len(boxes) > 0:
            
            if PLOT_IMAGES:
                # create annotations show the image using opencv and matplotlib
                annotated_frame = annotate(image_source=image_source, boxes=boxes, logits=logits, phrases=phrases)
                plt.imshow(cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB))
                plt.show()

            # write the image to the image folder
            image_path = os.path.join(image_folder, file)
            cv2.imwrite(image_path, cv2.cvtColor(image_source, cv2.COLOR_BGR2RGB))

            # create the annotation file
            annotation_file = file.replace('.jpg', '.txt')
            annotation_path = os.path.join(annotation_folder, annotation_file)

            # open the annotation file
            with open(annotation_path, 'w') as f:
                
                # convert out phrases to indices
                out_phrases = [classes.index(s) for s in phrases]

                # for each bounding box
                for box, logits, phrase in zip(boxes, logits, out_phrases):
                    # write the bounding box to the file
                    f.write(f"{phrase} {box[0]} {box[1]} {box[2]} {box[3]}\n")
        
    

    


