import os
import shutil
from tqdm import tqdm
import yaml
import zipfile


def zipdir(path, ziph, progress_bar):
    # ziph is zipfile handle
    for root, dirs, files in os.walk(path):
        for file in tqdm(files, unit="file", desc="Zipping"):
            full_path = os.path.join(root, file)
            relative_path = os.path.relpath(full_path, path)  # This will give you relative paths for zipped files
            ziph.write(full_path, relative_path)
            progress_bar.update(os.path.getsize(full_path))

def main():
    # Load the config file
    with open('config.yaml', 'r') as file:
        config = yaml.safe_load(file)

    # Parse paths
    image_folder =  config['paths']['image_folder']
    annotation_folder =  config['paths']['annotation_folder']
    output_temp_folder = config['paths']['output_temp_folder']
    obj_train_data_folder = config['paths']['obj_train_data_folder']

    # Parse classes
    classes = config['classes']

    # zip files path
    annotations_zip_path = os.path.join(os.path.dirname(output_temp_folder), 'annotations.zip')
    images_zip_path = os.path.join(os.path.dirname(output_temp_folder), 'images.zip')

    # if folder exists, delete it
    if os.path.exists(output_temp_folder):
        shutil.rmtree(output_temp_folder)

    # if zip exists, delete it
    if os.path.exists(annotations_zip_path):
        os.remove(annotations_zip_path)
    if os.path.exists(images_zip_path):
        os.remove(images_zip_path)

    #########################
    # obj_train_data folder #
    #########################

    # if folder exists, delete it
    if os.path.exists(obj_train_data_folder):
        shutil.rmtree(obj_train_data_folder)

    os.makedirs(output_temp_folder)

    # add a folder called obj_train_data
    os.makedirs(obj_train_data_folder)

    # get all files from the image folder
    files = os.listdir(image_folder)

    # list of all annoation paths
    annotation_paths = [os.path.join('data/obj_train_data/', file) for file in files]

    # save annotation paths to txt file
    with open(os.path.join(output_temp_folder, "train.txt"), "w") as outfile:
        outfile.write('\n'.join(annotation_paths))

    # get all files from the annotation folder
    files = os.listdir(annotation_folder)

    # copy all files in the annotation folder to the obj_train_data folder
    for file in tqdm(files):
        shutil.copy(os.path.join(annotation_folder, file), os.path.join(obj_train_data_folder, file))

    #########################
    ##### obj.data file #####
    #########################

    # Generate the content for the txt file
    lines = [
        f"classes = {len(classes)}",
        "train = data/train.txt",
        "names = data/obj.names",
        "backup = backup/"
    ]

    # Write the lines to a txt file
    with open(os.path.join(output_temp_folder,"obj.data"), "w") as outfile:
        outfile.write('\n'.join(lines))

    #########################
    #### obj.names file #####
    #########################
    # Write the lines to a txt file
    with open(os.path.join(output_temp_folder,"obj.names"), "w") as outfile:
        outfile.write('\n'.join(classes))

    # zip the annotations
    with tqdm(total=os.path.getsize(output_temp_folder), unit="B", unit_scale=True, desc="Zip Overall Progress", position=1, leave=True) as progress_overall:
        with zipfile.ZipFile(annotations_zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            zipdir(output_temp_folder, zipf, progress_overall)

    # zip the images
    with tqdm(total=os.path.getsize(image_folder), unit="B", unit_scale=True, desc="Zip Overall Progress", position=1, leave=True) as progress_overall:
        with zipfile.ZipFile(images_zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            zipdir(image_folder, zipf, progress_overall)


    # delete safety_temp folder
    shutil.rmtree(output_temp_folder)

if __name__ == '__main__':
    main()