import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os 




# Set up the paths for the input and output directories
# input_dir = '/content/drive/MyDrive/jellyfish_tech_dark_vs_light/DARK/toodark'
# output_dir = '/content/drive/MyDrive/jellyfish_tech_dark_vs_light/DARK/aug_toodark'
#input_dir = "C:/Users/Afzal/Downloads/toodark-20230502T121629Z-001"

def augmentation_img(input_dir, 
    output_dir, 
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest',
    num_of_copies):
    import os
    '''
    input_dir: path of input folder, 
    output_dir: path of output(augmnted) folder, 
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest',
    num_of_copies):
    import os

    '''
    import shutil
    import random
    import tensorflow as tf
    from tensorflow.keras.preprocessing.image import ImageDataGenerator

    # Create the output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Copy the original files to the output directory
    for filename in os.listdir(input_dir):
        input_path = os.path.join(input_dir, filename)
        output_path = os.path.join(output_dir, filename)
        shutil.copyfile(input_path, output_path)

    # Set up the data augmentation generator
    datagen = ImageDataGenerator(
        rotation_range=rotation_range,
        width_shift_range=width_shift_range,
        height_shift_range=height_shift_range,
        shear_range=shear_range,
        zoom_range=zoom_range,
        horizontal_flip=horizontal_flip,
        fill_mode=fill_mode
    )

    # Apply data augmentation to the files in the output directory
    for filename in os.listdir(output_dir):
        if filename.endswith(('.jpg','.png','.jpeg')):
            img = tf.keras.preprocessing.image.load_img(os.path.join(output_dir, filename))
            x = tf.keras.preprocessing.image.img_to_array(img)
            x = x.reshape((1,) + x.shape)
            i = 0
            for batch in datagen.flow(x, batch_size=1):
                save_path = os.path.join(output_dir, os.path.splitext(filename)[0] + f'_aug{i}.jpg')
                tf.keras.preprocessing.image.save_img(save_path, batch[0])
                i += 1
                if i >= num_of_copies:
                    break


