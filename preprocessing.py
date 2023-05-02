import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os 




# Set up the paths for the input and output directories
# input_dir = '/content/drive/MyDrive/jellyfish_tech_dark_vs_light/DARK/toodark'
# output_dir = '/content/drive/MyDrive/jellyfish_tech_dark_vs_light/DARK/aug_toodark'
#input_dir = "C:/Users/Afzal/Downloads/toodark-20230502T121629Z-001"

def augmentation_img(input_dir="path\input", 
    output_dir="path\output", 
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest',
    num_of_copies=4):
  
    """
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

    """
    import os
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




def split_test_train_image(
    input_dir = '/path/to/input',
    train_dir = '/path/to/train_data_set',
    val_dir = '/path/to/test_data_set',
    train_pct = 0.8):
  """
  Function can split the image data into test and train data set in given ratio
  """
    import os
    import shutil
    import random



    # Create the output directories if they don't exist
    if not os.path.exists(train_dir):
        os.makedirs(train_dir)
    if not os.path.exists(val_dir):
        os.makedirs(val_dir)

    # Set the percentage of images to use for training
    train_pct = train_pct

    # Iterate over each subdirectory in the input directory
    for filename in os.listdir(input_dir):
        filepath = os.path.join(input_dir, filename)


        if random.random() < train_pct:
                output_dir = train_dir
        else:
                output_dir = val_dir

        # Copy the image to the corresponding output directory
        input_path = os.path.join(input_dir, filename)
        output_path = os.path.join(output_dir, filename)
        shutil.copyfile(input_path, output_path)




# Make a function to predict on images and plot them (works with multi-class)
def pred_and_plot(model, filename, class_names):
  """
  Imports an image located at filename, makes a prediction on it with
  a trained model and plots the image with the predicted class as the title.
  """
  # Import the target image and preprocess it
  img = load_and_prep_image(filename)

  # Make a prediction
  pred = model.predict(tf.expand_dims(img, axis=0))

  # Get the predicted class
  if len(pred[0]) > 1: # check for multi-class
    pred_class = class_names[pred.argmax()] # if more than one output, take the max
  else:
    pred_class = class_names[int(tf.round(pred)[0][0])] # if only one output, round

  # Plot the image and predicted class
  plt.imshow(img)
  plt.title(f"Prediction: {pred_class}")
  plt.axis(False);
  


# Create a function to import an image and resize it to be able to be used with our model
def load_and_prep_image(filename, img_shape=256, scale=True):
  """
  Reads in an image from filename, turns it into a tensor and reshapes into
  (256, 256, 3).
  Parameters
  ----------
  filename (str): string filename of target image
  img_shape (int): size to resize target image to, default 224
  scale (bool): whether to scale pixel values to range(0, 1), default True
  """
  # Read in the image
  img = tf.io.read_file(filename)
  # Decode it into a tensor
  img = tf.image.decode_jpeg(img)
  # Resize the image
  img = tf.image.resize(img, [img_shape, img_shape])
  if scale:
    # Rescale the image (get all values between 0 and 1)
    return img/255.
  else:
    return img