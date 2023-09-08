import tensorflow as tf
import os
from PIL import Image
import numpy as np

source_directory = 'your path to the dataset directory'
destination = 'your path to the preprocessed directory'

target_size = (224, 224)

# Data augmentation
data_aug = tf.keras.Sequential([tf.keras.layers.experimental.preprocessing.RandomFlip("horizontal"),
tf.keras.layers.experimental.preprocessing.RandomRotation(0.2),
tf.keras.layers.experimental.preprocessing.RandomZoom(0.2),
tf.keras.layers.experimental.preprocessing.RandomTranslation(0.2, 0.2)])


# Going through all subdirectories and processing images, recursively

for root, directories, files in os.walk(source_directory):
    for filename in files:
        if(filename.endswith('.jpg')):

            # Loading and decoding the image
            image_path = os.path.join(root, filename)
            image = tf.io.read_file(image_path)
            try:
                image = tf.image.decode_image(image)
            except Exception as e:
                #print(f"Error loading image {image_path}: {e}")
                os.remove(image_path)
                continue



            # Resizing
            try:
                image = tf.image.resize(image, target_size)
            except Exception as e:
                print(f"Error Resizing {image_path}: {e}")
                continue

            # Normalizing
            try:
                image = tf.cast(image, tf.float32) / 255.0
            except Exception as e:
                os.remove(image_path)
                continue

            # Augmentation
            try:
                image = data_aug(tf.expand_dims(image, axis = 0))[0]
            except Exception as e:
                os.remove(image_path)
                continue

            # Converting all RGBA images to RGB
            if image.shape[-1] == 4:
                image = Image.fromarray((image * 255).numpy().astype('uint8'))
                image = image.convert('RGB')

            # Class label
            class_label = os.path.basename(root)
            class_dir = os.path.join(destination, class_label)
            os.makedirs(class_dir, exist_ok = True)

            image = np.array(image)  # Converting tensor to array

            # Save the image to the destination
            try:
                image = Image.fromarray((image * 255).astype('uint8'))
            except Exception as e:
                os.remove(image_path)
                continue

           # image = Image.fromarray(image)
            image.save(os.path.join(class_dir, filename))


