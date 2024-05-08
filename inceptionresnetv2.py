
pip install gdown

import gdown
import zipfile
import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import numpy as np
from tensorflow.keras.layers import Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard, ReduceLROnPlateau
from tensorflow.keras.applications import InceptionResNetV2

from transformers import AutoImageProcessor, ResNetForImageClassification
from tensorflow.keras.regularizers import l2
from sklearn.model_selection import StratifiedKFold
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense
from tensorflow.keras.models import Model
from sklearn.metrics import accuracy_score
import shutil

file_id = ""
zip_file = "train_images.zip"

extract_dir = "data_images"
gdown.download(f"https://drive.google.com/uc?id={file_id}", zip_file, quiet=False)
with zipfile.ZipFile(zip_file, "r") as zip_ref:
    zip_ref.extractall(extract_dir)

extracted_contents = os.listdir(extract_dir)
print("Extracted contents:", extracted_contents)

# root_directory_train = ''
# output_directory_train = ''
# target_size = (299, 299)
# os.makedirs(output_directory_train, exist_ok=True)


# def resize_and_save_image(image_path,output_path):
#     image = tf.io.read_file(image_path)
#     image = tf.image.decode_image(image, channels=3)  # Set channels=3 for RGB images.
#     image = tf.image.resize(image, target_size)
#     image = tf.image.convert_image_dtype(image, tf.uint8, saturate=False, name=None)
#     image = tf.cast(image, tf.float32)
#     image = tf.keras.applications.inception_resnet_v2.preprocess_input(image, data_format=None)
#     image = tf.keras.preprocessing.image.img_to_array(image)
#     tf.keras.preprocessing.image.save_img(output_path, image)


# for folder_name in os.listdir(root_directory_train):
#     folder_path = os.path.join(root_directory_train, folder_name)

#     if os.path.isdir(folder_path):
#         print(f"Processing images in folder: {folder_name}")
#         output_subdirectory = os.path.join(output_directory_train, folder_name)
#         os.makedirs(output_subdirectory, exist_ok=True)


#         for image_file in os.listdir(folder_path):
#             if image_file.endswith(('.jpg')):
#                 image_path = os.path.join(folder_path, image_file)
#                 output_path = os.path.join(output_subdirectory, image_file)
#                 resize_and_save_image(image_path, output_path)

# root_directory_val = ''
# output_directory_val = ''
# target_size = (299, 299)
# os.makedirs(output_directory_val, exist_ok=True)


# def resize_and_save_image(image_path,output_path):
#     image = tf.io.read_file(image_path)
#     image = tf.image.decode_image(image, channels=3)  # Set channels=3 for RGB images.
#     image = tf.image.resize(image, target_size)
#     image = tf.image.convert_image_dtype(image, tf.uint8, saturate=False, name=None)
#     image = tf.cast(image, tf.float32)
#     image = tf.keras.applications.inception_resnet_v2.preprocess_input(image, data_format=None)
#     image = tf.keras.preprocessing.image.img_to_array(image)
#     tf.keras.preprocessing.image.save_img(output_path, image)


# for folder_name in os.listdir(root_directory_val):
#     folder_path = os.path.join(root_directory_val, folder_name)

#     if os.path.isdir(folder_path):
#         print(f"Processing images in folder: {folder_name}")
#         output_subdirectory = os.path.join(output_directory_val, folder_name)
#         os.makedirs(output_subdirectory, exist_ok=True)


#         for image_file in os.listdir(folder_path):
#             if image_file.endswith(('.jpg')):
#                 image_path = os.path.join(folder_path, image_file)
#                 output_path = os.path.join(output_subdirectory, image_file)
#                 resize_and_save_image(image_path, output_path)

data_dir_training = ''
data_dir_val = ''
# # class_labels = os.listdir(data_dir)
# # class_to_int = {class_label: i for i, class_label in enumerate(class_labels)}
# # labels = [class_label for class_label, i in class_to_int.items()]

input_shape = (299, 299, 3)
num_classes = 10
batch_size = 35
k_folds = 10

datagen_train = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    vertical_flip=True,
    fill_mode='nearest')

datagen_val = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale=1./255,
    )

train_datagen = datagen_train.flow_from_directory(
        data_dir_training,
        target_size=input_shape[:2],
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=True,
        seed=42
    )




val_datagen = datagen_val.flow_from_directory(
        data_dir_val,
        target_size=input_shape[:2],
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=True,
        seed=42
    )

class_indices = train_datagen.class_indices

class_labels

# total_samples = 0
# labels = []
# x = sorted(os.listdir(data_dir_train))
# print(x)
# class_labels = {class_folder: i for i, class_folder in enumerate(sorted(os.listdir(data_dir_train)))}
# print(class_labels)


# for class_folder in sorted(os.listdir(data_dir_train)):

#     class_path = os.path.join(data_dir_train, class_folder)
#     if os.path.isdir(class_path):
#         samples_in_class = len(os.listdir(class_path))
#         labels.extend([class_labels[class_folder]] * samples_in_class)
#         total_samples += samples_in_class

def create_inceptionresnetv2_model(input_shape, num_classes):

    base_model = InceptionResNetV2(weights='imagenet', include_top=False, input_shape=input_shape)


    for layer in base_model.layers:
        layer.trainable = True

    x = GlobalAveragePooling2D()(base_model.output)
    x = Dropout(0.10)(x)
    x = Dense(128, activation='relu')(x)
    predictions = Dense(10, activation='softmax')(x)


    model = Model(inputs=base_model.input, outputs=predictions)
    opt = tf.keras.optimizers.Adam(learning_rate=0.0002)


    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])

    return model

checkpoint = ModelCheckpoint("model_checkpoint.keras", save_best_only=True,verbose =1 )
early_stopping = EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True,verbose =1)
reduce_lr = ReduceLROnPlateau(monitor="val_loss", factor=0.2, patience=3,verbose =1)


model = create_inceptionresnetv2_model(input_shape, num_classes)
num_epochs = 40

model.fit(
        train_datagen,
        epochs=num_epochs,
        batch_size=batch_size,
        validation_data=val_datagen,
        callbacks=[checkpoint,  early_stopping, reduce_lr],
        verbose=1
    )

# Save the model checkpoint
model.save('/.../model_inceptionresnet_97val.keras')

from IPython.display import FileLink

FileLink('model_inceptionresnet_97val.keras')

##### kf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=42)

# overall_loss = []
# overall_accuracy = []

# for fold, (train_indices, val_indices) in enumerate(kf.split(np.arange(total_samples), labels)):
#     print(f'Fold {fold + 1}')

#     checkpoint = ModelCheckpoint("model_checkpoint.keras", save_best_only=True,verbose =1 )
#     early_stopping = EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True,verbose =1)
#     reduce_lr = ReduceLROnPlateau(monitor="val_loss", factor=0.2, patience=3,verbose =1)


#     model = create_inceptionresnetv2_model(input_shape, num_classes)
#     num_epochs = 20

#     train_datagen = load_and_preprocess_data_train(train_indices)
#     val_datagen = load_and_preprocess_data_val( val_indices)

#     model.fit(
#         train_datagen,
#         epochs=num_epochs,
#         batch_size=batch_size,
#         validation_data=val_datagen,
#         callbacks=[checkpoint,  early_stopping, reduce_lr],
#         verbose=1
#     )

#     loss, accuracy = model.evaluate(val_datagen)
#     print(f'Validation loss: {loss}, Validation accuracy: {accuracy}')
#     overall_loss.append(loss)
#     overall_accuracy.append(accuracy)

# average_loss = sum(overall_loss) / len(overall_loss)
# average_accuracy = sum(overall_accuracy) / len(overall_accuracy)

# print(f'Average Loss: {average_loss}, Average Accuracy: {average_accuracy}')


import shutil

# Copy the model checkpoint to the output directory
shutil.copy('model_checkpoint.h5', '')

from IPython.display import FileLink

# Generate a download link for the model checkpoint
FileLink(r'')















