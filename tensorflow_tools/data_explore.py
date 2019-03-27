import os
import os
import random
import skimage.data
import skimage.transform
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
def load_data(data_dir):
    """Loads a data set and returns two lists:

    images: a list of Numpy arrays, each representing an image.
    labels: a list of numbers that represent the images labels.
    """
    # Get all subdirectories of data_dir. Each represents a label.
    directories = [d for d in os.listdir(data_dir)
                   if os.path.isdir(os.path.join(data_dir, d))]
    # Loop through the label directories and collect the data in
    # two lists, labels and images.
    labels = []
    images = []
    for d in directories:
        label_dir = os.path.join(data_dir, d)
        file_names = [os.path.join(label_dir, f)
                      for f in os.listdir(label_dir) if f.endswith(".ppm")]
        # For each label, load it's images and add them to the images list.
        # And add the label number (i.e. directory name) to the labels list.
        for f in file_names:
            images.append(skimage.data.imread(f))
            labels.append(int(d))
    return images, labels




def display_images_and_labels(images, labels):
    """Display the first image of each label."""
    unique_labels = set(labels)
    plt.figure(figsize=(15, 15))
    i = 1
    for label in unique_labels:
        # Pick the first image for each label.
        image = images[labels.index(label)]
        plt.subplot(8, 8, i)  # A grid of 8 rows x 8 columns
        plt.axis('off')
        plt.title("Label {0} ({1})".format(label, labels.count(label)))
        i += 1
        _ = plt.imshow(image)
    plt.show()

def show_images_size(images):
    for image in images[:5]:
        print("shape: {0}, min: {1}, max: {2}".format(image.shape, image.min(), image.max()))

def shringle_images(images,labels):
    # 调整图像
    images32 = [skimage.transform.resize(image, (32, 32)) for image in images]
    # display_images_and_labels(images32, labels)
    # show_images_size(images32)
    labels_a = np.array(labels)
    images_a = np.array(images32)
    print("labels: ", labels_a.shape, "\nimages: ", images_a.shape)
    return {'labels_a':labels_a,'images_a':images_a}
def display_label_images(images, labels,label):
    """Display images of a specific label."""
    limit = 24  # show a max of 24 images
    plt.figure(figsize=(15, 5))
    i = 1

    start = labels.index(label)
    end = start + labels.count(label)
    for image in images[start:end][:limit]:
        plt.subplot(3, 8, i)  # 3 rows, 8 per row
        plt.axis('off')
        i += 1
        plt.imshow(image)
    plt.show()

if __name__=='__main__':
    # Load training and testing datasets.
    ROOT_PATH = 'F:/陶士来文件/tsl_python_project/model_datas'
    train_data_dir = os.path.join(ROOT_PATH, "logo_recogniztion/BelgiumTSC_Training/Training")
    test_data_dir = os.path.join(ROOT_PATH, "logo_recogniztion/BelgiumTSC_Testing/Testing")

    images, labels = load_data(train_data_dir)

    print("Unique Labels: {0}\nTotal Images: {1}".format(len(set(labels)), len(images)))
    # display_images_and_labels(images, labels)
    # display_label_images(images,labels, 26)
    # show_images_size(images)
    data=shringle_images(images,labels)