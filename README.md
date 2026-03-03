# LW3_CUSTOM_IMAGE_CLASSIFIER_CSC120

# Part 1: Preparing and Loading Custom Images from Google Drive

GDRIVE link: https://drive.google.com/drive/folders/19oy7WYlHUFdy3h8TCm34t7I_FQAkSr_o?usp=sharing

Guide Questions (Student Reflection & Explanation)


1. Dataset Preparation

○ How did you organize your dataset in Google Drive?

Answer: I put it in a folder and named it the kinds of plants it contains and then I created a 20 subfolders inside the folder and each subfolder contains 250 images per class.


○ Why is folder structure important for TensorFlow image loading?

Answer: TensorFlow uses the folder structure to automatically infer class labels for image classification tasks. By organizing images into subdirectories where each subdirectory name corresponds to a specific class (e.g., train/cats/ and train/dogs/), functions like tf.keras.utils.image_dataset_from_directory can efficiently load the data and assign the correct labels without requiring separate label files.


2. Model Training

   
○ What is the role of convolutional layers in image classification?

Answer: Convolutional layers extract features from images. They use filters (kernels) that slide over the input data to identify patterns like edges, textures, and shapes.

○ Why do we split data into training and validation sets?

Answer: We split data into training and validation sets to train a machine learning model effectively while preventing overfitting.


3. Performance Analysis

   
○ What accuracy did your model achieve?

Answer:

○ How did the number of images affect the model’s performance?

Answer:
