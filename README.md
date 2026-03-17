   

---

# LW3_CUSTOM_IMAGE_CLASSIFIER_CSC120

# Part 1: Preparing and Loading Custom Images from Google Drive

GDRIVE link: https://drive.google.com/drive/folders/19oy7WYlHUFdy3h8TCm34t7I_FQAkSr_o?usp=sharing 

COLAB Link: https://colab.research.google.com/drive/1TJq8AiOhKOnPKB0m7CL4PoVuwdZ0rGMn?usp=sharing

Guide Questions (Student Reflection & Explanation)


## 1. Dataset Preparation

○ How did you organize your dataset in Google Drive?

Answer: I organized my dataset following the required folder structure with 20 mushroom species as subfolders under `ImageDataset/`. Each species folder (e.g., `Amanita_muscaria/`, `Agaricus_arvensis/`) contained 250 images named `angle_001.jpg` through `angle_250.jpg`, representing 360-degree rotational views of a single mushroom specimen per species.

○ Why is folder structure important for TensorFlow image loading?

Answer: TensorFlow uses the folder structure to automatically infer class labels for image classification tasks. By organizing images into subdirectories where each subdirectory name corresponds to a specific class, functions like `tf.keras.utils.image_dataset_from_directory` can efficiently load the data and assign the correct labels without requiring separate label files. The hierarchical structure allows TensorFlow to infer both the input images and their corresponding target labels in one function call.


## 2. Model Training

○ What is the role of convolutional layers in image classification?

Answer: Convolutional layers extract features from images. They use filters (kernels) that slide over the input data to identify patterns like edges, textures, and shapes. Early layers detect simple features, while deeper layers combine these into complex shapes and object parts. In my mushroom classification task, these layers learned to identify visual characteristics like cap shape, gill structure, and stem texture — though I later discovered the model was actually memorizing rotation-specific artifacts rather than true species features.

○ Why do we split data into training and validation sets?

Answer: We split data into training and validation sets to train a machine learning model effectively while preventing overfitting. The training set teaches the model patterns, while the validation set tests whether it learned generalizable features or simply memorized examples. In my case, the standard random split initially showed 100% validation accuracy, which was misleading due to data leakage — similar rotation angles appeared in both sets. I implemented an angle-based split (angles 1-200 for training, 201-250 for validation) to ensure true generalization testing.


## 3. Performance Analysis

○ What accuracy did your model achieve?

Answer: Initially, using the standard random split, the model achieved 100% training and validation accuracy. However, I identified this as invalid due to data leakage — the validation set contained different rotation angles of the same mushrooms seen in training. After implementing an angle-based split to prevent leakage, the true performance emerged: approximately 95-100% training accuracy but only **5% validation accuracy** (0.05000000074505806). This revealed the model memorized specific rotation angles rather than learning to identify mushroom species.

○ How did the number of images affect the model's performance?

Answer: The quantity of images (250 per class) appeared sufficient, but the **quality and independence** of images mattered more. Despite having 5,000 total images, there were only 20 unique mushroom specimens. The model failed to generalize because it lacked diverse examples of each species. This demonstrates that 250 independent photos of different mushrooms would be far more valuable than 250 rotations of a single mushroom.


## 4. Critical Thinking

○ What challenges did you encounter while using your own dataset?

Answer: The primary challenge was **data leakage caused by rotational duplicates**. My dataset structure (250 angles per specimen) violated the assumption that training and validation images are independent. The standard random validation split placed visually similar rotations in both sets, creating artificial performance. I had to implement a custom angle-based splitting strategy to obtain honest evaluation metrics. This revealed that my model had zero ability to generalize to new specimens — only to new angles of already-seen specimens.

○ How can data augmentation improve your model?

Answer: Data augmentation (random rotations, flips, zooms) can improve generalization by artificially increasing dataset diversity. However, in my specific case, augmentation would have limited impact because the fundamental problem was **lack of specimen diversity**, not lack of angle diversity. My dataset already contained extensive rotational coverage. Augmentation cannot create information about different individual mushrooms — it only transforms existing information. True improvement requires collecting photos of multiple distinct specimens per species.


## 5. Application

○ Suggest a real-world application for your trained model.

Answer: A properly trained mushroom classifier (with diverse specimen data) could support **mycology education and preliminary field identification**. Hikers and nature enthusiasts could photograph mushrooms and receive suggestions for possible species, along with safety warnings for toxic varieties like *Amanita muscaria*. However, my current model is unsuitable for deployment due to its inability to generalize beyond training specimens.

○ How can this system be integrated into a mobile or web application?

Answer: The trained TensorFlow model can be converted to **TensorFlow Lite** for mobile deployment or served via **TensorFlow Serving** for web applications. Users would upload a photo, the model would preprocess it (resize to 180×180, normalize), and return predicted class probabilities. For mycology applications, the system should include confidence thresholds (reject low-confidence predictions), safety warnings (flag potentially toxic species), geographic filtering (suggest species native to user's location), and disclaimer that automated identification is not 100% reliable.


---

# Activity 3A: Improving and Evaluating a Custom Image Classifier

## Guide Questions (Student Explanation & Reflection)

## 1 What signs indicated overfitting in your first model?

Answer: The classic sign was **100% training accuracy with 100% validation accuracy** using the random split. While this appears as "perfect fit," it was actually **data leakage masquerading as good performance**. The true overfitting was revealed when I used the angle-based split: training accuracy remained high (~95-100%) while validation accuracy collapsed to 5%. This massive gap confirmed severe overfitting — the model memorized specific rotation angles rather than learning generalizable species features.

## 2 How did data augmentation affect validation accuracy?

Answer: I did not apply data augmentation because my dataset already contained extensive rotational coverage (250 angles per specimen). Augmentation would provide limited benefit since the core issue was **lack of independent specimens**, not lack of rotational diversity. In a proper dataset with multiple specimens, augmentation would improve validation accuracy by teaching the model to ignore irrelevant variations (lighting, angle, background) and focus on species-defining characteristics.

## 3 What is the purpose of dropout layers?

Answer: Dropout randomly deactivates neurons during training to prevent co-adaptation and memorization. It forces the network to develop redundant, robust representations rather than relying on specific neurons. In my case, dropout might have slightly improved generalization, but it cannot compensate for the fundamental dataset limitation of having only one specimen per class.

## 4 Why does data augmentation improve generalization?

Answer: Augmentation increases effective dataset size by creating modified copies of training images (rotated, flipped, zoomed, brightened). This teaches the model that classification should be **invariant to these transformations** — a mushroom doesn't change species when photographed from a different angle. However, augmentation only helps when the base dataset contains diverse, independent examples. It cannot create new specimen information from rotational duplicates.

## 5 Compare accuracy before and after improvements.

Answer: Before implementing the angle-based split, the model showed 100% training accuracy and 100% validation accuracy using the random split. This was invalid due to data leakage. After implementing the angle-based split, the model showed approximately 98% training accuracy but only 5% validation accuracy. The "improvement" was not better accuracy but honest evaluation. The 5% result, while poor, correctly identified that my dataset was unsuitable for species classification.

## 6 Which technique contributed most to improvement?

Answer: **The angle-based data splitting strategy** contributed most by eliminating data leakage. Without this fix, I would have reported false 100% accuracy and deployed a useless model. Identifying and addressing the dataset structure problem was more valuable than any architectural change.

## 7 Why is saving the model important?

Answer: Model saving preserves trained weights and architecture for deployment without retraining. It enables consistent inference across sessions and allows version control for model iterations. For my project, saving is essential to demonstrate the final state after identifying the data leakage issue — serving as a checkpoint of this learning experience.

## 8 How can this model be deployed in a real-world system?

Answer: **It should not be deployed** in its current state due to 5% validation accuracy. However, the deployment pipeline would involve five key steps. First, model export using `model.save()` to SavedModel or HDF5 format. Second, conversion to TensorFlow Lite for mobile or TensorFlow.js for browser. Third, API wrapper using Flask or FastAPI for image upload and prediction. Fourth, preprocessing pipeline to resize, normalize, and batch incoming images. Fifth, post-processing with confidence thresholding and safety warnings. Before deployment, the model requires retraining on a proper dataset with multiple independent specimens per species to achieve usable accuracy.
