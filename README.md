# LW3_CUSTOM_IMAGE_CLASSIFIER_CSC120

# Part 1: Preparing and Loading Custom Images from Google Drive

GDRIVE link: https://drive.google.com/drive/folders/19oy7WYlHUFdy3h8TCm34t7I_FQAkSr_o?usp=sharing

COLAB Link: https://colab.research.google.com/drive/1TJq8AiOhKOnPKB0m7CL4PoVuwdZ0rGMn?usp=sharing

Guide Questions (Student Reflection & Explanation)


## 1. Dataset Preparation

○ How did you organize your dataset in Google Drive?

Answer: I organized my dataset following the required folder structure with 20 mushroom species as subfolders under `ImageDataset/`. Each species folder contained 250 images representing multiple rotational angles of a single mushroom specimen per species. The 20 classes are organized as follows:

```
ImageDataset/
├── Agaricus_arvensis/          (250 images)
├── Agaricus_augustus/          (250 images)
├── Agaricus_campestris/        (250 images)
├── Agaricus_sylvicola/         (250 images)
├── Agaricus_xanthodermus/      (250 images)
├── Amanita_battarrae/          (250 images)
├── Amanita_muscaria/           (250 images)
├── Auricularia_auricula-judae/ (250 images)
├── Auriscalpium_vulgare/       (250 images)
├── Cantharellus_lateritius/    (250 images)
├── Conocybe_apala/             (250 images)
├── Cortinarius_mucosus/        (250 images)
├── Cortinarius_violaceus/      (250 images)
├── Craterellus_tubaeformis/    (250 images)
├── Geastrum_rufescens/         (250 images)
├── Hericium_erinaceus/         (250 images)
├── Leucocoprinus_birnbaumii/   (250 images)
├── Omphalotus_olearius/        (250 images)
├── Ramaria_lutea/              (250 images)
└── Strobilomyces_strobilaceus/ (250 images)
```

TensorFlow's `image_dataset_from_directory()` automatically used these folder names as class labels, requiring no separate label files.

○ Why is folder structure important for TensorFlow image loading?

Answer: TensorFlow uses the folder structure to automatically infer class labels for image classification tasks. By organizing images into subdirectories where each subdirectory name corresponds to a specific mushroom species, functions like `tf.keras.utils.image_dataset_from_directory()` can efficiently load the data and assign the correct labels without requiring separate label files. The hierarchical structure allows TensorFlow to infer both the input images and their corresponding target labels in one function call. If the structure is incorrect or inconsistent, the model will misassign labels across all 20 mushroom classes, leading to incorrect predictions regardless of how well the model trains.


## 2. Model Training

○ What is the role of convolutional layers in image classification?

Answer: Convolutional layers extract features from images by using filters (kernels) that slide over the input data to identify patterns like edges, textures, and shapes. Early layers detect simple features such as edges and color gradients, while deeper layers combine these into complex shapes and object parts. In my mushroom classification task, these layers learned to identify visual characteristics like cap shape, gill structure, stem texture, and surface patterns across all 20 species — from the distinctive spiny appearance of *Hericium erinaceus* to the star-shaped structure of *Geastrum rufescens*. The three Conv2D layers (16 → 32 → 128 filters) progressively built richer feature representations with each layer.

○ Why do we split data into training and validation sets?

Answer: We split data into training and validation sets to train a machine learning model effectively while preventing overfitting. The training set (80% — 4,000 images) teaches the model patterns across all 20 mushroom species, while the validation set (20% — 1,000 images) tests whether it learned generalizable features or simply memorized training examples. Without a validation set, there would be no way to detect if the model was overfitting to the specific angles and lighting conditions in the training data rather than learning true species-defining features. The validation set acts as a proxy for real-world performance on new, unseen mushroom images.


## 3. Performance Analysis

○ What accuracy did your model achieve?

Answer: The model achieved a final validation accuracy of **93.20%** at Epoch 29 (best epoch restored by EarlyStopping), with a validation loss of **0.2215**. Training accuracy reached **80.23%** with a training loss of **0.5330**. The training started with only 21.63% accuracy in Epoch 1, with a dramatically high validation loss of 56.35, before stabilizing significantly by Epoch 4 with a breakthrough jump to 62.10% validation accuracy. Based on the standard benchmark ranges for image classification models, this places the model in the **Good to Excellent** category — particularly in the validation metrics which are most relevant to real-world performance.

○ How did the number of images affect the model's performance?

Answer: Having 250 images per class from a single specimen created a low-diversity dataset across all 20 species. Since all 250 images are different angles of the same mushroom, the model risked memorizing each specimen's specific characteristics — such as the exact gill pattern of one *Cortinarius violaceus* or the specific cap texture of one *Amanita muscaria* — rather than learning general class features. This is why aggressive data augmentation was critical. The augmentation strategy using `RandomFlip`, `RandomRotation(0.3)`, `RandomZoom(0.3)`, `RandomBrightness(0.3)`, and `RandomContrast(0.3)` artificially created diversity to compensate for the single-specimen-per-class limitation, ultimately allowing the model to achieve 93.20% validation accuracy despite this constraint.


## 4. Critical Thinking

○ What challenges did you encounter while using your own dataset?

Answer: The primary challenge was the **low specimen diversity** caused by using 250 rotational angles of a single mushroom per species. This created a risk where the model could memorize the specific specimen's unique characteristics — such as the particular coloring of one *Leucocoprinus birnbaumii* or the exact surface texture of one *Strobilomyces strobilaceus* — rather than learning the general visual features that define each species. Evidence of this challenge was visible in Epoch 1 where the validation loss was extremely high at 56.35, indicating the model initially struggled to generalize. The challenge was addressed through aggressive data augmentation and strong regularization (Dropout 0.5, BatchNormalization after every Conv layer, EarlyStopping, and ReduceLROnPlateau), which together pushed the validation accuracy to 93.20%.

○ How can data augmentation improve your model?

Answer: Data augmentation was applied using `RandomFlip("horizontal_and_vertical")`, `RandomRotation(0.3)`, `RandomZoom(0.3)`, `RandomBrightness(0.3)`, and `RandomContrast(0.3)`. These transformations simulated diverse real-world conditions — different lighting environments (bright sunlight vs. forest shadow), varied camera angles and distances, and different contrast levels across forest floor backgrounds. For a mushroom dataset where all 20 species were photographed from controlled rotational angles, augmentation was essential to teach the model that species identity — whether *Omphalotus olearius* or *Cantharellus lateritius* — remains the same regardless of viewing angle, lighting, or zoom level. This directly contributed to the model achieving 93.20% validation accuracy despite the single-specimen-per-class constraint.


## 5. Application

○ Suggest a real-world application for your trained model.

Answer: The trained model can be used as a **mushroom identification and safety app for hikers, foragers, and nature enthusiasts**. Users can take a photo of a wild mushroom and the app will identify the species from the 20 trained classes and provide safety information — for example, flagging *Amanita muscaria*, *Omphalotus olearius*, *Conocybe apala*, and *Agaricus xanthodermus* as toxic or potentially harmful, while indicating safe edible species like *Cantharellus lateritius* and *Hericium erinaceus*. This has direct public health applications in preventing accidental mushroom poisoning, which causes thousands of hospitalizations annually worldwide. The system could also support **mycology education** by helping students and researchers quickly identify field specimens.

○ How can this system be integrated into a mobile or web application?

Answer: The saved `.keras` model can be converted to **TensorFlow Lite (TFLite)** format for lightweight, offline deployment on Android or iOS mobile apps, allowing users to identify mushrooms in the field without an internet connection. For a web application, the model can be served using a **Flask or FastAPI REST API backend** where users upload a photo through a browser interface and receive real-time species predictions with confidence scores. The preprocessing pipeline would resize incoming images to 180×180, normalize pixel values, and return the top predicted species from the 20 classes along with toxicity warnings. The backend can be hosted on cloud platforms like **Google Cloud Run** or **AWS Lambda** for scalable, cost-effective deployment. Additional features could include geographic filtering (suggesting only species native to the user's region) and a confidence threshold to reject low-certainty predictions.


---

# Activity 3A: Improving and Evaluating a Custom Image Classifier

## Guide Questions (Student Explanation & Reflection)

## 1. What signs indicated overfitting in your first model?

Answer: The signs of overfitting were visible in the training curves across 30 epochs. In the early epochs, validation loss was extremely unstable and high — reaching 56.35 in Epoch 1 and 47.14 in Epoch 2 — while training accuracy was already climbing steadily. The large gap between training loss (0.5330) and validation loss (0.2215) at the final epoch, combined with training accuracy (80.23%) being noticeably lower than validation accuracy (93.20%), indicated that the Dropout(0.5) regularization was working correctly by suppressing training scores while the model generalized well on validation data. The unstable early validation loss was the clearest sign of initial overfitting before regularization effects stabilized training by Epoch 4.

## 2. How did data augmentation affect validation accuracy?

Answer: Data augmentation had a significant positive effect on validation accuracy. By applying `RandomFlip("horizontal_and_vertical")`, `RandomRotation(0.3)`, `RandomZoom(0.3)`, `RandomBrightness(0.3)`, and `RandomContrast(0.3)`, the model was exposed to artificial variations that simulated real-world diversity across all 20 mushroom species. This was especially critical given the single-specimen-per-class dataset structure. Without augmentation, the model would have been far more likely to memorize rotation-specific features of each specimen. With augmentation, the model learned species-defining features that generalized well, ultimately contributing to the 93.20% validation accuracy achieved at the best epoch.

## 3. What is the purpose of dropout layers?

Answer: Dropout randomly deactivates a proportion of neurons during each training step, forcing the network to develop redundant and distributed representations rather than relying on specific neurons or memorizing exact training patterns. In this model, `Dropout(0.5)` was applied both after the convolutional blocks and after the Dense(128) layer. This prevented co-adaptation between neurons and reduced the risk of overfitting to the specific rotational angles and specimen characteristics in the training set. The effectiveness of dropout was demonstrated by the model's strong validation accuracy of 93.20% despite the challenging single-specimen dataset structure.

## 4. Why does data augmentation improve generalization?

Answer: Data augmentation increases the effective diversity of the training dataset by creating modified copies of training images through transformations like flips, rotations, zooms, and brightness/contrast changes. This teaches the model that classification should be **invariant to these transformations** — a *Cortinarius violaceus* remains the same species regardless of whether it is photographed from above, from the side, in bright light, or in shadow. For a dataset where all 250 images per class are different angles of the same specimen, augmentation was essential to simulate the diversity that would normally come from photographing multiple different individual mushrooms of the same species.

## 5. Compare accuracy before and after improvements.

Answer: The baseline model provided in the lab instructions (without augmentation or dropout) consisted of a simple 3-layer CNN with no regularization. After applying improvements — stronger augmentation (`RandomBrightness`, `RandomContrast`, increased rotation and zoom ranges to 0.3), `BatchNormalization` after every Conv layer, `Dropout(0.5)`, `EarlyStopping`, and `ReduceLROnPlateau` — the improved model achieved **93.20% validation accuracy** and **0.2215 validation loss** at Epoch 29. The `ReduceLROnPlateau` callback triggered at Epoch 16, reducing the learning rate from 0.001 to 0.0003, which produced a second wave of accuracy improvement visible in the training curves from Epoch 17 onward.

## 6. Which technique contributed most to improvement?

Answer: **BatchNormalization combined with ReduceLROnPlateau** contributed most to the improvement. BatchNormalization stabilized training by normalizing activations between layers, preventing the extreme validation loss instability seen in early epochs (56.35 in Epoch 1). ReduceLROnPlateau then acted as a second-stage optimizer — when training plateaued around Epoch 16, it reduced the learning rate from 0.001 to 0.0003, enabling the model to make finer weight adjustments that pushed validation accuracy from around 86% to the final 93.20%. Without either of these techniques, the model would likely have stagnated at a lower accuracy level.

## 7. Why is saving the model important?

Answer: Saving the model with `model.save("/content/drive/MyDrive/mushroom_classifier_improved.keras")` preserves the trained weights, architecture, optimizer state, and all learned features without requiring retraining. This is essential for deployment — the saved model can be loaded and used to classify new mushroom images instantly. It also enables version control across lab activities, as the saved LW3 model is directly loaded and evaluated in Laboratory Work 4 for advanced metrics (Precision, Recall, F1, ROC/AUC) and Grad-CAM visualization. Without saving, all training progress would be lost when the Colab session ends.

## 8. How can this model be deployed in a real-world system?

Answer: The saved `.keras` model can be deployed through several pathways. For **mobile deployment**, the model can be converted to TensorFlow Lite using `tf.lite.TFLiteConverter` for integration into an Android or iOS app that allows offline mushroom identification in the field. For **web deployment**, the model can be served via a Flask or FastAPI REST API where users upload images and receive JSON responses containing the predicted species name and confidence score from the 20 classes. For **scalable cloud deployment**, the model can be containerized with Docker and hosted on Google Cloud Run or AWS Lambda. The preprocessing pipeline (resize to 180×180, normalize to [0,1], expand dims for batch) must be replicated exactly in production to match the training conditions. Safety features should include confidence thresholds (reject predictions below 80%) and toxicity warnings for dangerous species like *Amanita muscaria* and *Omphalotus olearius*.
