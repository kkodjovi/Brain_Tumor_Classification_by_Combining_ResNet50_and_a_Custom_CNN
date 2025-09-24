# Brain Tumor Classification using ResNet50 (Keras and PyTorch)

The goal of this project is to classify brain tumors from MRI images.
I leveraged **ResNet50 pretrained on ImageNet** to perform **transfer learning** and feature extraction.
The same dataset was trained using both **PyTorch** and **Keras** implementations in order to compare performance and highlight differences.

The dataset is available on [Kaggle](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset) and contains 4 classes:
- **glioma**
- **meningioma**
- **pituitary tumor**
- **no tumor**

This project was run on **Google Colab** using a **T4 GPU**.

------------------------------------------------------------------------

## Training Setup

-   **PyTorch**:
    -   Data augmentation with `RandomResizedCrop`, horizontal flips, and small rotations
    -   Normalization with ImageNet mean/std
    -   Optimizer: Adam, learning rate `1e-4`
    -   10 epochs.
-   **Keras**:
    -   Data augmentation with `ImageDataGenerator` (rescale, flips, rotations, shifts, zoom)
    -   Normalization based on `ResNet50.preprocess_input`
    -   Optimizer: Adam, learning rate `1e-4`
    -   10 epochs.

> Note: Preprocessing and normalization steps are not exactly identical between the two frameworks. This difference can partly explain the performance gap.

------------------------------------------------------------------------

## Results (10 epochs)

| Metric         | PyTorch | Keras  |
|----------------|---------|--------|
| Train Loss     | 0.0871  | 0.2343 |
| Train Accuracy | 97.02%  | 91.31% |
| Test Loss      | 0.0259  | 0.2721 |
| Test Accuracy  | 99.24%  | 89.09% |

------------------------------------------------------------------------

## Conclusion

-   Both frameworks successfully fine-tuned **ResNet50** for brain tumor classification.

-   **PyTorch achieved higher test accuracy (99.24% vs 89.1%)** with the same number of epochs.

-   The main factors influencing this difference probably are:

    -   **Normalization & preprocessing**: For PyTorch I used ImageNet's mean/std manually, while I used `preprocess_input` for Keras.
    -   **Data augmentation** strategies are similar but not strictly identical across frameworks.

Future improvements could include: 
- Aligning preprocessing pipelines between PyTorch and Keras.
- Extending training beyond 10 epochs - Lowering the Learning rate to see if the trend still the same

------------------------------------------------------------------------

## NB

I first tried to combine ResNet50 with a custom CNN built with Keras, but the training was very unstable and the model was struggling to learn.
This was most likely because: - The custom layers were not properly balanced with the pretrained ResNet50 backbone (risk of vanishing gradient).
- The preprocessing pipeline was not fully aligned with what ResNet50 expects (ImageNet normalization).
- The chosen optimizer and learning rate were not well tuned for this hybrid architecture.

For these reasons, I decided to focus instead on a **clean comparison of ResNet50 between PyTorch and Keras**, where the training process was much more stable and gave consistent results.

I plan to revisit this project later to improve the hybrid model approach (better normalization, careful layer freezing, and learning rate tuning).