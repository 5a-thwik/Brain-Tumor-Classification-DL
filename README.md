# Brain Tumor Classification using VGG16 and ResNet18 🧠

## Overview
This project focuses on automating the classification of brain tumors from MRI scans using deep learning. We developed and evaluated two Convolutional Neural Network (CNN) architectures, **VGG16** and **ResNet18**, to classify brain MRI images into four distinct categories: glioma, meningioma, pituitary tumor, and no tumor. The project leverages transfer learning and advanced image preprocessing techniques to achieve high classification accuracy, with ResNet18 demonstrating superior performance.

---

## Motivation
Brain tumors are among the most complex and fatal diseases, making early and accurate detection critical for effective treatment. Traditional diagnosis relies on manual MRI scan analysis, which is time-consuming and prone to human error due to subjectivity and fatigue. Deep learning, particularly CNNs, offers a powerful solution to automate and enhance medical image analysis, demonstrating remarkable accuracy in visual pattern recognition.

Our motivation is to assist medical professionals in making faster, more accurate diagnoses by developing a reliable, automated system using VGG16 and ResNet18.

---

## Dataset
* **Source:** A publicly available **Brain Tumor MRI Dataset**.
* **Classes:** The dataset contains images categorized into four classes:
    * Glioma
    * Meningioma
    * Pituitary Tumor
    * No Tumor
* **Format:** Folder-structured images, split into training and testing sets.
* **Preprocessing:**
    * Images resized to $224 \times 224$ pixels for compatibility with CNN input layers.
    * Normalization applied using ImageNet mean and standard deviation.
    * Data augmentation techniques like random rotation and horizontal flipping were used to improve generalization.

---

## Model Architectures
Both VGG16 and ResNet18 models were utilized, pre-trained on ImageNet, with their final classification layers replaced and fine-tuned for our specific 4-class dataset.

* **VGG16:**
    * A deep 16-layer CNN.
    * Composed of sequential stacked convolutional blocks.
    * Features are flattened and connected to a dense layer of size $512 \rightarrow 4$ for classification.
* **ResNet18:**
    * An 18-layer residual CNN.
    * Includes **skip connections** to combat vanishing gradient issues, enabling the training of deeper networks.
    * The final layer was modified to output 4-class predictions.

---

## Implementation Details
* **Platform:** Google Colab.
* **Hardware:** NVIDIA Tesla T4 GPU (used for faster computation via mixed precision training).
* **Frameworks & Libraries:**
    * Python
    * PyTorch
    * Torchvision
    * Matplotlib / Seaborn
    * scikit-learn

### Hyperparameters Used
| Hyperparameter | Value   |
| :------------- | :------ |
| Learning Rate  | 0.0001  |
| Optimizer      | Adam    |
| Batch Size     | 32      |
| Dropout Rate   | 0.5     |
| Epochs         | 10      |
*Note: Filter sizes are predefined by the VGG16 and ResNet18 architectures.*

---

## Results and Comparison
Model performance was evaluated using a **confusion matrix**, **F1-score**, and **training/validation accuracy/loss plots**. Both models were trained with consistent hyperparameters and augmentation techniques.

### ResNet18 Results
* **Final Training Loss:** 0.0253
* **F1-Score:** 0.9901
* **Accuracy:** ~99%
* **Validation Stability:** High, with stable convergence and no overfitting, indicating excellent generalization.

**ResNet18 Confusion Matrix:**
| Actual/Predicted | Glioma | Meningioma | Pituitary | No Tumor |
| :--------------- | :----- | :--------- | :-------- | :------- |
| Glioma           | 291    | 9          | 0         | 0        |
| Meningioma       | 1      | 303        | 1         | 1        |
| Pituitary        | 0      | 0          | 405       | 0        |
| No Tumor         | 1      | 0          | 0         | 299      |

**ResNet18 Training & Validation Graphs:**
*Replace with actual image links to your plots on GitHub*
![ResNet18 Loss per Epoch](https://github.com/5a-thwik/Brain-Tumor-Classification-DL/commit/45249f541e08d36bdf3abacd8ead95453b5b27f1#diff-0ce51d34a897bcf9a6679499a4b6e4ca22f92bd61ab5aa56bfe95c53f9c311d2)
![ResNet18 Accuracy per Epoch](https://github.com/5a-thwik/Brain-Tumor-Classification-DL/commit/45249f541e08d36bdf3abacd8ead95453b5b27f1#diff-5cd89315d60b6cb5d111a97aaccdd024031c245be669bf117c670033042f4c18 )


### VGG16 Results
* **Final Training Loss:** ~0.07
* **F1-Score:** ~0.95
* **Accuracy:** ~95%
* **Validation Stability:** Medium, validation accuracy fluctuated slightly after 5-6 epochs, suggesting mild overfitting.

**VGG16 Sample Confusion Matrix:**
| Actual/Predicted | Glioma | Meningioma | No Tumor | Pituitary |
| :--------------- | :----- | :--------- | :------- | :-------- |
| Glioma           | 290    | 9          | 1        | 0         |
| Meningioma       | 24     | 269        | 10       | 3         |
| No Tumor         | 0      | 0          | 405      | 0         |
| Pituitary        | 3      | 6          | 2        | 289       |
*Note: Glioma and meningioma classes showed a few misclassifications.*

**VGG16 Training & Validation Graphs:**
*Replace with actual image links to your plots on GitHub*
![VGG16 Loss per Epoch](link-to-your-vgg-loss-plot.png)
![VGG16 Accuracy per Epoch](link-to-your-vgg-accuracy-plot.png)

### Comparison Summary
| Metric             | VGG16  | ResNet18 |
| :----------------- | :----- | :------- |
| F1 Score           | ~0.95  | 0.99     |
| Accuracy           | ~95%   | 99%      |
| Validation Stability | Medium | High     |
| Speed (training)   | Slower | Faster   |
| Parameters         | High   | Lower    |

**Discussion:** ResNet18 showed superior generalization, faster training, and higher accuracy, making it more suitable for real-world deployment. The use of residual connections in ResNet18 made it more effective in extracting deep hierarchical features, resulting in better convergence and generalization.

---

## Conclusion
We successfully implemented and compared VGG16 and ResNet18 deep learning models for classifying brain MRI images into four tumor categories. Both models demonstrated high classification performance, but **ResNet18 significantly outperformed VGG16**, achieving an F1-score of 0.99 and a validation accuracy of ~99%. This project validates the effectiveness of deep learning, particularly with transfer learning and pre-trained CNNs, as a powerful tool for medical image classification. This automated system has the potential to assist radiologists in making faster and more accurate decisions during early tumor diagnosis.

---

## Future Enhancements
Several enhancements could further improve performance and applicability:
1.  **Multi-modal Data Integration:** Incorporate other medical data (CT scans, patient history) for more robust predictions.
2.  **Larger Dataset:** Train on a more diverse and extensive dataset to improve generalization to unseen data.
3.  **Real-time Deployment:** Deploy the trained model into a clinical decision support system via web or mobile interfaces for real-time inference.
4.  **Explainability Tools:** Integrate tools like Grad-CAM or saliency maps to visualize model focus, enhancing clinical trust.
5.  **Advanced Architectures:** Explore models such as EfficientNet, DenseNet, or transformer-based vision models for potentially better performance.

---
