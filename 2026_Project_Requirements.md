# Group Project

**Course:** Introduction to Deep Learning with Python

## Topic

**Skin Lesion Classification Using Deep Learning**

## Objective

This project aims to help students complete a full deep learning workflow for medical image classification. Using a skin lesion image dataset, each group is required to build, train, evaluate, and interpret deep learning models using PyTorch.

Skin lesion classification is an important task in medical image analysis. Deep learning models, especially convolutional neural networks and transfer learning models, have been widely used for image-based disease recognition. Through this project, students are expected to apply the knowledge learned in this course to a real-world image classification problem.

## Dataset

The project uses a dermoscopic skin lesion image dataset based on the ISIC 2018 Task 3 / HAM10000 dataset. The dataset contains skin lesion images from multiple diagnostic categories. Each image is associated with one diagnosis label, and the task is to classify each image into the correct skin lesion category.

The dataset contains the following seven diagnostic categories:

| Label | Full Name | Description |
|---|---|---|
| MEL | Melanoma | Melanoma |
| NV | Melanocytic nevus | Melanocytic nevus |
| BCC | Basal cell carcinoma | Basal cell carcinoma |
| AKIEC | Actinic keratosis / Bowen's disease | Intraepithelial carcinoma |
| BKL | Benign keratosis-like lesion | Solar lentigo / seborrheic keratosis / LPLK |
| DF | Dermatofibroma | Dermatofibroma |
| VASC | Vascular lesion | Vascular lesion |

Students may refer to the following official dataset pages:

- ISIC 2018 Task 3: Lesion Diagnosis:  
  <https://challenge.isic-archive.com/landing/2018/47/>
- ISIC Challenge Dataset Download Page:  
  <https://challenge.isic-archive.com/data/>
- HAM10000 Dataset Paper:  
  <https://www.nature.com/articles/sdata2018161>
- Kaggle mirror of HAM10000 dataset:  
  <https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000>

Students should use the provided training, validation, and test datasets. The training set should be used for model training, the validation set should be used for model selection and hyperparameter tuning, and the test set should be used only for final performance reporting.

The project will not be ranked by F1 score, accuracy, or any leaderboard. However, students should report their own test performance in the project report.

## Requirements

Each group should complete the following tasks:

1. Build at least one baseline deep learning model, such as a simple convolutional neural network.
2. Build at least one transfer learning model, such as ResNet, DenseNet, EfficientNet, MobileNet, or another suitable pretrained model.
3. Train and evaluate the models using PyTorch.
4. Report suitable evaluation metrics, such as accuracy, precision, recall, F1-score, Macro-F1, and confusion matrix.
5. Compare the performance of different models or different training strategies.
6. Discuss important experimental issues, such as class imbalance, overfitting, data augmentation, and model generalization.
7. Provide model interpretation, such as Grad-CAM or another appropriate visualization method.
8. Analyze both successful and failed prediction examples.
9. Submit all required files before the deadline.

## Report

Each group should submit a research report in PDF format. The report should include the following parts:

1. Title and Group Members(student ID and Chinese Name)
2. Abstract
3. Introduction and motivation
4. Dataset description and preprocessing
5. Methodology
6. Experimental results and analysis
7. Model interpretation and error analysis
8. Conclusion
9. References
10. AI tool usage statement

The report should clearly describe the complete deep learning workflow, including data preprocessing, model design, training settings, hyperparameters, evaluation results, and interpretation of the model behavior.

In the experimental results section, students should report both validation performance and test performance. The test set should not be used for model selection or hyperparameter tuning.

## Code and AI Dialogue Record

Each group should also submit the source code and AI dialogue record.

The code should be implemented in PyTorch and organized clearly. It should include the main steps of the project, such as data loading, preprocessing, model construction, training, validation, testing, evaluation, and model saving.

A README file should be provided to explain:

- the file structure;
- the required Python packages;
- how to train the model;
- how to evaluate the model;
- how to reproduce the reported results.

If students use AI tools, they must submit the relevant AI dialogue records. The dialogue records should show how AI tools were used in the project, for example for debugging, code explanation, language polishing, or idea generation. Students should make sure that the final report, code, experiments, and analysis reflect their own understanding.

## Submission Deadline

The submission deadline is **June 10, 12:00 noon**.

## Submission Files

Each group should submit one zip file that includes the following files:

1. A research report in PDF format
2. Source code implemented in PyTorch
3. Trained model weights, if applicable
4. A README file explaining how to run the code
5. AI dialogue records, if AI tools are used

## Academic Integrity

Students are allowed to use AI tools responsibly. However, all use of AI tools must be clearly acknowledged. Directly submitting AI-generated content without understanding, verification, or acknowledgement is not acceptable.

Plagiarism, data fraud, dishonest use of external labels, or direct copying of online solutions will be treated as academic misconduct. All group members are responsible for the submitted work.

## Marking

This project counts for **30%** of the final score.

The marking consists of two major components:

| Component | Marks |
|---|---:|
| Research report, including experimental analysis and model interpretation | 20 |
| Code implementation, reproducibility, and AI dialogue record | 10 |
| Total | 30 |
