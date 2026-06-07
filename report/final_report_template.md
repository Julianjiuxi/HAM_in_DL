# Skin Lesion Classification Using Deep Learning

## Title and Group Members

- Project title: Skin Lesion Classification Using Deep Learning
- Group members: [FILL IN STUDENT ID AND CHINESE NAME]

## Abstract

[FILL IN AFTER TRAINING]

Briefly summarize the task, dataset, models, main validation/test results, Grad-CAM findings, and conclusion.

## Introduction

[FILL IN]

Discuss skin lesion classification, medical image analysis, why CNNs and transfer learning are useful, and the project objective.

## Dataset Description and Preprocessing

[FILL IN]

Describe the ISIC 2018 Task 3 / HAM10000 seven-class dataset, Kaggle HAM10000 training source, provided TestSet, class labels, train/validation split, image resizing, augmentation, ImageNet normalization, and class imbalance.

## Methodology

[FILL IN]

Describe:

- Baseline CNN architecture
- ResNet18 transfer learning model
- Loss function and optional class weights
- Optimizer, learning rate, batch size, epochs, early stopping
- Validation macro-F1 checkpoint selection
- Test set used only for final evaluation

## Experimental Results and Analysis

[FILL IN AFTER TRAINING]

### Validation Results

| Model | Accuracy | Macro Precision | Macro Recall | Macro-F1 | Weighted-F1 |
|---|---:|---:|---:|---:|---:|
| Baseline CNN | [FILL IN] | [FILL IN] | [FILL IN] | [FILL IN] | [FILL IN] |
| ResNet18 | [FILL IN] | [FILL IN] | [FILL IN] | [FILL IN] | [FILL IN] |

### Test Results

| Model | Accuracy | Macro Precision | Macro Recall | Macro-F1 | Weighted-F1 |
|---|---:|---:|---:|---:|---:|
| Baseline CNN | [FILL IN] | [FILL IN] | [FILL IN] | [FILL IN] | [FILL IN] |
| ResNet18 | [FILL IN] | [FILL IN] | [FILL IN] | [FILL IN] | [FILL IN] |

### Confusion Matrix Discussion

[FILL IN AFTER EVALUATION]

Discuss which classes are often confused and relate this to class imbalance or visual similarity.

### Comparison and Generalization

[FILL IN AFTER TRAINING]

Compare baseline CNN and ResNet18. Discuss overfitting, validation/test gap, augmentation, class weights, and limitations.

## Model Interpretation and Error Analysis

[FILL IN AFTER GRAD-CAM AND ERROR ANALYSIS]

Include Grad-CAM overlays and representative high-confidence correct and failed predictions. Explain whether the model focuses on lesion regions or irrelevant background.

## Conclusion

[FILL IN]

Summarize findings, limitations, and possible improvements such as stronger augmentation, more transfer models, balanced sampling, or additional validation.

## References

- ISIC 2018 Task 3: Lesion Diagnosis, https://challenge.isic-archive.com/landing/2018/47/
- HAM10000 dataset paper, https://www.nature.com/articles/sdata2018161
- Kaggle HAM10000 mirror, https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000
- PyTorch documentation, https://pytorch.org/docs/stable/index.html
- ResNet paper: He et al., Deep Residual Learning for Image Recognition.

## AI Tool Usage Statement

[FILL IN / ADAPT FROM ai_dialogue_records/ai_usage_statement.md]
