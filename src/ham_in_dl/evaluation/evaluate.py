"""Model evaluation on validation or test split."""

import torch
from tqdm import tqdm


@torch.no_grad()
def predict(model, dataloader, device) -> tuple[list[int], list[int], list[float]]:
    """Run model inference and return targets, predictions, and confidence scores."""
    model.eval()
    y_true: list[int] = []
    y_pred: list[int] = []
    confidences: list[float] = []

    for images, labels in tqdm(dataloader, desc="predict", leave=False):
        images = images.to(device)
        logits = model(images)
        probabilities = torch.softmax(logits, dim=1)
        confidence, predictions = probabilities.max(dim=1)

        y_true.extend(int(x) for x in labels.cpu().tolist())
        y_pred.extend(int(x) for x in predictions.cpu().tolist())
        confidences.extend(float(x) for x in confidence.cpu().tolist())

    return y_true, y_pred, confidences


def evaluate_model(model, dataloader, device) -> tuple[list[int], list[int], list[float]]:
    """Compatibility wrapper around predict."""
    return predict(model, dataloader, device)
