"""Image preprocessing and augmentation."""

from torchvision import transforms


def build_train_transforms(image_size: int = 224):
    """TODO: Tune data augmentation for training."""
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
    ])


def build_eval_transforms(image_size: int = 224):
    """Transforms for validation and test."""
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
    ])
