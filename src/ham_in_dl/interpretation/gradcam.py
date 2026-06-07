"""Grad-CAM implementation or wrapper."""

import torch
import torch.nn.functional as F


class GradCAM:
    """Minimal Grad-CAM implementation for convolutional classifiers."""

    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.activations = None
        self.gradients = None
        self.handles = [
            target_layer.register_forward_hook(self._save_activations),
            target_layer.register_full_backward_hook(self._save_gradients),
        ]

    def _save_activations(self, _module, _inputs, output):
        self.activations = output.detach()

    def _save_gradients(self, _module, _grad_inputs, grad_output):
        self.gradients = grad_output[0].detach()

    def remove_hooks(self) -> None:
        for handle in self.handles:
            handle.remove()

    def __call__(
        self,
        input_tensor: torch.Tensor,
        *,
        target_class: int | None = None,
    ) -> tuple[torch.Tensor, int, float]:
        self.model.eval()
        self.model.zero_grad(set_to_none=True)

        logits = self.model(input_tensor)
        probabilities = torch.softmax(logits, dim=1)
        confidence, predicted = probabilities.max(dim=1)
        if target_class is None:
            target_class = int(predicted.item())

        score = logits[:, target_class].sum()
        score.backward()

        if self.activations is None or self.gradients is None:
            raise RuntimeError("Grad-CAM hooks did not capture activations/gradients.")

        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = F.relu(cam)
        cam = F.interpolate(
            cam,
            size=input_tensor.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        cam = cam.squeeze()
        cam_min = cam.min()
        cam_max = cam.max()
        cam = (cam - cam_min) / (cam_max - cam_min + 1e-8)
        return cam.detach().cpu(), target_class, float(confidence.item())


def generate_gradcam(
    model,
    input_tensor: torch.Tensor,
    target_layer,
    *,
    target_class: int | None = None,
) -> tuple[torch.Tensor, int, float]:
    """Generate a Grad-CAM heatmap for a trained model and input image."""
    gradcam = GradCAM(model, target_layer)
    try:
        return gradcam(input_tensor, target_class=target_class)
    finally:
        gradcam.remove_hooks()
