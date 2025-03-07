import torch

class EdgeDetector:
    def __init__(self, threshold: float = 0.5):
        self.threshold = threshold

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: shape (n, c, h, w)

        # Convert to grayscale if more than one channel.
        if x.shape[1] > 1:
            x = torch.mean(x, dim=1, keepdim=True)
        else:
            x = x.clone()

        # Compute difference with left neighbor: (n, c, h, w-1)
        leftDiff = torch.abs(x[:, :, :, 1:] - x[:, :, :, :-1])
        # Compute difference with bottom neighbor: (n, c, h-1, w)
        bottomDiff = torch.abs(x[:, :, 1:, :] - x[:, :, :-1, :])

        # Initialize output tensor with zeros.
        output = torch.zeros_like(x)

        # Create white (255) and black (0) constants.
        white = torch.tensor(1.0, device=x.device, dtype=x.dtype)
        black = torch.tensor(0.0, device=x.device, dtype=x.dtype)

        # Apply threshold: for left differences assign to columns 1:.
        output[:, :, :, 1:] = torch.where(leftDiff > self.threshold, white, black)
        # For bottom differences assign to rows 1: and combine with left-edge results.
        bottomEdges = torch.where(bottomDiff > self.threshold, white, black)
        output[:, :, 1:, :] = torch.maximum(output[:, :, 1:, :], bottomEdges)

        return output
