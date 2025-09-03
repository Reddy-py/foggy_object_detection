# dehaze_aodnet.py (example skeleton)

import torch
import torch.nn as nn
import torchvision.transforms as T


class AODNet(nn.Module):
    def __init__(self):
        super(AODNet, self).__init__()
        # Define your AOD-Net layers here (this is just a placeholder)
        # You would load the pretrained weights in real code.
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        # ... more layers ...

    def forward(self, x):
        # Perform the forward pass
        x = self.conv1(x)
        # ... the rest of AOD-Net ...
        return x


class DehazerAOD:
    """A wrapper class for loading and applying AOD-Net."""

    def __init__(self, checkpoint_path, device='cpu'):
        self.device = device
        self.model = AODNet().to(device)
        # Load pretrained weights
        self.model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        self.model.eval()

        # Define image transforms as needed for your network
        self.transform = T.Compose([
            T.ToTensor()
            # Possibly add normalization
        ])

    def dehaze(self, bgr_frame):
        """Dehaze an OpenCV BGR frame and return the enhanced image."""
        # Convert BGR (OpenCV) to RGB for PyTorch
        rgb_frame = bgr_frame[:, :, ::-1]

        # Transform to tensor
        input_tensor = self.transform(rgb_frame).unsqueeze(0).to(self.device)

        # Forward pass
        with torch.no_grad():
            output_tensor = self.model(input_tensor)

        # Convert back to NumPy (and BGR) for OpenCV display
        output_tensor = output_tensor.squeeze(0).cpu().numpy().transpose(1, 2, 0)
        output_tensor = (output_tensor * 255).clip(0, 255).astype('uint8')

        # Convert RGB back to BGR
        enhanced_frame = output_tensor[:, :, ::-1]
        return enhanced_frame
