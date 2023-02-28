import os
import numpy as np
import cv2
import torch

from deeplsd.models.deeplsd import DeepLSD


class DeepLSD_detector():
    """ Wrapper class for line detection and matching with SOLD2
        (https://github.com/cvg/SOLD2). """
    def __init__(self, conf, device):
        self.conf = conf
        self.device = device
        self.max_size = 1200
        self.net = DeepLSD(conf)
        ckpt = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            'weights/deeplsd_md.tar')
        ckpt = torch.load(ckpt, map_location='cpu')
        self.net.load_state_dict(ckpt['model'])
        self.net = self.net.to(device).eval()

    def detect_lines(self, image):
        """ Detect line segments and a dense descriptor in a given image. """
        # Restrict input_image to 2D np arrays
        if ((not len(image.shape) == 2)
            or (not isinstance(image, np.ndarray))):
            raise ValueError(
                "The input image should be a 2D numpy array")

        # Pre-process the image
        scale_factor = 1
        if max(image.shape) > self.max_size:
            # Resize the image
            scale_factor = self.max_size / max(image.shape)
            new_size = tuple(np.round(np.array(image.shape)
                                      * scale_factor).astype(int)[[1, 0]])
            image = cv2.resize(image, new_size, interpolation=cv2.INTER_AREA)
        torch_img = torch.as_tensor(image, dtype=torch.float,
                                    device=self.device)[None, None] / 255.

        # Forward pass of the CNN backbone and line detection
        with torch.no_grad():
            lines = self.net({'image': torch_img})['lines'][0]
            lines = lines[:, :, [1, 0]] / scale_factor

        return lines
