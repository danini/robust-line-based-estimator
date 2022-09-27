import numpy as np
import cv2
import torch
from torch.nn.functional import softmax

from third_party.SOLD2.sold2.model.model_util import SOLD2Net
from third_party.SOLD2.sold2.model.line_detection import LineSegmentDetectionModule
from third_party.SOLD2.sold2.model.line_matching import WunschLineMatcher
from third_party.SOLD2.sold2.train import convert_junc_predictions
from third_party.SOLD2.sold2.model.line_detector import line_map_to_segments


class Sold2Wrapper():
    """ Wrapper class for line detection and matching with SOLD2
        (https://github.com/cvg/SOLD2). """
    def __init__(self, conf):
        self.conf = conf
        self.max_size = 800
        self.model = SOLD2Net(conf.model_cfg)
        self.device = conf.device
        ckpt = torch.load(conf.ckpt_path, map_location=self.device)
        self.model.load_state_dict(ckpt["model_state_dict"], strict=False)
        self.model = self.model.to(self.device)
        self.model = self.model.eval()

        # Initialize the line detector
        self.line_detector = LineSegmentDetectionModule(
            **self.conf.line_detector_cfg)

        # Initialize the line matcher
        self.line_matcher = WunschLineMatcher(**self.conf.line_matcher_cfg)

    def detect_lines(self, image, desc_only=False):
        """ Detect line segments and a dense descriptor in a given image. """
        # Restrict input_image to 4D torch tensor
        if ((not len(image.shape) == 2)
            or (not isinstance(image, np.ndarray))):
            raise ValueError(
                "The input image should be a 2D numpy array")

        # Pre-process the image
        scale_factor = 1
        if max(image.shape) > self.max_size:
            # Resize the image
            orig_shape = image.shape
            scale_factor = self.max_size / max(image.shape)
            new_size = tuple(np.round(np.array(image.shape)
                                      * scale_factor).astype(int)[[1, 0]])
            image = cv2.resize(image, new_size, interpolation=cv2.INTER_AREA)
        torch_img = torch.as_tensor(image, dtype=torch.float,
                                    device=self.device) / 255.
        torch_img = torch_img[None, None]

        # Forward of the CNN backbone
        with torch.no_grad():
            net_outputs = self.model(torch_img)

        outputs = {"descriptor": net_outputs["descriptors"][0],
                   "scale_factor": scale_factor}

        if not desc_only:
            junc_np = convert_junc_predictions(
                net_outputs["junctions"], self.conf.model_cfg.grid_size,
                self.conf.model_cfg.detection_thresh,
                self.conf.model_cfg.max_num_junctions)
            junctions = np.where(junc_np["junc_pred_nms"].squeeze())
            junctions = np.concatenate([junctions[0][..., None],
                                        junctions[1][..., None]], axis=-1)

            heatmap = softmax(
                net_outputs["heatmap"],
                dim=1)[:, 1:, :, :].cpu().numpy()[0, 0]

            # Run the line detector.
            line_map, junctions, heatmap = self.line_detector.detect(
                junctions, heatmap, device=self.device)
            line_map = line_map.cpu().numpy()
            junctions = junctions.cpu().numpy()
            outputs["heatmap"] = heatmap.cpu().numpy()
            outputs["junctions"] = junctions

            # Convert to line segments
            line_segments = line_map_to_segments(junctions, line_map)
            outputs["line_segments"] = line_segments / scale_factor

        return outputs

    def match_lines(self, lines0, lines1, desc0=None, desc1=None,
                    img0=None, img1=None, scale_factor0=1, scale_factor1=1):
        """ Match lines between two images.
            Either the dense descriptor or the input images should be provided.
            If the image dense descriptors desc0 and desc1 are not provided,
            they will be automatically be computed from the images. """
        if desc0 is None and img0 is None:
            raise ValueError("No dense descriptor or image provided.")
        if desc1 is None and img1 is None:
            raise ValueError("No dense descriptor or image provided.")

        if desc0 is None:
            out0 = self.detect_lines(img0, desc_only=True)
            desc0 = out0["descriptor"][None]
            scale_factor0 = out0["scale_factor"]
        if desc1 is None:
            out1 = self.detect_lines(img1, desc_only=True)
            desc1 = out1["descriptor"][None]
            scale_factor1 = out1["scale_factor"]

        # Match the lines between the two images, taking into account
        # the difference of scale between lines and descriptors
        matches = self.line_matcher.forward(
            lines0 * scale_factor0, lines1 * scale_factor1, desc0, desc1)
        return matches
