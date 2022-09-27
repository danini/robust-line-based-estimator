import numpy as np
import cv2
import torch
from copy import deepcopy
from omegaconf import OmegaConf
from torch.nn.functional import softmax, interpolate

from sold2.model.model_util import SOLD2Net
from sold2.model.line_detection import LineSegmentDetectionModule
from sold2.model.line_matching import WunschLineMatcher
from sold2.train import convert_junc_predictions
from sold2.model.line_detector import line_map_to_segments
from pytlsd import lsd
from pytlbd import lbd_multiscale_pyr, lbd_matching_multiscale


default_conf = {
    'min_length': 15,
    'sold2': {
        'ckpt_path': 'third_party/SOLD2/pretrained_models/sold2_wireframe.tar',
        'device': 'cuda',
        'model_cfg': {
            'model_name': "lcnn_simple",
            'model_architecture': "simple",
            # Backbone related config
            'backbone': "lcnn",
            'backbone_cfg': {
                'input_channel': 1, # Use RGB images or grayscale images.
                'depth': 4,
                'num_stacks': 2,
                'num_blocks': 1,
                'num_classes': 5
            },
            # Junction decoder related config
            'junction_decoder': "superpoint_decoder",
            'junc_decoder_cfg': {},
            # Heatmap decoder related config
            'heatmap_decoder': "pixel_shuffle",
            'heatmap_decoder_cfg': {},
            # Descriptor decoder related config
            'descriptor_decoder': "superpoint_descriptor",
            'descriptor_decoder_cfg': {},
            # Shared configurations
            'grid_size': 8,
            'keep_border_valid': True,
            # Threshold of junction detection
            'detection_thresh': 0.0153846, # 1/65
            'max_num_junctions': 300,
            # Threshold of heatmap detection
            'prob_thresh': 0.5,
            # Weighting related parameters
            'weighting_policy': 'dynamic',
            # [Heatmap loss]
            'w_heatmap': 0.,
            'w_heatmap_class': 1,
            'heatmap_loss_func': "cross_entropy",
            'heatmap_loss_cfg': {
                'policy': 'dynamic'
            },
            # [Heatmap consistency loss]
            # [Junction loss]
            'w_junc': 0.,
            'junction_loss_func': "superpoint",
            'junction_loss_cfg': {
                'policy': 'dynamic'
            },
            # [Descriptor loss]
            'w_desc': 0.,
            'descriptor_loss_func': "regular_sampling",
            'descriptor_loss_cfg': {
                'dist_threshold': 8,
                'grid_size': 4,
                'margin': 1,
                'policy': 'dynamic'
            },
        },
        'line_detector_cfg': {
            'detect_thresh': 0.25,  # depending on your images, you might need to tune this parameter
            'num_samples': 64,
            'sampling_method': "local_max",
            'inlier_thresh': 0.9,
            "use_candidate_suppression": True,
            "nms_dist_tolerance": 3.,
            "use_heatmap_refinement": True,
            "heatmap_refine_cfg": {
                "mode": "local",
                "ratio": 0.2,
                "valid_thresh": 1e-3,
                "num_blocks": 20,
                "overlap_ratio": 0.5
            }
        },
        'multiscale': False,
        'line_matcher_cfg': {
            'cross_check': True,
            'num_samples': 5,
            'min_dist_pts': 8,
            'top_k_candidates': 10,
            'grid_size': 4
        }
    }
}


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


class LBDWrapper():
    def to_multiscale_lines(self, lines):
        ms_lines = []
        for l in lines.reshape(-1, 4):
            ll = np.append(l, [0, np.linalg.norm(l[:2] - l[2:4])])
            ms_lines.append([(0, ll)] + [(i, ll / (i * np.sqrt(2))) for i in range(1, 5)])
        return ms_lines

    def process_pyramid(self, img, n_levels=5,  level_scale=np.sqrt(2)):
        octave_img = img.copy()
        pre_sigma2 = 0
        cur_sigma2 = 1.0
        pyramid = []
        for i in range(n_levels):
            increase_sigma = np.sqrt(cur_sigma2 - pre_sigma2)
            blurred = cv2.GaussianBlur(octave_img, (5, 5), increase_sigma,
                                       borderType=cv2.BORDER_REPLICATE)
            pyramid.append(blurred)

            # Down sample the current octave image to get the next octave image
            new_size = (int(octave_img.shape[1] / level_scale),
                        int(octave_img.shape[0] / level_scale))
            octave_img = cv2.resize(blurred, new_size, 0, 0,
                                    interpolation=cv2.INTER_NEAREST)
            pre_sigma2 = cur_sigma2
            cur_sigma2 = cur_sigma2 * 2
        return pyramid

    def describe_lines(self, img, lines):
        ##########################################################################
        ms_lines = self.to_multiscale_lines(lines[:, :, [1, 0]].reshape(-1, 4))
        pyramid = self.process_pyramid(img)
        descriptors = lbd_multiscale_pyr(pyramid, ms_lines, 9, 7)
        return ms_lines, np.array(descriptors)


class LineMatcher():
    """ A generic line matcher that can detect
        and match different kinds of lines. """
    def __init__(self, line_detector='lsd', line_matcher='lbd',
                 conf=default_conf):
        self.detector = line_detector
        self.matcher = line_matcher
        self.conf = OmegaConf.merge(OmegaConf.create(deepcopy(default_conf)),
                                    OmegaConf.create(conf))
        if line_detector == 'sold2' or line_matcher == 'sold2':
            self.sold2 = Sold2Wrapper(self.conf.sold2)
        if line_matcher == 'lbd':
            self.lbd = LBDWrapper()

    def detect_and_describe_lines(self, image):
        """ Detect line segments and optionally compute descriptors from a
            given image. The line segments are given as a [N, 2, 2] np.array
            in row-col coordinates convention. """
        outputs = {}
        if self.detector == 'lsd':
            outputs["line_segments"] = lsd(
                image)[:, [1, 0, 3, 2]].reshape(-1, 2, 2)
        elif self.detector == 'sold2':
            outputs = self.sold2.detect_lines(image)
        else:
            raise ValueError("Unknown line detector: " + self.detector)

        # Remove short lines
        lines = outputs["line_segments"]
        line_lengths = np.linalg.norm(lines[:, 0] - lines[:, 1], axis=1)
        outputs["line_segments"] = lines[line_lengths > self.conf.min_length]

        # Describe lines with LBD
        if self.matcher == 'lbd':
            outputs["ms_lines"], outputs["descriptor"] = self.lbd.describe_lines(
                image, outputs["line_segments"])

        return outputs

    def match_lines(self, img0, img1, line_features0, line_features1):
        """ Match lines given two sets of pre-extracted line features. """
        if self.matcher == 'lbd':
            lbd_matches = np.array(lbd_matching_multiscale(
                line_features0["ms_lines"], line_features1["ms_lines"],
                list(line_features0["descriptor"]),
                list(line_features1["descriptor"])))
            matches = -np.ones(len(line_features0["descriptor"]), dtype=int)
            matches[lbd_matches[:, 0]] = lbd_matches[:, 1]
        elif self.matcher == 'sold2' and self.detector == 'sold2':
            matches = self.sold2.match_lines(
                line_features0["line_segments"],
                line_features1["line_segments"],
                desc0=line_features0["descriptor"][None],
                desc1=line_features1["descriptor"][None],
                scale_factor0=line_features0["scale_factor"],
                scale_factor1=line_features1["scale_factor"])
        elif self.matcher == 'sold2' and self.detector == 'lsd':
            matches = self.sold2.match_lines(
                line_features0["line_segments"],
                line_features1["line_segments"],
                img0=img0, img1=img1)
        else:
            raise ValueError("Unknown line matcher: " + self.matcher)
        
        # Retrieve the matched lines
        m_lines0 = line_features0["line_segments"][matches != -1]
        m_lines1 = line_features1["line_segments"][matches[matches != -1]]
        return matches, m_lines0, m_lines1
