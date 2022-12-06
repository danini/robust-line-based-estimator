import numpy as np
from copy import deepcopy
from omegaconf import OmegaConf

from pytlsd import lsd
from pytlbd import lbd_matching_multiscale

from .lbd import LBDWrapper
from .sold2 import Sold2Wrapper
from .superglue_endpoints import SuperGlueEndpoints
from .gluestick import GlueStick


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
    },
    'sp_params': {
        'descriptor_dim': 256,
        'nms_radius': 4,
        'keypoint_threshold': 0.005,
        'max_keypoints': -1,
        'remove_borders': 4,
    },
    'sg_params': {
        'descriptor_dim': 256,
        'weights': 'outdoor',
        'keypoint_encoder': [32, 64, 128, 256],
        'GNN_layers': ['self', 'cross'] * 9,
        'sinkhorn_iterations': 100,
        'match_threshold': 0.2,
    },
    'device': 'cuda',
}


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
        if line_matcher == 'superglue_endpoints':
            self.sg_endpoints = SuperGlueEndpoints(self.conf)
        if line_matcher == 'gluestick':
            self.gluestick = GlueStick(self.conf)

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

        # Describe line endpoints with SuperPoint
        if self.matcher == 'superglue_endpoints':
            outputs["descriptor"] = self.sg_endpoints.describe_lines(
                image, outputs["line_segments"])

        # Describe line endpoints with SuperPoint
        if self.matcher == 'gluestick':
            outputs["descriptor"] = self.gluestick.describe_lines(
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
        elif self.matcher == 'superglue_endpoints':
            matches = self.sg_endpoints.match_lines(
                line_features0["line_segments"],
                line_features1["line_segments"],
                line_features0["descriptor"],
                line_features1["descriptor"],
                img0.shape, img1.shape)
        elif self.matcher == 'gluestick':
            matches = self.gluestick.match_lines(
                line_features0["descriptor"],
                line_features1["descriptor"])
        else:
            raise ValueError("Unknown line matcher: " + self.matcher)
        
        # Retrieve the matched lines
        m_lines0 = line_features0["line_segments"][matches != -1]
        m_lines1 = line_features1["line_segments"][matches[matches != -1]]
        return matches, m_lines0, m_lines1
