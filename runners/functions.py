import numpy as np
import cv2
import torch
import pyprogressivex


def verify_pyprogressivex(img_width, img_height, lines_segments, threshold = 2.0):
    lines = []
    weights = []
    for i in range(lines_segments.shape[0]):
        p0 = lines_segments[i,0]
        p1 = lines_segments[i,1]
        len = np.linalg.norm(p1 - p0)
        lines.append([p0[0], p0[1], p1[0], p1[1]])
        weights.append(len)

    lines = np.array(lines)
    weights = np.array(weights)

    vanishing_points, labeling = pyprogressivex.findVanishingPoints(
        np.ascontiguousarray(lines),
        np.ascontiguousarray(weights),
        img_width, img_height,
        threshold = threshold,
        conf = 0.99,
        spatial_coherence_weight = 0.0,
        neighborhood_ball_radius = 1.0,
        maximum_tanimoto_similarity = 1.0,
        max_iters = 1000,
        minimum_point_number = 5,
        maximum_model_number = -1,
        sampler_id = 0,
        scoring_exponent = 1.0,
        do_logging = False)
    return vanishing_points, labeling


def joint_vp_detection_and_matching(img_width, img_height, m_lines0,
                                    m_lines1, threshold = 2.0):
    """ m_lines0 and m_lines1 are two sets of matching [N, 2, 2] lines in x-y coordinate convention.
        Returns a set of matching VPs (of size [N, 3] each) and their line labels.
    """
    lines = []
    weights = []
    for i in range(len(m_lines0)):
        l0, l1 = m_lines0[i], m_lines1[i]
        weights.append((np.linalg.norm(l0[1] - l0[1]) + np.linalg.norm(l1[1] - l1[1])) / 2)
        lines.append([l0[0, 0], l0[0, 1], l0[1, 0], l0[1, 1], l1[0, 0], l1[0, 1], l1[1, 0], l1[1, 1]])

    lines = np.array(lines)
    weights = np.array(weights)

    vanishing_points, labeling = pyprogressivex.findCommonVanishingPoints(
        np.ascontiguousarray(lines), 
        np.ascontiguousarray(weights), 
        img_width, img_height,
        threshold = threshold,
        conf = 0.99,
        spatial_coherence_weight = 0.0,
        neighborhood_ball_radius = 1.0,
        maximum_tanimoto_similarity = 1.0,
        max_iters = 1000,
        minimum_point_number = 5,
        maximum_model_number = -1,
        sampler_id = 0,
        scoring_exponent = 1.0,
        do_logging = False)

    # Set the line labeling to -1 when the line has no corresponding VP
    n_vp = len(vanishing_points)
    labeling[labeling == n_vp] = -1
    vanishing_points = np.array(vanishing_points)
    vp1 = vanishing_points[:, :3]
    vp2 = vanishing_points[:, 3:]

    return vp1, vp2, labeling


def find_homography_points(lines0, lines1, img1_size, img2_size, threshold = 3.0,
                           confidence = 0.5, minimum_point_number = 10):
    if lines0.shape[0] != lines1.shape[0]:
        print("The number of line segments do not match.")
        return

    line_number = lines0.shape[0]
    correspondences = []

    for i in range(line_number):
        correspondences.append([lines0[i][0][0], lines0[i][0][1], lines1[i][0][0], lines1[i][0][1]])
        correspondences.append([lines0[i][1][0], lines0[i][1][1], lines1[i][1][0], lines1[i][1][1]])
    correspondences = np.array(correspondences).astype(np.float64)

    homographies, labeling = pyprogressivex.findHomographies(
        np.ascontiguousarray(correspondences),
        img1_size[1], img1_size[0],
        img2_size[1], img2_size[0],
        threshold = threshold,
        conf = confidence,
        spatial_coherence_weight = 0.0,
        neighborhood_ball_radius = 200.0,
        maximum_tanimoto_similarity = 0.4,
        max_iters = 1000,
        minimum_point_number = minimum_point_number,
        maximum_model_number = 10,
        sampler_id = 3,
        do_logging = False)

    model_number = max(labeling)
    line_labeling = []
    for i in range(line_number):
        idx = 2 * i

        if labeling[idx] == labeling[idx + 1]:
            line_labeling.append(labeling[idx])
        else:
            line_labeling.append(model_number)

    line_labeling = np.array(line_labeling)
    return homographies, line_labeling


def normalize_keypoints(keypoints, K):
    '''Normalize keypoints using the calibration data.'''
    C_x = K[0, 2]
    C_y = K[1, 2]
    f_x = K[0, 0]
    f_y = K[1, 1]
    keypoints = (keypoints - np.array([[C_x, C_y]])) / np.array([[f_x, f_y]])
    return keypoints

def find_relative_pose_from_points(kp_matches, K1, K2, kp_scores=None):
    if kp_scores is not None:
        # Select the points with lowest ratio score
        good_matches = kp_scores < 0.8
        pts1 = kp_matches[good_matches, :2]
        pts2 = kp_matches[good_matches, 2:]
    else:
        pts1 = kp_matches[:, :2]
        pts2 = kp_matches[:, 2:]

    if len(pts1) < 5:
        return None, None, None, None

    # Normalize KP
    p1n = normalize_keypoints(pts1, K1)
    p2n = normalize_keypoints(pts2, K2)

    # Find the essential matrix with OpenCV RANSAC
    E, inl_mask = cv2.findEssentialMat(p1n, p2n, np.eye(3), cv2.RANSAC, 0.999, 1e-3)
    if E is None:
        return None, None, None, None

    # Obtain the corresponding pose
    best_num_inliers = 0
    ret = None
    mask = np.array(inl_mask)[:, 0].astype(bool)
    for _E in np.split(E, len(E) / 3):
        n, R, t, _ = cv2.recoverPose(
            _E, p1n, p2n, np.eye(3), 1e9, mask=inl_mask[:, 0])
        if n > best_num_inliers:
            best_num_inliers = n
            ret = (R, t[:, 0], pts1[mask], pts2[mask])
    return ret

def sg_matching(img1, img2, superglue_matcher, device):
    with torch.no_grad():
        inputs = {
            'image0': torch.tensor(img1, dtype=torch.float, device=device)[None, None] / 255.,
            'image1': torch.tensor(img2, dtype=torch.float, device=device)[None, None] / 255.
        }
        pred = superglue_matcher(inputs)
        pred = {k: v[0].cpu().numpy() for k, v in pred.items()}
        kpts0, kpts1 = pred['keypoints0'], pred['keypoints1']
        matches, conf = pred['matches0'], pred['matching_scores0']
    # Keep the matching keypoints.
    valid = matches > -1
    mkpts0 = kpts0[valid]
    mkpts1 = kpts1[matches[valid]]
    mconf = conf[valid]
    return np.concatenate([mkpts0, mkpts1], axis=1), mconf

