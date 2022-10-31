import numpy as np
import sys
import cv2
import pymagsac

def normalize_keypoints(keypoints, K):
    '''Normalize keypoints using the calibration data.'''
    C_x = K[0, 2]
    C_y = K[1, 2]
    f_x = K[0, 0]
    f_y = K[1, 1]
    return (keypoints - np.array([[C_x, C_y]])) / np.array([[f_x, f_y]])

def run_point_based_relative_pose(K1, K2, point_matches, junction_matches, th_pixel=1.0, best_k = np.inf, config=0):
    
    matches = []
    if config == 0 or len(junction_matches) == 0:
        matches = point_matches
    elif config == 1:
        best_k = min(best_k, junction_matches.shape[0])
        matches = junction_matches[:best_k,:]
    elif config == 2:
        best_k = min(best_k, junction_matches.shape[0])
        matches = np.concatenate((point_matches, junction_matches[:best_k,:]), axis=0)

    try:
        #norm_matches = np.zeros(matches.shape)
        #norm_matches[:, :2] = normalize_keypoints(matches[:, :2], K1)
        #norm_matches[:, 2:] = normalize_keypoints(matches[:, 2:], K2)
        #threshold_normalizer = 0.25 * (K1[0, 0] + K1[1, 1] + K2[0, 0] + K2[1, 1])

        #E, mask = cv2.findEssentialMat(matches[:,:2], matches[:,2:], np.eye(3), cv2.USAC_MAGSAC, 0.999, th_pixel / threshold_normalizer)
        E, mask = pymagsac.findEssentialMatrix(
            np.ascontiguousarray(matches).astype(np.float64), # Point correspondences in the two images
            np.ascontiguousarray(K1).astype(np.float64), # Intrinsic camera parameters of the source image
            np.ascontiguousarray(K2).astype(np.float64), # Intrinsic camera parameters of the destination image
            int(K1[1, 2] * 2), int(K1[0, 2] * 2), int(K2[1, 2] * 2), int(K2[0, 2] * 2), # The sizes of the images
            probabilities = [],
            sigma_th = th_pixel, # Inlier-outlier threshold
            use_sprt = False,
            conf = 0.999, # RANSAC confidence
            min_iters = 5000, # The minimum iteration number in RANSAC. If time does not matter, I suggest setting it to, e.g., 1000
            max_iters = 5000, # The maximum iteration number in RANSAC
            sampler = 1) # Sampler index (0 - Uniform, 1 - PROSAC, 2 - P-NAPSAC, 3 - NG-RANSAC, 4 - AR-Sampler)

        _, R, t, _ = cv2.recoverPose(E, matches[mask, :2], matches[mask, 2:])
    except:
        return None, None
    
    return R, t

