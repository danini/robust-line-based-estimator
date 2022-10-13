import numpy as np
import cv2
import h5py
import math
import threading

def load_h5(filename):
    '''Loads dictionary from hdf5 file'''
    dict_to_load = {}
    try:
        #with self.lock:
        with h5py.File(filename, 'r') as f:
            keys = [key for key in f.keys()]
            for key in keys:
                dict_to_load[key] = f[key][()]
    except:
        print('Cannot find file {}'.format(filename))
    return dict_to_load

def append_h5(dict_to_save, filename):
    '''Saves dictionary to HDF5 file'''

    with h5py.File(filename, 'a') as f:
        #with self.lock:
        for key in dict_to_save:
            f.create_dataset(key, data=dict_to_save[key])

def read_h5(key, filename):
    '''Saves dictionary to HDF5 file'''

    with h5py.File(filename, 'r') as f:
        #with self.lock:
        if key in f.keys():
            return np.array(f.get(key))
        else:
            return None

def convert_ms_lines(ms_lines):
    np_ms_lines = np.zeros((len(ms_lines), 5 * 6))
    for row_idx in range(np_ms_lines.shape[0]):
        for item_idx in range(5):
            np_ms_lines[row_idx, 6 * item_idx : 6 * (item_idx + 1)] = ms_lines[row_idx][item_idx][1]
    return np_ms_lines

def parse_ms_lines(np_ms_lines):    
    ms_lines = []
    for row_idx in range(np_ms_lines.shape[0]):
        inner_list = []
        for item_idx in range(5):
            inner_list.append((item_idx, np_ms_lines[row_idx, 6 * item_idx : 6 * (item_idx + 1)]))
        ms_lines.append(inner_list)
    return ms_lines

def normalize_keypoints(keypoints, K):
    '''Normalize keypoints using the calibration data.'''

    C_x = K[0, 2]
    C_y = K[1, 2]
    f_x = K[0, 0]
    f_y = K[1, 1]
    keypoints = (keypoints - np.array([[C_x, C_y]])) / np.array([[f_x, f_y]])

    return keypoints

def sampson_epipolar_distance(pts1, pts2, F, squared = False, eps = 1e-8):
    r"""Return Sampson distance for correspondences given the fundamental matrix.
    Args:
        pts1: correspondences from the left images with shape
          (*, N, 2 or 3). If they are not homogeneous, converted automatically.
        pts2: correspondences from the right images with shape
          (*, N, 2 or 3). If they are not homogeneous, converted automatically.
        Fm: Fundamental matrices with shape :math:`(*, 3, 3)`. Called Fm to
          avoid ambiguity with torch.nn.functional.
        squared: if True (default), the squared distance is returned.
        eps: Small constant for safe sqrt.

    Returns:
        the computed Sampson distance with shape :math:`(*, N)`.
    """
    if pts1.shape[1] == 2:
        pts1 = np.concatenate((pts1, np.ones((pts1.shape[0], 1), np.float64)), axis=1)

    if pts2.shape[1] == 2:
        pts2 = np.concatenate((pts2, np.ones((pts2.shape[0], 1), np.float64)), axis=1)

    # From Hartley and Zisserman, Sampson error (11.9)
    # sam =  (x'^T F x) ** 2 / (  (((Fx)_1**2) + (Fx)_2**2)) +  (((F^Tx')_1**2) + (F^Tx')_2**2)) )
    # line1_in_2 = (F @ pts1.transpose(dim0=-2, dim1=-1)).transpose(dim0=-2, dim1=-1)
    # line2_in_1 = (F.transpose(dim0=-2, dim1=-1) @ pts2.transpose(dim0=-2, dim1=-1)).transpose(dim0=-2, dim1=-1)

    # Instead we can just transpose F once and switch the order of multiplication
    F_t = F.T
    line1_in_2 = pts1 @ F_t
    line2_in_1 = pts2 @ F

    # numerator = (x'^T F x) ** 2
    numerator = (pts2 * line1_in_2).sum(axis=-1) ** 2

    # denominator = (((Fx)_1**2) + (Fx)_2**2)) +  (((F^Tx')_1**2) + (F^Tx')_2**2))
    denominator = np.linalg.norm(line1_in_2[..., :2], 2, axis=-1) ** 2 + np.linalg.norm(line2_in_1[..., :2], 2, axis=-1) ** 2
    out = numerator / denominator
    if squared:
        return out
    return (out + eps) ** (1/2)

def sampson_errors(keypoints1, keypoints2, P1, P2):
    P1_full = np.eye(4); P1_full[0:3, :] = P1[0:3, :]    # convert to 4x4
    P2_full = np.eye(4); P2_full[0:3, :] = P2[0:3, :]    # convert to 4x4
    P_canon = P2_full.dot(cv2.invert(P1_full)[1])    # find canonical P which satisfies P2 = P_canon * P1
    
    # "F = [t]_cross * R" [HZ 9.2.4]; transpose is needed for numpy
    F = np.cross(P_canon[0:3, 3], P_canon[0:3, 0:3], axisb=0).T
    return sampson_epipolar_distance(keypoints1, keypoints2, F)

#def sampson_errors(keypoints1, keypoints2, R, T, K1, K2):
#    P1_full = np.eye(4); P1_full[0:3, :] = P1[0:3, :]    # convert to 4x4
#    P2_full = np.eye(4); P2_full[0:3, :] = P2[0:3, :]    # convert to 4x4
#    P_canon = P2_full.dot(cv2.invert(P1_full)[1])    # find canonical P which satisfies P2 = P_canon * P1
#    
#    # "F = [t]_cross * R" [HZ 9.2.4]; transpose is needed for numpy
#    F = np.cross(P_canon[0:3, 3], P_canon[0:3, 0:3], axisb=0).T
#    return sampson_epipolar_distance(keypoints1, keypoints2, F)

def reprojection_errors(keypoints1, keypoints2, points3d, P1, P2):
    if points3d.shape[1] == 3:
        points3d = np.concatenate((points3d, np.ones((points3d.shape[0], 1), np.float64)), axis=1)

    # Projecting the 3D points in the images
    projections1 = points3d @ P1.T
    projections2 = points3d @ P2.T

    behind_camera = (projections1[:, 2] < 0) & (projections2[:, 2] < 0)

    # Homogeneous division
    projections1 = projections1[:, :2] / projections1[:, 2, None]
    projections2 = projections2[:, :2] / projections2[:, 2, None]

    # Calculating the difference between the original keypoints and the projections
    diff1 = keypoints1 - projections1
    diff2 = keypoints2 - projections2    
    
    norm1 = np.linalg.norm(diff1, axis=1)
    norm2 = np.linalg.norm(diff2, axis=1)
    avg_norms = 0.5 * (norm1 + norm2)
    avg_norms[behind_camera] = 1e10
    return avg_norms

def linear_eigen_triangulation(u1, P1, u2, P2, max_coordinate_value=1.e16):
    """
    Linear Eigenvalue based (using SVD) triangulation.
    Wrapper to OpenCV's "triangulatePoints()" function.
    Relative speed: 1.0
    
    (u1, P1) is the reference pair containing normalized image coordinates (x, y) and the corresponding camera matrix.
    (u2, P2) is the second pair.
    "max_coordinate_value" is a threshold to decide whether points are at infinity
    
    u1 and u2 are matrices: amount of points equals #rows and should be equal for u1 and u2.
    
    The status-vector is based on the assumption that all 3D points have finite coordinates.
    """
    x = cv2.triangulatePoints(P1[0:3, 0:4], P2[0:3, 0:4], u1.T, u2.T)    # OpenCV's Linear-Eigen triangl
    
    x[0:3, :] /= x[3:4, :]    # normalize coordinates
    x_status = (np.max(abs(x[0:3, :]), axis=0) <= max_coordinate_value)    # NaN or Inf will receive status False
    
    return x[0:3, :].T.astype(np.float64), x_status

def polynomial_triangulation(u1, P1, u2, P2):
    """
    Polynomial (Optimal) triangulation.
    Uses Linear-Eigen for final triangulation.
    Relative speed: 0.1
    
    (u1, P1) is the reference pair containing normalized image coordinates (x, y) and the corresponding camera matrix.
    (u2, P2) is the second pair.
    
    u1 and u2 are matrices: amount of points equals #rows and should be equal for u1 and u2.
    
    The status-vector is based on the assumption that all 3D points have finite coordinates.
    """
    P1_full = np.eye(4); P1_full[0:3, :] = P1[0:3, :]    # convert to 4x4
    P2_full = np.eye(4); P2_full[0:3, :] = P2[0:3, :]    # convert to 4x4
    P_canon = P2_full.dot(cv2.invert(P1_full)[1])    # find canonical P which satisfies P2 = P_canon * P1
    
    # "F = [t]_cross * R" [HZ 9.2.4]; transpose is needed for numpy
    F = np.cross(P_canon[0:3, 3], P_canon[0:3, 0:3], axisb=0).T
    
    # Other way of calculating "F" [HZ (9.2)]
    #op1 = (P2[0:3, 3:4] - P2[0:3, 0:3] .dot (cv2.invert(P1[0:3, 0:3])[1]) .dot (P1[0:3, 3:4]))
    #op2 = P2[0:3, 0:4] .dot (cv2.invert(P1_full)[1][0:4, 0:3])
    #F = np.cross(op1.reshape(-1), op2, axisb=0).T
    
    # Project 2D matches to closest pair of epipolar lines
    u1_new, u2_new = cv2.correctMatches(F, u1.reshape(1, len(u1), 2), u2.reshape(1, len(u1), 2))
    
    # For a purely sideways trajectory of 2nd cam, correctMatches() returns NaN for all possible points!
    if np.isnan(u1_new).all() or np.isnan(u2_new).all():
        F = cv2.findFundamentalMat(u1, u2, cv2.FM_8POINT)[0]    # so use a noisy version of the fund mat
        u1_new, u2_new = cv2.correctMatches(F, u1.reshape(1, len(u1), 2), u2.reshape(1, len(u1), 2))
    
    # Triangulate using the refined image points
    return linear_eigen_triangulation(u1_new[0], P1, u2_new[0], P2)    # TODO: replace with linear_LS: better results for points not at Inf


def get_implicit_line(endpoint1, endpoint2):
    v = endpoint2 - endpoint1
    v /= np.linalg.norm(v)
    n = [-v[1], v[0]]
    c = -n[0] * endpoint1[0] - n[1] * endpoint1[1]
    return np.array([n[0], n[1], c])

def get_endpoint_correspondences(m_lines1, m_lines2):
    num_matches = m_lines1.shape[0]

    if num_matches == 0:
        return np.array([])

    keypoints1 = []
    keypoints2 = []
    for idx in range(num_matches):
        # Endpoints in Source Image
        end_src_1 = m_lines1[idx][0][:]
        end_src_2 = m_lines1[idx][1][:]
        # Endpoints in Destination Image
        end_dst_1 = m_lines2[idx][0][:]
        end_dst_2 = m_lines2[idx][1][:]

        keypoints1.append(end_src_1)
        keypoints2.append(end_dst_1)
        keypoints1.append(end_src_2)
        keypoints2.append(end_dst_2)

    keypoints1 = np.array(keypoints1)
    keypoints2 = np.array(keypoints2)
    keypoints1 = keypoints1[:, [1, 0]]
    keypoints2 = keypoints2[:, [1, 0]]
    keypoints = np.concatenate((keypoints1, keypoints2), axis=1)
    return keypoints


def get_line_junctions(m_lines1, m_lines2, valid_pairs = None):
    num_matches = m_lines1.shape[0]

    if valid_pairs is None:
        keypoints1 = []
        keypoints2 = []
        distances = []
        for line_idx1 in range(num_matches - 1):
            # Line in the source image
            line11 = get_implicit_line(m_lines1[line_idx1][0][:], m_lines1[line_idx1][1][:])
            # Line in the destination image
            line21 = get_implicit_line(m_lines2[line_idx1][0][:], m_lines2[line_idx1][1][:])

            for line_idx2 in range(line_idx1 + 1, num_matches):
                # Line in the source image
                line12 = get_implicit_line(m_lines1[line_idx2][0][:], m_lines1[line_idx2][1][:])
                # Line in the destination image
                line22 = get_implicit_line(m_lines2[line_idx2][0][:], m_lines2[line_idx2][1][:])

                # Intersection in the source image
                intersection1 = np.cross(line11, line12)
                if abs(intersection1[2]) < 1e-9:
                    continue
                intersection1 /= intersection1[2]
                # Intersection in the destination image
                intersection2 = np.cross(line21, line22)
                if abs(intersection2[2]) < 1e-9:
                    continue
                intersection2 /= intersection2[2]
                # Save the junctions
                keypoints1.append(intersection1[:2])
                keypoints2.append(intersection2[:2])
        return np.array(keypoints1), np.array(keypoints2), None
    else:
        keypoints1 = []
        keypoints2 = []
        distances = []
        for (line_idx1, line_idx2, dist) in valid_pairs:
            # Line in the source image
            line11 = get_implicit_line(m_lines1[line_idx1][0][:], m_lines1[line_idx1][1][:])
            # Line in the destination image
            line21 = get_implicit_line(m_lines2[line_idx1][0][:], m_lines2[line_idx1][1][:])

            # Line in the source image
            line12 = get_implicit_line(m_lines1[line_idx2][0][:], m_lines1[line_idx2][1][:])
            # Line in the destination image
            line22 = get_implicit_line(m_lines2[line_idx2][0][:], m_lines2[line_idx2][1][:])

            # Intersection in the source image
            intersection1 = np.cross(line11, line12)
            if abs(intersection1[2]) < 1e-9:
                continue
            intersection1 /= intersection1[2]
            # Intersection in the destination image
            intersection2 = np.cross(line21, line22)
            if abs(intersection2[2]) < 1e-9:
                continue
            intersection2 /= intersection2[2]
            # Save the junctions
            keypoints1.append(intersection1[:2])
            keypoints2.append(intersection2[:2])
            distances.append(dist)
        return np.array(keypoints1), np.array(keypoints2), np.array(distances)

def filter_points_by_relative_pose(K1, K2, R, t, keypoints1, keypoints2, threshold = 3.0):
    # Normalize the obtained keypoints by the camera matrix
    kp1n = normalize_keypoints(keypoints1, K1).astype(np.float64)
    kp2n = normalize_keypoints(keypoints2, K2).astype(np.float64)
    
    # Composing the projection matrices from the rotations and translations
    P1 = np.concatenate([np.identity(3), np.zeros((3, 1))], axis=1)
    P2 = np.concatenate((R, t[:, np.newaxis]), axis=1)
    
    points3d, status = polynomial_triangulation(kp1n, P1, kp2n, P2)
    errors = reprojection_errors(keypoints1, keypoints2, points3d, K1 @ P1, K2 @ P2)
    inlier_mask = errors < threshold    
    return inlier_mask, errors


def bilinear_interpolation(r, c, depth_img):
    '''Interpolate (r,c) from values associated with four points.
    The four points are a list of four triplets:  (r, c, value).
    The four points can be in any order.  They should form a rectangle.
    '''
    # See formula at:  http://en.wikipedia.org/wiki/Bilinear_interpolation

    points = [(math.floor(r), math.floor(c), 0),
        (math.floor(r) + 1, math.floor(c), 0),
        (math.floor(r) + 1, math.floor(c) + 1, 0),
        (math.floor(r), math.floor(c) + 1, 0)]

    try:
        for i in range(4):
            points[i] = (points[i][0], points[i][1], depth_img[points[i][0], points[i][1]])
    except:
        return 0

    points = sorted(points)               # order points by r, then by y
    (r1, c1, q11), (_r1, c2, q12), (r2, _c1, q21), (_r2, _c2, q22) = points

    if r1 != _r1 or r2 != _r2 or c1 != _c1 or c2 != _c2:
        raise ValueError('points do not form a rectangle')
    if not r1 <= r <= r2 or not c1 <= c <= c2:
        raise ValueError('(r, c) not within the rectangle')

    return (q11 * (r2 - r) * (c2 - c) +
            q21 * (r - r1) * (c2 - c) +
            q12 * (r2 - r) * (c - c1) +
            q22 * (r - r1) * (c - c1)
           ) / ((r2 - r1) * (c2 - c1) + 0.0)

def depth_consistency_check(m_lines1, m_lines2, K1, K2, img1, img2, depth_img1, depth_img2, threshold, line_pairs = None):
    # The number of line correspondences
    num_matches = m_lines1.shape[0]
    valid_line_pairs = []

    C1x = K1[0, 2]
    C1y = K1[1, 2]
    f1x = K1[0, 0]
    f1y = K1[1, 1]
    C2x = K2[0, 2]
    C2y = K2[1, 2]
    f2x = K2[0, 0]
    f2y = K2[1, 1]

    # Calculating the resize ratios for the feature points to the depth maps
    ratios1 = tuple(float(ele1) / float(ele2) for ele1, ele2 in zip(depth_img1.shape, img1.shape))
    ratios2 = tuple(float(ele1) / float(ele2) for ele1, ele2 in zip(depth_img2.shape, img2.shape))

    if line_pairs == None:
        # Iterate through all line correspondences and check if they are on the same plane
        for line_idx1 in range(0, num_matches):
            # Depth at the endpoints in the source image
            rs11 = m_lines1[line_idx1][0][0] * ratios1[0]
            cs11 = m_lines1[line_idx1][0][1] * ratios1[1]
            re11 = m_lines1[line_idx1][1][0] * ratios1[0]
            ce11 = m_lines1[line_idx1][1][1] * ratios1[1]
            depth11s = bilinear_interpolation(rs11, cs11, depth_img1) / 1000.0
            depth11e = bilinear_interpolation(re11, ce11, depth_img1) / 1000.0

            nxs11 = (m_lines1[line_idx1][0][1] - C1x) / f1x
            nys11 = (m_lines1[line_idx1][0][0] - C1y) / f1y
            nxe11 = (m_lines1[line_idx1][1][1] - C1x) / f1x
            nye11 = (m_lines1[line_idx1][1][0] - C1y) / f1y
            tangent11 = np.array([nxe11, nye11, depth11e]) - np.array([nxs11, nys11, depth11s])
            
            if depth11s < 1e-9 or depth11e < 1e-9:
                continue

            # Depth at the endpoints in the destination image
            rs21 = m_lines2[line_idx1][0][0] * ratios2[0]
            cs21 = m_lines2[line_idx1][0][1] * ratios2[1]
            re21 = m_lines2[line_idx1][1][0] * ratios2[0]
            ce21 = m_lines2[line_idx1][1][1] * ratios2[1]
            depth21s = bilinear_interpolation(rs21, cs21, depth_img2) / 1000.0
            depth21e = bilinear_interpolation(re21, ce21, depth_img2) / 1000.0
            
            nxs21 = (m_lines2[line_idx1][0][1] - C2x) / f2x
            nys21 = (m_lines2[line_idx1][0][0] - C2y) / f2y
            nxe21 = (m_lines2[line_idx1][1][1] - C2x) / f2x
            nye21 = (m_lines2[line_idx1][1][0] - C2y) / f2y
            tangent21 = np.array([nxe21, nye21, depth21e]) - np.array([nxs21, nys21, depth21s])

            if depth21s < 1e-9 or depth21e < 1e-9:
                continue

            for line_idx2 in range(line_idx1 + 1, num_matches):
                # Depth at the endpoints in the source image
                rs12 = m_lines1[line_idx2][0][0] * ratios1[0]
                cs12 = m_lines1[line_idx2][0][1] * ratios1[1]
                re12 = m_lines1[line_idx2][1][0] * ratios1[0]
                ce12 = m_lines1[line_idx2][1][1] * ratios1[1]
                depth12s = bilinear_interpolation(rs12, cs12, depth_img1) / 1000.0
                depth12e = bilinear_interpolation(re12, ce12, depth_img1) / 1000.0

                nxs12 = (m_lines1[line_idx2][0][1] - C1x) / f1x
                nys12 = (m_lines1[line_idx2][0][0] - C1y) / f1y
                nxe12 = (m_lines1[line_idx2][1][1] - C1x) / f1x
                nye12 = (m_lines1[line_idx2][1][0] - C1y) / f1y
                tangent12 = np.array([nxs12, nys12, depth12s]) - np.array([nxs11, nys11, depth11s])

                if depth12s < 1e-9 or depth12e < 1e-9:
                    continue
                
                # Depth at the endpoints in the destination image
                rs22 = m_lines2[line_idx2][0][0] * ratios2[0]
                cs22 = m_lines2[line_idx2][0][1] * ratios2[1]
                re22 = m_lines2[line_idx2][1][0] * ratios2[0]
                ce22 = m_lines2[line_idx2][1][1] * ratios2[1]
                depth22s = bilinear_interpolation(rs22, cs22, depth_img2) / 1000.0
                depth22e = bilinear_interpolation(re22, ce22, depth_img2) / 1000.0

                nxs22 = (m_lines2[line_idx2][0][1] - C2x) / f2x
                nys22 = (m_lines2[line_idx2][0][0] - C2y) / f2y
                nxe22 = (m_lines2[line_idx2][1][1] - C2x) / f2x
                nye22 = (m_lines2[line_idx2][1][0] - C2y) / f2y
                tangent22 = np.array([nxs22, nys22, depth22s]) - np.array([nxs21, nys21, depth21s])

                if depth22s < 1e-9 or depth22e < 1e-9:
                    continue
                
                # Estimate the plane normals
                normal1 = np.cross(tangent11, tangent12)
                if np.linalg.norm(normal1) < 1e-9:
                    continue
                normal1 /= np.linalg.norm(normal1)
                normal2 = np.cross(tangent21, tangent22)
                if np.linalg.norm(normal2) < 1e-9:
                    continue
                normal2 /= np.linalg.norm(normal2)

                # Calculate the plane offset w.r.t. the origin
                d1 = -normal1[0] * nxs11 - normal1[1] * nys11 - normal1[2] * depth11s
                d2 = -normal2[0] * nxs21 - normal2[1] * nys21 - normal2[2] * depth21s

                # Check the point-to-plane distance of the 4th points
                distance1 = abs(normal1[0] * nxe12 + normal1[1] * nye12 + normal1[2] * depth12e + d1)
                distance2 = abs(normal2[0] * nxe22 + normal2[1] * nye22 + normal2[2] * depth22e + d2)

                #print(distance1, distance2, threshold)
                if max(distance1, distance2) < threshold:
                    valid_line_pairs.append((line_idx1, line_idx2, max(distance1, distance2)))
    else:
        for line_idx1, line_idx2, distance in line_pairs:
            # Depth at the endpoints in the source image
            rs11 = m_lines1[line_idx1][0][0] * ratios1[0]
            cs11 = m_lines1[line_idx1][0][1] * ratios1[1]
            re11 = m_lines1[line_idx1][1][0] * ratios1[0]
            ce11 = m_lines1[line_idx1][1][1] * ratios1[1]
            depth11s = bilinear_interpolation(rs11, cs11, depth_img1) / 1000.0
            depth11e = bilinear_interpolation(re11, ce11, depth_img1) / 1000.0

            nxs11 = (m_lines1[line_idx1][0][1] - C1x) / f1x
            nys11 = (m_lines1[line_idx1][0][0] - C1y) / f1y
            nxe11 = (m_lines1[line_idx1][1][1] - C1x) / f1x
            nye11 = (m_lines1[line_idx1][1][0] - C1y) / f1y
            tangent11 = np.array([nxe11, nye11, depth11e]) - np.array([nxs11, nys11, depth11s])
            
            if depth11s < 1e-9 or depth11e < 1e-9:
                continue

            # Depth at the endpoints in the destination image
            rs21 = m_lines2[line_idx1][0][0] * ratios2[0]
            cs21 = m_lines2[line_idx1][0][1] * ratios2[1]
            re21 = m_lines2[line_idx1][1][0] * ratios2[0]
            ce21 = m_lines2[line_idx1][1][1] * ratios2[1]
            depth21s = bilinear_interpolation(rs21, cs21, depth_img2) / 1000.0
            depth21e = bilinear_interpolation(re21, ce21, depth_img2) / 1000.0
            
            nxs21 = (m_lines2[line_idx1][0][1] - C2x) / f2x
            nys21 = (m_lines2[line_idx1][0][0] - C2y) / f2y
            nxe21 = (m_lines2[line_idx1][1][1] - C2x) / f2x
            nye21 = (m_lines2[line_idx1][1][0] - C2y) / f2y
            tangent21 = np.array([nxe21, nye21, depth21e]) - np.array([nxs21, nys21, depth21s])

            if depth21s < 1e-9 or depth21e < 1e-9:
                continue

            # Depth at the endpoints in the source image
            rs12 = m_lines1[line_idx2][0][0] * ratios1[0]
            cs12 = m_lines1[line_idx2][0][1] * ratios1[1]
            re12 = m_lines1[line_idx2][1][0] * ratios1[0]
            ce12 = m_lines1[line_idx2][1][1] * ratios1[1]
            depth12s = bilinear_interpolation(rs12, cs12, depth_img1) / 1000.0
            depth12e = bilinear_interpolation(re12, ce12, depth_img1) / 1000.0

            nxs12 = (m_lines1[line_idx2][0][1] - C1x) / f1x
            nys12 = (m_lines1[line_idx2][0][0] - C1y) / f1y
            nxe12 = (m_lines1[line_idx2][1][1] - C1x) / f1x
            nye12 = (m_lines1[line_idx2][1][0] - C1y) / f1y
            tangent12 = np.array([nxs12, nys12, depth12s]) - np.array([nxs11, nys11, depth11s])

            if depth12s < 1e-9 or depth12e < 1e-9:
                continue
            
            # Depth at the endpoints in the destination image
            rs22 = m_lines2[line_idx2][0][0] * ratios2[0]
            cs22 = m_lines2[line_idx2][0][1] * ratios2[1]
            re22 = m_lines2[line_idx2][1][0] * ratios2[0]
            ce22 = m_lines2[line_idx2][1][1] * ratios2[1]
            depth22s = bilinear_interpolation(rs22, cs22, depth_img2) / 1000.0
            depth22e = bilinear_interpolation(re22, ce22, depth_img2) / 1000.0

            nxs22 = (m_lines2[line_idx2][0][1] - C2x) / f2x
            nys22 = (m_lines2[line_idx2][0][0] - C2y) / f2y
            nxe22 = (m_lines2[line_idx2][1][1] - C2x) / f2x
            nye22 = (m_lines2[line_idx2][1][0] - C2y) / f2y
            tangent22 = np.array([nxs22, nys22, depth22s]) - np.array([nxs21, nys21, depth21s])

            if depth22s < 1e-9 or depth22e < 1e-9:
                continue
            
            # Estimate the plane normals
            normal1 = np.cross(tangent11, tangent12)
            if np.linalg.norm(normal1) < 1e-9:
                continue
            normal1 /= np.linalg.norm(normal1)
            normal2 = np.cross(tangent21, tangent22)
            if np.linalg.norm(normal2) < 1e-9:
                continue
            normal2 /= np.linalg.norm(normal2)

            # Calculate the plane offset w.r.t. the origin
            d1 = -normal1[0] * nxs11 - normal1[1] * nys11 - normal1[2] * depth11s
            d2 = -normal2[0] * nxs21 - normal2[1] * nys21 - normal2[2] * depth21s

            # Check the point-to-plane distance of the 4th points
            distance1 = abs(normal1[0] * nxe12 + normal1[1] * nye12 + normal1[2] * depth12e + d1)
            distance2 = abs(normal2[0] * nxe22 + normal2[1] * nye22 + normal2[2] * depth22e + d2)

            #print(distance1, distance2, threshold)
            if max(distance1, distance2) < threshold:
                valid_line_pairs.append((line_idx1, line_idx2, max(distance1, distance2)))
        
    return valid_line_pairs


def angular_check(m_lines1, m_lines2, K1, K2):
    # The number of line correspondences
    num_matches = m_lines1.shape[0]
    valid_line_pairs = []

    C1x = K1[0, 2]
    C1y = K1[1, 2]
    f1x = K1[0, 0]
    f1y = K1[1, 1]
    C2x = K2[0, 2]
    C2y = K2[1, 2]
    f2x = K2[0, 0]
    f2y = K2[1, 1]

    # Iterate through all line correspondences and check if they are on the same plane
    for line_idx1 in range(0, num_matches):
        # Direction in the source image
        nxs11 = (m_lines1[line_idx1][0][1] - C1x) / f1x
        nys11 = (m_lines1[line_idx1][0][0] - C1y) / f1y
        nxe11 = (m_lines1[line_idx1][1][1] - C1x) / f1x
        nye11 = (m_lines1[line_idx1][1][0] - C1y) / f1y
        tangent11 = np.array([nxe11, nye11]) - np.array([nxs11, nys11])
       
        # Direction in the destination image
        nxs21 = (m_lines2[line_idx1][0][1] - C2x) / f2x
        nys21 = (m_lines2[line_idx1][0][0] - C2y) / f2y
        nxe21 = (m_lines2[line_idx1][1][1] - C2x) / f2x
        nye21 = (m_lines2[line_idx1][1][0] - C2y) / f2y
        tangent21 = np.array([nxe21, nye21]) - np.array([nxs21, nys21])

        for line_idx2 in range(line_idx1 + 1, num_matches):
            # Direction in the source image
            nxs12 = (m_lines1[line_idx2][0][1] - C1x) / f1x
            nys12 = (m_lines1[line_idx2][0][0] - C1y) / f1y
            nxe12 = (m_lines1[line_idx2][1][1] - C1x) / f1x
            nye12 = (m_lines1[line_idx2][1][0] - C1y) / f1y
            tangent12 = np.array([nxe12, nye12]) - np.array([nxs12, nys12])

            # Direction in the destination image
            nxs22 = (m_lines2[line_idx2][0][1] - C2x) / f2x
            nys22 = (m_lines2[line_idx2][0][0] - C2y) / f2y
            nxe22 = (m_lines2[line_idx2][1][1] - C2x) / f2x
            nye22 = (m_lines2[line_idx2][1][0] - C2y) / f2y
            tangent22 = np.array([nxe22, nye22]) - np.array([nxs22, nys22])

            # Estimate the angles between the lines in the source image
            cos_source = np.dot(tangent11, tangent12) / np.linalg.norm(tangent11) / np.linalg.norm(tangent12)
            angle_source = math.acos(max(min(cos_source, 1.0), -1.0))
            
            # Estimate the angles between the lines in the destination image
            cos_destination = np.dot(tangent21, tangent22) / np.linalg.norm(tangent21) / np.linalg.norm(tangent22)
            angle_destination = math.acos(max(min(cos_destination, 1.0), -1.0))
                        
            valid_line_pairs.append((line_idx1, line_idx2, angle_source, angle_destination))
    return valid_line_pairs