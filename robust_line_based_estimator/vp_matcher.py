import numpy as np
from scipy.optimize import linear_sum_assignment


def vp_matching(vp0, label0, vp1, label1):
    """ Match the vanishing points between two images,
        given a corresponding set of lines. The lines corresponding
        to label0 and label1 should be matching. """
    assert len(label0) == len(label1), "1:1 correspondence between lines is not respected."

    # Build a matrix of common lines
    n_vp0, n_vp1 = len(vp0), len(vp1)
    if n_vp0 == 0 or n_vp1 == 0:
        m_vp0, m_vp1 = [], []
        m_label0, m_label1 = -np.ones_like(label0), -np.ones_like(label1)
        return m_vp0, m_label0, m_vp1, m_label1

    common_lines = np.zeros((n_vp0, n_vp1))
    for l0, l1 in zip(label0, label1):
        if l0 == n_vp0 or l1 == n_vp1:
            continue
        common_lines[l0, l1] += 1

    # Compute the optimal assignment
    row_assignment, col_assignment = linear_sum_assignment(common_lines,
                                                           maximize=True)

    # Get the matched VPs and new labeling
    # -1 means that a line is not shared by the same VP (or has no VP)
    m_vp0 = vp0[row_assignment]
    m_vp1 = vp1[col_assignment]
    m_label0, m_label1 = -np.ones_like(label0), -np.ones_like(label1)
    for i in range(len(row_assignment)):
        m_label0[label0 == row_assignment[i]] = i
        m_label1[label1 == col_assignment[i]] = i

    return m_vp0, m_label0, m_vp1, m_label1


def associate_lines_to_vps(lines, vp, thresh=1.5):
    """ Associate a set of lines to a set of VPs in homogeneous format.
    Args:
        lines: [N, 2, 2] array in ij convention.
        vp: [M, 3] array in homogeneous format.
    Returns:
        An [N] array indicating the associated VP (-1 if not associated).
    """
    # Center of the lines
    centers = ((lines[:, 0] + lines[:, 1]) / 2)
    centers = np.concatenate([centers[:, [1, 0]],
                              np.ones_like(centers[:, :1])], axis=1)

    # Line passing through the VP and the center of the lines
    # l = cross(center, vp)
    # l is [N, M, 3]
    line_vp = np.cross(centers[:, None], vp[None])
    line_vp_norm = np.linalg.norm(line_vp[:, :, :2], axis=2)

    # Orthogonal distance of the lines to l
    endpts = np.concatenate([lines[:, 0][:, [1, 0]],
                             np.ones_like(lines[:, 0, :1])], axis=1)
    orth_dist = np.abs(np.sum(endpts[:, None] * line_vp,
                              axis=2))
    orth_dist[line_vp_norm < 1e-4] = 0
    line_vp_norm[line_vp_norm < 1e-4] = 1
    orth_dist /= line_vp_norm  # [N, M] matrix

    # Find the best assignment
    closest_vp = np.argmin(orth_dist, axis=1)
    closest_vp[np.amin(orth_dist, axis=1) > thresh] = -1

    return closest_vp
