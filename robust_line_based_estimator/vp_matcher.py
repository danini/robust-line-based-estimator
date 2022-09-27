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
