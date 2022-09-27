import numpy as np
import cv2

from pytlbd import lbd_multiscale_pyr


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
