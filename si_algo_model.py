"""Training phase of proposed SI algorithm
"""
import cv2
import numpy as np
import itertools
import pickle
import scipy


class Operator(object):
    pass


class ImageOperator(Operator):
    def __call__(self, img: np.ndarray) -> np.ndarray:
        return img


class Downscale(ImageOperator):

    def __init__(self, fx=0.5, fy=0.5):
        assert(0 < fx < 1 and 0 < fy < 1)
        self.scale_factor: tuple[(float, float)] = (fx, fy)

    def __call__(self, img: np.ndarray):
        dsize = (int(img.shape[1] * self.scale_factor[0]), int(img.shape[0] * self.scale_factor[1]))
        return cv2.resize(img, dsize, interpolation=cv2.INTER_LINEAR)


def LR_HR_patch_pairs(LR_img: np.ndarray, HR_img: np.ndarray, lsize = 3, hsize = 2):
    """Generator that generates LR HR patch pairs, notice HR is 2x larger than LR.
    lsize must be odd.

    Return:
        (lpatch, hpatch)
    """
    assert(lsize % 2 == 1)
    # For simplicit, variables with a ending underline belongs to HR image, vice versa
    h, w, c = LR_img.shape

    w_ = w * 2
    h_ = h * 2

    for i, j in itertools.product(range(h_//hsize), range(w_//hsize)):

        y_ = i * hsize
        x_ = j * hsize
        y = y_ // 2 - lsize // 2
        x = x_ // 2 - lsize // 2

        for c in range(LR_img.shape[2]):
            if 0 <= x < w - lsize and 0 <= y < h - lsize:  # return a view of the LR image if not exceeding the boundry
                lpatch = LR_img[y:y+lsize, x:x+lsize, c]
            else:   # padding the image if the patch is over the boundray
                lpatch = np.empty((lsize, lsize))
                for k, l in itertools.product(range(lsize), range(lsize)):
                    yp = y + k
                    xp = x + l

                    if yp < 0:
                        yp = 0
                    elif yp >= h:
                        yp = h-1

                    if xp < 0:
                        xp = 0
                    elif xp >= w:
                        xp = w-1

                    lpatch[k, l] = LR_img[yp, xp, c]

            yield lpatch, HR_img[y_:y_+hsize, x_:x_+hsize, c]


def classify_patch_3x3(patch, threshhold=15):
    assert(patch.shape[0] == 3 and patch.shape[1] == 3)
    k_h = np.array([[ 1, -1],
                    [ 1, -1]], dtype=float)
    k_v = np.array([[ 1,  1],
                    [-1, -1]], dtype=float)
    grad_h = scipy.signal.convolve2d(patch, k_h, mode='valid')
    grad_v = scipy.signal.convolve2d(patch, k_v, mode='valid')

    magnitude = np.sqrt(np.square(grad_h), np.square(grad_v))
    direction = np.arctan(np.divide(grad_h, grad_v + 0.0000001)) / np.pi * 180  # unit: degree, range: [-180, 180]

    # quantitize direction
    direction[direction < 0] += 180
    direction = direction / 45 + 1
    direction = direction.astype(int)

    # apply mask according to threshhold
    direction[magnitude < threshhold] = 0
    direction = direction.reshape(-1)

    # calculate class index
    c = 0
    for i in range(direction.shape[0]):
        c += direction[i] * 5 ** i

    return c


class SuperInterpolation(object):

    def __init__(self, lsize=3, hsize=2):
        self._linear_mappings = []  # classes are represented as indexes
        self.lsize = lsize
        self.hsize = hsize
        for i in range(5 ** ((lsize - 1) ** 2)):
            self._linear_mappings.append(np.eye(hsize**2, lsize**2))


    def fit(self, dataloader, lambda_=1):
        classified_data: list[list[tuple[np.ndarray, np.ndarray]]] = [[] for _ in range(len(self._linear_mappings))]
        self._train_data_cache: list[tuple[np.ndarray, np.ndarray]] = []  # lists of samples are combined into a big matrix 

        for lpatch, hpatch in dataloader:
            idx = classify_patch_3x3(lpatch)
            assert(idx < len(classified_data))
            classified_data[idx].append((lpatch, hpatch))
            
        for i, d in enumerate(classified_data):
            if len(d) > 8000:
                choices = np.random.choice(len(d), 8000)
                d_ = []
                for c in choices:
                    d_.append(d[c])
                classified_data[i] = d_

        max_num_samples = 0
        min_num_samples = 0
        for i in range(len(self._linear_mappings)):
            num_samples = len(classified_data[i])

            if num_samples > max_num_samples:
                max_num_samples = num_samples
            if min_num_samples == 0:
                min_num_samples = num_samples
            elif num_samples < min_num_samples:
                min_num_samples = num_samples

            X = np.empty((self.hsize**2, num_samples))
            Y = np.empty((self.lsize**2, num_samples))
            for k in range(num_samples):
                lpatch, hpatch = classified_data[i][k]
                X[:, k] = hpatch.reshape(-1)
                Y[:, k] = lpatch.reshape(-1)

            self._train_data_cache.append((Y, X))

            self._linear_mappings[i] = X @ Y.T @ np.linalg.inv(Y @ Y.T + lambda_*np.eye(self.lsize**2, self.lsize**2))

        print(f"Training information:\n{max_num_samples=} {min_num_samples=}")


    def predict(self, LR_img: np.ndarray):
        HR_img = np.empty((LR_img.shape[0]*2, LR_img.shape[1]*2, LR_img.shape[2]))

        for lpatch, hpatch in LR_HR_patch_pairs(LR_img, HR_img):
            class_ = classify_patch_3x3(lpatch)
            hpatch[:] = (self._linear_mappings[class_] @ lpatch.reshape(-1)).reshape(self.hsize, self.hsize)
        return np.clip(HR_img, 0, 255)

    def load(self, filename):
        with open(filename, 'rb') as f:
            si = pickle.load(f)
            self.lsize = si.lsize
            self.hsize = si.hsize
            self._linear_mappings = si._linear_mappings

    def save(self, filename):
        with open(filename, 'wb') as f:
            pickle.dump(self, f)

    def __eq__(self, other):
        flag = True
        for l1, l2 in zip(self._linear_mappings, other._linear_mappings):
            flag &= (l1 == l2).all()
        return self.lsize == other.lsize and self.hsize == other.hsize and flag

