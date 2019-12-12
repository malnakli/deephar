import os

import numpy as np
import scipy.io as sio
import torch
from PIL import Image
from torch.utils.data import Dataset

from .transform import T, normalize_channels, transform_2d_points
from .utils import pa16j2d

TEST_MODE = 0
TRAIN_MODE = 1
VALID_MODE = 2


class MpiiSinglePerson(Dataset):
    """
    Implementation of the MPII dataset for single person.
    """

    def __init__(
        self,
        dataset_path,
        dataconf,
        y_dictkeys,
        poselayout=pa16j2d,
        mode=TRAIN_MODE,
        remove_outer_joints=True,
        num_predictions=1,
        transform=None,
    ):

        self.dataset_path = dataset_path
        self.dataconf = dataconf
        self.poselayout = poselayout
        self.mode = mode
        self.y_dictkeys = y_dictkeys
        self.num_predictions = len(self.y_dictkeys) * [num_predictions]
        self.remove_outer_joints = remove_outer_joints
        self.transform = transform
        self.load_annotations(os.path.join(dataset_path, "annotations.mat"))

    def load_annotations(self, filename):
        try:
            rectidxs, images, annorect = self._load_mpii_mat_annotation(filename)

            self.samples = {}
            self.samples[TEST_MODE] = []  # No samples for test
            self.samples[TRAIN_MODE] = self._serialize_annorect(
                rectidxs[TRAIN_MODE], annorect[TRAIN_MODE]
            )
            self.samples[VALID_MODE] = self._serialize_annorect(
                rectidxs[VALID_MODE], annorect[VALID_MODE]
            )
            self.images = images

        except Exception:
            print("Error loading the MPII dataset!")
            raise

    def _load_mpii_mat_annotation(self, filename):
        mat = sio.loadmat(filename)
        annot_tr = mat["annot_tr"]
        annot_val = mat["annot_val"]

        # Respect the order of TEST (0), TRAIN (1), and VALID (2)
        rectidxs = [None, annot_tr[0, :], annot_val[0, :]]
        images = [None, annot_tr[1, :], annot_val[1, :]]
        annorect = [None, annot_tr[2, :], annot_val[2, :]]

        return rectidxs, images, annorect

    def _serialize_annorect(self, rectidxs, annorect):
        assert len(rectidxs) == len(annorect)

        sample_list = []

        for i in range(len(rectidxs)):
            rec = rectidxs[i]
            for j in range(rec.size):
                idx = rec[j, 0] - 1  # Convert idx from Matlab
                ann = annorect[i][idx, 0]
                annot = {}
                annot["head"] = ann["head"][0, 0][0]
                annot["objpos"] = ann["objpos"][0, 0][0]
                annot["scale"] = ann["scale"][0, 0][0, 0]
                annot["pose"] = ann["pose"][0, 0]
                annot["imgidx"] = i
                sample_list.append(annot)

        return sample_list

    def _load_image(self, key, mode):
        try:
            annot = self.samples[mode][key]
            image = self.images[mode][annot["imgidx"]][0]
            imgt = T(Image.open(os.path.join(self.dataset_path, "images", image)))
        except Exception:
            print("Error loading sample key/mode: %d/%d" % (key, mode))
            raise

        return imgt

    def _objposwin_to_bbox(self, objpos, winsize):
        x1 = objpos[0] - winsize[0] / 2
        y1 = objpos[1] - winsize[1] / 2
        x2 = objpos[0] + winsize[0] / 2
        y2 = objpos[1] + winsize[1] / 2

        return np.array([x1, y1, x2, y2])

    @classmethod
    def _get_visible_joints(cls, x, margin=0.0):
        def _func_and(x):
            if x.all():
                return 1
            return 0

        visible = np.apply_along_axis(_func_and, axis=1, arr=(x > margin))
        visible *= np.apply_along_axis(_func_and, axis=1, arr=(x < 1 - margin))

        return visible

    @classmethod
    def _calc_head_size(cls, head_annot):
        head = np.array(
            [
                float(head_annot[0]),
                float(head_annot[1]),
                float(head_annot[2]),
                float(head_annot[3]),
            ]
        )
        return 0.6 * np.linalg.norm(head[0:2] - head[2:4])

    def __len__(self):
        return len(self.samples[self.mode])

    def __getitem__(self, idx):
        output = {}

        if self.mode == TRAIN_MODE:
            dconf = self.dataconf.random_data_generator()
        else:
            dconf = self.dataconf.get_fixed_config()

        imgt = self._load_image(idx, self.mode)
        annot = self.samples[self.mode][idx]

        scale = 1.25 * annot["scale"]
        objpos = np.array([annot["objpos"][0], annot["objpos"][1] + 12 * scale])
        objpos += scale * np.array([dconf["transx"], dconf["transy"]])
        winsize = 200 * dconf["scale"] * scale
        winsize = (winsize, winsize)
        output["bbox"] = self._objposwin_to_bbox(objpos, winsize)

        imgt.rotate_crop(dconf["angle"], objpos, winsize)
        imgt.resize(self.dataconf.crop_resolution)

        if dconf["hflip"] == 1:
            imgt.horizontal_flip()

        imgt.normalize_affinemap()
        X = normalize_channels(imgt.asarray(), channel_power=dconf["chpower"])

        _p = np.empty((self.poselayout.num_joints, self.poselayout.dim))
        _p[:] = np.nan

        # head = annot["head"]
        _p[self.poselayout.map_to_mpii, 0:2] = transform_2d_points(
            imgt.afmat, annot["pose"].T, transpose=True
        )
        if imgt.hflip:
            _p = _p[self.poselayout.map_hflip, :]

        # Set invalid joints and NaN values as an invalid value
        _p[np.isnan(_p)] = -1e9
        _v = np.expand_dims(self._get_visible_joints(_p[:, 0:2]), axis=-1)
        if self.remove_outer_joints:
            _p[(_v == 0)[:, 0], :] = -1e9

        output["pose"] = np.concatenate((_p, _v), axis=-1)
        output["headsize"] = self._calc_head_size(annot["head"])
        output["afmat"] = imgt.afmat.copy()

        y_batch = []
        for i, dkey in enumerate(self.y_dictkeys):
            for _ in range(self.num_predictions[i]):
                y_batch.append(output[dkey])

        if self.transform is not None:
            X = self.transform(X)

        if self.mode == TRAIN_MODE:
            y_batch = torch.from_numpy(y_batch[0])

        return X, y_batch

