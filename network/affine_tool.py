import numpy as np
import torch
import cv2


class AffineAugmentation(object):
    def __init__(self,
                 patch_ratio=1.2,
                 scaling_sample_num=5, #5,
                 scaling_low=1.0, #0.8
                 scaling_up=1.0, #1.2
                 translation_overflow=0, #0.05,
                 rotation_sample_num=25, #25,
                 rotation_max_angle=np.pi/6, #np.pi/2,
                 do_scaling=False,
                 do_rotation=True,
                 do_translation=False,
                 allow_artifacts=True,
                 rotation=None
                 ):
        self.patch_ratio = patch_ratio
        self.scaling_sample_num = scaling_sample_num
        self.scaling_low = scaling_low
        self.scaling_up = scaling_up
        self.translation_overflow = translation_overflow
        self.rotation_sample_num = rotation_sample_num
        if rotation is None:
            self.rotation_min_angle = -rotation_max_angle
            self.rotation_max_angle = rotation_max_angle
        else:
            self.rotation_min_angle = rotation[0]
            self.rotation_max_angle = rotation[1]
        self.do_scaling = do_scaling
        self.do_rotation = do_rotation
        if self.rotation_max_angle == self.rotation_min_angle == 0:
            self.do_rotation = False
        self.do_translation = do_translation
        self.allow_artifacts = allow_artifacts

    def __call__(self, h,w):
        homography = self.sample(height=h, width=w)
        return homography


    def sample(self, height, width):

        pts_1 = np.array(((0, 0), (0, 1), (1, 1)), dtype=np.float)  
        margin = (1 - self.patch_ratio) / 2
        pts_2 = margin + np.array(((0, 0), (0, self.patch_ratio),
                                   (self.patch_ratio, self.patch_ratio)),
                                  dtype=np.float)

        if self.do_scaling:
            random_scales = torch.ones((self.scaling_sample_num,), dtype=torch.float).uniform_(
                self.scaling_low, self.scaling_up).numpy()
            scales = np.concatenate((random_scales, np.ones((1,))), axis=0)

            center = np.mean(pts_2, axis=0, keepdims=True)
            scaled = np.expand_dims(pts_2 - center, axis=0) * np.expand_dims(np.expand_dims(scales, 1), 1) + center
            if self.allow_artifacts:
                valid = np.arange(self.scaling_sample_num + 1)
            else:
                valid = np.where(np.all((scaled >= 0.) & (scaled < 1.), axis=(1, 2)))[0]
            random_idx = torch.randint(0, valid.shape[0], size=[]).item()
            idx = valid[random_idx]
            pts_2 = scaled[idx]


        if self.do_translation:
            t_min, t_max = np.min(np.abs(pts_2), axis=0), np.min(np.abs(1 - pts_2), axis=0)

            if self.allow_artifacts:
                t_min += self.translation_overflow
                t_max += self.translation_overflow
            random_t_0 = torch.ones([]).uniform_(-t_min[0], t_max[0]).item()
            random_t_1 = torch.ones([]).uniform_(-t_min[1], t_max[1]).item()
            pts_2 += np.expand_dims(np.stack((random_t_0, random_t_1)), axis=0)

        if self.do_rotation:
            angles = torch.ones((self.rotation_sample_num,), dtype=torch.float).uniform_(
                self.rotation_min_angle, self.rotation_max_angle).numpy()

            angles = np.concatenate((angles, np.zeros((1,))), axis=0)  # in case no rotation is valid
            center = np.mean(pts_2, axis=0, keepdims=True)
            rot_mat = np.reshape(np.stack((np.cos(angles), -np.sin(angles),
                                           np.sin(angles), np.cos(angles)), axis=1), newshape=(-1, 2, 2))

            rotated = np.matmul(
                np.tile(np.expand_dims(pts_2 - center, axis=0), reps=(self.rotation_sample_num + 1, 1, 1)), rot_mat
            ) + center
            if self.allow_artifacts:
                valid = np.arange(self.rotation_sample_num)
            else:
                valid = np.where(np.all((rotated >= 0.) & (rotated < 1.), axis=(1, 2)))[0]
            random_idx = torch.randint(0, valid.shape[0], size=[]).item()
            idx = valid[random_idx]
            pts_2 = rotated[idx]

        M_shear = cv2.getAffineTransform(np.float32(pts_1), np.float32(pts_2))

        return M_shear










