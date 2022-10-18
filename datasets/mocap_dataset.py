from torch.utils.data import Dataset

# from common.imutils import process_image
# from common.utils import estimate_focal_length
import constants
import config
import numpy as np
from os.path import join
import cv2
import torch

class MocapDataset(Dataset):
    def __init__(self, options, dataset, ignore_3d=False, use_augmentation=False, is_train=True):
        super(MocapDataset, self).__init__()
        self.dataset = dataset
        self.is_train = is_train
        self.options = options
        self.img_dir = config.DATASET_FOLDERS[dataset]
        # self.normalize_img = Normalize(mean=constants.IMG_NORM_MEAN, std=constants.IMG_NORM_STD)
        self.data = np.load(config.DATASET_FILES[is_train][dataset])
        self.imgname = self.data['imgname']
        self.keys = self.data.files
        
        # Get paths to gt masks, if available
        if 'maskname' in self.keys:
            self.maskname = self.data['maskname']
        else:
            pass
        if 'partname' in self.keys:
            self.partname = self.data['partname']
        else:
            pass

        # Bounding boxes are assumed to be in the center and scale format
        self.scale = self.data['scale']
        self.center = self.data['center']
        
        # If False, do not do augmentation
        self.use_augmentation = use_augmentation
        
        # Get gt SMPL parameters, if available
        try:
            self.pose = self.data['pose'].astype(np.float)
            self.betas = self.data['shape'].astype(np.float)
            if 'has_smpl' in self.data:
                self.has_smpl = self.data['has_smpl']
            else:
                self.has_smpl = np.ones(len(self.imgname))
        except:
            self.has_smpl = np.zeros(len(self.imgname))
        if ignore_3d:
            self.has_smpl = np.zeros(len(self.imgname))
        
        # Get gt 3D pose, if available
        if 'S' in self.keys:
            self.pose_3d = self.data['S']
            self.has_pose_3d = 1
        else:
            self.has_pose_3d = 0
        if ignore_3d:
            self.has_pose_3d = 0
        
        # Get 2D keypoints
        if 'part' in self.keys:
            keypoints_gt = self.data['part']
        else:
            keypoints_gt = np.zeros((len(self.imgname), 24, 3))
        if 'openpose' in self.keys:
            keypoints_openpose = self.data['openpose']
        else:
            keypoints_openpose = np.zeros((len(self.imgname), 25, 3))
        self.keypoints = np.concatenate([keypoints_openpose, keypoints_gt], axis=1)

        # Get gender data, if available
        if 'gender' in self.keys:
            gender = self.data['gender']
            self.gender = np.array([0 if str(g) == 'm' else 1 for g in gender]).astype(np.int32)
        else:
            self.gender = -1*np.ones(len(self.imgname)).astype(np.int32)
        
        self.length = self.scale.shape[0]

    def augm_params(self):
        """Get augmentation parameters."""
        flip = 0            # flipping
        pn = np.ones(3)  # per channel pixel-noise
        rot = 0            # rotation
        sc = 1            # scaling
        if self.is_train:
            # We flip with probability 1/2
            if np.random.uniform() <= 0.5:
                flip = 1
            
            # Each channel is multiplied with a number 
            # in the area [1-opt.noiseFactor,1+opt.noiseFactor]
            pn = np.random.uniform(1-self.options.noise_factor, 1+self.options.noise_factor, 3)
            
            # The rotation is a number in the area [-2*rotFactor, 2*rotFactor]
            rot = min(2*self.options.rot_factor,
                    max(-2*self.options.rot_factor, np.random.randn()*self.options.rot_factor))
            
            # The scale is multiplied with a number
            # in the area [1-scaleFactor,1+scaleFactor]
            sc = min(1+self.options.scale_factor,
                    max(1-self.options.scale_factor, np.random.randn()*self.options.scale_factor+1))
            # but it is zero with probability 3/5
            if np.random.uniform() <= 0.6:
                rot = 0
        
        return flip, pn, rot, sc


    def get_transform(self, center, scale, res, rot=0):
        """Generate transformation matrix."""
        # res: (height, width), (rows, cols)
        crop_aspect_ratio = res[0] / float(res[1])
        h = 200 * scale
        w = h / crop_aspect_ratio
        t = np.zeros((3, 3))
        t[0, 0] = float(res[1]) / w
        t[1, 1] = float(res[0]) / h
        t[0, 2] = res[1] * (-float(center[0]) / w + .5)
        t[1, 2] = res[0] * (-float(center[1]) / h + .5)
        t[2, 2] = 1
        if not rot == 0:
            rot = -rot  # To match direction of rotation from cropping
            rot_mat = np.zeros((3, 3))
            rot_rad = rot * np.pi / 180
            sn, cs = np.sin(rot_rad), np.cos(rot_rad)
            rot_mat[0, :2] = [cs, -sn]
            rot_mat[1, :2] = [sn, cs]
            rot_mat[2, 2] = 1
            # Need to rotate around center
            t_mat = np.eye(3)
            t_mat[0, 2] = -res[1] / 2
            t_mat[1, 2] = -res[0] / 2
            t_inv = t_mat.copy()
            t_inv[:2, 2] *= -1
            t = np.dot(t_inv, np.dot(rot_mat, np.dot(t_mat, t)))
        return t
        
    def transform(self, pt, center, scale, res, invert=0, rot=0):
        """Transform pixel location to different reference."""
        t = self.get_transform(center, scale, res, rot=rot)
        if invert:
            t = np.linalg.inv(t)
        new_pt = np.array([pt[0] - 1, pt[1] - 1, 1.]).T
        new_pt = np.dot(t, new_pt)
        return np.array([round(new_pt[0]), round(new_pt[1])], dtype=int) + 1
        
    def crop_cliff(self, img, center, scale, res):
        """
        Crop image according to the supplied bounding box.
        res: [rows, cols]
        """
        # Upper left point
        ul = np.array(self.transform([1, 1], center, scale, res, invert=1)) - 1
        # Bottom right point
        br = np.array(self.transform([res[1] + 1, res[0] + 1], center, scale, res, invert=1)) - 1

        # Padding so that when rotated proper amount of context is included
        pad = int(np.linalg.norm(br - ul) / 2 - float(br[1] - ul[1]) / 2)

        new_shape = [br[1] - ul[1], br[0] - ul[0]]
        if len(img.shape) > 2:
            new_shape += [img.shape[2]]
        new_img = np.zeros(new_shape, dtype=np.float32)

        # Range to fill new array
        new_x = max(0, -ul[0]), min(br[0], len(img[0])) - ul[0]
        new_y = max(0, -ul[1]), min(br[1], len(img)) - ul[1]
        # Range to sample from original image
        old_x = max(0, ul[0]), min(len(img[0]), br[0])
        old_y = max(0, ul[1]), min(len(img), br[1])
        try:
            new_img[new_y[0]:new_y[1], new_x[0]:new_x[1]] = img[old_y[0]:old_y[1], old_x[0]:old_x[1]]
        except Exception as e:
            print(e)

        new_img = cv2.resize(new_img, (res[1], res[0]))  # (cols, rows)

        return new_img, ul, br

    def estimate_focal_length(self, img_h, img_w):
        return (img_w * img_w + img_h * img_h) ** 0.5  # fov: 55 degree

    def process_image(self, orig_img_rgb, center, scale, rot, flip, pn,
                    crop_height=constants.IMG_H,
                    crop_width=constants.IMG_W):
        """
        Read image, do preprocessing and possibly crop it according to the bounding box.
        If there are bounding box annotations, use them to crop the image.
        If no bounding box is specified but openpose detections are available, use them to get the bounding box.
        """

        img, ul, br = self.crop_cliff(orig_img_rgb, center, scale, (crop_height, crop_width))
        # crop_img = img.copy()
        # print(img.shape)
        # # flip the image 
        # if flip:
        #     img = flip_img(img)

        img = img / 255.
        mean = np.array(constants.IMG_NORM_MEAN, dtype=np.float32)
        std = np.array(constants.IMG_NORM_STD, dtype=np.float32)
        norm_img = (img - mean) / std
        norm_img = np.transpose(norm_img, (2, 0, 1))
        # print(img.shape, norm_img.shape)
        return norm_img

    def __getitem__(self, index):
        """
        bbox: [batch_id, min_x, min_y, max_x, max_y, det_conf, nms_conf, category_id]
        :param idx:
        :return:
        """
        item = {}
        # img_idx = int(self.detection_list[idx][0].item())
        # img_bgr = self.img_bgr_list[img_idx]
        # img_rgb = img_bgr[:, :, ::-1]
        # img_h, img_w, _ = img_rgb.shape
        # focal_length = estimate_focal_length(img_h, img_w)

        # bbox = self.detection_list[idx][1:5]
        # norm_img, center, scale, crop_ul, crop_br, _ = process_image(img_rgb, bbox)


        # get scale, center, pose, beta following spin 
        item = {}
        scale = self.scale[index].copy()
        center = self.center[index].copy()

        # Get augmentation parameters
        flip, pn, rot, sc = self.augm_params()
        
        # Load image
        imgname = join(self.img_dir, self.imgname[index])
        try:
            img = cv2.imread(imgname)[:,:,::-1].copy().astype(np.float32)
        except TypeError:
            print(imgname)
        orig_shape = np.array(img.shape)[:2]

        # Get SMPL parameters, if available
        if self.has_smpl[index]:
            pose = self.pose[index].copy()
            betas = self.betas[index].copy()
        else:
            pose = np.zeros(72)
            betas = np.zeros(10)
        item['pose'] = torch.from_numpy(pose).float()# torch.from_numpy(self.pose_processing(pose, rot, flip)).float()
        item['betas'] = torch.from_numpy(betas).float()
        item['imgname'] = imgname

        # Process image
        # img = self.rgb_processing(img, center, sc*scale, rot, flip, pn)
        img = self.process_image(img, center, sc*scale, rot, flip, pn)
        img = torch.from_numpy(img).float()
        # print(rot, flip)
        # Store image before normalization to use it in visualization
        # item['img'] = self.normalize_img(img)
        item['img'] = img

        # to get img_h, img_w, focal_lenght, bbox_info
        img_h, img_w = orig_shape
        focal_length = float(self.estimate_focal_length(img_h, img_w))
        item['img_h'] = img_h
        item['img_w'] = img_w
        item['focal_length'] = focal_length
        cx, cy = center.astype(np.float32)
        s = float(sc * scale)

        bbox_info = np.stack([cx - img_w / 2., cy - img_h / 2., s*200])
        bbox_info[:2] = bbox_info[:2] / focal_length * 2.8  # [-1, 1]
        bbox_info[2] = (bbox_info[2] - 0.24 * focal_length) / (
            0.06 * focal_length)  # [-1, 1]

        item['bbox_info'] = np.float32(bbox_info)
        item['center'] = center
        item["scale"] = scale
        item['gender'] = self.gender[index]

        # item["img"] = norm_img
        return item

    def __len__(self):
        return len(self.imgname)