import copy
import os
import numpy as np
import torch
from typing import Any, Dict, List, Union
from yacs.config import CfgNode
import braceexpand
import cv2

from .dataset import Dataset
from .utils import get_example, expand_to_aspect_ratio
from .smplh_prob_filter import poses_check_probable, load_amass_hist_smooth
import itertools

def expand(s):
    return os.path.expanduser(os.path.expandvars(s))
def expand_urls(urls: Union[str, List[str]]):
    if isinstance(urls, str):
        urls = [urls]
    urls = [u for url in urls for u in braceexpand.braceexpand(expand(url))]
    return urls

AIC_TRAIN_CORRUPT_KEYS = {
    '0a047f0124ae48f8eee15a9506ce1449ee1ba669',
    '1a703aa174450c02fbc9cfbf578a5435ef403689',
    '0394e6dc4df78042929b891dbc24f0fd7ffb6b6d',
    '5c032b9626e410441544c7669123ecc4ae077058',
    'ca018a7b4c5f53494006ebeeff9b4c0917a55f07',
    '4a77adb695bef75a5d34c04d589baf646fe2ba35',
    'a0689017b1065c664daef4ae2d14ea03d543217e',
    '39596a45cbd21bed4a5f9c2342505532f8ec5cbb',
    '3d33283b40610d87db660b62982f797d50a7366b',
}
CORRUPT_KEYS = {
    *{f'aic-train/{k}' for k in AIC_TRAIN_CORRUPT_KEYS},
    *{f'aic-train-vitpose/{k}' for k in AIC_TRAIN_CORRUPT_KEYS},
}

# 镜像翻转后，节点会变成当前对应的索引
body_permutation = [0, 1, 5, 6, 7, 2, 3, 4, 8, 12, 13, 14, 9, 10, 11, 16, 15, 18, 17, 22, 23, 24, 19, 20, 21]
extra_permutation = [5, 4, 3, 2, 1, 0, 11, 10, 9, 8, 7, 6, 12, 13, 14, 15, 16, 17, 18]
FLIP_KEYPOINT_PERMUTATION = body_permutation + [25 + i for i in extra_permutation]

# 归一化的均值
DEFAULT_MEAN = 255. * np.array([0.485, 0.456, 0.406])
DEFAULT_STD = 255. * np.array([0.229, 0.224, 0.225])
DEFAULT_IMG_SIZE = 256

class SiameseImageDataset(Dataset):

    def __init__(self,
                 cfg: CfgNode,
                 dataset_file: str,
                 img_dir: str,
                 train: bool = True,
                 prune: Dict[str, Any] = {},
                 **kwargs):
        """
        Dataset class used for loading images and corresponding annotations.
        Args:
            cfg (CfgNode): Model config file.
            dataset_file (str): Path to npz file containing dataset info.
            img_dir (str): Path to image folder.
            train (bool): Whether it is for training or not (enables data augmentation).
        """
        super(SiameseImageDataset, self).__init__()
        self.train = train
        self.cfg = cfg

        self.img_size = cfg.MODEL.IMAGE_SIZE
        self.mean = 255. * np.array(self.cfg.MODEL.IMAGE_MEAN)
        self.std = 255. * np.array(self.cfg.MODEL.IMAGE_STD)

        self.img_dir = img_dir
        self.data = np.load(dataset_file, allow_pickle=True)

        self.imgname = self.data['imgname']
        self.personid = np.zeros(len(self.imgname), dtype=np.int32)
        self.extra_info = self.data.get('extra_info', [{} for _ in range(len(self.imgname))])

        self.flip_keypoint_permutation = copy.copy(FLIP_KEYPOINT_PERMUTATION)

        num_pose = 3 * (self.cfg.SMPL.NUM_BODY_JOINTS + 1)

        # Bounding boxes are assumed to be in the center and scale format
        self.center = self.data['center']
        self.scale = self.data['scale'].reshape(len(self.center), -1) / 200.0
        if self.scale.shape[1] == 1:
            self.scale = np.tile(self.scale, (1, 2))
        assert self.scale.shape == (len(self.center), 2)

        # Get gt SMPLX parameters, if available
        try:
            self.body_pose = self.data['body_pose'].astype(np.float32)
            self.has_body_pose = self.data['has_body_pose'].astype(np.float32)
        except KeyError:
            self.body_pose = np.zeros((len(self.imgname), num_pose), dtype=np.float32)
            self.has_body_pose = np.zeros(len(self.imgname), dtype=np.float32)
        try:
            self.betas = self.data['betas'].astype(np.float32)
            self.has_betas = self.data['has_betas'].astype(np.float32)
        except KeyError:
            self.betas = np.zeros((len(self.imgname), 10), dtype=np.float32)
            self.has_betas = np.zeros(len(self.imgname), dtype=np.float32)

        # Try to get 2d keypoints, if available
        try:
            body_keypoints_2d = self.data['body_keypoints_2d']
        except KeyError:
            body_keypoints_2d = np.zeros((len(self.center), 25, 3))
        # Try to get extra 2d keypoints, if available
        try:
            extra_keypoints_2d = self.data['extra_keypoints_2d']
        except KeyError:
            extra_keypoints_2d = np.zeros((len(self.center), 19, 3))

        self.keypoints_2d = np.concatenate((body_keypoints_2d, extra_keypoints_2d), axis=1).astype(np.float32)

        # Try to get 3d keypoints, if available
        try:
            body_keypoints_3d = self.data['body_keypoints_3d'].astype(np.float32)
        except KeyError:
            body_keypoints_3d = np.zeros((len(self.center), 25, 4), dtype=np.float32)
        # Try to get extra 3d keypoints, if available
        try:
            extra_keypoints_3d = self.data['extra_keypoints_3d'].astype(np.float32)
        except KeyError:
            extra_keypoints_3d = np.zeros((len(self.center), 19, 4), dtype=np.float32)

        body_keypoints_3d[:, [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14], -1] = 0

        self.keypoints_3d = np.concatenate((body_keypoints_3d, extra_keypoints_3d), axis=1).astype(np.float32)

    def __len__(self) -> int:
        return len(self.scale)

    def __getitem__(self, idx: int) -> Dict:
        """
        Returns an example from the dataset.
        """
        try:
            image_file_rel = self.imgname[idx].decode('utf-8')
        except AttributeError:
            image_file_rel = self.imgname[idx]
        image_file = os.path.join(self.img_dir, image_file_rel)
        keypoints_2d = self.keypoints_2d[idx].copy()
        keypoints_3d = self.keypoints_3d[idx].copy()

        center = self.center[idx].copy()
        center_x = center[0]
        center_y = center[1]
        scale = self.scale[idx]
        BBOX_SHAPE = self.cfg.MODEL.get('BBOX_SHAPE', None)
        bbox_size = expand_to_aspect_ratio(scale*200, target_aspect_ratio=BBOX_SHAPE).max()
        bbox_expand_factor = bbox_size / ((scale*200).max())
        body_pose = self.body_pose[idx].copy().astype(np.float32)
        betas = self.betas[idx].copy().astype(np.float32)

        has_body_pose = self.has_body_pose[idx].copy()
        has_betas = self.has_betas[idx].copy()

        smpl_params = {'global_orient': body_pose[:3],
                       'body_pose': body_pose[3:],
                       'betas': betas
                      }

        has_smpl_params = {'global_orient': has_body_pose,
                           'body_pose': has_body_pose,
                           'betas': has_betas
                           }

        smpl_params_is_axis_angle = {'global_orient': True,
                                     'body_pose': True,
                                     'betas': False
                                    }

        augm_config = self.cfg.DATASETS.CONFIG
        # Crop image and (possibly) perform data augmentation
        img_patch, keypoints_2d, keypoints_3d, smpl_params, has_smpl_params, img_size = get_example(image_file,
                                                                                                    center_x, center_y,
                                                                                                    bbox_size, bbox_size,
                                                                                                    keypoints_2d, keypoints_3d,
                                                                                                    smpl_params, has_smpl_params,
                                                                                                    self.flip_keypoint_permutation,
                                                                                                    self.img_size, self.img_size,
                                                                                                    self.mean, self.std, self.train, augm_config)

        item = {}
        # These are the keypoints in the original image coordinates (before cropping)
        orig_keypoints_2d = self.keypoints_2d[idx].copy()

        item['img'] = img_patch
        item['keypoints_2d'] = keypoints_2d.astype(np.float32)
        item['keypoints_3d'] = keypoints_3d.astype(np.float32)
        item['orig_keypoints_2d'] = orig_keypoints_2d
        item['box_center'] = self.center[idx].copy()
        item['box_size'] = bbox_size
        item['bbox_expand_factor'] = bbox_expand_factor
        item['img_size'] = 1.0 * img_size[::-1].copy()
        item['smpl_params'] = smpl_params
        item['has_smpl_params'] = has_smpl_params
        item['smpl_params_is_axis_angle'] = smpl_params_is_axis_angle
        item['imgname'] = image_file
        item['imgname_rel'] = image_file_rel
        item['personid'] = int(self.personid[idx])
        item['extra_info'] = copy.deepcopy(self.extra_info[idx])
        item['idx'] = idx
        item['_scale'] = scale
        return item


    @staticmethod
    def load_tars_as_webdataset(cfg: CfgNode, urls: Union[str, List[str]], train: bool,
            resampled=False,
            epoch_size=None,
            cache_dir=None,
            **kwargs) -> Dataset:
        """
        Loads the dataset from a webdataset tar file.
        """

        IMG_SIZE = cfg.MODEL.IMAGE_SIZE  # 256
        BBOX_SHAPE = cfg.MODEL.get('BBOX_SHAPE', None)   # None
        MEAN = 255. * np.array(cfg.MODEL.IMAGE_MEAN)  # [123.675, 116.28, 103.53]
        STD = 255. * np.array(cfg.MODEL.IMAGE_STD)  # [58.395, 57.12, 57.375]

        # 处理有mask的数据集，判断是否存在这个npz文件
        def split_data(source):
            source1, source2 = itertools.tee(source)
            item = next(source2)  # 令source1和source2错位一个，避免两个数据完全相同
            for item1 in source1:
                try:
                    item2 = next(source2)
                except Exception:
                    print("iteration end!")
                    item2 = item
                # 查看image
                key1 = os.path.basename(item1['__key__'])
                key2 = os.path.basename(item2['__key__'])
                print("key1:", key1)
                print("key2:", key2)
                sub_names1 = key1.split("_")
                sub_names2 = key2.split("_")
                label = 0  # 0：表示两个item不是一个动作  1：表示两个item是一个动作
                if (len(sub_names1) == 5) and (len(sub_names2) == 5):
                    motion_name1 = "_".join(sub_names1[:3])
                    view_name1 = sub_names1[3]
                    frame_idx1 = sub_names1[4]
                    motion_name2 = "_".join(sub_names2[:3])
                    view_name2 = sub_names2[3]
                    frame_idx2 = sub_names2[4]
                    # 若motion_name和frame_idx相同，view_name不同，表示是不同视角下的同一个动作，其3d动作(或者pose)应该一样
                    if (motion_name1 == motion_name2) and (frame_idx1 == frame_idx2):
                        label = 1
                datas1 = item1['data.pyd']
                datas2 = item2['data.pyd']
                for data1, data2 in zip(datas1, datas2):
                    # 归一化keypoints3d到(-1, 1)
                    # keypoints3d = data["keypoints_3d"][:, :3]
                    # keypoints3d = keypoints3d / 100  # 单位由米变成厘米
                    # min_value = np.min(keypoints3d)
                    # max_value = np.max(keypoints3d)
                    # keypoints3d = 2 * (keypoints3d - min_value) / (max_value - min_value) - 1   # [-0.5, 0.5]
                    # njoints, _ = keypoints3d.shape
                    # confidence = data["keypoints_3d"][:, 3]
                    # confidence = np.expand_dims(confidence, axis=1)
                    # keypoints3d = np.concatenate((keypoints3d, confidence), axis=1)
                    # keypoints3d.fill(0)
                    # data["keypoints_3d"] = keypoints3d
                    data1["keypoints_3d"].fill(0)
                    data2["keypoints_3d"].fill(0)
                    data1["body_pose"].fill(0)
                    data2["body_pose"].fill(0)
                    data1["betas"].fill(0)
                    data2["betas"].fill(0)
                    data1["has_body_pose"] = 0
                    data2["has_betas"] = 0
                    if 'detection.npz' in item1:
                        det_idx = data1['extra_info']['detection_npz_idx']
                        mask1 = item1['detection.npz']['masks'][det_idx]
                    else:
                        mask1 = np.ones_like(item1['jpg'][:, :, 0], dtype=bool)
                    if 'detection.npz' in item2:
                        det_idx = data2['extra_info']['detection_npz_idx']
                        mask2 = item2['detection.npz']['masks'][det_idx]
                    else:
                        mask2 = np.ones_like(item2['jpg'][:, :, 0], dtype=bool)

                    yield {
                        '__key__': item1['__key__'],
                        '__key1__': item1['__key__'],
                        'jpg1': item1['jpg'],
                        'data1.pyd': data1,
                        'mask1': mask1,
                        '__key2__': item2['__key__'],
                        'jpg2': item2['jpg'],
                        'data2.pyd': data2,
                        'mask2': mask2,
                        'label': label
                    }

        def suppress_bad_kps(item, thresh=0.0):
            # 把置信度低于thresh的keypoints2d置信度置为0
            if thresh > 0:
                kp2d = item['data1.pyd']['keypoints_2d']
                kp2d_conf = np.where(kp2d[:, 2] < thresh, 0.0, kp2d[:, 2])
                item['data1.pyd']['keypoints_2d'] = np.concatenate([kp2d[:,:2], kp2d_conf[:,None]], axis=1)
                kp2d = item['data2.pyd']['keypoints_2d']
                kp2d_conf = np.where(kp2d[:, 2] < thresh, 0.0, kp2d[:, 2])
                item['data2.pyd']['keypoints_2d'] = np.concatenate([kp2d[:, :2], kp2d_conf[:, None]], axis=1)
            return item

        def filter_numkp(item, numkp=4, thresh=0.0):  # 可用的keypoints2d要大于4个
            kp_conf1 = item['data1.pyd']['keypoints_2d'][:, 2]
            kp_conf2 = item['data2.pyd']['keypoints_2d'][:, 2]
            return ((kp_conf1 > thresh).sum() > numkp) and ((kp_conf2 > thresh).sum() > numkp)

        def filter_reproj_error(item, thresh=10**4.5):  # 重投影误差要小于阈值
            losses1 = item['data1.pyd'].get('extra_info', {}).get('fitting_loss', np.array({})).item()
            reproj_loss1 = losses1.get('reprojection_loss', None)
            losses2 = item['data2.pyd'].get('extra_info', {}).get('fitting_loss', np.array({})).item()
            reproj_loss2 = losses2.get('reprojection_loss', None)
            return (reproj_loss1 is None or reproj_loss1 < thresh) and (reproj_loss2 is None or reproj_loss2 < thresh)

        def filter_bbox_size(item, thresh=1):
            bbox_size_min1 = item['data1.pyd']['scale'].min().item() * 200.
            bbox_size_min2 = item['data2.pyd']['scale'].min().item() * 200.
            return (bbox_size_min1 > thresh) and (bbox_size_min2 > thresh)

        def filter_no_poses(item):  # 当前帧是否含有pose
            return (item['data1.pyd']['has_body_pose'] > 0) and (item['data2.pyd']['has_body_pose'] > 0)

        def supress_bad_betas(item, thresh=3):
            has_betas = item['data1.pyd']['has_betas']   # 当前帧是否有shape参数
            if thresh > 0 and has_betas:
                betas_abs = np.abs(item['data1.pyd']['betas'])
                if (betas_abs > thresh).any():
                    item['data1.pyd']['has_betas'] = False
            has_betas = item['data2.pyd']['has_betas']  # 当前帧是否有shape参数
            if thresh > 0 and has_betas:
                betas_abs = np.abs(item['data2.pyd']['betas'])
                if (betas_abs > thresh).any():
                    item['data2.pyd']['has_betas'] = False
            return item

        amass_poses_hist100_smooth = load_amass_hist_smooth()
        def supress_bad_poses(item):  # pose是否合理
            has_body_pose = item['data1.pyd']['has_body_pose']
            if has_body_pose:
                body_pose = item['data1.pyd']['body_pose']
                pose_is_probable = poses_check_probable(torch.from_numpy(body_pose)[None, 3:], amass_poses_hist100_smooth).item()
                if not pose_is_probable:
                    item['data1.pyd']['has_body_pose'] = False
            has_body_pose = item['data2.pyd']['has_body_pose']
            if has_body_pose:
                body_pose = item['data2.pyd']['body_pose']
                pose_is_probable = poses_check_probable(torch.from_numpy(body_pose)[None, 3:],
                                                        amass_poses_hist100_smooth).item()
                if not pose_is_probable:
                    item['data2.pyd']['has_body_pose'] = False
            return item

        def poses_betas_simultaneous(item):
            # We either have both body_pose and betas, or neither
            # 是否同时具有betas和body_pose
            has_betas = item['data1.pyd']['has_betas']
            has_body_pose = item['data1.pyd']['has_body_pose']
            item['data1.pyd']['has_betas'] = item['data1.pyd']['has_body_pose'] = np.array(float((has_body_pose>0) and (has_betas>0)))
            has_betas = item['data2.pyd']['has_betas']
            has_body_pose = item['data2.pyd']['has_body_pose']
            item['data2.pyd']['has_betas'] = item['data2.pyd']['has_body_pose'] = np.array(
                float((has_body_pose > 0) and (has_betas > 0)))
            return item

        def set_betas_for_reg(item):
            # Always have betas set to true
            has_betas = item['data1.pyd']['has_betas']
            betas = item['data1.pyd']['betas']
            if not (has_betas>0):
                item['data1.pyd']['has_betas'] = np.array(float((True)))
                item['data1.pyd']['betas'] = betas * 0

            has_betas = item['data2.pyd']['has_betas']
            betas = item['data2.pyd']['betas']
            if not (has_betas > 0):
                item['data2.pyd']['has_betas'] = np.array(float((True)))
                item['data2.pyd']['betas'] = betas * 0
            return item

        # Load the dataset
        if epoch_size is not None:
            resampled = True  # 重新根据数据集的weight来采样
        corrupt_filter = lambda sample: (sample['__key__'] not in CORRUPT_KEYS)
        import webdataset as wds
        dataset = wds.WebDataset(expand_urls(urls),    # xxxdatatset/xxxxxx.tar
                                nodesplitter=wds.split_by_node,  # 将数据集划分为多个节点，以便在多个设备上并行处理
                                shardshuffle=True,  # 开启了shard级别的数据洗牌，这样可以在每个epoch中获得不同的样本顺序
                                resampled=resampled,  # 是否对数据集进行重采样
                                cache_dir=cache_dir,  # 指定了用于缓存数据的目录
                              ).select(corrupt_filter)
        #if train:
        #    dataset = dataset.shuffle(100)
        dataset = dataset.decode('rgb8').rename(jpg='jpg;jpeg;png')  # numpy uint8的格式

        # Process the dataset
        dataset = dataset.compose(split_data)

        # Filter/clean the dataset
        # 对于输入数据的预处理（例：对于置信度低于阈值的点，就把置信度置为0），过滤和清洗整个数据集
        SUPPRESS_KP_CONF_THRESH = cfg.DATASETS.get('SUPPRESS_KP_CONF_THRESH', 0.0)
        SUPPRESS_BETAS_THRESH = cfg.DATASETS.get('SUPPRESS_BETAS_THRESH', 0.0)
        SUPPRESS_BAD_POSES = cfg.DATASETS.get('SUPPRESS_BAD_POSES', False)
        POSES_BETAS_SIMULTANEOUS = cfg.DATASETS.get('POSES_BETAS_SIMULTANEOUS', False)
        BETAS_REG = cfg.DATASETS.get('BETAS_REG', False)
        FILTER_NO_POSES = cfg.DATASETS.get('FILTER_NO_POSES', False)
        FILTER_NUM_KP = cfg.DATASETS.get('FILTER_NUM_KP', 4)
        FILTER_NUM_KP_THRESH = cfg.DATASETS.get('FILTER_NUM_KP_THRESH', 0.0)
        FILTER_REPROJ_THRESH = cfg.DATASETS.get('FILTER_REPROJ_THRESH', 0.0)
        FILTER_MIN_BBOX_SIZE = cfg.DATASETS.get('FILTER_MIN_BBOX_SIZE', 0.0)
        if SUPPRESS_KP_CONF_THRESH > 0:
            dataset = dataset.map(lambda x: suppress_bad_kps(x, thresh=SUPPRESS_KP_CONF_THRESH))
        if SUPPRESS_BETAS_THRESH > 0:
            dataset = dataset.map(lambda x: supress_bad_betas(x, thresh=SUPPRESS_BETAS_THRESH))
        if SUPPRESS_BAD_POSES:
            dataset = dataset.map(lambda x: supress_bad_poses(x))
        if POSES_BETAS_SIMULTANEOUS:
            dataset = dataset.map(lambda x: poses_betas_simultaneous(x))
        if FILTER_NO_POSES:
            dataset = dataset.select(lambda x: filter_no_poses(x))
        if FILTER_NUM_KP > 0:
            dataset = dataset.select(lambda x: filter_numkp(x, numkp=FILTER_NUM_KP, thresh=FILTER_NUM_KP_THRESH))
        if FILTER_REPROJ_THRESH > 0:
            dataset = dataset.select(lambda x: filter_reproj_error(x, thresh=FILTER_REPROJ_THRESH))
        if FILTER_MIN_BBOX_SIZE > 0:
            dataset = dataset.select(lambda x: filter_bbox_size(x, thresh=FILTER_MIN_BBOX_SIZE))
        if BETAS_REG:
            dataset = dataset.map(lambda x: set_betas_for_reg(x))       # NOTE: Must be at the end

        use_skimage_antialias = cfg.DATASETS.get('USE_SKIMAGE_ANTIALIAS', False)
        border_mode = {
            'constant': cv2.BORDER_CONSTANT,
            'replicate': cv2.BORDER_REPLICATE,
        }[cfg.DATASETS.get('BORDER_MODE', 'constant')]

        # Process the dataset further
        dataset = dataset.map(lambda x: SiameseImageDataset.process_webdataset_tar_item(x, train,
                                                        augm_config=cfg.DATASETS.CONFIG,
                                                        MEAN=MEAN, STD=STD, IMG_SIZE=IMG_SIZE,
                                                        BBOX_SHAPE=BBOX_SHAPE,
                                                        use_skimage_antialias=use_skimage_antialias,
                                                        border_mode=border_mode,
                                                        ))
        if epoch_size is not None:
            dataset = dataset.with_epoch(epoch_size)

        return dataset


    # 数据最后处理的方式，进行数据增强和返回数据
    @staticmethod
    def process_webdataset_tar_item(item, train, 
                                    augm_config=None, 
                                    MEAN=DEFAULT_MEAN, 
                                    STD=DEFAULT_STD, 
                                    IMG_SIZE=DEFAULT_IMG_SIZE,
                                    BBOX_SHAPE=None,
                                    use_skimage_antialias=False,
                                    border_mode=cv2.BORDER_CONSTANT,
                                    ):
        # Read data from item
        key1 = item['__key1__']
        image1 = item['jpg1']
        data1 = item['data1.pyd']
        mask1 = item['mask1']
        key2 = item['__key2__']
        image2 = item['jpg2']
        data2 = item['data2.pyd']
        mask2 = item['mask2']
        label = item['label']

        keypoints1_2d = data1['keypoints_2d'].astype(np.float32)
        keypoints1_3d = data1['keypoints_3d'].astype(np.float32)
        center1 = data1['center'].astype(np.float32)
        scale1 = data1['scale'].astype(np.float32)
        body_pose1 = data1['body_pose'].astype(np.float32)
        betas1 = data1['betas'].astype(np.float32)
        has_body_pose1 = data1['has_body_pose'].astype(np.float32)
        has_betas1 = data1['has_betas'].astype(np.float32)
        # image_file = data['image_file']
        keypoints2_2d = data2['keypoints_2d'].astype(np.float32)
        keypoints2_3d = data2['keypoints_3d'].astype(np.float32)
        center2 = data2['center'].astype(np.float32)
        scale2 = data2['scale'].astype(np.float32)
        body_pose2 = data2['body_pose'].astype(np.float32)
        betas2 = data2['betas'].astype(np.float32)
        has_body_pose2 = data2['has_body_pose'].astype(np.float32)
        has_betas2 = data2['has_betas'].astype(np.float32)
        # image_file = data['image_file']

        # Process data
        orig_keypoints1_2d = keypoints1_2d.copy()
        center1_x = center1[0]
        center1_y = center1[1]
        bbox_size1 = expand_to_aspect_ratio(scale1*200, target_aspect_ratio=BBOX_SHAPE).max()
        orig_keypoints2_2d = keypoints2_2d.copy()
        center2_x = center2[0]
        center2_y = center2[1]
        bbox_size2 = expand_to_aspect_ratio(scale2 * 200, target_aspect_ratio=BBOX_SHAPE).max()
        # if bbox_size < 1:  # debug
        #     breakpoint()


        smpl_params1 = {'global_orient': body_pose1[:3],
                    'body_pose': body_pose1[3:],
                    'betas': betas1
                    }

        has_smpl_params1 = {'global_orient': has_body_pose1,
                        'body_pose': has_body_pose1,
                        'betas': has_betas1
                        }
        smpl_params2 = {'global_orient': body_pose2[:3],
                       'body_pose': body_pose2[3:],
                       'betas': betas2
                       }

        has_smpl_params2 = {'global_orient': has_body_pose2,
                           'body_pose': has_body_pose2,
                           'betas': has_betas2
                           }

        smpl_params_is_axis_angle = {'global_orient': True,
                                    'body_pose': True,
                                    'betas': False
                                    }

        augm_config = copy.deepcopy(augm_config)
        # Crop image and (possibly) perform data augmentation
        img_rgba1 = np.concatenate([image1, mask1.astype(np.uint8)[:,:,None]*255], axis=2)
        img_patch_rgba1, keypoints_2d1, keypoints_3d1, smpl_params1, has_smpl_params1, img_size1, trans1 = get_example(img_rgba1,
                                                                                                    center1_x, center1_y,
                                                                                                    bbox_size1, bbox_size1,
                                                                                                    keypoints1_2d, keypoints1_3d,
                                                                                                    smpl_params1, has_smpl_params1,
                                                                                                    FLIP_KEYPOINT_PERMUTATION,
                                                                                                    IMG_SIZE, IMG_SIZE,
                                                                                                    MEAN, STD, train, augm_config,
                                                                                                    is_bgr=False, return_trans=True,
                                                                                                    use_skimage_antialias=use_skimage_antialias,
                                                                                                    border_mode=border_mode,
                                                                                                    )
        img_patch1 = img_patch_rgba1[:3,:,:]
        mask_patch1 = (img_patch_rgba1[3,:,:] / 255.0).clip(0,1)
        if (mask_patch1 < 0.5).all():
            mask_patch1 = np.ones_like(mask_patch1)

        img_rgba2 = np.concatenate([image2, mask2.astype(np.uint8)[:, :, None] * 255], axis=2)
        img_patch_rgba2, keypoints_2d2, keypoints_3d2, smpl_params2, has_smpl_params2, img_size2, trans2 = get_example(
            img_rgba2,
            center2_x, center2_y,
            bbox_size2, bbox_size2,
            keypoints2_2d, keypoints2_3d,
            smpl_params2, has_smpl_params2,
            FLIP_KEYPOINT_PERMUTATION,
            IMG_SIZE, IMG_SIZE,
            MEAN, STD, train, augm_config,
            is_bgr=False, return_trans=True,
            use_skimage_antialias=use_skimage_antialias,
            border_mode=border_mode,
            )
        img_patch2 = img_patch_rgba2[:3, :, :]
        mask_patch2 = (img_patch_rgba2[3, :, :] / 255.0).clip(0, 1)
        if (mask_patch2 < 0.5).all():
            mask_patch2 = np.ones_like(mask_patch2)

        item = {}

        item['img1'] = img_patch1
        item['mask1'] = mask_patch1
        # item['img_og'] = image
        # item['mask_og'] = mask
        item['keypoints_2d1'] = keypoints_2d1.astype(np.float32)
        item['keypoints_3d1'] = keypoints_3d1.astype(np.float32)
        item['orig_keypoints_2d1'] = orig_keypoints1_2d.astype(np.float32)
        item['box_center1'] = center1.copy()
        item['box_size1'] = bbox_size1
        item['img_size1'] = 1.0 * img_size1[::-1].copy()
        item['smpl_params1'] = smpl_params1
        item['has_smpl_params1'] = has_smpl_params1
        item['smpl_params_is_axis_angle1'] = smpl_params_is_axis_angle
        item['_scale1'] = scale1
        item['_trans1'] = trans1
        item['imgname1'] = key1
        # item['idx'] = idx
        item['img2'] = img_patch2
        item['mask2'] = mask_patch2
        # item['img_og'] = image
        # item['mask_og'] = mask
        item['keypoints_2d2'] = keypoints_2d2.astype(np.float32)
        item['keypoints_3d2'] = keypoints_3d2.astype(np.float32)
        item['orig_keypoints_2d2'] = orig_keypoints2_2d.astype(np.float32)
        item['box_center2'] = center2.copy()
        item['box_size2'] = bbox_size2
        item['img_size2'] = 1.0 * img_size2[::-1].copy()
        item['smpl_params2'] = smpl_params2
        item['has_smpl_params2'] = has_smpl_params2
        item['smpl_params_is_axis_angle2'] = smpl_params_is_axis_angle
        item['_scale2'] = scale2
        item['_trans2'] = trans2
        item['imgname2'] = key2
        item['label'] = label
        return item
