from .modelnet import ModelNet40
from .kitti import KittiOdometryDataset
from .threedmatch import ThreeDMatch


DATASETS = {
    'kitti': KittiOdometryDataset,
    'modelnet40': ModelNet40,
    '3dmatch': ThreeDMatch,
}
