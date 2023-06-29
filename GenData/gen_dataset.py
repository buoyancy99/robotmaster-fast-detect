import os
from pathlib import Path
from utils import rm_dataset_utils
from torchvision.transforms import ColorJitter

dataset_dir = '/run/media/boyuan/data/Datasets/'
# dataset_dir = '/shared/boyuan/Datasets/'
robot_dataset_dir = os.path.join(dataset_dir, 'RMRobot')
mscoco_imgae_dir = os.path.join(dataset_dir, 'MSCOCO', 'images')
rmcoco_imgae_dir = os.path.join(dataset_dir, 'RMCOCO6', 'images')
rmcoco_annotation_dir = os.path.join(dataset_dir, 'RMCOCO6', 'annotations')

Path(rmcoco_imgae_dir).mkdir(parents=True, exist_ok=True)
Path(rmcoco_annotation_dir).mkdir(parents=True, exist_ok=True)

util = rm_dataset_utils(mscoco_imgae_dir,
                        robot_dataset_dir,
                        rmcoco_imgae_dir,
                        rmcoco_annotation_dir,
                        ColorJitter(brightness=0.3, contrast=0.2, saturation=0.15, hue=0.05))

if __name__ == "__main__":
    util.generate()