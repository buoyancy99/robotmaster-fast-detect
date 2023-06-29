import numpy as np
import os
import cv2
import pandas as pd
from glob import glob
from tqdm import tqdm, trange
import json
import torchvision
from PIL import Image

class rm_dataset_utils:
    def __init__(self, background_dir, robot_dataset_dir, result_imgae_dir, result_annotation_dir, transform=None,):
        self.background_dir = background_dir
        self.robot_dataset_dir = robot_dataset_dir
        self.result_imgae_dir = result_imgae_dir
        self.result_annotation_dir = result_annotation_dir
        self.robot_dataset = pd.read_csv(os.path.join(robot_dataset_dir, 'labels_c8.csv'))
        self.background_files = glob(os.path.join(self.background_dir, '*.jpg'))
        self.transform = transform
        self.label_idx = {'red_1': 1, 'red_2': 2, 'blue_1': 3, 'blue_2': 4, 'red_armor': 5, 'blue_armor': 6}
        self.annotation_counter = 0
        self.annotations = {'images': [],
                            "annotations": [],
                            "info": {
                                "description": "RM Berkeley 2020 Dataset",
                                "url": "",
                                "version": "1.0",
                                "year": 2020,
                                "contributor": "Boyuan Chen",
                                "date_created": "2020/05/01"
                            },
                            "licenses": [{
                                "url": "http://boyuan.space",
                                "id": 0,
                                "name": "Dummy License"
                            }],
                            "categories": [{"supercategory": label,
                                            "id": id,
                                            "name": label
                                            } for label, id in self.label_idx.items()]
                            }

    def generate(self):
        for i in trange(len(self.background_files) * 10):
            image = cv2.imread(self.background_files[i % len(self.background_files)])
            image = cv2.resize(image, dsize=(int(image.shape[1] * 0.5), int(image.shape[0] * 0.5)))
            image, annotation = self._generate_with_background(image, i)
            image_file_name = str(i) + '.jpg'
            self.annotations["annotations"].extend(annotation)
            self.annotations["images"].append({
                "license": 0,
                "file_name": image_file_name,
                "coco_url": "",
                "height": image.shape[0],
                "width": image.shape[1],
                "date_captured": "",
                "flickr_url": "",
                "id": i
            })
            image = self._jitter_image(image)
            cv2.imwrite(os.path.join(self.result_imgae_dir, image_file_name), image)

        with open(os.path.join(self.result_annotation_dir, 'annotations.json'), 'w') as f:
            json.dump(self.annotations, f)

    def _jitter_image(self, image):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        im_pil = Image.fromarray(image)
        im_pil = self.transform(im_pil)
        image = np.asarray(im_pil)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        return image


    def _generate_with_background(self, background, image_id):
        num_robots = np.random.randint(1, 4)
        annotations = []
        background_H, background_W = background.shape[:2]
        for _ in range(num_robots):
            robot_data = self.robot_dataset.loc[np.random.randint(len(self.robot_dataset))]
            robot_image = cv2.imread(os.path.join(self.robot_dataset_dir, robot_data.file), cv2.IMREAD_UNCHANGED)
            H, W = robot_image.shape[:2]

            armor_xmin1 = robot_data.xmin1 / W
            armor_xmax1 = robot_data.xmax1 / W
            armor_ymin1 = robot_data.ymin1 / H
            armor_ymax1 = robot_data.ymax1 / H
            armor_xmin2 = robot_data.xmin2 / W
            armor_xmax2 = robot_data.xmax2 / W
            armor_ymin2 = robot_data.ymin2 / H
            armor_ymax2 = robot_data.ymax2 / H

            if np.random.rand() < 0.3:
                # random crop robot left & right
                crop_cap = 0.6  # we should see at least 0.4 of the robot
                left_prop = np.random.rand()
                right_prop = 1 - left_prop
                left_margin, right_margin = int(W * left_prop * crop_cap), int(W * right_prop * crop_cap)
                top_margin, bottom_margin = 0, 0
                if np.random.rand() < 0.2:
                    # crop up and down
                    crop_cap = 0.3  # we should see at least 1/2 of the robot
                    top_prop = np.random.rand()
                    bottom_prop = 1 - top_prop
                    top_margin, bottom_margin = int(H * top_prop * crop_cap), int(H * bottom_prop * crop_cap)
                new_image = np.zeros_like(robot_image)
                new_image[top_margin:H - bottom_margin, left_margin:W - right_margin] = robot_image[
                                                                                        top_margin:H - bottom_margin,
                                                                                        left_margin:W - right_margin]
                robot_image = new_image

            scale = background_H / H / np.random.uniform(1, 6)
            H_scaled, W_scaled = int(scale * H), int(scale * W)
            robot_image = cv2.resize(robot_image, (W_scaled, H_scaled))

            armor_xmin1 = int(armor_xmin1 * W_scaled)
            armor_xmax1 = int(armor_xmax1 * W_scaled)
            armor_ymin1 = int(armor_ymin1 * H_scaled)
            armor_ymax1 = int(armor_ymax1 * H_scaled)
            armor_xmin2 = int(armor_xmin2 * W_scaled)
            armor_xmax2 = int(armor_xmax2 * W_scaled)
            armor_ymin2 = int(armor_ymin2 * H_scaled)
            armor_ymax2 = int(armor_ymax2 * H_scaled)

            pos_hori, pos_vert = int(np.random.random() * background_W) - W_scaled // 2, int(
                np.random.random() * background_H) - H_scaled // 2
            left_offset = 0 - min(pos_hori, 0)
            top_offset = 0 - min(pos_vert, 0)
            paste_W = min(pos_hori + W_scaled, background_W) - max(pos_hori, 0)
            paste_H = min(pos_vert + H_scaled, background_H) - max(pos_vert, 0)
            pos_hori_clip = max(pos_hori, 0)
            pos_vert_clip = max(pos_vert, 0)
            robot_image = robot_image[top_offset:top_offset + paste_H, left_offset:left_offset + paste_W]
            alpha = robot_image[:, :, -1:] > 127
            background[pos_vert_clip:pos_vert_clip + paste_H, pos_hori_clip:pos_hori_clip + paste_W] = \
                robot_image[:, :, :3] * alpha + background[pos_vert_clip:pos_vert_clip + \
                paste_H, pos_hori_clip:pos_hori_clip + paste_W] * (1 - alpha)

            robot_xmin = pos_hori
            robot_ymin = pos_vert
            robot_xmax = pos_hori + W_scaled
            robot_ymax = pos_vert + H_scaled
            record = {"segmentation": [],
                      "area": 1,
                      "iscrowd": 0,
                      "image_id": image_id,
                      "bbox": [robot_xmin, robot_ymin, robot_xmax - robot_xmin, robot_ymax - robot_ymin],
                      "category_id": self.label_idx[robot_data.label],
                      "id": self.annotation_counter
                      }
            self.annotation_counter += 1
            annotations.append(record)

            armor_xmin1 = pos_hori + armor_xmin1
            armor_ymin1 = pos_vert + armor_ymin1
            armor_xmax1 = pos_hori + armor_xmax1
            armor_ymax1 = pos_vert + armor_ymax1
            armor_xmin2 = pos_hori + armor_xmin2
            armor_ymin2 = pos_vert + armor_ymin2
            armor_xmax2 = pos_hori + armor_xmax2
            armor_ymax2 = pos_vert + armor_ymax2

            armor_color = robot_data.label.split("_")[0] + "_armor"
            if robot_data.num_plates >= 1:
                record = {"segmentation": [],
                          "area": 1,
                          "iscrowd": 0,
                          "image_id": image_id,
                          "bbox": [armor_xmin1, armor_ymin1, armor_xmax1 - armor_xmin1, armor_ymax1 - armor_ymin1],
                          "category_id": self.label_idx[armor_color],
                          "id": self.annotation_counter
                          }
                self.annotation_counter += 1
                annotations.append(record)

            elif robot_data.num_plates == 2:
                record = {"segmentation": [],
                          "area": 1,
                          "iscrowd": 0,
                          "image_id": image_id,
                          "bbox": [armor_xmin2, armor_ymin2, armor_xmax2 - armor_xmin2, armor_ymax2 - armor_ymin2],
                          "category_id": self.label_idx[armor_color],
                          "id": self.annotation_counter
                          }
                self.annotation_counter += 1
                annotations.append(record)

        return background, annotations