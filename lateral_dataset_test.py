import os
import cv2
import numpy as np
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset

class LateralDatasetTest(Dataset):
    def __init__(self, df, args, training=True):
        """
        For external testing (classification only).
        Expects a DataFrame with columns: 'img_dcm' and 'binary_label'.
        """
        self.df = df.reset_index(drop=True)
        self.training = training
        self.min_side = args.size
        self.args = args
        self.transforms = self.build_transforms()
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        label = row['binary_label']
        img_path = row['img_dcm']
        dcm_name = os.path.basename(img_path)
        
        # Read the image in grayscale.
        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise ValueError(f"Image not found or unable to read: {img_path}")
        
        # Apply Albumentations transforms.
        transformed = self.transforms(image=image)
        image_transformed = transformed['image']
        
        # Prepare the sample dict.
        sample = {
            'image': image_transformed,
            'label': torch.tensor(label, dtype=torch.float32),
            'dcm_name': dcm_name
        }
        return sample
    
    def build_transforms(self):
        """
        Build Albumentations transform pipeline for test data.
        When training is True, a set of augmentations is applied;
        otherwise, only resizing, normalization, and tensor conversion.
        """
        if self.training:
            return A.Compose([
                A.Resize(self.min_side, self.min_side, p=1.0),
                A.Rotate(limit=45, p=0.8),
                A.ShiftScaleRotate(
                    shift_limit=0.1,
                    scale_limit=0.0,
                    rotate_limit=0.0,
                    p=0.8,
                    border_mode=cv2.BORDER_CONSTANT
                ),
                A.RandomBrightnessContrast(
                    brightness_limit=0.2,
                    contrast_limit=0.2,
                    p=0.5
                ),
                A.OneOf([
                    A.ElasticTransform(
                        alpha=30,
                        sigma=30 * 0.05,
                        alpha_affine=30 * 0.03,
                        p=0.5
                    ),
                    A.GaussNoise(var_limit=(10.0, 50.0), p=0.5),
                    A.CoarseDropout(
                        max_holes=4,
                        max_height=8,
                        max_width=8,
                        fill_value=0,
                        p=0.5
                    ),
                ], p=0.5),
                A.Normalize(mean=0.0, std=1.0),
                ToTensorV2()
            ])
        else:
            return A.Compose([
                A.Resize(self.min_side, self.min_side, p=1.0),
                A.Normalize(mean=0.0, std=1.0),
                ToTensorV2()
            ])
