import torch
import albumentations as A 
from albumentations.pytorch import ToTensorV2 
import cv2   
from kmeans_anchor import auto_anchor
import os


class Config:
    _initialized = False

    @classmethod
    def initialize(self):
        print(f"Initialized: {self._initialized}")
        if not self._initialized:
            print("Initializing config...")
            self._initialized = True
            
            dataset = "pascal voc"
            if os.name == 'nt':
                base_dir = os.getcwd()
                #self.train_csv_file = os.path.join(base_dir, "data", dataset, "100examples.csv")
                #self.test_csv_file = os.path.join(base_dir, "data", dataset, "100examples_test.csv")
                self.train_csv_file = os.path.join(base_dir, "data", dataset, "train.csv")
                self.test_csv_file = os.path.join(base_dir, "data", dataset, "test.csv")
                self.image_dir = os.path.join(base_dir, "data", dataset, "images")
                self.label_dir = os.path.join(base_dir, "data", dataset, "labels")
                self.checkpoint_file = os.path.join(base_dir, "v5", f"{dataset}_checkpoint.pth.tar")
            else:
                #self.train_csv_file = f"../data/{dataset}/100examples.csv"
                #self.test_csv_file = f"../data/{dataset}/100examples_test.csv"
                self.train_csv_file = f"../data/{dataset}/train.csv"
                self.test_csv_file = f"../data/{dataset}/test.csv"
                self.image_dir = f"../data/{dataset}/images/"
                self.label_dir = f"../data/{dataset}/labels/"  
                self.checkpoint_file = f"{dataset}_checkpoint.pth.tar"

            self.PAN_channels = [256, 512, 1024]

            self.load_model = True
            self.save_model = True

            self.epochs = 600
            self.batch_size = 4
            self.min_leanring_rate = 1e-4
            self.max_leanring_rate = self.min_leanring_rate * 5
            self.numerical_stability = 1e-6
            self.image_size = 640
            self.s = [self.image_size // 32, self.image_size // 16, self.image_size // 8] 

            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            print(f"-{self.device}-")

            self.ANCHORS = auto_anchor(self.num_anchors, self.label_dir, self.s)
            self.num_anchors = len(self.ANCHORS)
            self.num_scales = len(self.s)

            # ANCHORS = [ 
            # 	[(0.28, 0.22), (0.38, 0.48), (0.9, 0.78)], 
            # 	[(0.07, 0.15), (0.15, 0.11), (0.14, 0.29)], 
            # 	[(0.02, 0.03), (0.04, 0.07), (0.08, 0.06)], 
            # ] 

            self.scaled_anchors = ( 
                torch.tensor(self.ANCHORS) *
                torch.tensor(self.s).unsqueeze(1).unsqueeze(1).repeat(1,3,2) 
            ).to(self.device) 

            self.class_labels = [ 
                "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", 
                "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person", 
                "pottedplant", "sheep", "sofa", "train", "tvmonitor"
            ]
            self.num_classes = len(self.class_labels)

            self.valid_prediction_threshold = 0.6
            self.enough_overlap_threshold = 0.6

            self.train_transform = A.Compose( 
                [ 
                    A.Resize(height=self.image_size, width=self.image_size),
                    A.ColorJitter(
                        brightness=0.5, contrast=0.5,
                        saturation=0.5, hue=0.5, p=0.5
                    ),
                    A.Rotate(limit=45, p=0.5),
                    A.HorizontalFlip(p=0.5),
                    A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),  
                    A.Normalize(mean=[0, 0, 0], std=[1, 1, 1], max_pixel_value=255), 
                    ToTensorV2() 
                ], 
                bbox_params=A.BboxParams(format="yolo", min_visibility=0.4, label_fields=[]) 
            ) 

            self.test_transform = A.Compose( 
                [ 
                    A.LongestMaxSize(max_size=self.image_size), 
                    A.PadIfNeeded(min_height=self.image_size, min_width=self.image_size, border_mode=cv2.BORDER_CONSTANT), 
                    A.Normalize(mean=[0, 0, 0], std=[1, 1, 1], max_pixel_value=255), 
                    ToTensorV2() 
                ], 
                bbox_params=A.BboxParams(format="yolo", min_visibility=0.4, label_fields=[]) 
            )

Config.initialize()





