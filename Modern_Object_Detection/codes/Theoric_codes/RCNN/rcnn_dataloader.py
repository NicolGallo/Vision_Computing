"""
Dataset loaders for R-CNN demo
Recommended: PASCAL VOC (what original R-CNN used)
"""

import os
import xml.etree.ElementTree as ET
import cv2
import numpy as np
from pathlib import Path
import urllib.request
import tarfile
import json


# ==================== OPTION 1: PASCAL VOC (RECOMMENDED) ====================
class PASCALVOCLoader:
    """
    Load PASCAL VOC dataset for R-CNN
    
    PASCAL VOC classes (20):
    ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat',
     'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike', 'person',
     'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor']
    """
    
    CLASSES = [
        'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat',
        'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike', 'person',
        'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor'
    ]
    
    def __init__(self, root_dir='./VOCdevkit/VOC2007'):
        """
        Args:
            root_dir: Path to VOC2007 or VOC2012 directory
        """
        self.root_dir = Path(root_dir)
        self.images_dir = self.root_dir / 'JPEGImages'
        self.annotations_dir = self.root_dir / 'Annotations'
        self.class_to_idx = {cls: i for i, cls in enumerate(self.CLASSES)}
    
    @staticmethod
    def download_voc2007(save_dir='./'):
        """Download PASCAL VOC 2007 dataset"""
        url = 'http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar'
        save_path = os.path.join(save_dir, 'VOCtrainval_06-Nov-2007.tar')
        
        if not os.path.exists(save_path):
            print(f"Downloading PASCAL VOC 2007 (~450MB)...")
            urllib.request.urlretrieve(url, save_path)
            print("Download complete!")
        
        print("Extracting...")
        with tarfile.open(save_path) as tar:
            tar.extractall(save_dir)
        print(f"Dataset extracted to {save_dir}/VOCdevkit/VOC2007")
    
    def parse_annotation(self, ann_path):
        """Parse VOC XML annotation file"""
        tree = ET.parse(ann_path)
        root = tree.getroot()
        
        boxes = []
        labels = []
        
        for obj in root.findall('object'):
            class_name = obj.find('name').text
            if class_name not in self.class_to_idx:
                continue
            
            bbox = obj.find('bndbox')
            x1 = int(bbox.find('xmin').text)
            y1 = int(bbox.find('ymin').text)
            x2 = int(bbox.find('xmax').text)
            y2 = int(bbox.find('ymax').text)
            
            # Convert to [x, y, w, h]
            boxes.append([x1, y1, x2 - x1, y2 - y1])
            labels.append(self.class_to_idx[class_name])
        
        return boxes, labels
    
    def load_image_and_annotation(self, image_name):
        """Load image and its annotation"""
        img_path = self.images_dir / f"{image_name}.jpg"
        ann_path = self.annotations_dir / f"{image_name}.xml"
        
        image = cv2.imread(str(img_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        boxes, labels = self.parse_annotation(ann_path)
        
        return image, {'boxes': boxes, 'labels': labels}
    
    def get_trainval_split(self, split='train', max_images=None):
        """
        Get train or val split
        
        Args:
            split: 'train' or 'val'
            max_images: Limit number of images (useful for quick demos)
        """
        split_file = self.root_dir / 'ImageSets' / 'Main' / f'{split}.txt'
        
        with open(split_file, 'r') as f:
            image_names = [line.strip() for line in f.readlines()]
        
        if max_images:
            image_names = image_names[:max_images]
        
        images = []
        annotations = []
        
        for img_name in image_names:
            try:
                img, ann = self.load_image_and_annotation(img_name)
                if len(ann['boxes']) > 0:  # Only include images with objects
                    images.append(img)
                    annotations.append(ann)
            except Exception as e:
                print(f"Error loading {img_name}: {e}")
                continue
        
        return images, annotations


# ==================== OPTION 2: COCO (More Modern) ====================
class COCOLoader:
    """
    Load COCO dataset (subset for demo)
    
    Note: COCO is large (80 classes, 118k training images).
    For R-CNN demo, use a small subset.
    """
    
    def __init__(self, root_dir='./coco', split='train2017'):
        """
        Args:
            root_dir: COCO root directory
            split: 'train2017' or 'val2017'
        """
        self.root_dir = Path(root_dir)
        self.images_dir = self.root_dir / split
        self.ann_file = self.root_dir / 'annotations' / f'instances_{split}.json'
        
        # Load annotations
        with open(self.ann_file, 'r') as f:
            self.coco_data = json.load(f)
        
        # Create mappings
        self.cat_id_to_idx = {cat['id']: i for i, cat in enumerate(self.coco_data['categories'])}
        self.img_id_to_anns = self._create_img_to_anns_mapping()
        
        self.class_names = [cat['name'] for cat in self.coco_data['categories']]
    
    def _create_img_to_anns_mapping(self):
        """Create mapping from image_id to annotations"""
        img_to_anns = {}
        for ann in self.coco_data['annotations']:
            img_id = ann['image_id']
            if img_id not in img_to_anns:
                img_to_anns[img_id] = []
            img_to_anns[img_id].append(ann)
        return img_to_anns
    
    def load_image_and_annotation(self, img_id):
        """Load image and annotation by image ID"""
        # Find image info
        img_info = next(img for img in self.coco_data['images'] if img['id'] == img_id)
        
        # Load image
        img_path = self.images_dir / img_info['file_name']
        image = cv2.imread(str(img_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Get annotations
        boxes = []
        labels = []
        
        if img_id in self.img_id_to_anns:
            for ann in self.img_id_to_anns[img_id]:
                x, y, w, h = ann['bbox']
                boxes.append([x, y, w, h])
                labels.append(self.cat_id_to_idx[ann['category_id']])
        
        return image, {'boxes': boxes, 'labels': labels}
    
    def get_subset(self, max_images=100, min_objects=1):
        """Get a small subset for demo"""
        images = []
        annotations = []
        
        count = 0
        for img_info in self.coco_data['images']:
            if count >= max_images:
                break
            
            img_id = img_info['id']
            if img_id in self.img_id_to_anns and len(self.img_id_to_anns[img_id]) >= min_objects:
                try:
                    img, ann = self.load_image_and_annotation(img_id)
                    images.append(img)
                    annotations.append(ann)
                    count += 1
                except Exception as e:
                    print(f"Error loading image {img_id}: {e}")
        
        return images, annotations


# ==================== OPTION 3: Custom Mini-Dataset ====================
class CustomMiniDataset:
    """
    Create a small custom dataset for quick R-CNN demo
    
    Uses a few hand-picked images with simple objects
    """
    
    def __init__(self):
        self.class_names = ['person', 'car', 'dog', 'cat', 'bicycle']
        self.images = []
        self.annotations = []
    
    def create_sample_dataset(self):
        """Create synthetic but realistic-looking dataset"""
        import matplotlib.pyplot as plt
        from matplotlib.patches import Rectangle
        
        # Dataset with realistic scenarios
        scenarios = [
            # Scenario 1: Person with dog
            {
                'background': 'park',
                'objects': [
                    {'class': 0, 'box': [50, 50, 80, 150]},   # person
                    {'class': 2, 'box': [140, 120, 60, 50]}   # dog
                ]
            },
            # Scenario 2: Car and bicycle
            {
                'background': 'street',
                'objects': [
                    {'class': 1, 'box': [30, 80, 120, 70]},   # car
                    {'class': 4, 'box': [160, 90, 50, 60]}    # bicycle
                ]
            },
            # Scenario 3: Multiple people
            {
                'background': 'office',
                'objects': [
                    {'class': 0, 'box': [40, 40, 60, 120]},   # person 1
                    {'class': 0, 'box': [120, 50, 65, 130]},  # person 2
                ]
            },
        ]
        
        for scenario in scenarios:
            # Create image
            img = np.random.randint(180, 240, (200, 250, 3), dtype=np.uint8)
            
            boxes = []
            labels = []
            
            for obj in scenario['objects']:
                x, y, w, h = obj['box']
                # Draw colored rectangle
                color = np.random.randint(50, 200, 3)
                cv2.rectangle(img, (x, y), (x+w, y+h), color.tolist(), -1)
                
                boxes.append(obj['box'])
                labels.append(obj['class'])
            
            self.images.append(img)
            self.annotations.append({'boxes': boxes, 'labels': labels})
        
        return self.images, self.annotations


# ==================== USAGE EXAMPLES ====================

def demo_pascal_voc():
    """Demo with PASCAL VOC dataset"""
    print("="*60)
    print("OPTION 1: PASCAL VOC Dataset (RECOMMENDED FOR R-CNN)")
    print("="*60)
    
    # Download if needed (only once)
    # PASCALVOCLoader.download_voc2007()
    
    # Load dataset
    voc_loader = PASCALVOCLoader(root_dir='./VOCdevkit/VOC2007')
    
    # Get a small subset for demo (10 training images)
    print("\nLoading 10 training images for demo...")
    images, annotations = voc_loader.get_trainval_split(split='train', max_images=10)
    
    print(f"Loaded {len(images)} images")
    for i, ann in enumerate(annotations):
        print(f"  Image {i+1}: {len(ann['boxes'])} objects")
    
    # Train R-CNN
    from rcnn import RCNN  # Your R-CNN implementation
    
    rcnn = RCNN(num_classes=20)
    rcnn.train(images, annotations)
    
    # Test
    test_images, test_annotations = voc_loader.get_trainval_split(split='val', max_images=5)
    detections = rcnn.detect(test_images[0])
    
    print(f"\nDetections: {len(detections)}")
    return images, annotations


def demo_coco():
    """Demo with COCO dataset"""
    print("="*60)
    print("OPTION 2: COCO Dataset")
    print("="*60)
    
    coco_loader = COCOLoader(root_dir='./coco', split='val2017')
    
    # Get small subset
    print("\nLoading 20 images for demo...")
    images, annotations = coco_loader.get_subset(max_images=20)
    
    print(f"Loaded {len(images)} images")
    print(f"Classes: {len(coco_loader.class_names)}")
    
    return images, annotations


def demo_custom():
    """Demo with custom mini-dataset"""
    print("="*60)
    print("OPTION 3: Custom Mini-Dataset (FASTEST FOR DEMO)")
    print("="*60)
    
    dataset = CustomMiniDataset()
    images, annotations = dataset.create_sample_dataset()
    
    print(f"Created {len(images)} synthetic images")
    print(f"Classes: {dataset.class_names}")
    
    # This is fastest for live demo
    from rcnn import RCNN
    
    rcnn = RCNN(num_classes=5)
    rcnn.train(images, annotations)
    
    return images, annotations


# ==================== QUICK START GUIDE ====================

if __name__ == "__main__":
    print("""
    DATASET RECOMMENDATIONS FOR R-CNN DEMO:
    
    1. PASCAL VOC 2007 (BEST - matches original paper)
       - 20 classes (person, car, dog, etc.)
       - ~2500 training images
       - Well-annotated, clean dataset
       - Use 10-20 images for quick demo
       
       Setup:
       - Download: PASCALVOCLoader.download_voc2007()
       - Use: demo_pascal_voc()
    
    2. COCO (Modern alternative)
       - 80 classes
       - Use small subset (20-50 images)
       - Good for showing scalability
       
       Setup:
       - Download from https://cocodataset.org
       - Use: demo_coco()
    
    3. Custom Synthetic (FASTEST)
       - No download needed
       - Instant demo
       - Good for algorithm explanation
       
       Setup:
       - Just run: demo_custom()
    
    RECOMMENDATION FOR LECTURE:
    - Start with Option 3 (synthetic) to show algorithm
    - Then show Option 1 (PASCAL VOC) for real results
    """)
    
    # Quick demo with synthetic data
    print("\n" + "="*60)
    print("Running Quick Demo with Synthetic Data")
    print("="*60)
    demo_custom()