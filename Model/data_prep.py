import os
import shutil
from pathlib import Path
import random
from typing import Tuple

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


def split_dataset(
    cats_dir: str,
    dogs_dir: str,
    output_dir: str,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42
) -> None:
    random.seed(seed)
    
    # Splits the cats and dogs dataset into thetrain/val/test sets

    for split in ['train', 'val', 'test']:
        for category in ['cats', 'dogs']:
            os.makedirs(os.path.join(output_dir, split, category), exist_ok=True)
    
    categories = [
        ('cats', cats_dir),
        ('dogs', dogs_dir)
    ]
    
    for category_name, category_dir in categories:
        print(f"\nProcessing {category_name}...")
        
        files = [f for f in os.listdir(category_dir) 
                if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        
        random.shuffle(files)
        
        # get num of files and split them appropriately 
        n_total = len(files)
        n_train = int(n_total * train_ratio)
        n_val = int(n_total * val_ratio)
        train_files = files[:n_train]
        val_files = files[n_train:n_train + n_val]
        test_files = files[n_train + n_val:]
        
        # copy the files to the new dirs
        for file in train_files:
            src = os.path.join(category_dir, file)
            dst = os.path.join(output_dir, 'train', category_name, file)
            shutil.copy2(src, dst)
        
        for file in val_files:
            src = os.path.join(category_dir, file)
            dst = os.path.join(output_dir, 'val', category_name, file)
            shutil.copy2(src, dst)
        
        for file in test_files:
            src = os.path.join(category_dir, file)
            dst = os.path.join(output_dir, 'test', category_name, file)
            shutil.copy2(src, dst)
        
        print(f"  Total: {n_total}")
        print(f"  Train: {len(train_files)} ({len(train_files)/n_total*100:.1f}%)")
        print(f"  Val: {len(val_files)} ({len(val_files)/n_total*100:.1f}%)")
        print(f"  Test: {len(test_files)} ({len(test_files)/n_total*100:.1f}%)")
    
    print(f"\nDataset split complete, output saved to: {output_dir}")


# data augmentation step:
def get_train_transforms(image_size: int = 224) -> transforms.Compose:
    # data augmentation may consist of horizontal flip, 
    # roll rotation of +-15°
    # random brightness, contrast, saturation or hue changes
    # linear translation and scaling 

    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(
            brightness=0.2,
            contrast=0.2,
            saturation=0.2,
            hue=0.1
        ),
        transforms.RandomAffine(
            degrees=0,
            translate=(0.1, 0.1),
            scale=(0.9, 1.1)
        ),
        transforms.ToTensor(),
        transforms.Normalize( #normalize with ImageNet stats
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])


def get_val_test_transforms(image_size: int = 224) -> transforms.Compose:
    # similar augmentation
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

# create the dataloaders and datasets
def create_dataloaders(
    data_dir: str,
    batch_size: int = 32,
    image_size: int = 224,
    num_workers: int = 4,
    shuffle_train: bool = True
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    
    train_transform = get_train_transforms(image_size)
    val_test_transform = get_val_test_transforms(image_size)
    
    train_dataset = datasets.ImageFolder(
        os.path.join(data_dir, 'train'),
        transform=train_transform
    )
    val_dataset = datasets.ImageFolder(
        os.path.join(data_dir, 'val'),
        transform=val_test_transform
    )
    test_dataset = datasets.ImageFolder(
        os.path.join(data_dir, 'test'),
        transform=val_test_transform
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle_train,
        num_workers=num_workers,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    print(f"\nDataLoaders created:")
    print(f"  Train: {len(train_dataset)} images, {len(train_loader)} batches")
    print(f"  Val: {len(val_dataset)} images, {len(val_loader)} batches")
    print(f"  Test: {len(test_dataset)} images, {len(test_loader)} batches")
    print(f"  Batch size: {batch_size}")
    print(f"  Image size: {image_size}x{image_size}")
    
    return train_loader, val_loader, test_loader


def main():
    CATS_DIR = "PetImages/Cat"
    DOGS_DIR = "PetImages/Dog"
    OUTPUT_DIR = "data/cats_vs_dogs_split"
    
    print("Cats vs Dogs Dataset Preparation")
    
    print("\nStep 1: Splitting dataset into train/val/test...")
    split_dataset(
        cats_dir=CATS_DIR,
        dogs_dir=DOGS_DIR,
        output_dir=OUTPUT_DIR,
        train_ratio=0.7,
        val_ratio=0.15,
        test_ratio=0.15,
        seed=42
    )
    
    print("\nStep 2: Creating DataLoaders...")
    train_loader, val_loader, test_loader = create_dataloaders(
        data_dir=OUTPUT_DIR,
        batch_size=32,
        image_size=224,
        num_workers=4
    )
    
    print("\nStep 3: Verifying data loading...")
    images, labels = next(iter(train_loader))
    print(f"  Batch shape: {images.shape}")
    print(f"  Labels shape: {labels.shape}")
    print(f"  Image range: [{images.min():.3f}, {images.max():.3f}]")
    print(f"  Label values: {labels.unique().tolist()}")
    
    print("Data preparation complete!")


if __name__ == "__main__":
    main()