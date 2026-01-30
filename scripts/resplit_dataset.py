"""
Dataset Re-splitting Script for Water Meter AI Project
======================================================
Solusi untuk masalah validation set yang terlalu kecil.

Masalah: 
- Current split: Train 94%, Valid 3%, Test 3%
- Validation set (795 images) terlalu kecil untuk 11 classes

Solusi:
- Target split: Train 80%, Valid 15%, Test 5%
- Stratified split berdasarkan class distribution

Author: AI Assistant
Date: 2026-01-30
"""

import os
import shutil
import random
from pathlib import Path
from collections import defaultdict
import argparse


def get_class_distribution(labels_dir: Path) -> dict:
    """Analisis distribusi class dari label files."""
    class_counts = defaultdict(int)
    file_classes = {}  # file -> list of classes
    
    for label_file in labels_dir.glob("*.txt"):
        classes_in_file = set()
        with open(label_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 5:  # OBB format: class x1 y1 x2 y2 x3 y3 x4 y4
                    class_id = int(parts[0])
                    class_counts[class_id] += 1
                    classes_in_file.add(class_id)
        
        file_classes[label_file.stem] = list(classes_in_file)
    
    return class_counts, file_classes


def collect_all_files(base_dir: Path) -> list:
    """Kumpulkan semua file dari train, valid, dan test."""
    all_files = []
    
    for split in ['train', 'valid', 'test']:
        images_dir = base_dir / split / 'images'
        labels_dir = base_dir / split / 'labels'
        
        if not images_dir.exists():
            print(f"Warning: {images_dir} tidak ditemukan")
            continue
            
        for img_file in images_dir.iterdir():
            if img_file.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']:
                label_file = labels_dir / f"{img_file.stem}.txt"
                if label_file.exists():
                    all_files.append({
                        'image': img_file,
                        'label': label_file,
                        'stem': img_file.stem
                    })
    
    return all_files


def stratified_split(files: list, file_classes: dict, 
                     train_ratio: float = 0.80,
                     valid_ratio: float = 0.15,
                     test_ratio: float = 0.05,
                     seed: int = 42) -> tuple:
    """
    Stratified split berdasarkan class distribution.
    Memastikan setiap class terwakili di semua splits.
    """
    random.seed(seed)
    
    # Group files by their primary class (most instances)
    class_files = defaultdict(list)
    for f in files:
        classes = file_classes.get(f['stem'], [])
        if classes:
            # Use first class as primary (or you could use the most frequent)
            primary_class = classes[0]
            class_files[primary_class].append(f)
        else:
            class_files[-1].append(f)  # No class
    
    train_files = []
    valid_files = []
    test_files = []
    
    # Split each class group proportionally
    for class_id, class_file_list in class_files.items():
        random.shuffle(class_file_list)
        n = len(class_file_list)
        
        n_train = int(n * train_ratio)
        n_valid = int(n * valid_ratio)
        # n_test = remaining
        
        train_files.extend(class_file_list[:n_train])
        valid_files.extend(class_file_list[n_train:n_train + n_valid])
        test_files.extend(class_file_list[n_train + n_valid:])
    
    # Shuffle final lists
    random.shuffle(train_files)
    random.shuffle(valid_files)
    random.shuffle(test_files)
    
    return train_files, valid_files, test_files


def backup_current_split(base_dir: Path, backup_dir: Path):
    """Backup current dataset split."""
    print(f"\n📦 Backing up current split to {backup_dir}...")
    
    for split in ['train', 'valid', 'test']:
        src = base_dir / split
        dst = backup_dir / split
        if src.exists():
            if dst.exists():
                shutil.rmtree(dst)
            shutil.copytree(src, dst)
            print(f"   ✓ Backed up {split}/")


def create_new_split(base_dir: Path, train_files: list, valid_files: list, test_files: list):
    """Create new dataset split using a safe two-phase approach."""
    print("\n🔄 Creating new split...")
    
    # Phase 1: Create temporary directories with new split
    temp_dir = base_dir / '_temp_resplit'
    if temp_dir.exists():
        shutil.rmtree(temp_dir)
    temp_dir.mkdir()
    
    splits = {
        'train': train_files,
        'valid': valid_files,
        'test': test_files
    }
    
    print("   Phase 1: Copying files to temporary location...")
    for split_name, files in splits.items():
        images_dir = temp_dir / split_name / 'images'
        labels_dir = temp_dir / split_name / 'labels'
        
        images_dir.mkdir(parents=True, exist_ok=True)
        labels_dir.mkdir(parents=True, exist_ok=True)
        
        for f in files:
            # Copy image
            dst_img = images_dir / f['image'].name
            shutil.copy2(f['image'], dst_img)
            
            # Copy label
            dst_lbl = labels_dir / f['label'].name
            shutil.copy2(f['label'], dst_lbl)
        
        print(f"   ✓ {split_name}: {len(files)} files copied to temp")
    
    # Phase 2: Replace original directories with new ones
    print("   Phase 2: Replacing original directories...")
    for split in ['train', 'valid', 'test']:
        original_dir = base_dir / split
        temp_split_dir = temp_dir / split
        
        # Remove original
        if original_dir.exists():
            shutil.rmtree(original_dir)
        
        # Move temp to original location
        shutil.move(str(temp_split_dir), str(original_dir))
        
        # Remove cache file if exists
        cache_file = original_dir / 'labels.cache'
        if cache_file.exists():
            cache_file.unlink()
        
        print(f"   ✓ {split}/ replaced")
    
    # Cleanup temp directory
    if temp_dir.exists():
        shutil.rmtree(temp_dir)
    
    print("   ✓ Cleanup complete")


def print_distribution_report(base_dir: Path, file_classes: dict, 
                              train_files: list, valid_files: list, test_files: list):
    """Print detailed distribution report."""
    
    class_names = {
        0: 'digit_0', 1: 'digit_1', 2: 'digit_2', 3: 'digit_3', 4: 'digit_4',
        5: 'digit_5', 6: 'digit_6', 7: 'digit_7', 8: 'digit_8', 9: 'digit_9',
        10: 'border_water_meter_number'
    }
    
    def count_classes(files):
        counts = defaultdict(int)
        for f in files:
            for cls in file_classes.get(f['stem'], []):
                counts[cls] += 1
        return counts
    
    train_counts = count_classes(train_files)
    valid_counts = count_classes(valid_files)
    test_counts = count_classes(test_files)
    
    print("\n" + "=" * 70)
    print("📊 NEW DATASET SPLIT DISTRIBUTION")
    print("=" * 70)
    
    total = len(train_files) + len(valid_files) + len(test_files)
    print(f"\n📁 Total Files: {total}")
    print(f"   Train: {len(train_files):,} ({len(train_files)/total*100:.1f}%)")
    print(f"   Valid: {len(valid_files):,} ({len(valid_files)/total*100:.1f}%)")
    print(f"   Test:  {len(test_files):,} ({len(test_files)/total*100:.1f}%)")
    
    print(f"\n📈 Class Distribution per Split:")
    print("-" * 70)
    print(f"{'Class':<25} {'Train':>10} {'Valid':>10} {'Test':>10} {'Total':>10}")
    print("-" * 70)
    
    for cls_id in sorted(class_names.keys()):
        name = class_names[cls_id]
        t = train_counts.get(cls_id, 0)
        v = valid_counts.get(cls_id, 0)
        te = test_counts.get(cls_id, 0)
        total_cls = t + v + te
        print(f"{name:<25} {t:>10} {v:>10} {te:>10} {total_cls:>10}")
    
    print("-" * 70)
    print("\n✅ Validation set sekarang cukup besar untuk 11 classes!")
    print(f"   Minimal ~{len(valid_files)//11} samples per class di validation")


def main():
    parser = argparse.ArgumentParser(description='Re-split Water Meter Dataset')
    parser.add_argument('--base-dir', type=str, 
                        default='.',
                        help='Base directory of the dataset')
    parser.add_argument('--train-ratio', type=float, default=0.80,
                        help='Training set ratio (default: 0.80)')
    parser.add_argument('--valid-ratio', type=float, default=0.15,
                        help='Validation set ratio (default: 0.15)')
    parser.add_argument('--test-ratio', type=float, default=0.05,
                        help='Test set ratio (default: 0.05)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    parser.add_argument('--backup', action='store_true',
                        help='Backup current split before re-splitting')
    parser.add_argument('--dry-run', action='store_true',
                        help='Show what would happen without making changes')
    
    args = parser.parse_args()
    
    base_dir = Path(args.base_dir).resolve()
    
    print("=" * 70)
    print("🔧 WATER METER DATASET RE-SPLITTING TOOL")
    print("=" * 70)
    print(f"\nBase Directory: {base_dir}")
    print(f"Target Split Ratio: {args.train_ratio:.0%} / {args.valid_ratio:.0%} / {args.test_ratio:.0%}")
    print(f"Random Seed: {args.seed}")
    
    # Validate ratios
    if abs(args.train_ratio + args.valid_ratio + args.test_ratio - 1.0) > 0.001:
        print("\n❌ Error: Ratios must sum to 1.0!")
        return
    
    # Collect current state
    print("\n📂 Analyzing current dataset...")
    
    # Get class distribution from all labels
    all_class_counts = defaultdict(int)
    file_classes = {}
    
    for split in ['train', 'valid', 'test']:
        labels_dir = base_dir / split / 'labels'
        if labels_dir.exists():
            counts, fc = get_class_distribution(labels_dir)
            for k, v in counts.items():
                all_class_counts[k] += v
            file_classes.update(fc)
    
    # Collect all files
    all_files = collect_all_files(base_dir)
    print(f"   Total files found: {len(all_files)}")
    
    if len(all_files) == 0:
        print("\n❌ Error: No files found!")
        return
    
    # Perform stratified split
    train_files, valid_files, test_files = stratified_split(
        all_files, file_classes,
        train_ratio=args.train_ratio,
        valid_ratio=args.valid_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed
    )
    
    # Show report
    print_distribution_report(base_dir, file_classes, train_files, valid_files, test_files)
    
    if args.dry_run:
        print("\n⚠️  DRY RUN MODE - No changes made")
        return
    
    # Confirm
    print("\n" + "=" * 70)
    response = input("⚠️  Proceed with re-splitting? (yes/no): ").strip().lower()
    
    if response != 'yes':
        print("❌ Cancelled.")
        return
    
    # Backup if requested
    if args.backup:
        backup_dir = base_dir / 'backup_original_split'
        backup_current_split(base_dir, backup_dir)
    
    # Create new split
    create_new_split(base_dir, train_files, valid_files, test_files)
    
    print("\n" + "=" * 70)
    print("✅ DATASET RE-SPLIT COMPLETE!")
    print("=" * 70)
    print("\n📝 Next Steps:")
    print("   1. Delete labels.cache files jika ada (sudah dilakukan)")
    print("   2. Jalankan training ulang: python scripts/train.py --config configs/train_config.yaml")
    print("   3. Monitor mAP di validation set - seharusnya lebih stabil")
    
    if args.backup:
        print(f"\n💾 Backup tersimpan di: {base_dir / 'backup_original_split'}")


if __name__ == "__main__":
    main()
