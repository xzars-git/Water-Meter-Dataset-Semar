"""
Training script for Water Meter AI
Supports both local and cloud (Colab) training

Usage:
    python scripts/train.py --config configs/train_config.yaml
"""
import argparse
import os
from pathlib import Path

from ultralytics import YOLO

# Add project root to path
import sys
sys.path.append(str(Path(__file__).parent.parent))

from src.utils.config import Config
from src.utils.logger import setup_logger, get_logger


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Train Water Meter Detection Model')
    parser.add_argument(
        '--config',
        type=str,
        default='configs/train_config.yaml',
        help='Path to training configuration file'
    )
    parser.add_argument(
        '--resume',
        type=str,
        default=None,
        help='Resume training from checkpoint'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='',
        help='Device to use (cuda:0, cpu, etc.)'
    )
    return parser.parse_args()


def main():
    """Main training function"""
    # Parse arguments
    args = parse_args()
    
    # Setup logging
    setup_logger(log_file='logs/training.log', level='INFO')
    logger = get_logger(__name__)
    
    logger.info("="*80)
    logger.info("Water Meter AI - Training Pipeline")
    logger.info("Author: Arsenius Purbandono")
    logger.info("="*80)
    
    # Load configuration
    logger.info(f"Loading configuration from: {args.config}")
    config = Config(args.config)
    
    # Initialize model
    model_type = config.get('model.type', 'yolov8n-obb')
    logger.info(f"Initializing model: {model_type}")
    
    if args.resume:
        logger.info(f"Resuming from checkpoint: {args.resume}")
        model = YOLO(args.resume)
    else:
        model = YOLO(f"{model_type}.pt")
    
    # Prepare training arguments
    train_args = {
        # Data
        'data': config.get('data.yaml_path'),
        'imgsz': config.get('data.img_size', 512),
        
        # Training
        'epochs': config.get('training.epochs', 100),
        'batch': config.get('training.batch_size', 16),
        'patience': config.get('training.patience', 20),
        
        # Optimizer
        'optimizer': config.get('training.optimizer', 'AdamW'),
        'lr0': config.get('training.lr0', 0.01),
        'lrf': config.get('training.lrf', 0.01),
        'momentum': config.get('training.momentum', 0.937),
        'weight_decay': config.get('training.weight_decay', 0.0005),
        
        # Augmentation
        'hsv_h': config.get('augmentation.hsv_h', 0.015),
        'hsv_s': config.get('augmentation.hsv_s', 0.7),
        'hsv_v': config.get('augmentation.hsv_v', 0.4),
        'degrees': config.get('augmentation.degrees', 15.0),
        'translate': config.get('augmentation.translate', 0.1),
        'scale': config.get('augmentation.scale', 0.5),
        'shear': config.get('augmentation.shear', 10.0),
        'flipud': config.get('augmentation.flipud', 0.0),
        'fliplr': config.get('augmentation.fliplr', 0.5),
        'mosaic': config.get('augmentation.mosaic', 1.0),
        
        # Hardware
        'device': args.device or config.get('hardware.device', ''),
        'workers': config.get('data.workers', 8),
        'amp': config.get('hardware.amp', True),
        
        # Logging
        'project': config.get('logging.project', 'water-meter-ai'),
        'name': config.get('logging.name', 'exp'),
        'exist_ok': config.get('logging.exist_ok', False),
        'save': config.get('logging.save', True),
        'save_period': config.get('logging.save_period', -1),
        
        # Validation
        'val': config.get('validation.val_interval', 1) > 0,
        'plots': config.get('validation.plots', True),
        
        # Misc
        'seed': config.get('seed', 42),
        'verbose': config.get('verbose', True),
    }
    
    # Log training configuration
    logger.info("Training Configuration:")
    for key, value in train_args.items():
        logger.info(f"  {key}: {value}")
    
    # Start training
    logger.info("Starting training...")
    try:
        results = model.train(**train_args)
        logger.info("Training completed successfully!")
        logger.info(f"Best weights saved at: {model.trainer.best}")
        
        # Log final metrics
        logger.info("Final Metrics:")
        logger.info(f"  mAP50: {results.results_dict.get('metrics/mAP50(B)', 'N/A')}")
        logger.info(f"  mAP50-95: {results.results_dict.get('metrics/mAP50-95(B)', 'N/A')}")
        
    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        raise
    
    logger.info("="*80)
    logger.info("Training pipeline finished!")
    logger.info("="*80)


if __name__ == '__main__':
    main()
