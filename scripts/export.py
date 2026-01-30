"""
Export script for Water Meter AI
Converts trained model to TFLite format for mobile deployment

Usage:
    python scripts/export.py --weights runs/train/exp/weights/best.pt
"""
import argparse
from pathlib import Path

from ultralytics import YOLO

# Add project root to path
import sys
sys.path.append(str(Path(__file__).parent.parent))

from src.utils.config import Config
from src.utils.logger import setup_logger, get_logger


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Export Water Meter Model to TFLite')
    parser.add_argument(
        '--weights',
        type=str,
        required=True,
        help='Path to trained model weights (.pt file)'
    )
    parser.add_argument(
        '--config',
        type=str,
        default='configs/export_config.yaml',
        help='Path to export configuration file'
    )
    parser.add_argument(
        '--format',
        type=str,
        default='tflite',
        choices=['onnx', 'tflite', 'both'],
        help='Export format'
    )
    parser.add_argument(
        '--int8',
        action='store_true',
        help='Use INT8 quantization'
    )
    return parser.parse_args()


def main():
    """Main export function"""
    # Parse arguments
    args = parse_args()
    
    # Setup logging
    setup_logger(log_file='logs/export.log', level='INFO')
    logger = get_logger(__name__)
    
    logger.info("="*80)
    logger.info("Water Meter AI - Model Export Pipeline")
    logger.info("Author: Arsenius Purbandono")
    logger.info("="*80)
    
    # Load configuration
    logger.info(f"Loading configuration from: {args.config}")
    config = Config(args.config)
    
    # Check if weights exist
    weights_path = Path(args.weights)
    if not weights_path.exists():
        raise FileNotFoundError(f"Weights not found: {weights_path}")
    
    logger.info(f"Loading model from: {weights_path}")
    model = YOLO(str(weights_path))
    
    # Prepare export arguments
    export_args = {
        'imgsz': config.get('export.input.image_size', [512, 512]),
        'optimize': config.get('export.optimization.optimize_for_mobile', True),
        'simplify': config.get('export.optimization.simplify', True),
        'int8': args.int8 or config.get('export.quantization.enabled', True),
        'dynamic': config.get('export.input.dynamic_batch', False),
    }
    
    # Export to ONNX first (intermediate format)
    if args.format in ['onnx', 'both']:
        logger.info("Exporting to ONNX format...")
        try:
            onnx_path = model.export(
                format='onnx',
                **export_args
            )
            logger.info(f"ONNX export successful: {onnx_path}")
        except Exception as e:
            logger.error(f"ONNX export failed: {str(e)}")
            if args.format == 'onnx':
                raise
    
    # Export to TFLite (target format)
    if args.format in ['tflite', 'both']:
        logger.info("Exporting to TFLite format...")
        logger.info(f"  INT8 Quantization: {export_args['int8']}")
        
        try:
            tflite_path = model.export(
                format='tflite',
                **export_args
            )
            logger.info(f"TFLite export successful: {tflite_path}")
            
            # Get model size
            tflite_file = Path(tflite_path)
            if tflite_file.exists():
                size_mb = tflite_file.stat().st_size / (1024 * 1024)
                logger.info(f"Model size: {size_mb:.2f} MB")
                
                # Check against target
                target_size = config.get('targets.model_size_mb', 10)
                if size_mb > target_size:
                    logger.warning(f"Model size ({size_mb:.2f} MB) exceeds target ({target_size} MB)")
                else:
                    logger.info(f"Model size meets target (<{target_size} MB) ✓")
            
        except Exception as e:
            logger.error(f"TFLite export failed: {str(e)}")
            raise
    
    logger.info("="*80)
    logger.info("Export pipeline finished!")
    logger.info("="*80)
    logger.info("\nNext steps:")
    logger.info("1. Test the exported model with scripts/inference.py")
    logger.info("2. Integrate .tflite file into Flutter app")
    logger.info("3. Test on real mobile devices")


if __name__ == '__main__':
    main()
