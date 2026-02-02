"""
Test script to verify all module imports work correctly
"""
import sys
import traceback

def test_import(module_path, class_name):
    """Test importing a specific class from a module"""
    try:
        exec(f"from {module_path} import {class_name}")
        print(f"✅ {class_name} from {module_path} - OK")
        return True
    except Exception as e:
        print(f"❌ {class_name} from {module_path} - FAILED")
        print(f"   Error: {type(e).__name__}: {e}")
        traceback.print_exc()
        return False

def main():
    print("=" * 60)
    print("Testing Module Imports")
    print("=" * 60)
    
    tests = [
        ("src.inference", "DetectionParser"),
        ("src.visualization", "FrameAnnotator"),
        ("src.utils", "VideoProcessor"),
        ("src.core", "WaterMeterSystem"),
    ]
    
    results = []
    for module_path, class_name in tests:
        print(f"\nTesting: {module_path}.{class_name}")
        success = test_import(module_path, class_name)
        results.append((module_path, class_name, success))
    
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    
    passed = sum(1 for _, _, success in results if success)
    total = len(results)
    
    for module_path, class_name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{status}: {module_path}.{class_name}")
    
    print(f"\nTotal: {passed}/{total} passed")
    
    if passed == total:
        print("\n🎉 All imports successful! Ready for cleanup.")
        return 0
    else:
        print(f"\n⚠️ {total - passed} import(s) failed. Need to fix before cleanup.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
