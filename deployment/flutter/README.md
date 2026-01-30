# Flutter Integration Guide

## Prerequisites

- Flutter 3.0+
- Android Studio / Xcode
- TFLite model file (`.tflite`)

## Step 1: Add Dependencies

Add to `pubspec.yaml`:

```yaml
dependencies:
  tflite_flutter: ^0.10.0
  image: ^4.0.0
  camera: ^0.10.0
```

## Step 2: Add Model to Assets

1. Create `assets/models/` directory
2. Copy `best.tflite` to `assets/models/water_meter.tflite`
3. Update `pubspec.yaml`:

```yaml
flutter:
  assets:
    - assets/models/water_meter.tflite
```

## Step 3: Create Detector Class

```dart
import 'package:tflite_flutter/tflite_flutter.dart';
import 'package:image/image.dart' as img;

class WaterMeterDetector {
  Interpreter? _interpreter;

  // Model configuration
  static const int INPUT_SIZE = 512;
  static const int NUM_CLASSES = 10; // 0-9 digits

  Future<void> loadModel() async {
    try {
      _interpreter = await Interpreter.fromAsset(
        'assets/models/water_meter.tflite',
        options: InterpreterOptions()
          ..threads = 4
          ..useNnApiForAndroid = true,
      );
      print('✅ Model loaded successfully');
    } catch (e) {
      print('❌ Error loading model: $e');
    }
  }

  Future<List<Detection>> detect(img.Image image) async {
    if (_interpreter == null) {
      throw Exception('Model not loaded');
    }

    // Preprocess image
    var input = preprocessImage(image);

    // Prepare output buffers
    var output = List.generate(1, (_) =>
      List.generate(300, (_) => List.filled(6, 0.0))
    );

    // Run inference
    _interpreter!.run(input, output);

    // Post-process results
    return postProcess(output);
  }

  List<List<List<List<double>>>> preprocessImage(img.Image image) {
    // Resize to 512x512
    var resized = img.copyResize(image,
      width: INPUT_SIZE,
      height: INPUT_SIZE
    );

    // Normalize to [0, 1]
    var input = List.generate(1, (_) =>
      List.generate(INPUT_SIZE, (y) =>
        List.generate(INPUT_SIZE, (x) {
          var pixel = resized.getPixel(x, y);
          return [
            img.getRed(pixel) / 255.0,
            img.getGreen(pixel) / 255.0,
            img.getBlue(pixel) / 255.0,
          ];
        })
      )
    );

    return input;
  }

  List<Detection> postProcess(List output) {
    List<Detection> detections = [];

    // Filter by confidence threshold
    for (var det in output[0]) {
      double confidence = det[4];
      if (confidence > 0.5) { // Confidence threshold
        detections.add(Detection(
          bbox: [det[0], det[1], det[2], det[3]],
          classId: det[5].toInt(),
          confidence: confidence,
        ));
      }
    }

    return detections;
  }

  void dispose() {
    _interpreter?.close();
  }
}

class Detection {
  final List<double> bbox;
  final int classId;
  final double confidence;

  Detection({
    required this.bbox,
    required this.classId,
    required this.confidence,
  });
}
```

## Step 4: Use in UI

```dart
import 'package:camera/camera.dart';

class WaterMeterScreen extends StatefulWidget {
  @override
  _WaterMeterScreenState createState() => _WaterMeterScreenState();
}

class _WaterMeterScreenState extends State<WaterMeterScreen> {
  WaterMeterDetector? _detector;
  CameraController? _cameraController;
  List<Detection>? _detections;
  bool _isDetecting = false;

  @override
  void initState() {
    super.initState();
    _initializeDetector();
    _initializeCamera();
  }

  Future<void> _initializeDetector() async {
    _detector = WaterMeterDetector();
    await _detector!.loadModel();
  }

  Future<void> _initializeCamera() async {
    final cameras = await availableCameras();
    _cameraController = CameraController(
      cameras.first,
      ResolutionPreset.high,
    );
    await _cameraController!.initialize();
    setState(() {});

    // Start continuous detection
    _startDetection();
  }

  void _startDetection() {
    _cameraController!.startImageStream((CameraImage cameraImage) async {
      if (_isDetecting) return;
      _isDetecting = true;

      try {
        // Convert CameraImage to img.Image
        var image = convertCameraImage(cameraImage);

        // Run detection
        var detections = await _detector!.detect(image);

        setState(() {
          _detections = detections;
        });
      } catch (e) {
        print('Detection error: $e');
      }

      _isDetecting = false;
    });
  }

  @override
  Widget build(BuildContext context) {
    if (_cameraController == null || !_cameraController!.value.isInitialized) {
      return Center(child: CircularProgressIndicator());
    }

    return Scaffold(
      appBar: AppBar(title: Text('Water Meter Scanner')),
      body: Stack(
        children: [
          // Camera preview
          CameraPreview(_cameraController!),

          // Detections overlay
          if (_detections != null)
            CustomPaint(
              painter: DetectionPainter(_detections!),
              child: Container(),
            ),

          // Display reading
          Positioned(
            bottom: 20,
            left: 20,
            right: 20,
            child: Card(
              child: Padding(
                padding: EdgeInsets.all(16),
                child: Text(
                  'Reading: ${_getReading()}',
                  style: TextStyle(fontSize: 24, fontWeight: FontWeight.bold),
                  textAlign: TextAlign.center,
                ),
              ),
            ),
          ),
        ],
      ),
    );
  }

  String _getReading() {
    if (_detections == null || _detections!.isEmpty) {
      return '--';
    }

    // Sort detections by x-coordinate (left to right)
    var sorted = List<Detection>.from(_detections!)
      ..sort((a, b) => a.bbox[0].compareTo(b.bbox[0]));

    // Concatenate digits
    return sorted.map((d) => d.classId.toString()).join();
  }

  @override
  void dispose() {
    _cameraController?.dispose();
    _detector?.dispose();
    super.dispose();
  }
}

class DetectionPainter extends CustomPainter {
  final List<Detection> detections;

  DetectionPainter(this.detections);

  @override
  void paint(Canvas canvas, Size size) {
    for (var detection in detections) {
      // Draw bounding box
      var paint = Paint()
        ..color = Colors.green
        ..strokeWidth = 2.0
        ..style = PaintingStyle.stroke;

      var rect = Rect.fromLTWH(
        detection.bbox[0] * size.width,
        detection.bbox[1] * size.height,
        detection.bbox[2] * size.width,
        detection.bbox[3] * size.height,
      );

      canvas.drawRect(rect, paint);

      // Draw label
      var textPainter = TextPainter(
        text: TextSpan(
          text: '${detection.classId} (${(detection.confidence * 100).toInt()}%)',
          style: TextStyle(color: Colors.white, fontSize: 14),
        ),
        textDirection: TextDirection.ltr,
      );
      textPainter.layout();
      textPainter.paint(canvas, Offset(rect.left, rect.top - 20));
    }
  }

  @override
  bool shouldRepaint(covariant CustomPainter oldDelegate) => true;
}
```

## Step 5: Test & Deploy

### Testing Checklist

- [ ] Model loads successfully
- [ ] Camera preview works
- [ ] Detections displayed correctly
- [ ] Reading extracted accurately
- [ ] Performance >30 FPS on target device

### Optimization Tips

1. **Use GPU Delegate** (Android):

```dart
InterpreterOptions()
  ..addDelegate(GpuDelegateV2())
```

2. **Reduce Input Size** (if needed):

```dart
static const int INPUT_SIZE = 416; // Instead of 512
```

3. **Debounce Detection**:

```dart
Timer? _debounce;
if (_debounce?.isActive ?? false) _debounce!.cancel();
_debounce = Timer(Duration(milliseconds: 200), () {
  // Run detection
});
```

## Performance Targets

- **Inference Time:** <100ms on mid-range devices
- **FPS:** >20 FPS for real-time experience
- **Accuracy:** >90% correct digit recognition

---

**For issues or questions, contact: Arsenius Purbandono**
