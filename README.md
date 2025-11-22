# DeepStream Triton YOLO

Run YOLO models (YOLOv7, YOLOv8, YOLOv9) with NVIDIA DeepStream 7.0 and Triton Inference Server using gRPC.

## Author
**Shiva** (shivashankarar@ai4mtech.com)

## Features
- Support for YOLOv7, YOLOv8, YOLOv9 detection models
- End-to-end TensorRT inference with EfficientNMS
- Triton Inference Server integration via gRPC
- **Standalone mode** - Run without separate Triton server using embedded backend
- Custom bounding box parser for YOLO NMS output

## Requirements
- NVIDIA DeepStream SDK 7.0
- Triton Inference Server 2.x
- TensorRT 8.x+
- CUDA 12.x
- Docker/Podman

## Directory Structure
```
deepstream-triton-yolo/
├── deepstream_yolo_det.txt       # Main DeepStream config for detection
├── pgie_yolov8_standalone.txt    # Standalone config (no Triton server needed)
├── labels.txt                    # COCO class labels
├── tools/                        # Export scripts
│   └── export_yolov9.py          # YOLOv9 export with fixed input support
├── nvdsinfer_yolo/               # Custom parser library
│   ├── nvdsinfer_yolo.cpp
│   ├── Makefile
│   └── libnvds_infer_yolo.so
├── triton-grpc/                  # DeepStream pgie configs
│   ├── yolov7/
│   ├── yolov8/
│   └── yolov9/
└── triton_models/                # Triton model repository
    ├── yolov7/
    │   ├── config.pbtxt
    │   └── 1/model.plan          # Place your engine here
    ├── yolov8/
    │   ├── config.pbtxt
    │   └── 1/model.plan
    └── yolov9/
        ├── config.pbtxt
        └── 1/model.plan
```

## Quick Start

### 1. Export YOLO Model to ONNX with End-to-End NMS

**YOLOv7:**
```bash
git clone https://github.com/WongKinYiu/yolov7
cd yolov7
python export.py --weights yolov7.pt --grid --end2end --simplify \
    --topk-all 100 --iou-thres 0.65 --conf-thres 0.35 \
    --img-size 640 640
```

**YOLOv8 (using triple-Mu/YOLOv8-TensorRT):**
```bash
git clone https://github.com/triple-Mu/YOLOv8-TensorRT
cd YOLOv8-TensorRT
# Copy the modified export script from this repo
cp /path/to/deepstream-triton-yolo/export-det.py .
python3 export-det.py --weights yolov8s.pt --iou-thres 0.65 \
    --conf-thres 0.25 --topk 100 --opset 11 --sim \
    --input-shape 1 3 640 640 --device cuda:0 --dynamic
```

**YOLOv9:**
```bash
git clone https://github.com/WongKinYiu/yolov9
cd yolov9
# Copy the modified export script from this repo
cp /path/to/deepstream-triton-yolo/tools/export_yolov9.py .
python export_yolov9.py --weights yolov9-c.pt --include onnx_end2end \
    --topk-all 100 --iou-thres 0.65 --conf-thres 0.25 \
    --imgsz 640 640 --device 0 --fixed-input
```

### 2. Convert ONNX to TensorRT Engine
```bash
trtexec --onnx=yolov7.onnx \
    --minShapes=images:1x3x640x640 \
    --optShapes=images:8x3x640x640 \
    --maxShapes=images:8x3x640x640 \
    --fp16 --workspace=4096 \
    --saveEngine=yolov7-fp16-1x8x8.engine
```

### 3. Setup Triton Model Repository
Copy your TensorRT engine to the model directory:
```bash
cp yolov7.engine triton_models/yolov7/1/model.plan
cp yolov8.engine triton_models/yolov8/1/model.plan
cp yolov9.engine triton_models/yolov9/1/model.plan
```

### 4. Start Triton Server (gRPC Mode)
```bash
tritonserver --model-repository=./triton_models
```

Or with Docker:
```bash
docker run --gpus all --rm -p 8000:8000 -p 8001:8001 -p 8002:8002 \
  -v $(pwd)/triton_models:/models \
  nvcr.io/nvidia/tritonserver:24.08-py3 \
  tritonserver --model-repository=/models
```

### 4b. Standalone Mode (No Triton Server Required)
Use `pgie_yolov8_standalone.txt` to run inference directly without starting a separate Triton server. This config uses the embedded Triton backend via `model_repo` instead of gRPC:

```bash
# Edit deepstream_yolo_det.txt and set:
config-file=./pgie_yolov8_standalone.txt

# Run directly - no tritonserver needed!
deepstream-app -c deepstream_yolo_det.txt
```

### 5. Build Custom Parser Library
```bash
cd nvdsinfer_yolo
make all && make install 
```

### 6. Run DeepStream
```bash
deepstream-app -c deepstream_yolo_det.txt
```

## Configuration

### Selecting YOLO Model
Edit `deepstream_yolo_det.txt` and uncomment the desired model config:

```ini
[primary-gie]
# YOLOv7
config-file=./triton-grpc/yolov7/pgie_yolov7_det.txt

# YOLOv8
#config-file=./triton-grpc/yolov8/pgie_yolov8n_det.txt

# YOLOv9
#config-file=./triton-grpc/yolov9/pgie_yolov9-c_det.txt
```

### Triton gRPC Settings
Edit the pgie config file to set Triton server address:
```
grpc {
  url: "127.0.0.1:8001"
  enable_cuda_buffer_sharing: true
}
```

## Output Format
The end-to-end NMS model outputs:
- `num_dets`: Number of detections [batch_size, 1]
- `det_boxes`: Bounding boxes [batch_size, topk, 4] (x1, y1, x2, y2)
- `det_scores`: Confidence scores [batch_size, topk]
- `det_classes`: Class indices [batch_size, topk]

## Acknowledgements & Credits

This project builds upon the excellent work of the following developers and projects. Special thanks to:

### Core Dependencies
- **[Levi Pereira](https://github.com/levipereira)** - This project is heavily based on his work:
  - [deepstream-triton-server-yolo](https://github.com/levipereira/deepstream-triton-server-yolo) - Original DeepStream + Triton + YOLO implementation
  - [nvdsinfer_yolo](https://github.com/levipereira/nvdsinfer_yolo) - Custom bounding box parser for YOLO NMS output

### YOLO Models
- **[WongKinYiu](https://github.com/WongKinYiu)** - Creator of YOLOv7 and YOLOv9:
  - [YOLOv7](https://github.com/WongKinYiu/yolov7)
  - [YOLOv9](https://github.com/WongKinYiu/yolov9)
- **[Ultralytics](https://github.com/ultralytics)** - [YOLOv8](https://github.com/ultralytics/ultralytics)
- **[triple-Mu](https://github.com/triple-Mu)** - [YOLOv8-TensorRT](https://github.com/triple-Mu/YOLOv8-TensorRT) export scripts

### NVIDIA
- [NVIDIA DeepStream SDK](https://developer.nvidia.com/deepstream-sdk)
- [Triton Inference Server](https://github.com/triton-inference-server/server)
- [TensorRT](https://developer.nvidia.com/tensorrt)

## License
Apache-2.0 License

This project respects and follows the licensing terms of all dependencies listed above.
