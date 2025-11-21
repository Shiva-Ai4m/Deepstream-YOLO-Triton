# DeepStream Triton YOLO

Run YOLO models (YOLOv7, YOLOv8, YOLOv9) with NVIDIA DeepStream 7.0 and Triton Inference Server using gRPC.

## Author
**Shiva** (shivashankarar@ai4mtech.com)

## Features
- Support for YOLOv7, YOLOv8, YOLOv9 detection models
- Instance segmentation support (YOLOv7/v9 mask)
- End-to-end TensorRT inference with EfficientNMS
- Triton Inference Server integration via gRPC
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
├── deepstream_yolo_mask.txt      # DeepStream config for segmentation
├── labels.txt                    # COCO class labels
├── nvdsinfer_yolo/               # Custom parser library
│   ├── nvdsinfer_yolo.cpp
│   ├── Makefile
│   └── libnvds_infer_yolo.so
└── triton-grpc/
    ├── yolov7/
    │   ├── pgie_yolov7_det.txt
    │   └── nvdsinfer_yolo/
    ├── yolov8/
    │   └── pgie_yolov8*.txt
    └── yolov9/
        └── pgie_yolov9*.txt
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
python3 export-det.py --weights yolov8n.pt --iou-thres 0.65 \
    --conf-thres 0.25 --topk 100 --opset 11 --sim \
    --input-shape 1 3 640 640 --device cuda:0
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
```
triton_models/
└── yolov7/
    ├── config.pbtxt
    └── 1/
        └── model.plan
```

**config.pbtxt:**
```
name: "yolov7"
platform: "tensorrt_plan"
max_batch_size: 8

input [
  {
    name: "images"
    data_type: TYPE_FP32
    dims: [ 3, 640, 640 ]
  }
]

output [
  {
    name: "num_dets"
    data_type: TYPE_INT32
    dims: [ 1 ]
  },
  {
    name: "det_boxes"
    data_type: TYPE_FP32
    dims: [ 100, 4 ]
  },
  {
    name: "det_scores"
    data_type: TYPE_FP32
    dims: [ 100 ]
  },
  {
    name: "det_classes"
    data_type: TYPE_INT32
    dims: [ 100 ]
  }
]

instance_group [
  {
    count: 1
    kind: KIND_GPU
  }
]
```

### 4. Start Triton Server
```bash
tritonserver --model-repository=/path/to/triton_models
```

### 5. Build Custom Parser Library
```bash
cd nvdsinfer_yolo
make
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

## Acknowledgements
- [NVIDIA DeepStream SDK](https://developer.nvidia.com/deepstream-sdk)
- [Triton Inference Server](https://github.com/triton-inference-server/server)
- [YOLOv7](https://github.com/WongKinYiu/yolov7)
- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)
- [YOLOv8-TensorRT](https://github.com/triple-Mu/YOLOv8-TensorRT)
- [nvdsinfer_yolo](https://github.com/levipereira/nvdsinfer_yolo) - Custom parser by Levi Pereira

## License
Apache-2.0 License
