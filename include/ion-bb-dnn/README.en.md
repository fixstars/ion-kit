# Building Block Reference Manual
<!-- ion-bb-dnn -->

## Object Detection

Recognizes objects by DNN for input input, and draws bounding boxes for output.
The following models are supported.

- [SSD MobileNet v2](https://arxiv.org/pdf/1801.04381.pdf)

Execution on the device is done by using one of the following runtime libraries.

1. TensorFlow Lite
2. ONNXRuntime (CPU Provier, CUDA Provider)

As shown below, the model file itself uses a model that has been converted for each platform in advance.

- ONNXRuntime: [ssd_mobilenet_v2_coco_2018_03_29.onnx](http://ion-archives.s3-us-west-2.amazonaws.com/models/ssd_mobilenet_v2_coco_2018_03_29.onnx)
- TensorFlow Lite: [ssd_mobilenet_v2_coco_quant_postprocess.tflite](http://ion-archives.s3-us-west-2.amazonaws.com/models/ssd_mobilenet_v2_coco_quant_postprocess.tflite)
- TensorFlow Lite (EdgeTPU): [ssd_mobilenet_v2_coco_quant_postprocess_edgetpu.tflite](http://ion-archives.s3-us-west-2.amazonaws.com/models/ssd_mobilenet_v2_coco_quant_postprocess_edgetpu.tflite)

The training dataset is MS COCO, and the detection targets are the following 80 classes of objects.

This specification is subject to change in the future.

```
"person",
"bicycle",
"car",
"motorcycle",
"airplane",
"bus",
"train",
"truck",
"boat",
"traffic light",
"fire hydrant",
"stop sign",
"parking meter",
"bench",
"bird",
"cat",
"dog",
"horse",
"sheep",
"cow",
"elephant",
"bear",
"zebra",
"giraffe",
"backpack",
"umbrella",
"handbag",
"tie",
"suitcase",
"frisbee",
"skis",
"snowboard",
"sports ball",
"kite",
"baseball bat",
"baseball glove",
"skateboard",
"surfboard",
"tennis racket",
"bottle",
"wine glass",
"cup",
"fork",
"knife",
"spoon",
"bowl",
"banana",
"apple",
"sandwich",
"orange",
"broccoli",
"carrot",
"hot dog",
"pizza",
"donut",
"cake",
"chair",
"couch",
"potted plant",
"bed",
"dining table",
"toilet",
"tv",
"laptop",
"mouse",
"remote",
"keyboard",
"cell phone",
"microwave",
"oven",
"toaster",
"sink",
"refrigerator",
"book",
"clock",
"vase",
"scissors",
"teddy bear",
"hair drier",
"toothbrush",
```

### Input

- input
  - Element type: float32
  - Dimension: 3
  - Format
    - Shape: `[width: 416, height: 416, chennle: 3]` (column-major notation, W is innermost)
    - color space: RGB (column-major notation, R is innermost)
    - value range: standardized (divide by 255) values in [0..1]

### Output

- output
  - Element type: uint8
  - Dimension: 3
  - Format: RGB data with CHW as dimension and [0..255] as value range

### Parameter

- model_base_url
  - The base URL path where the trained model files to use are located (do not change this unless you have a reason to)
  - Element type: string
- cache_root
  - Path to the directory where the cache file used by ONNXRuntime to use the TensorRT provider is stored or loaded.
  - Element type: string
- width
  - Width of the image (pixel)
  - Element type: int32
- height
  - Height of the image (pixel)
  - Element type: int32

### Performances
| Model                     | Input Resolution | mAP (fp16) | mAP (int8) | Raspberry Pi 4 int8 (ms) | Jetson Nano fp16 (ms) | Jetson TX2 fp16 (ms) | Jetson Xavier NX fp16 (ms) | Coral Dev Board int8 (ms) | Ultra96 int8+fp32 (ms) | KV260 int8+fp32 (ms) |
| :------------------------ | ---------------: | ---------: | ---------: | -----------------------: | --------------------: | -------------------: | -------------------------: | ------------------------: | ---------------------: | -------------------: |
| SSD MobileNet V1          |          300x300 |       17.9 |       17.5 |                   112.63 |                 26.37 |                13.95 |                       4.23 |                     25.13 |                  33.70 |                19.57 |
| SSD MobileNet V2          |          300x300 |       18.6 |       18.0 |                   218.52 |                 34.67 |                15.91 |                       6.04 |                     54.04 |                  56.13 |                27.12 |
| SSDLite MobileNet V2      |          300x300 |       18.0 |       16.7 |                    95.10 |                 30.12 |                14.38 |                       5.53 |                     26.13 |                  35.50 |                21.46 |
| SSD MobileNet V3 Small    |          320x320 |       13.5 |        0.0 |                    35.68 |                 21.86 |                12.40 |                       7.30 |                           |                  85.96 |               560.62 |
| SSD MobileNet V3 Large    |          320x320 |       22.3 |        1.5 |                    80.21 |                 33.41 |                16.25 |                       7.60 |                           |                 249.81 |              1928.32 |
| SSDLite MobileDet CPU     |          320x320 |       24.5 |        0.1 |                    96.37 |                 40.28 |                18.77 |                      10.40 |                           |                        |                      |
| SSDLite MobileDet EdgeTPU |          320x320 |       26.3 |       26.1 |                   126.66 |                 35.30 |                16.21 |                       7.31 |                     43.36 |                        |                26.13 |
| SSDLite MobileDet DSP     |          320x320 |       30.9 |       13.2 |                          |                 46.44 |                20.06 |                       9.00 |                     50.58 |                  60.58 |                26.03 |
| SSDLite MobileDet GPU     |          320x320 |       22.5 |       22.4 |                          |                       |                19.21 |                       8.01 |                     46.08 |                  75.40 |                34.08 |
| YOLO v2                   |          608x608 |       24.6 |            |                          |                180.49 |                68.23 |                      27.51 |                           |                        |              2551.60 |
| tiny-YOLO v3              |          416x416 |    1.2[^1] |            |                          |                 25.93 |                11.34 |                       6.44 |                           |                  44.35 |                19.58 |
| YOLO v3                   |          416x416 |       29.5 |            |                          |                200.43 |                80.71 |                      34.97 |                           |                        |               144.96 |

## TLT Object Detection SSD

For input input, it performs object recognition using DNN optimized for NVIDIA GPUs, and draws a bounding box for output.
The following models are supported.

- [Pretrained Object Detection (Resnet18 SSD)](https://ngc.nvidia.com/catalog/models/nvidia:tlt_pretrained_object_detection)

Execution on the device is done using the TensorRT environment.
If the TensorRT runtime is not present in the runtime environment, it will not work properly.

The model used internally takes a color image of 1248 pixels in width and 384 pixels in height as input for inference.
If the input shape of the BB is different from that of the internal model, it will be scaled and padded before being inferred.

Please refer to the above link for detailed model specifications.

### Input

- input
  - Element type: float32
  - Dimension: 3
  - Format.
    - Shape: `[panel: 3, width: *, height: *]` (column-major notation, C is innermost)
    - Colorspace: RGB
    - value range: standardized (divide by 255) values in [0..1.0]

### Output

- output
  - Element type: uint8
  - Dimension: 3
  - Format:
    - Shape: `[panel: 3, width: *, height: *]` (column-major notation, C is innermost)
    - Colorspace: RGB
    - value range: standardized (divide by 255) values in [0..1.0].

### Parameter

- model_base_url
  - The base URL path where the trained model files to use are located (do not change this unless you have a reason to)
  - Element type: string
- cache_root
  - Path of the directory where the TensorRT model file cache is stored or loaded.
  - Element type: string

### Performances
| Model                     | Input Resolution | mAP (fp16) | mAP (int8) | Raspberry Pi 4 int8 (ms) | Jetson Nano fp16 (ms) | Jetson TX2 fp16 (ms) | Jetson Xavier NX fp16 (ms) | Coral Dev Board int8 (ms) | Ultra96 int8+fp32 (ms) | KV260 int8+fp32 (ms) |
| :------------------------ | ---------------: | ---------: | ---------: | -----------------------: | --------------------: | -------------------: | -------------------------: | ------------------------: | ---------------------: | -------------------: |
| SSD MobileNet V1          |          300x300 |       17.9 |       17.5 |                   112.63 |                 26.37 |                13.95 |                       4.23 |                     25.13 |                  33.70 |                19.57 |
| SSD MobileNet V2          |          300x300 |       18.6 |       18.0 |                   218.52 |                 34.67 |                15.91 |                       6.04 |                     54.04 |                  56.13 |                27.12 |
| SSDLite MobileNet V2      |          300x300 |       18.0 |       16.7 |                    95.10 |                 30.12 |                14.38 |                       5.53 |                     26.13 |                  35.50 |                21.46 |
| SSD MobileNet V3 Small    |          320x320 |       13.5 |        0.0 |                    35.68 |                 21.86 |                12.40 |                       7.30 |                           |                  85.96 |               560.62 |
| SSD MobileNet V3 Large    |          320x320 |       22.3 |        1.5 |                    80.21 |                 33.41 |                16.25 |                       7.60 |                           |                 249.81 |              1928.32 |
| SSDLite MobileDet CPU     |          320x320 |       24.5 |        0.1 |                    96.37 |                 40.28 |                18.77 |                      10.40 |                           |                        |                      |
| SSDLite MobileDet EdgeTPU |          320x320 |       26.3 |       26.1 |                   126.66 |                 35.30 |                16.21 |                       7.31 |                     43.36 |                        |                26.13 |
| SSDLite MobileDet DSP     |          320x320 |       30.9 |       13.2 |                          |                 46.44 |                20.06 |                       9.00 |                     50.58 |                  60.58 |                26.03 |
| SSDLite MobileDet GPU     |          320x320 |       22.5 |       22.4 |                          |                       |                19.21 |                       8.01 |                     46.08 |                  75.40 |                34.08 |
| YOLO v2                   |          608x608 |       24.6 |            |                          |                180.49 |                68.23 |                      27.51 |                           |                        |              2551.60 |
| tiny-YOLO v3              |          416x416 |    1.2[^1] |            |                          |                 25.93 |                11.34 |                       6.44 |                           |                  44.35 |                19.58 |
| YOLO v3                   |          416x416 |       29.5 |            |                          |                200.43 |                80.71 |                      34.97 |                           |                        |               144.96 |

## TLT PeopleNet

For input input, it recognizes three classes of objects (human, face, and bag) using DNN optimized for NVIDIA GPUs, and draws a bounding box for output.
The following models are supported.

- [PeopleNet (Resnet18 DetectNet)](https://ngc.nvidia.com/catalog/models/nvidia:tlt_peoplenet)

Execution on the device is done using the TensorRT environment.
If the TensorRT runtime is not present in the runtime environment, it will not work properly.

The model used internally performs inference using a color image of 960 pixels in width and 544 pixels in height as input.
If the input shape of the BB is different from that of the internal model, it will be scaled and padded before being inferred.

Please refer to the above link for detailed model specifications.

### Input

- input
  - Element type: float32
  - Dimension: 3
  - Format.
    - Shape: `[panel: 3, width: *, height: *]` (column-major notation, C is innermost)
    - Colorspace: RGB
    - value range: standardized (divide by 255) values in [0..1.0]

### Output

- output
  - Element type: uint8
  - Dimension: 3
  - Format:
    - Shape: `[panel: 3, width: *, height: *]` (column-major notation, C is innermost)
    - Colorspace: RGB
    - value range: standardized (divide by 255) values in [0..1.0].

### Parameter

- model_base_url
  - The base URL path where the trained model files to use are located (do not change this unless you have a reason to)
  - Element type: string
- cache_root
  - Path of the directory where the TensorRT model file cache is stored or loaded.
  - Element type: string


## TLT PeopleNet metadata version

It performs basically the same kind of inference as TLT PeopleNet in the previous section, but outputs metadata including recognition information instead of images with bounding boxes drawn on them.

The output data is an ASCII string formatted in JSON format and converted to a byte array.
If the buffer size of the output data is not enough, the program execution will be aborted with an error message.

### Input

- input
  - Element type: float32
  - Dimension: 3
  - Format.
    - Shape: `[panel: 3, width: *, height: *]` (column-major notation, C is innermost)
    - Colorspace: RGB
    - value range: standardized (divide by 255) values in [0..1.0]

### Output

- output
  - Element type: uint8
  - Dimension: 1
  - Format:
    - Shape: `[N: *]` (can be changed by parameter `output_size`)
    - Schema: DetectionBox Array

DetectionBox Array:
```
{
  "$schema": "http://json-schema.org/draft-04/schema#",
  "type": "array",
  "items": [
    {
      "type": "object",
      "properties": {
        "c": {
          "type": "number"
        },
        "id": {
          "type": "integer"
        },
        "x1": {
          "type": "number"
        },
        "x2": {
          "type": "number"
        },
        "y1": {
          "type": "number"
        },
        "y2": {
          "type": "number"
        }
      },
      "required": [
        "c",
        "id",
        "x1",
        "x2",
        "y1",
        "y2"
      ]
    }
  ]
}
```

### Parameter

- model_base_url
  - The base URL path where the trained model files to use are located (do not change this unless you have a reason to)
  - Element type: string
- cache_root
  - Path of the directory where the TensorRT model file cache is stored or loaded.
  - Element type: string
- width
  - Width of the image (pixel)
  - Element type: int32
- height
  - Height of the image (pixel)
  - Element type: int32
- output_size
  - Size of the output buffer
  - Element type: int32


## ClassifyGender

DetectionBox Cuts out a rectangular face image from an image using Array format recognition information, and determines the gender.

The output data is an ASCII string formatted in JSON format and converted to a byte array.
If the buffer size of the output data is not enough, the program execution will be aborted with an error message.

### Input

- input_img
  - Element type: float32
  - Dimension: 3
  - format
    - Shape: `[panel: 3, width: *, height: *]` (column-major notation, C is innermost)
    - Colorspace: RGB
    - range: standardized (divide by 255) values in [0..1.0]
- input_md
  - element type: uint8
  - Dimension: 1
  - Format:
    - Shape: `[N: *]` (can be changed by parameter `input_md_size`)
    - Schema: DetectionBox Array

### Output

- output
  - Element type: uint8
  - Dimension: 1
  - Format:
    - Shape: `[N: *]` (can be changed by parameter `output_size`)
    - Schema: ClassifyResult

ClassifyResult
```
{
  "$schema": "http://json-schema.org/draft-04/schema#",
  "type": "object",
  "properties": {
    "Female": {
      "type": "integer"
    },
    "Male": {
      "type": "integer"
    }
  },
  "required": [
    "Female",
    "Male"
  ]
}
```

### Parameter

- model_base_url
  - The base URL path where the trained model files to use are located (do not change this unless you have a reason to)
  - Element type: string
- cache_root
  - Path of the directory where the TensorRT model file cache is stored or loaded.
  - Element type: string
- width
  - Width of the image (pixel)
  - Element type: int32
- height
  - Height of the image (pixel)
  - Element type: int32
- input_md_size
  - Size of the input metadata buffer
  - Element type: int32
- output_size
  - Size of the output buffer
  - element type: int32

## JSONDictAverageRegulator

For data input in dictionary format, this function outputs the total value within the interval for each specified number of seconds.
Valid output is only for each second, and null is set for all other output.

### Input

- input
  - Element type: uint8
  - Dimension: 1
  - Format:
    - Shape: `[N: *]` (can be changed by parameter `input_md_size`)
    - Schema: NumberDict

NumberDict:
```
{
  "$schema": "http://json-schema.org/draft-04/schema#",
  "type": ["object", "null"],
  "additionalProperties": {
    "anyOf": [
      {"type": "number"},
    ],
  },
}
```

### Output

- output
  - Element type: uint8
  - Dimension: 1
  - Format:
    - Shape: `[N: *]` (can be changed by parameter `output_size`)
    - Schema: NumberDict

### Parameter

- io_md_size
  - Size of the input/output metadata buffer
  - Element type: int32
- period_in_sec
  - Interval between integration and output (in seconds)
  - Element type: int32

## IFTTT WebHook Uploader

Sends input data as a message to the specified IFTTT WebHook URL.
If the input data is null, it will not be sent.

### Input

- input_md
  - Element type: uint8
  - Dimension: 1
  - Format:
    - Shape: `[N: *]` (can be changed by parameter `input_md_size`)
    - Schema: arbitrary JSON data

### Output

- output
  - Element type: int32
  - Dimension: 0
  - Format: unconstrained

### Parameter

- input_md_size
  - Size of input/output metadata buffer
  - Element type: int32
- ifttt_webhook_url
  - IFTTT WebHook URL
  - Element type: string
