# Building Block リファレンスマニュアル
<!-- ion-bb-dnn -->

## Object Detection

入力 input に対してDNN によるオブジェクト認識を行い、バウンディングボックスを描画して出力します。
以下のモデルに対応しています。

- [SSD MobileNet v2](https://arxiv.org/pdf/1801.04381.pdf)

デバイス上での実行は、以下のランタイムライブラリのどちらかを利用して行われます。

1. TensorFlow Lite
2. ONNXRuntime (CPU Provier, CUDA Provider)

以下の通り、モデルファイル自体は事前にプラットフォームごとに変換されたモデルを利用します。

- ONNXRuntime: [ssd_mobilenet_v2_coco_2018_03_29.onnx](http://ion-archives.s3-us-west-2.amazonaws.com/models/ssd_mobilenet_v2_coco_2018_03_29.onnx)
- TensorFlow Lite: [ssd_mobilenet_v2_coco_quant_postprocess.tflite](http://ion-archives.s3-us-west-2.amazonaws.com/models/ssd_mobilenet_v2_coco_quant_postprocess.tflite)
- TensorFlow Lite (EdgeTPU): [ssd_mobilenet_v2_coco_quant_postprocess_edgetpu.tflite](http://ion-archives.s3-us-west-2.amazonaws.com/models/ssd_mobilenet_v2_coco_quant_postprocess_edgetpu.tflite)

学習データセットはMS COCOであり、検出対象は以下の80クラスのオブジェクトです。

この仕様は将来的に変更される可能性があります。

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

### 入力

- input
  - 要素型: float32
  - 次元: 3
  - フォーマット
    - シェイプ: `[width: 416, height: 416, chennle: 3]` (column-major表記、Wが最内)
    - 色空間: RGB (column-major表記、Rが最内)
    - 値範囲: 標準化 (除数255で除算) された [0..1] の値

### 出力

- output
  - 要素型: uint8
  - 次元: 3
  - フォーマット: 次元配置はCHW、値範囲は [0..255] で表される RGB データ

### パラメータ

- model_base_url
  - 使用する学習済みモデルファイルを配置しているベースURLパス (こちらは理由がない限り変更しないでください)
  - 要素型: string
- cache_root
  - ONNXRuntimeでTensorRTプロバイダを使用する際に用いるキャッシュファイルを保存 or ロードするディレクトリのパス
  - 要素型: string
- width
  - 画像の横幅 (pixel)
  - 要素型: int32
- height
  - 画像の縦幅 (pixel)
  - 要素型: int32

### パフォーマンス
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

入力 input に対して、NVIDIA GPU に最適化された DNN によるオブジェクト認識を行い、バウンディングボックスを描画して出力します。
以下のモデルに対応しています。

- [Pretrained Object Detection (Resnet18 SSD)](https://ngc.nvidia.com/catalog/models/nvidia:tlt_pretrained_object_detection)

デバイス上での実行は、TensorRT 環境を利用して行われます。
実行環境にTensorRTランタイムが存在しない場合、正しく動作しません。

内部で使用されているモデルは 横1248ピクセル, 縦384ピクセルのカラー画像を入力として推論を行います。
BBの入力シェイプが内部モデルのものと異なる場合には、拡大縮小およびパディングが行われたのち推論が行われます。

詳細なモデルの仕様は上記リンクを参照してください。

### 入力

- input
  - 要素型: float32
  - 次元: 3
  - フォーマット
    - シェイプ: `[chanel: 3, width: *, height: *]` (column-major表記、Cが最内)
    - 色空間: RGB
    - 値範囲: 標準化 (除数255で除算) された [0..1.0] の値

### 出力

- output
  - 要素型: uint8
  - 次元: 3
  - フォーマット:
    - シェイプ: `[chanel: 3, width: *, height: *]` (column-major表記、Cが最内)
    - 色空間: RGB
    - 値範囲: 標準化 (除数255で除算) された [0..1.0] の値

### パラメータ

- model_base_url
  - 使用する学習済みモデルファイルを配置しているベースURLパス (こちらは理由がない限り変更しないでください)
  - 要素型: string
- cache_root
  - TensorRT モデルファイルのキャッシュを保存 or ロードするディレクトリのパス
  - 要素型: string

### パフォーマンス
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

入力 input に対して、NVIDIA GPU に最適化された DNN によって、人・顔・バッグの3クラスのオブジェクトの認識を行い、バウンディングボックスを描画して出力します。
以下のモデルに対応しています。

- [PeopleNet (Resnet18 DetectNet)](https://ngc.nvidia.com/catalog/models/nvidia:tlt_peoplenet)

デバイス上での実行は、TensorRT 環境を利用して行われます。
実行環境にTensorRTランタイムが存在しない場合、正しく動作しません。

内部で使用されているモデルは 横960ピクセル, 縦544ピクセルのカラー画像を入力として推論を行います。
BBの入力シェイプが内部モデルのものと異なる場合には、拡大縮小およびパディングが行われたのち推論が行われます。

詳細なモデルの仕様は上記リンクを参照してください。

### 入力

- input
  - 要素型: float32
  - 次元: 3
  - フォーマット
    - シェイプ: `[chanel: 3, width: *, height: *]` (column-major表記、Cが最内)
    - 色空間: RGB
    - 値範囲: 標準化 (除数255で除算) された [0..1.0] の値

### 出力

- output
  - 要素型: uint8
  - 次元: 3
  - フォーマット:
    - シェイプ: `[chanel: 3, width: *, height: *]` (column-major表記、Cが最内)
    - 色空間: RGB
    - 値範囲: 標準化 (除数255で除算) された [0..1.0] の値

### パラメータ

- model_base_url
  - 使用する学習済みモデルファイルを配置しているベースURLパス (こちらは理由がない限り変更しないでください)
  - 要素型: string
- cache_root
  - TensorRT モデルファイルのキャッシュを保存 or ロードするディレクトリのパス
  - 要素型: string

## TLT PeopleNet metadata version

前項の TLT PeopleNet と基本的には同じ種類の推論を行いますが、バウンディングボックスが描画された画像ではなく、認識情報を含むメタデータを出力します。

出力データはJSON形式でフォーマットされたASCII文字列をbyte arrayに変換したものです。
出力データのバッファサイズが足りない場合は、エラーメッセージとともにプログラムの実行が中止されます。

### 入力

- input
  - 要素型: float32
  - 次元: 3
  - フォーマット
    - シェイプ: `[chanel: 3, width: *, height: *]` (column-major表記、Cが最内)
    - 色空間: RGB
    - 値範囲: 標準化 (除数255で除算) された [0..1.0] の値

### 出力

- output
  - 要素型: uint8
  - 次元: 1
  - フォーマット:
    - シェイプ: `[N: *]` (パラメータ `output_size` によって変更可能)
    - スキーマ: DetectionBox Array

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

### パラメータ

- model_base_url
  - 使用する学習済みモデルファイルを配置しているベースURLパス (こちらは理由がない限り変更しないでください)
  - 要素型: string
- cache_root
  - TensorRT モデルファイルのキャッシュを保存 or ロードするディレクトリのパス
  - 要素型: string
- width
  - 画像の横幅 (pixel)
  - 要素型: int32
- height
  - 画像の縦幅 (pixel)
  - 要素型: int32
- output_size
  - 出力バッファのサイズ
  - 要素型: int32


## ClassifyGender

DetectionBox Array形式の認識情報を使用して画像から矩形の顔画像を切り出し、性別を判定します。

出力データはJSON形式でフォーマットされたASCII文字列をbyte arrayに変換したものです。
出力データのバッファサイズが足りない場合は、エラーメッセージとともにプログラムの実行が中止されます。

### 入力

- input_img
  - 要素型: float32
  - 次元: 3
  - フォーマット
    - シェイプ: `[chanel: 3, width: *, height: *]` (column-major表記、Cが最内)
    - 色空間: RGB
    - 値範囲: 標準化 (除数255で除算) された [0..1.0] の値
- input_md
  - 要素型: uint8
  - 次元: 1
  - フォーマット:
    - シェイプ: `[N: *]` (パラメータ `input_md_size` によって変更可能)
    - スキーマ: DetectionBox Array

### 出力

- output
  - 要素型: uint8
  - 次元: 1
  - フォーマット:
    - シェイプ: `[N: *]` (パラメータ `output_size` によって変更可能)
    - スキーマ: ClassifyResult

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

### パラメータ

- model_base_url
  - 使用する学習済みモデルファイルを配置しているベースURLパス (こちらは理由がない限り変更しないでください)
  - 要素型: string
- cache_root
  - TensorRT モデルファイルのキャッシュを保存 or ロードするディレクトリのパス
  - 要素型: string
- width
  - 画像の横幅 (pixel)
  - 要素型: int32
- height
  - 画像の縦幅 (pixel)
  - 要素型: int32
- input_md_size
  - 入力メタデータバッファのサイズ
  - 要素型: int32
- output_size
  - 出力バッファのサイズ
  - 要素型: int32

## JSONDictAverageRegulator

辞書形式で入力されるデータに対して、指定された秒数ごとに区間内の積算値を出力します。
有効な出力は秒数ごとにのみ行われ、それ以外 の出力にはnullが設定されます。

### 入力

- input
  - 要素型: uint8
  - 次元: 1
  - フォーマット:
    - シェイプ: `[N: *]` (パラメータ `input_md_size` によって変更可能)
    - スキーマ: NumberDict

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

### 出力

- output
  - 要素型: uint8
  - 次元: 1
  - フォーマット:
    - シェイプ: `[N: *]` (パラメータ `output_size` によって変更可能)
    - スキーマ: NumberDict

### パラメータ

- io_md_size
  - 入出力メタデータバッファのサイズ
  - 要素型: int32
- period_in_sec
  - 積算と出力のインターバル (秒)
  - 要素型: int32

## IFTTT WebHook Uploader

指定されたIFTTT WebHook URLに対して入力データをメッセージとして送信します。
入力データがnullのときには送信は行いません。

### 入力

- input_md
  - 要素型: uint8
  - 次元: 1
  - フォーマット:
    - シェイプ: `[N: *]` (パラメータ `input_md_size` によって変更可能)
    - スキーマ: 任意のJSONデータ

### 出力

- output
  - 要素型: int32
  - 次元: 0
  - フォーマット: 制約なし

### パラメータ

- input_md_size
  - 入出力メタデータバッファのサイズ
  - 要素型: int32
- ifttt_webhook_url
  - IFTTT WebHook URL
  - 要素型: string
