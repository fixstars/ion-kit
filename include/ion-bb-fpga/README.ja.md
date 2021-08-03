# Building Block リファレンスマニュアル
<!-- ion-bb-fpga -->

## Simple ISP(FPGA)

下記のISP処理をFPGA上で処理します。

各処理の内容については同名のBBのリファレンスを御参照ください。

本BBはFPGA向けのためオリジナルのBBとは計算精度が異なりますのでご注意ください。

- Normalize RAW
- BayerOffset
- BayerWhiteBalance
- BayerDemosaicSimple
- GammaCorrection3D

### 入力

- input
  - 要素型: uint16
  - 次元: 2
  - フォーマット: ベイヤー画像

### 出力

- output
  - 要素型: uint8
  - 次元: 3
  - フォーマット: 次元配置は**HWC**、 値範囲は [0..255] で表される RGB データ

### パラメータ

- width
  - 画像の横幅 (pixel)
  - 要素型: int32
- height
  - 画像の縦幅 (pixel)
  - 要素型: int32
- bayer_pattern
  - ベイヤーフォーマット
  - 下記の中から選択
    - 0: RGGB
    - 1: BGGR
    - 2: GRBG
    - 3: GBRG
  - 要素型: int32
- normalize_input_bits
  - RAW画像のビット幅
  - 例: 10bit上詰めの場合 10 を指定
  - 要素型: int32
- normalize_input_shift
  - RAW画像の最下位ビット位置
  - 例: 10bit上詰めの場合 6 を指定
  - 要素型: int32
- offset_offset_r
  - R 画素のオフセット値
  - 要素型: uint16
- offset_offset_g
  - G 画素のオフセット値
  - 要素型: uint16
- offset_offset_b
  - B 画素のオフセット値
  - 要素型: uint16
- white_balance_gain_r
  - R 画素のゲイン値
  - 固定小数点で指定(4bit整数部、12bit小数部)
  - 要素型: uint16
- white_balance_gain_g
  - G 画素のゲイン値
  - 固定小数点で指定(4bit整数部、12bit小数部)
  - 要素型: uint16
- white_balance_gain_b
  - B 画素のゲイン値
  - 固定小数点で指定(4bit整数部、12bit小数部)
  - 要素型: uint16
- gamma_gamma
  - ガンマ補正値
  - 要素型: double
- unroll_level
  - 演算器の展開数
    - 0から3の範囲で指定
    - 大きいほど性能が向上するが回路使用量も増加する
    - FPGA実行の場合のみ有効
  - 要素型: int32

## Simple ISP with unsharp mask(FPGA)

下記のISP処理をFPGA上で処理します。

各処理の内容については同名のBBのリファレンスを御参照ください。

本BBはFPGA向けのためオリジナルのBBとは計算精度が異なりますのでご注意ください。

- Normalize RAW
- BayerOffset
- BayerWhiteBalance
- BayerDemosaicSimple
- Convolution3D (unsharp mask)
- GammaCorrection3D

unsharp maskとして下記の3x3カーネルを使用します。

```
-0.11 -0.11 -0.11
-0.11  1.88 -0.11
-0.11 -0.11 -0.11
```

### 入力

- input
  - 要素型: uint16
  - 次元: 2
  - フォーマット: ベイヤー画像

### 出力

- output
  - 要素型: uint8
  - 次元: 3
  - フォーマット: 次元配置は**HWC**、 値範囲は [0..255] で表される RGB データ

### パラメータ

- width
  - 画像の横幅 (pixel)
  - 要素型: int32
- height
  - 画像の縦幅 (pixel)
  - 要素型: int32
- bayer_pattern
  - ベイヤーフォーマット
  - 下記の中から選択
    - 0: RGGB
    - 1: BGGR
    - 2: GRBG
    - 3: GBRG
  - 要素型: int32
- normalize_input_bits
  - RAW画像のビット幅
  - 例: 10bit上詰めの場合 10 を指定
  - 要素型: int32
- normalize_input_shift
  - RAW画像の最下位ビット位置
  - 例: 10bit上詰めの場合 6 を指定
  - 要素型: int32
- offset_offset_r
  - R 画素のオフセット値
  - 要素型: uint16
- offset_offset_g
  - G 画素のオフセット値
  - 要素型: uint16
- offset_offset_b
  - B 画素のオフセット値
  - 要素型: uint16
- white_balance_gain_r
  - R 画素のゲイン値
  - 固定小数点で指定(4bit整数部、12bit小数部)
  - 要素型: uint16
- white_balance_gain_g
  - G 画素のゲイン値
  - 固定小数点で指定(4bit整数部、12bit小数部)
  - 要素型: uint16
- white_balance_gain_b
  - B 画素のゲイン値
  - 固定小数点で指定(4bit整数部、12bit小数部)
  - 要素型: uint16
- gamma_gamma
  - ガンマ補正値
  - 要素型: double
- unroll_level
  - 演算器の展開数
    - 0から3の範囲で指定
    - 大きいほど性能が向上するが回路使用量も増加する
    - FPGA実行の場合のみ有効
  - 要素型: int32
