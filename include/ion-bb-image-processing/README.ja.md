# Building Block リファレンスマニュアル
<!-- ion-bb-image-processing -->

## BayerOffset

ベイヤー画像の画素値から指定した値を**減算**します。
出力は [0.0, 1.0] にクランプされます。

### 入力

- input
  - 要素型: float32
  - 次元: 2
  - フォーマット: 値範囲は [0.0, 1.0] で表されるベイヤー画像
- offset_r
  - R 画素のオフセット値
  - 要素型: float32
- offset_g
  - G 画素のオフセット値
  - 要素型: float32
- offset_b
  - B 画素のオフセット値
  - 要素型: float32

### 出力

- output
  - 要素型: float32
  - 次元: 2
  - フォーマット: 値範囲は [0.0, 1.0] で表されるベイヤー画像

### パラメータ

- bayer_pattern
  - ベイヤーフォーマット
  - 下記の中から選択
    - 0: RGGB
    - 1: BGGR
    - 2: GRBG
    - 3: GBRG
  - 要素型: int32

## BayerWhiteBalance

ベイヤー画像の画素値に指定した値を**乗算**します。
出力は [0.0, 1.0] にクランプされます。

### 入力

- input
  - 要素型: float32
  - 次元: 2
  - フォーマット: 値範囲は [0.0, 1.0] で表されるベイヤー画像
- gain_r
  - R 画素のゲイン値
  - 要素型: float32
- gain_g
  - G 画素のゲイン値
  - 要素型: float32
- gain_b
  - B 画素のゲイン値
  - 要素型: float32

### 出力

- output
  - 要素型: float32
  - 次元: 2
  - フォーマット: 値範囲は [0.0, 1.0] で表されるベイヤー画像

### パラメータ

- bayer_pattern
  - ベイヤーフォーマット
  - 下記の中から選択
    - 0: RGGB
    - 1: BGGR
    - 2: GRBG
    - 3: GBRG
  - 要素型: int32

## BayerDemosaicSimple

ベイヤー画像のデモザイクを行い、RGB画像を出力します。

縮小してデモザイクを行います。
出力サイズは縦横ともに入力サイズの1/2になります。

### 入力

- input
  - 要素型: float32
  - 次元: 2
  - フォーマット: 値範囲は [0.0, 1.0] で表されるベイヤー画像

### 出力

- output
  - 要素型: float32
  - 次元: 3
  - フォーマット: 次元配置はCHW、 値範囲は [0.0, 1.0] で表される RGB データ

### パラメータ

- width
  - 入力画像の横幅 (pixel)
  - 要素型: int32
- height
  - 入力画像の縦幅 (pixel)
  - 要素型: int32
- bayer_pattern
  - ベイヤーフォーマット
  - 下記の中から選択
    - 0: RGGB
    - 1: BGGR
    - 2: GRBG
    - 3: GBRG
  - 要素型: int32

## BayerDemosaicLinear

ベイヤー画像のデモザイクを行い、RGB画像を出力します。

線形補間を使用してデモザイクを行います。

### 入力

- input
  - 要素型: float32
  - 次元: 2
  - フォーマット: 値範囲は [0.0, 1.0] で表されるベイヤー画像

### 出力

- output
  - 要素型: float32
  - 次元: 3
  - フォーマット: 次元配置はCHW、 値範囲は [0.0, 1.0] で表される RGB データ

### パラメータ

- width
  - 入力画像の横幅 (pixel)
  - 要素型: int32
- height
  - 入力画像の縦幅 (pixel)
  - 要素型: int32
- bayer_pattern
  - ベイヤーフォーマット
  - 下記の中から選択
    - 0: RGGB
    - 1: BGGR
    - 2: GRBG
    - 3: GBRG
  - 要素型: int32

## BayerDemosaicFilter

ベイヤー画像のデモザイクを行い、RGB画像を出力します。

フィルターを使用してデモザイクを行います。

### 入力

- input
  - 要素型: float32
  - 次元: 2
  - フォーマット: 値範囲は [0.0, 1.0] で表されるベイヤー画像

### 出力

- output
  - 要素型: float32
  - 次元: 3
  - フォーマット: 次元配置はCHW、 値範囲は [0.0, 1.0] で表される RGB データ

### パラメータ

- width
  - 入力画像の横幅 (pixel)
  - 要素型: int32
- height
  - 入力画像の縦幅 (pixel)
  - 要素型: int32
- bayer_pattern
  - ベイヤーフォーマット
  - 下記の中から選択
    - 0: RGGB
    - 1: BGGR
    - 2: GRBG
    - 3: GBRG
  - 要素型: int32

## GammaCorrection2D

与えられた入力 `gamma` を用いて、各要素に対して以下の式で表されるガンマ補正を行います。

```
output = clamp(pow(input, gamma), 0.0, 1.0);
```

### 入力

- input
  - 要素型: float32
  - 次元: 2
  - フォーマット: 値範囲は [0..1.0] で表されるデータ
- gamma
  - ガンマ補正値
  - 要素型: float32

### 出力

- output
  - 要素型: float32
  - 次元: 2
  - フォーマット: 値範囲は [0..1.0] で表されるデータ

## GammaCorrection3D

与えられた入力 `gamma` を用いて、各要素に対して以下の式で表されるガンマ補正を行います。

```
output = clamp(pow(input, gamma), 0.0, 1.0);
```

### 入力

- input
  - 要素型: float32
  - 次元: 3
  - フォーマット: 値範囲は [0..1.0] で表されるデータ
- gamma
  - ガンマ補正値
  - 要素型: float32

### 出力

- output
  - 要素型: float32
  - 次元: 3
  - フォーマット: 値範囲は [0..1.0] で表されるデータ

## LensShadingCorrectionLinear

ベイヤー画像に対して、周辺光量補正を行います。

光量が画像中心からの距離の二乗に反比例する(=ゲインが距離の二乗に比例する)と仮定して補正します。

距離は最大値が1になるように正規化されます。

### 入力

- input
  - 要素型: float32
  - 次元: 2
  - フォーマット: 値範囲は [0.0, 1.0] で表されるベイヤー画像
- slope_r
  - R 画素のゲインの傾き
  - 要素型: float32
- slope_g
  - G 画素のゲインの傾き
  - 要素型: float32
- slope_b
  - B 画素のゲインの傾き
  - 要素型: float32
- offset_r
  - R 画素のゲインのオフセット値
  - 要素型: float32
- offset_g
  - G 画素のゲインのオフセット値
  - 要素型: float32
- offset_b
  - B 画素のゲインのオフセット値
  - 要素型: float32

### 出力

- output
  - 要素型: float32
  - 次元: 2
  - フォーマット: 値範囲は [0.0, 1.0] で表されるベイヤー画像

### パラメータ

- width
  - 入力画像の横幅 (pixel)
  - 要素型: int32
- height
  - 入力画像の縦幅 (pixel)
  - 要素型: int32
- bayer_pattern
  - ベイヤーフォーマット
  - 下記の中から選択
    - 0: RGGB
    - 1: BGGR
    - 2: GRBG
    - 3: GBRG
  - 要素型: int32

## ColorMatrix

RGB画像に色変換行列を適用します。

### 入力

- input
  - 要素型: float32
  - 次元: 3
  - フォーマット: 次元配置はCHW、 値範囲は [0.0, 1.0] で表される RGB データ
- matrix
  - 変換行列
  - 要素型: float32
  - 次元: 2
  - フォーマット: サイズ3x3

### 出力

- output
  - 要素型: float32
  - 次元: 3
  - フォーマット: 次元配置はCHW、 値範囲は [0.0, 1.0] で表される RGB データ

## CalcLuminance

RGB画像から輝度画像に変換します。

変換式は下記のいずれかです。

```
Max
  Luminance = max(R, G, B)

Average
  Luminance = (R + G + B) / 3

SimpleY
  Luminance = (3R + 12G + B) / 16

Y
  Luminance = 0.2126R + 0.7152G + 0.0722B
```

### 入力

- input
  - 要素型: float32
  - 次元: 3
  - フォーマット: 次元配置はCHW、 値範囲は [0.0, 1.0] で表される RGB データ

### 出力

- output
  - 要素型: float32
  - 次元: 2
  - フォーマット: 値範囲 [0.0, 1.0] で表される輝度データ

### パラメータ

- luminance_method
  - 計算式
  - 下記の中から選択
    - 0: Max
    - 1: Average
    - 2: SimpleY
    - 3: Y
  - 要素型: int32

## BilateralFilter2D

画像にバイラテラルフィルタを適用します。

### 入力

- input
  - 要素型: float32
  - 次元: 2
  - フォーマット: 次元配置はCHW(３次元の場合)、 値範囲は [0.0, 1.0] で表される画像データ
- sigma
  - 各画素のシグマ値
  - 要素型: float32
  - 次元: 2

### 出力

- output
  - 要素型: float32
  - 次元: 2
  - フォーマット: 次元配置はCHW(３次元の場合)、 値範囲は [0.0, 1.0] で表される画像データ

### パラメータ

- width
  - 入力画像の横幅 (pixel)
  - 要素型: int32
- height
  - 入力画像の縦幅 (pixel)
  - 要素型: int32
- window_size
  - ウィンドウサイズ
  - 実際のウィンドウサイズはn^2+1になる
  - 例: 5x5の場合はwindow_size=2とする
  - 要素型: int32
- coef_color
  - 色方向の係数
  - 要素型: float32
- coef_space
  - 空間方向の係数
  - 要素型: float32
- color_difference_method(BilateralFilter3Dのみ)
  - 色差の計算方法
  - 下記の中から選択
    - 0: PerChannel
      - チャンネルごとの差の二乗の合計
    - 1: Average
      - 平均の差の二乗

## BilateralFilter3D

画像にバイラテラルフィルタを適用します。

### 入力

- input
  - 要素型: float32
  - 次元: 3
  - フォーマット: 次元配置はCHW(３次元の場合)、 値範囲は [0.0, 1.0] で表される画像データ
- sigma
  - 各画素のシグマ値
  - 要素型: float32
  - 次元: 2

### 出力

- output
  - 要素型: float32
  - 次元: 3
  - フォーマット: 次元配置はCHW(３次元の場合)、 値範囲は [0.0, 1.0] で表される画像データ

### パラメータ

- width
  - 入力画像の横幅 (pixel)
  - 要素型: int32
- height
  - 入力画像の縦幅 (pixel)
  - 要素型: int32
- window_size
  - ウィンドウサイズ
  - 実際のウィンドウサイズはn^2+1になる
  - 例: 5x5の場合はwindow_size=2とする
  - 要素型: int32
- coef_color
  - 色方向の係数
  - 要素型: float32
- coef_space
  - 空間方向の係数
  - 要素型: float32
- color_difference_method(BilateralFilter3Dのみ)
  - 色差の計算方法
  - 下記の中から選択
    - 0: PerChannel
      - チャンネルごとの差の二乗の合計
    - 1: Average
      - 平均の差の二乗

## Convolution2D

画像の畳み込みを行います。

### 入力

- input
  - 要素型: float32
  - 次元: 2
  - フォーマット: 次元配置はCHW(３次元の場合)、 値範囲は [0.0, 1.0] で表される画像データ
- kernel
  - 畳み込みカーネル
  - 要素型: float32
  - 次元: 2

### 出力

- output
  - 要素型: float32
  - 次元: 2
  - フォーマット: 次元配置はCHW(３次元の場合)、 値範囲は [0.0, 1.0] で表される画像データ

### パラメータ

- width
  - 入力画像の横幅 (pixel)
  - 要素型: int32
- height
  - 入力画像の縦幅 (pixel)
  - 要素型: int32
- window_size
  - ウィンドウサイズ
  - 実際のウィンドウサイズはn^2+1になる
  - 例: 5x5の場合はwindow_size=2とする
  - 要素型: int32
- boundary_conditions_method
  - 境界条件
  - 下記の中から選択
    - 0: RepeatEdge
    - 1: RepeatImage
    - 2: MirrorImage
    - 3: MirrorInterior
    - 4: Zero

## Convolution3D

画像の畳み込みを行います。

### 入力

- input
  - 要素型: float32
  - 次元: 3
  - フォーマット: 次元配置はCHW(３次元の場合)、 値範囲は [0.0, 1.0] で表される画像データ
- kernel
  - 畳み込みカーネル
  - 要素型: float32
  - 次元: 2

### 出力

- output
  - 要素型: float32
  - 次元: 3
  - フォーマット: 次元配置はCHW(３次元の場合)、 値範囲は [0.0, 1.0] で表される画像データ

### パラメータ

- width
  - 入力画像の横幅 (pixel)
  - 要素型: int32
- height
  - 入力画像の縦幅 (pixel)
  - 要素型: int32
- window_size
  - ウィンドウサイズ
  - 実際のウィンドウサイズはn^2+1になる
  - 例: 5x5の場合はwindow_size=2とする
  - 要素型: int32
- boundary_conditions_method
  - 境界条件
  - 下記の中から選択
    - 0: RepeatEdge
    - 1: RepeatImage
    - 2: MirrorImage
    - 3: MirrorInterior
    - 4: Zero

## LensDistortionCorrectionModel2D

パラメータに従ってレンズ歪補正を行います。

### 入力

- input
  - 要素型: float32
  - 次元: 2
  - フォーマット: 次元配置はCHW(３次元の場合)、 値範囲は [0.0, 1.0] で表される画像データ
- k1
  - 補正パラメータ
  - 要素型: float32
- k2
  - 補正パラメータ
  - 要素型: float32
- k3
  - 補正パラメータ
  - 要素型: float32
- p1
  - 補正パラメータ
  - 要素型: float32
- p2
  - 補正パラメータ
  - 要素型: float32
- fx
  - 補正パラメータ
  - 要素型: float32
- fy
  - 補正パラメータ
  - 要素型: float32
- cx
  - 補正パラメータ
  - 要素型: float32
- cy
  - 補正パラメータ
  - 要素型: float32
- output_scale
  - 出力スケール
  - 要素型: float32

### 出力

- output
  - 要素型: float32
  - 次元: 2
  - フォーマット: 次元配置はCHW(３次元の場合)、 値範囲は [0.0, 1.0] で表される画像データ

### パラメータ

- width
  - 入力画像の横幅 (pixel)
  - 要素型: int32
- height
  - 入力画像の縦幅 (pixel)
  - 要素型: int32

## LensDistortionCorrectionModel3D

パラメータに従ってレンズ歪補正を行います。

### 入力

- input
  - 要素型: float32
  - 次元: 3
  - フォーマット: 次元配置はCHW(３次元の場合)、 値範囲は [0.0, 1.0] で表される画像データ
- k1
  - 補正パラメータ
  - 要素型: float32
- k2
  - 補正パラメータ
  - 要素型: float32
- k3
  - 補正パラメータ
  - 要素型: float32
- p1
  - 補正パラメータ
  - 要素型: float32
- p2
  - 補正パラメータ
  - 要素型: float32
- fx
  - 補正パラメータ
  - 要素型: float32
- fy
  - 補正パラメータ
  - 要素型: float32
- cx
  - 補正パラメータ
  - 要素型: float32
- cy
  - 補正パラメータ
  - 要素型: float32
- output_scale
  - 出力スケール
  - 要素型: float32

### 出力

- output
  - 要素型: float32
  - 次元: 3
  - フォーマット: 次元配置はCHW(３次元の場合)、 値範囲は [0.0, 1.0] で表される画像データ

### パラメータ

- width
  - 入力画像の横幅 (pixel)
  - 要素型: int32
- height
  - 入力画像の縦幅 (pixel)
  - 要素型: int32

## ResizeNearest2D

画像のリサイズをニアレストネイバー法で行います。

### 入力

- input
  - 要素型: float32
  - 次元: 2
  - フォーマット: 次元配置はCHW(３次元の場合)、 値範囲は [0.0, 1.0] で表される画像データ

### 出力

- output
  - 要素型: float32
  - 次元: 2
  - フォーマット: 次元配置はCHW(３次元の場合)、 値範囲は [0.0, 1.0] で表される画像データ

### パラメータ

- width
  - 入力画像の横幅 (pixel)
  - 要素型: int32
- height
  - 入力画像の縦幅 (pixel)
  - 要素型: int32
- scale
  - 拡大縮小比率
  - 例: 1/2 縮小の場合 0.5 を指定
  - 要素型: float32

## ResizeNearest3D

画像のリサイズをニアレストネイバー法で行います。

### 入力

- input
  - 要素型: float32
  - 次元: 3
  - フォーマット: 次元配置はCHW(３次元の場合)、 値範囲は [0.0, 1.0] で表される画像データ

### 出力

- output
  - 要素型: float32
  - 次元: 3
  - フォーマット: 次元配置はCHW(３次元の場合)、 値範囲は [0.0, 1.0] で表される画像データ

### パラメータ

- width
  - 入力画像の横幅 (pixel)
  - 要素型: int32
- height
  - 入力画像の縦幅 (pixel)
  - 要素型: int32
- scale
  - 拡大縮小比率
  - 例: 1/2 縮小の場合 0.5 を指定
  - 要素型: float32

## ResizeBilinear2D

RGB画像のリサイズをバイリニア法で行います。

### 入力

- input
  - 要素型: float32
  - 次元: 2
  - フォーマット: 次元配置はCHW(３次元の場合)、 値範囲は [0.0, 1.0] で表される画像データ

### 出力

- output
  - 要素型: float32
  - 次元: 2
  - フォーマット: 次元配置はCHW(３次元の場合)、 値範囲は [0.0, 1.0] で表される画像データ

### パラメータ

- width
  - 入力画像の横幅 (pixel)
  - 要素型: int32
- height
  - 入力画像の縦幅 (pixel)
  - 要素型: int32
- scale
  - 拡大縮小比率
  - 例: 1/2 縮小の場合 0.5 を指定
  - 要素型: float32

## ResizeBilinear3D

RGB画像のリサイズをバイリニア法で行います。

### 入力

- input
  - 要素型: float32
  - 次元: 3
  - フォーマット: 次元配置はCHW(３次元の場合)、 値範囲は [0.0, 1.0] で表される画像データ

### 出力

- output
  - 要素型: float32
  - 次元: 3
  - フォーマット: 次元配置はCHW(３次元の場合)、 値範囲は [0.0, 1.0] で表される画像データ

### パラメータ

- width
  - 入力画像の横幅 (pixel)
  - 要素型: int32
- height
  - 入力画像の縦幅 (pixel)
  - 要素型: int32
- scale
  - 拡大縮小比率
  - 例: 1/2 縮小の場合 0.5 を指定
  - 要素型: float32

## ResizeAreaAverage2D

RGB画像のリサイズを平均画素法で行います。

### 入力

- input
  - 要素型: float32
  - 次元: 2
  - フォーマット: 次元配置はCHW(３次元の場合)、 値範囲は [0.0, 1.0] で表される画像データ

### 出力

- output
  - 要素型: float32
  - 次元: 2
  - フォーマット: 次元配置はCHW(３次元の場合)、 値範囲は [0.0, 1.0] で表される画像データ

### パラメータ

- width
  - 入力画像の横幅 (pixel)
  - 要素型: int32
- height
  - 入力画像の縦幅 (pixel)
  - 要素型: int32
- scale
  - 拡大縮小比率
  - 例: 1/2 縮小の場合 0.5 を指定
  - 要素型: float32

## ResizeAreaAverage3D

RGB画像のリサイズを平均画素法で行います。

### 入力

- input
  - 要素型: float32
  - 次元: 3
  - フォーマット: 次元配置はCHW(３次元の場合)、 値範囲は [0.0, 1.0] で表される画像データ

### 出力

- output
  - 要素型: float32
  - 次元: 3
  - フォーマット: 次元配置はCHW(３次元の場合)、 値範囲は [0.0, 1.0] で表される画像データ

### パラメータ

- width
  - 入力画像の横幅 (pixel)
  - 要素型: int32
- height
  - 入力画像の縦幅 (pixel)
  - 要素型: int32
- scale
  - 拡大縮小比率
  - 例: 1/2 縮小の場合 0.5 を指定
  - 要素型: float32

## BayerDownscaleUInt16

ベイヤー画像のダウンスケールを行います。

### 入力

- input
  - 要素型: uint16
  - 次元: 2
  - フォーマット: 2x2のベイヤー配列

### 出力

- output
  - 要素型: uint16
  - 次元: 2
  - フォーマット: 2x2のベイヤー配列

### パラメータ

- input_width
  - 入力画像の横幅 (pixel)
  - 要素型: int32
- input_height
  - 入力画像の縦幅 (pixel)
  - 要素型: int32
- downscale_factor
  - 縮小比率
  - 例: 1/2 縮小の場合 2 を指定
  - 要素型: int32

## Normalize RAW

RAW画像を [0.0, 1.0] に正規化して出力します。

### 入力

- input
  - 要素型: uint16
  - 次元: 2
  - フォーマット: 制約なし

### 出力

- output
  - 要素型: float32
  - 次元: 2
  - フォーマット: 制約なし

### パラメータ

- bit_width
  - RAW画像のビット幅
  - 例: 10bit上詰めの場合 10 を指定
  - 要素型: uint8
- bit_shift
  - RAW画像の最下位ビット位置
  - 例: 10bit上詰めの場合 6 を指定
  - 要素型: uint8

## FitImageToCenter2DUInt8

画像を中央に配置します。

入力サイズが大きい場合はクロップ、小さい場合は0でパディングされます。

### 入力

- input
  - 要素型: uint8
  - 次元: 2
  - フォーマット: 次元配置はCHW(３次元の場合)である画像データ

### 出力

- output
  - 要素型: uint8
  - 次元: 2
  - フォーマット: 次元配置はCHW(３次元の場合)である画像データ

### パラメータ

- input_width
  - 入力画像の横幅 (pixel)
- input_height
  - 入力画像の縦幅 (pixel)
- output_width
  - 出力画像の横幅 (pixel)
- output_height
  - 出力画像の縦幅 (pixel)

## FitImageToCenter3DUInt8

画像を中央に配置します。

入力サイズが大きい場合はクロップ、小さい場合は0でパディングされます。

### 入力

- input
  - 要素型: uint8
  - 次元: 3
  - フォーマット: 次元配置はCHW(３次元の場合)である画像データ

### 出力

- output
  - 要素型: uint8
  - 次元: 3
  - フォーマット: 次元配置はCHW(３次元の場合)である画像データ

### パラメータ

- input_width
  - 入力画像の横幅 (pixel)
- input_height
  - 入力画像の縦幅 (pixel)
- output_width
  - 出力画像の横幅 (pixel)
- output_height
  - 出力画像の縦幅 (pixel)

## FitImageToCenter2DFloat

画像を中央に配置します。

入力サイズが大きい場合はクロップ、小さい場合は0でパディングされます。

### 入力

- input
  - 要素型: float32
  - 次元: 2
  - フォーマット: 次元配置はCHW(３次元の場合)である画像データ

### 出力

- output
  - 要素型: float32
  - 次元: 2
  - フォーマット: 次元配置はCHW(３次元の場合)である画像データ

### パラメータ

- input_width
  - 入力画像の横幅 (pixel)
- input_height
  - 入力画像の縦幅 (pixel)
- output_width
  - 出力画像の横幅 (pixel)
- output_height
  - 出力画像の縦幅 (pixel)

## FitImageToCenter3DFloat

画像を中央に配置します。

入力サイズが大きい場合はクロップ、小さい場合は0でパディングされます。

### 入力

- input
  - 要素型: float32
  - 次元: 3
  - フォーマット: 次元配置はCHW(３次元の場合)である画像データ

### 出力

- output
  - 要素型: float32
  - 次元: 3
  - フォーマット: 次元配置はCHW(３次元の場合)である画像データ

### パラメータ

- input_width
  - 入力画像の横幅 (pixel)
- input_height
  - 入力画像の縦幅 (pixel)
- output_width
  - 出力画像の横幅 (pixel)
- output_height
  - 出力画像の縦幅 (pixel)

## ReorderColorChannel3DUInt8

RGB画像のチャネル順序を逆順に変換します。

RGB->BGR変換、BGR->RGB変換の両方に使用できます。

### 入力

- input
  - 要素型: uint8
  - 次元: 3
  - フォーマット: RGB データ

### 出力

- output
  - 要素型: uint8
  - 次元: 3
  - フォーマット: RGB データ

### パラメータ

- color_dim
  - カラーチャンネルを表す次元
  - 要素型: int32

## ReorderColorChannel3DFloat

RGB画像のチャネル順序を逆順に変換します。

RGB->BGR変換、BGR->RGB変換の両方に使用できます。

### 入力

- input
  - 要素型: float32
  - 次元: 3
  - フォーマット: RGB データ

### 出力

- output
  - 要素型: float32
  - 次元: 3
  - フォーマット: RGB データ

### パラメータ

- color_dim
  - カラーチャンネルを表す次元
  - 要素型: int32

## OverlayImage2DUInt8

２つの画像を１つにまとめます。

input0が背景、input1が前景として合成されます。

範囲外は0埋めされます。

### 入力

- input0
  - 要素型: uint8
  - 次元: 2
  - フォーマット: 画像データ
- input1
  - 要素型: uint8
  - 次元: 2
  - フォーマット: 画像データ

### 出力

- output
  - 要素型: uint8
  - 次元: 2
  - フォーマット: 画像データ

### パラメータ

- x_dim
  - 横方向を表す次元
  - CHWフォーマットであれば0
  - 要素型: int32
- y_dim
  - 縦方向を表す次元
  - CHWフォーマットであれば1
  - 要素型: int32
- input0_width
  - 背景の横幅 (pixel)
  - 要素型: int32
- input0_height
  - 背景の縦幅 (pixel)
  - 要素型: int32
- input1_left
  - 前景の横位置 (pixel)
  - 要素型: int32
- input1_top
  - 前景の縦位置 (pixel)
  - 要素型: int32
- input1_width
  - 前景の横幅 (pixel)
  - 要素型: int32
- input1_height
  - 前景の縦幅 (pixel)
  - 要素型: int32

## OverlayImage3DUInt8

２つの画像を１つにまとめます。

input0が背景、input1が前景として合成されます。

範囲外は0埋めされます。

### 入力

- input0
  - 要素型: uint8
  - 次元: 3
  - フォーマット: 画像データ
- input1
  - 要素型: uint8
  - 次元: 3
  - フォーマット: 画像データ

### 出力

- output
  - 要素型: uint8
  - 次元: 3
  - フォーマット: 画像データ

### パラメータ

- x_dim
  - 横方向を表す次元
  - CHWフォーマットであれば0
  - 要素型: int32
- y_dim
  - 縦方向を表す次元
  - CHWフォーマットであれば1
  - 要素型: int32
- input0_width
  - 背景の横幅 (pixel)
  - 要素型: int32
- input0_height
  - 背景の縦幅 (pixel)
  - 要素型: int32
- input1_left
  - 前景の横位置 (pixel)
  - 要素型: int32
- input1_top
  - 前景の縦位置 (pixel)
  - 要素型: int32
- input1_width
  - 前景の横幅 (pixel)
  - 要素型: int32
- input1_height
  - 前景の縦幅 (pixel)
  - 要素型: int32

## OverlayImage2DFloat

２つの画像を１つにまとめます。

input0が背景、input1が前景として合成されます。

範囲外は0埋めされます。

### 入力

- input0
  - 要素型: float32
  - 次元: 2
  - フォーマット: 画像データ
- input1
  - 要素型: float32
  - 次元: 2
  - フォーマット: 画像データ

### 出力

- output
  - 要素型: float32
  - 次元: 2
  - フォーマット: 画像データ

### パラメータ

- x_dim
  - 横方向を表す次元
  - CHWフォーマットであれば0
  - 要素型: int32
- y_dim
  - 縦方向を表す次元
  - CHWフォーマットであれば1
  - 要素型: int32
- input0_width
  - 背景の横幅 (pixel)
  - 要素型: int32
- input0_height
  - 背景の縦幅 (pixel)
  - 要素型: int32
- input1_left
  - 前景の横位置 (pixel)
  - 要素型: int32
- input1_top
  - 前景の縦位置 (pixel)
  - 要素型: int32
- input1_width
  - 前景の横幅 (pixel)
  - 要素型: int32
- input1_height
  - 前景の縦幅 (pixel)
  - 要素型: int32

## OverlayImage3DFloat

２つの画像を１つにまとめます。

input0が背景、input1が前景として合成されます。

範囲外は0埋めされます。

### 入力

- input0
  - 要素型: float32
  - 次元: 3
  - フォーマット: 画像データ
- input1
  - 要素型: float32
  - 次元: 3
  - フォーマット: 画像データ

### 出力

- output
  - 要素型: float32
  - 次元: 3
  - フォーマット: 画像データ

### パラメータ

- x_dim
  - 横方向を表す次元
  - CHWフォーマットであれば0
  - 要素型: int32
- y_dim
  - 縦方向を表す次元
  - CHWフォーマットであれば1
  - 要素型: int32
- input0_width
  - 背景の横幅 (pixel)
  - 要素型: int32
- input0_height
  - 背景の縦幅 (pixel)
  - 要素型: int32
- input1_left
  - 前景の横位置 (pixel)
  - 要素型: int32
- input1_top
  - 前景の縦位置 (pixel)
  - 要素型: int32
- input1_width
  - 前景の横幅 (pixel)
  - 要素型: int32
- input1_height
  - 前景の縦幅 (pixel)
  - 要素型: int32

## TileImageHorizontal2DUInt8

２つの画像を１つにまとめます。

画像は横方向に連結されます。

範囲外は0埋めされます。

### 入力

- input0
  - 要素型: uint8
  - 次元: 2
  - フォーマット: 画像データ
- input1
  - 要素型: uint8
  - 次元: 2
  - フォーマット: 画像データ

### 出力

- output
  - 要素型: uint8
  - 次元: 2
  - フォーマット: 画像データ

### パラメータ

- x_dim
  - 横方向を表す次元
  - CHWフォーマットであれば0
  - 要素型: int32
- y_dim
  - 縦方向を表す次元
  - CHWフォーマットであれば1
  - 要素型: int32
- input0_width
  - 画像0の横幅 (pixel)
  - 要素型: int32
- input0_height
  - 画像0の縦幅 (pixel)
  - 要素型: int32
- input1_width
  - 画像1の横幅 (pixel)
  - 要素型: int32
- input1_height
  - 画像1の縦幅 (pixel)
  - 要素型: int32

## TileImageHorizontal3DUInt8

２つの画像を１つにまとめます。

画像は横方向に連結されます。

範囲外は0埋めされます。

### 入力

- input0
  - 要素型: uint8
  - 次元: 3
  - フォーマット: 画像データ
- input1
  - 要素型: uint8
  - 次元: 3
  - フォーマット: 画像データ

### 出力

- output
  - 要素型: uint8
  - 次元: 3
  - フォーマット: 画像データ

### パラメータ

- x_dim
  - 横方向を表す次元
  - CHWフォーマットであれば0
  - 要素型: int32
- y_dim
  - 縦方向を表す次元
  - CHWフォーマットであれば1
  - 要素型: int32
- input0_width
  - 画像0の横幅 (pixel)
  - 要素型: int32
- input0_height
  - 画像0の縦幅 (pixel)
  - 要素型: int32
- input1_width
  - 画像1の横幅 (pixel)
  - 要素型: int32
- input1_height
  - 画像1の縦幅 (pixel)
  - 要素型: int32

## TileImageHorizontal2DFloat

２つの画像を１つにまとめます。

画像は横方向に連結されます。

範囲外は0埋めされます。

### 入力

- input0
  - 要素型: float32
  - 次元: 2
  - フォーマット: 画像データ
- input1
  - 要素型: float32
  - 次元: 2
  - フォーマット: 画像データ

### 出力

- output
  - 要素型: uint8 / float32
  - 次元: 2 / 3
  - フォーマット: 画像データ

### パラメータ

- x_dim
  - 横方向を表す次元
  - CHWフォーマットであれば0
  - 要素型: int32
- y_dim
  - 縦方向を表す次元
  - CHWフォーマットであれば1
  - 要素型: int32
- input0_width
  - 画像0の横幅 (pixel)
  - 要素型: int32
- input0_height
  - 画像0の縦幅 (pixel)
  - 要素型: int32
- input1_width
  - 画像1の横幅 (pixel)
  - 要素型: int32
- input1_height
  - 画像1の縦幅 (pixel)
  - 要素型: int32

## TileImageHorizontal3DFloat

２つの画像を１つにまとめます。

画像は横方向に連結されます。

範囲外は0埋めされます。

### 入力

- input0
  - 要素型: float32
  - 次元: 3
  - フォーマット: 画像データ
- input1
  - 要素型: float32
  - 次元: 3
  - フォーマット: 画像データ

### 出力

- output
  - 要素型: float32
  - 次元: 3
  - フォーマット: 画像データ

### パラメータ

- x_dim
  - 横方向を表す次元
  - CHWフォーマットであれば0
  - 要素型: int32
- y_dim
  - 縦方向を表す次元
  - CHWフォーマットであれば1
  - 要素型: int32
- input0_width
  - 画像0の横幅 (pixel)
  - 要素型: int32
- input0_height
  - 画像0の縦幅 (pixel)
  - 要素型: int32
- input1_width
  - 画像1の横幅 (pixel)
  - 要素型: int32
- input1_height
  - 画像1の縦幅 (pixel)
  - 要素型: int32

## TileImageVertical2DUInt8

２つの画像を１つにまとめます。

画像は縦方向に連結されます。

範囲外は0埋めされます。

### 入力

- input0
  - 要素型: uint8
  - 次元: 2
  - フォーマット: 画像データ
- input1
  - 要素型: uint8
  - 次元: 2
  - フォーマット: 画像データ

### 出力

- output
  - 要素型: uint8
  - 次元: 2
  - フォーマット: 画像データ

### パラメータ

- x_dim
  - 横方向を表す次元
  - CHWフォーマットであれば0
  - 要素型: int32
- y_dim
  - 縦方向を表す次元
  - CHWフォーマットであれば1
  - 要素型: int32
- input0_width
  - 画像0の横幅 (pixel)
  - 要素型: int32
- input0_height
  - 画像0の縦幅 (pixel)
  - 要素型: int32
- input1_width
  - 画像1の横幅 (pixel)
  - 要素型: int32
- input1_height
  - 画像1の縦幅 (pixel)
  - 要素型: int32

## TileImageVertical3DUInt8

２つの画像を１つにまとめます。

画像は縦方向に連結されます。

範囲外は0埋めされます。

### 入力

- input0
  - 要素型: uint8
  - 次元: 3
  - フォーマット: 画像データ
- input1
  - 要素型: uint8
  - 次元: 3
  - フォーマット: 画像データ

### 出力

- output
  - 要素型: uint8
  - 次元: 3
  - フォーマット: 画像データ

### パラメータ

- x_dim
  - 横方向を表す次元
  - CHWフォーマットであれば0
  - 要素型: int32
- y_dim
  - 縦方向を表す次元
  - CHWフォーマットであれば1
  - 要素型: int32
- input0_width
  - 画像0の横幅 (pixel)
  - 要素型: int32
- input0_height
  - 画像0の縦幅 (pixel)
  - 要素型: int32
- input1_width
  - 画像1の横幅 (pixel)
  - 要素型: int32
- input1_height
  - 画像1の縦幅 (pixel)
  - 要素型: int32

## TileImageVertical2DFloat

２つの画像を１つにまとめます。

画像は縦方向に連結されます。

範囲外は0埋めされます。

### 入力

- input0
  - 要素型: float32
  - 次元: 2
  - フォーマット: 画像データ
- input1
  - 要素型: float32
  - 次元: 2
  - フォーマット: 画像データ

### 出力

- output
  - 要素型: float32
  - 次元: 2
  - フォーマット: 画像データ

### パラメータ

- x_dim
  - 横方向を表す次元
  - CHWフォーマットであれば0
  - 要素型: int32
- y_dim
  - 縦方向を表す次元
  - CHWフォーマットであれば1
  - 要素型: int32
- input0_width
  - 画像0の横幅 (pixel)
  - 要素型: int32
- input0_height
  - 画像0の縦幅 (pixel)
  - 要素型: int32
- input1_width
  - 画像1の横幅 (pixel)
  - 要素型: int32
- input1_height
  - 画像1の縦幅 (pixel)
  - 要素型: int32

## TileImageVertical3DFloat

２つの画像を１つにまとめます。

画像は縦方向に連結されます。

範囲外は0埋めされます。

### 入力

- input0
  - 要素型: float32
  - 次元: 3
  - フォーマット: 画像データ
- input1
  - 要素型: float32
  - 次元: 3
  - フォーマット: 画像データ

### 出力

- output
  - 要素型: float32
  - 次元: 3
  - フォーマット: 画像データ

### パラメータ

- x_dim
  - 横方向を表す次元
  - CHWフォーマットであれば0
  - 要素型: int32
- y_dim
  - 縦方向を表す次元
  - CHWフォーマットであれば1
  - 要素型: int32
- input0_width
  - 画像0の横幅 (pixel)
  - 要素型: int32
- input0_height
  - 画像0の縦幅 (pixel)
  - 要素型: int32
- input1_width
  - 画像1の横幅 (pixel)
  - 要素型: int32
- input1_height
  - 画像1の縦幅 (pixel)
  - 要素型: int32

## CropImage2DUInt8

画像の一部領域をクロップします。

範囲外は0埋めされます。

### 入力

- input
  - 要素型: uint8
  - 次元: 2
  - フォーマット: 画像データ

### 出力

- output
  - 要素型: uint8
  - 次元: 2
  - フォーマット: 画像データ

### パラメータ

- x_dim
  - 横方向を表す次元
  - CHWフォーマットであれば0
  - 要素型: int32
- y_dim
  - 縦方向を表す次元
  - CHWフォーマットであれば1
  - 要素型: int32
- input_width
  - 入力画像の横幅 (pixel)
  - 要素型: int32
- input_height
  - 入力画像の縦幅 (pixel)
  - 要素型: int32
- left
  - 領域の横位置 (pixel)
  - 要素型: int32
- top
  - 領域の縦位置 (pixel)
  - 要素型: int32
- output_width
  - 出力画像の横幅 (pixel)
  - 要素型: int32
- output_height
  - 出力画像の縦幅 (pixel)
  - 要素型: int32

## CropImage3DUInt8

画像の一部領域をクロップします。

範囲外は0埋めされます。

### 入力

- input
  - 要素型: uint8
  - 次元: 3
  - フォーマット: 画像データ

### 出力

- output
  - 要素型: uint8
  - 次元: 3
  - フォーマット: 画像データ

### パラメータ

- x_dim
  - 横方向を表す次元
  - CHWフォーマットであれば0
  - 要素型: int32
- y_dim
  - 縦方向を表す次元
  - CHWフォーマットであれば1
  - 要素型: int32
- input_width
  - 入力画像の横幅 (pixel)
  - 要素型: int32
- input_height
  - 入力画像の縦幅 (pixel)
  - 要素型: int32
- left
  - 領域の横位置 (pixel)
  - 要素型: int32
- top
  - 領域の縦位置 (pixel)
  - 要素型: int32
- output_width
  - 出力画像の横幅 (pixel)
  - 要素型: int32
- output_height
  - 出力画像の縦幅 (pixel)
  - 要素型: int32

## CropImage2DFloat

画像の一部領域をクロップします。

範囲外は0埋めされます。

### 入力

- input
  - 要素型: float32
  - 次元: 2
  - フォーマット: 画像データ

### 出力

- output
  - 要素型: float32
  - 次元: 2
  - フォーマット: 画像データ

### パラメータ

- x_dim
  - 横方向を表す次元
  - CHWフォーマットであれば0
  - 要素型: int32
- y_dim
  - 縦方向を表す次元
  - CHWフォーマットであれば1
  - 要素型: int32
- input_width
  - 入力画像の横幅 (pixel)
  - 要素型: int32
- input_height
  - 入力画像の縦幅 (pixel)
  - 要素型: int32
- left
  - 領域の横位置 (pixel)
  - 要素型: int32
- top
  - 領域の縦位置 (pixel)
  - 要素型: int32
- output_width
  - 出力画像の横幅 (pixel)
  - 要素型: int32
- output_height
  - 出力画像の縦幅 (pixel)
  - 要素型: int32

## CropImage3DFloat

画像の一部領域をクロップします。

範囲外は0埋めされます。

### 入力

- input
  - 要素型: float32
  - 次元: 3
  - フォーマット: 画像データ

### 出力

- output
  - 要素型: float32
  - 次元: 3
  - フォーマット: 画像データ

### パラメータ

- x_dim
  - 横方向を表す次元
  - CHWフォーマットであれば0
  - 要素型: int32
- y_dim
  - 縦方向を表す次元
  - CHWフォーマットであれば1
  - 要素型: int32
- input_width
  - 入力画像の横幅 (pixel)
  - 要素型: int32
- input_height
  - 入力画像の縦幅 (pixel)
  - 要素型: int32
- left
  - 領域の横位置 (pixel)
  - 要素型: int32
- top
  - 領域の縦位置 (pixel)
  - 要素型: int32
- output_width
  - 出力画像の横幅 (pixel)
  - 要素型: int32
- output_height
  - 出力画像の縦幅 (pixel)
  - 要素型: int32

## ColorSpaceConverter RGB to HSV

RGB 色空間から HSV 色空間への変換を行います。

### 入力

- input
  - 要素型: float32
  - 次元: 3
  - フォーマット: 次元配置はCHW, 値範囲は [0..1.0] で表される RGB データ

### 出力

- output
  - 要素型: float32
  - 次元: 3
  - フォーマット: 次元配置はCHW, 値範囲は [0..1.0] で表される HSV データ

### パラメータ

なし

## ColorSpaceConverter HSV to RGB

HSV 色空間から RGB 色空間への変換を行います。

### 入力

- input
  - 要素型: float32
  - 次元: 3
  - フォーマット: 次元配置はCHW, 値範囲は [0..1.0] で表される HSV データ

### 出力

- output
  - 要素型: float32
  - 次元: 3
  - フォーマット: 次元配置はCHW, 値範囲は [0..1.0] で表される RGB データ

### パラメータ

なし

## Color Adjustment

与えられたパラメータ `adjustment_value` を用いて `target_channel` に対して色補正を適用します。

```
output = select(c == target_channel, clamp(pow(input, adjustment_value), 0.0, 1.0), input);
```

### 入力

- input
  - 要素型: float32
  - 次元: 3
  - フォーマット: 次元配置はCHW、 値範囲は [0..1.0] で表されるデータ

### 出力

- output
  - 要素型: float32
  - 次元: 3
  - フォーマット: 次元配置はCHW、 値範囲は [0..1.0] で表されるデータ

### パラメータ

- adjustment_value
  - 補正パラメータ
  - 要素型: float32
- target_channel
  - 対象チャンネル
  - 要素型: int32
