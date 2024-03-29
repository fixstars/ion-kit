<!-- ion-bb-image-io -->

# IMX219

## 説明

接続されているIMX219からRAW画像を取り込みます。

シミュレーションモードでは画像URLを指定することで画像からRAW画像を生成できます。

カメラが検出されない場合は自動的にシミュレーションモードになります。

### 入力

なし

### 出力

- output
  - 要素型: uint16
  - 次元: 2
  - フォーマット: 上詰め10bit、サイズ3264x2464のRGGBベイヤー画像

### パラメータ

- fps
  - 動画のフレームレート
  - 要素型: int32
- width
  - 画像の横幅 (pixel)
  - 要素型: int32
- height
  - 画像の縦幅 (pixel)
  - 要素型: int32
- index
  - カメラ番号
  - 要素型: int32
- force_sim_mode
  - カメラが接続されている場合であってもシミュレーションモードを使用する
  - 要素型: bool
- url
  - シミュレーションモードで使用する画像URL
  - **https非対応**
  - 要素型: string

# D435

## 説明

接続されているD435からステレオ画像と深度画像を取り込みます。

カメラが検出されない場合は自動的にシミュレーションモードになります。

### 入力

なし

### 出力

- output_l
  - 要素型: uint８
  - 次元: 2
  - フォーマット: サイズ1280x720の左画像
- output_r
  - 要素型: uint８
  - 次元: 2
  - フォーマット: サイズ1280x720の右画像
- output_d
  - 要素型: uint16
  - 次元: 2
  - フォーマット: サイズ1280x720の深度画像

### パラメータ

なし

# USBCamera

## 説明

USBカメラを取り扱うためのブロックです。

`/dev/video<index>` デバイスから指定された画像サイズでデータを取得し、RGB8へと変換します。

カメラが検出されない場合は自動的にシミュレーションモードになります。

!> **ベンチマーク時の挙動**  
ベンチマーク環境には実デバイスが接続されていないため、シミュレーションモードとなります。
したがって、本ブロック自体のベンチマーク結果は実デバイスのものとは異なります。

### 入力

なし

### 出力

- output
  - 要素型: uint8
  - 次元: 3
  - フォーマット: 次元配置はCHW、値範囲は [0..255] で表される RGB データ

### パラメータ

- fps
  - 動画のフレームレート
  - 要素型: int32
- width
  - 画像の横幅 (pixel)
  - 要素型: int32
- height
  - 画像の縦幅 (pixel)
  - 要素型: int32
- index
  - カメラ番号
  - 要素型: int32
- url
  - シミュレーションモードで使用する画像URL
  - **https非対応**
  - 要素型: string

# GenericV4L2Bayer

## 説明

RAWカメラを取り扱うためのブロックです。

`/dev/video<index>` デバイスから指定された画像サイズでRAWデータを取得し、Bayer画像を出力します。

カメラが検出されない場合は自動的にシミュレーションモードになります。

!> **ベンチマーク時の挙動**  
ベンチマーク環境には実デバイスが接続されていないため、シミュレーションモードとなります。
したがって、本ブロック自体のベンチマーク結果は実デバイスのものとは異なります。

### 入力

なし

### 出力

- output
  - 要素型: uint16
  - 次元: 2
  - フォーマット: ベイヤー画像

### パラメータ

- fps
  - 動画のフレームレート
  - 要素型: int32
- width
  - 画像の横幅 (pixel)
  - 要素型: int32
- height
  - 画像の縦幅 (pixel)
  - 要素型: int32
- index
  - カメラ番号
  - 要素型: int32
- bit_width
  - RAWビット幅
  - 要素型: int32
- format
  - ベイヤーフォーマット
  - 下記の中から選択
    - 0: RGGB
    - 1: BGGR
    - 2: GRBG
    - 3: GBRG
  - 要素型: int32
- url
  - シミュレーションモードで使用する画像URL
  - **https非対応**
  - 要素型: string

# CameraSimulation

## 説明

RAWカメラをシミュレーションするブロックです。

シミュレーション用の詳細なパラメータを持ちます。

### 入力

なし

### 出力

- output
  - 要素型: uint16
  - 次元: 2
  - フォーマット: ベイヤー画像

### パラメータ

- fps
  - 動画のフレームレート
  - 要素型: int32
- width
  - 画像の横幅 (pixel)
  - 要素型: int32
- height
  - 画像の縦幅 (pixel)
  - 要素型: int32
- bit_width
  - RAWビット幅
  - 要素型: int32
- bit_shift
  - ビットシフト量
  - 要素型: int32
- bayer_pattern
  - ベイヤーフォーマット
  - 下記の中から選択
    - 0: RGGB
    - 1: BGGR
    - 2: GRBG
    - 3: GBRG
  - 要素型: int32
- offset
  - 画素のオフセット
  - 要素型: float32
- gain_r
  - R 画素のゲイン
  - 要素型: float32
- gain_g
  - G 画素のゲイン
  - 要素型: float32
- gain_b
  - B 画素のゲイン
  - 要素型: float32
- url
  - 画像URL
  - **https非対応**
  - 要素型: string

# GUI Display

## 説明

RGB画像をウィンドウに表示します。

!> **ベンチマーク時の挙動**  
ベンチマーク環境には実デバイスが接続されていないため、本ブロックは単に何も行いません。
したがって、本ブロック自体のベンチマーク結果は実デバイスのものとは異なります。

### 入力

- input
  - 要素型: uint8
  - 次元: 3
  - フォーマット: 次元配置はCHW、値範囲は [0..255] で表される RGB データ

### 出力

- output
  - 要素型: int32
  - 次元: 0
  - フォーマット: 制約なし

### パラメータ

- idx
  - ウィンドウ
  - 要素型: int32
- width
  - 画像の横幅 (pixel)
  - 要素型: int32
- height
  - 画像の縦幅 (pixel)
  - 要素型: int32

# FBDisplay

## 説明

フレームバッファにデータを出力します。
RGB データを受け取り、`/dev/fb0` デバイスに指定された画像サイズでデータを書き込みます。

?> **パイプライン構成時の制約**   
現在、本ブロックは複数のフレームバッファに対応していません。
パイプライン中には単一の Display ブロックみが含まれるように注意してください。

!> **ベンチマーク時の挙動**   
ベンチマーク環境には実デバイスが接続されていないため、本ブロックは単に何も行いません。
したがって、本ブロック自体のベンチマーク結果は実デバイスのものとは異なります。

### 入力

- input
  - 要素型: uint8
  - 次元: 3
  - フォーマット: 次元配置はCHW、値範囲は [0..255] で表される RGB データ

### 出力

- output
  - 要素型: uint32
  - 次元: 0
  - フォーマット: 制約なし

### パラメータ

- width
  - 画像の横幅 (pixel)
  - 要素型: int32
- height
  - 画像の縦幅 (pixel)
  - 要素型: int32

# Data Loader / Grayscale

## 説明

与えられたパラメータ `url` で指定される画像ファイル (bmp, png, jpg 等) を読み取り、グレイスケール データへと変換して出力します。

### 入力

なし

### 出力

- output
  - 要素型: uint16
  - 次元: 2
  - フォーマット: 次元配置はHW、値範囲は [0.. `dynamic_range` ] で表される グレイスケール データ

### パラメータ

- width
  - 画像の横幅 (pixel)
  - 要素型: int32
- height
  - 画像の縦幅 (pixel)
  - 要素型: int32
- dynamic_range
  - 画像のダイナミックレンジ (dB)
  - 要素型: int32
- url
  - 読み込む画像のURL
    - 指定可能なファイルは、bmp, png, jpg, raw 等の画像ファイル、またはそれらを複数枚まとめて圧縮した zip 形式ファイルです。
      - raw ファイルは、イメージセンサから取得されるデータではなく、ピクセルの生データをダンプしたものを指しますのでご注意ください。期待される画像フォーマットは下記の通りです。
        - 要素型: uint8 または uint16
        - チャンネル数: 1
      - zip ファイルを指定した場合、zip ファイルに含まれるファイルを名前順に読み取り連続で出力します。プレビュー作成時に出力されるファイル数は、通常のプレビューの場合は1枚、連続プレビューの場合はプレビュー回数で指定した数になります。入力されたプレビュー回数が zip ファイル内のファイル数よりも大きい場合は、ファイル名順に繰り返し出力します。
  - **https非対応**
  - 要素型: string

# Data Loader / Color

## 説明

与えられたパラメータ `url` で指定される画像ファイル (bmp, png, jpg 等) を読み取り、RGB データへと変換して出力します。

### 入力

なし

### 出力

- output
  - 要素型: uint8
  - 次元: 3
  - フォーマット: 次元配置はCHW、値範囲は [0..255] で表される RGB データ

### パラメータ

- width
  - 画像の横幅 (pixel)
  - 要素型: int32
- height
  - 画像の縦幅 (pixel)
  - 要素型: int32
- url
  - 読み込む画像のURL
    - 指定可能なファイルは、bmp, png, jpg, raw 等の画像ファイル、またはそれらを複数枚まとめて圧縮した zip 形式ファイルです。
      - raw ファイルは、イメージセンサから取得されるデータではなく、ピクセルの生データをダンプしたものを指しますのでご注意ください。期待される画像フォーマットは下記の通りです。
        - 要素型: uint8
        - チャンネル数: 3
      - zip ファイルを指定した場合、zip ファイルに含まれるファイルを名前順に読み取り連続で出力します。プレビュー作成時に出力されるファイル数は、通常のプレビューの場合は1枚、連続プレビューの場合はプレビュー回数で指定した数になります。入力されたプレビュー回数が zip ファイル内のファイル数よりも大きい場合は、ファイル名順に繰り返し出力します。
  - **https非対応**
  - 要素型: string

# Image Saver

## 説明

RGB の入力データを、与えられたパラメータ `path` で指定されるパスに画像ファイルとして書き出します。
書き出されるファイルフォーマットは拡張子から自動的に判断されます。

### 入力

- input
  - 要素型: uint8
  - 次元: 3
  - フォーマット: 次元配置はCHW、値範囲は [0..255] で表される RGB データ

### 出力

なし

### パラメータ

- width
  - 画像の横幅 (pixel)
  - 要素型: int32
- height
  - 画像の縦幅 (pixel)
  - 要素型: int32
- path
  - 書き出す画像のファイルパス
  - 要素型: string
