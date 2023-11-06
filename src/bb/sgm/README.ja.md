<!-- ion-bb-sgm -->

# Stereo Matching

## 説明

Semi-Global Matching アルゴリズムを使用して、ステレオ画像から深度情報を推定します。

### 入力

- input_l
  - 要素型: uint8
  - 次元: 2
  - フォーマット: Mono8、次元配置はHW、値範囲は [0..255] で表されるモノクロデータ
- input_r
  - 要素型: uint8
  - 次元: 2
  - フォーマット: Mono8、次元配置はHW、値範囲は [0..255] で表されるモノクロデータ

### 出力

- output
  - 要素型: uint8
  - 次元: 2
  - フォーマット: 次元配置はHW、値範囲は [0..255] で表される深度データ

### パラメータ

- disp
  - 深度パラメータ
  - 要素型: int32
- width
  - 画像の横幅 (pixel)
  - 要素型: int32
- height
  - 画像の縦幅 (pixel)
  - 要素型: int32
