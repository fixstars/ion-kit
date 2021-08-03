# Building Block リファレンスマニュアル
<!-- ion-bb-core -->

## ReorderBuffer3DUInt8

入力データの次元を入れ替えます。

HWC <-> CHW 変換などに使用できます。

### 入力

- input
  - 要素型: uint8
  - 次元: 3
  - フォーマット: 制約なし

### 出力

- output
  - 要素型: uint8
  - 次元: 3
  - フォーマット: 制約なし

### パラメータ

- dim0
  - 出力の次元0に紐付ける入力次元
  - 要素型: int32
- dim1
  - 出力の次元1に紐付ける入力次元
  - 要素型: int32
- dim2
  - 出力の次元2に紐付ける入力次元
  - 要素型: int32

### パラメータ例

- HWC -> CHW変換
  - dim0 = 1
  - dim1 = 2
  - dim2 = 0
- CHW -> HWC変換
  - dim0 = 2
  - dim1 = 0
  - dim2 = 1

## ReorderBuffer3DFloat

入力データの次元を入れ替えます。

HWC <-> CHW 変換などに使用できます。

### 入力

- input
  - 要素型: float32
  - 次元: 3
  - フォーマット: 制約なし

### 出力

- output
  - 要素型: float32
  - 次元: 3
  - フォーマット: 制約なし

### パラメータ

- dim0
  - 出力の次元0に紐付ける入力次元
  - 要素型: int32
- dim1
  - 出力の次元1に紐付ける入力次元
  - 要素型: int32
- dim2
  - 出力の次元2に紐付ける入力次元
  - 要素型: int32

### パラメータ例

- HWC -> CHW変換
  - dim0 = 1
  - dim1 = 2
  - dim2 = 0
- CHW -> HWC変換
  - dim0 = 2
  - dim1 = 0
  - dim2 = 1

## Denormalize2DUInt8

[0..1.0] に正規化されている入力データを [0..255] に拡大して出力します。

### 入力

- input
  - 要素型: float32
  - 次元: 2
  - フォーマット: 制約なし

### 出力

- output
  - 要素型: uint8
  - 次元: 2
  - フォーマット: 制約なし

### パラメータ

なし

## Denormalize3DUInt8

[0..1.0] に正規化されている入力データを [0..255] に拡大して出力します。

### 入力

- input
  - 要素型: float32
  - 次元: 3
  - フォーマット: 制約なし

### 出力

- output
  - 要素型: uint8
  - 次元: 3
  - フォーマット: 制約なし

### パラメータ

なし

## Normalize2DUInt8

[0..255] の値レンジをもつ入力データを [0..1.0] に正規化して出力します。

### 入力

- input
  - 要素型: uint8
  - 次元: 2
  - フォーマット: 制約なし

### 出力

- output
  - 要素型: float32
  - 次元: 2
  - フォーマット: 制約なし

### パラメータ

なし

## Normalize3DUInt8

[0..255] の値レンジをもつ入力データを [0..1.0] に正規化して出力します。

### 入力

- input
  - 要素型: uint8
  - 次元: 3
  - フォーマット: 制約なし

### 出力

- output
  - 要素型: float32
  - 次元: 3
  - フォーマット: 制約なし

### パラメータ

なし

## ExtendDimension2DUInt8

入力バッファの次元を拡張します。

### 入力

- input
  - 要素型: uint8
  - 次元: 2
  - フォーマット: 制約なし

### 出力

- output
  - 要素型: uint8
  - 次元: 3
  - フォーマット: 制約なし

### パラメータ

- new_dim
  - 追加する次元
  - 要素型: int32
- extent
  - 追加次元のサイズ
  - 要素型: int32

## ExtendDimension2DFloat

入力バッファの次元を拡張します。

### 入力

- input
  - 要素型: float32
  - 次元: 2
  - フォーマット: 制約なし

### 出力

- output
  - 要素型: float32
  - 次元: 3
  - フォーマット: 制約なし

### パラメータ

- new_dim
  - 追加する次元
  - 要素型: int32
- extent
  - 追加次元のサイズ
  - 要素型: int32

## ConstantBuffer2DFloat

パラメータで指定した値を持つバッファを生成します。

出力サイズに対して値が足りない場合は繰り返されます。

### 入力

なし

### 出力

- output
  - 要素型: float32
  - 次元: 2
  - フォーマット: 制約なし

### パラメータ

- values
  - **スペース区切り**の値リスト
  - 要素型: string
- extent0
  - 出力サイズ(0次元目)
  - 要素型: int32
- extent1
  - 出力サイズ(1次元目)
  - 要素型: int32
- extent2 (ConstantBuffer3DFloatのみ)
  - 出力サイズ(2次元目)
  - 要素型: int32

## ConstantBuffer3DFloat

パラメータで指定した値を持つバッファを生成します。

出力サイズに対して値が足りない場合は繰り返されます。

### 入力

なし

### 出力

- output
  - 要素型: float32
  - 次元: 3
  - フォーマット: 制約なし

### パラメータ

- values
  - **スペース区切り**の値リスト
  - 要素型: string
- extent0
  - 出力サイズ(0次元目)
  - 要素型: int32
- extent1
  - 出力サイズ(1次元目)
  - 要素型: int32
- extent2 (ConstantBuffer3DFloatのみ)
  - 出力サイズ(2次元目)
  - 要素型: int32

## Add2DFloat

バッファ同士を加算します。

### 入力

- input0
  - 要素型: float32
  - 次元: 2
  - フォーマット: 制約なし

- input1
  - 要素型: float32
  - 次元: 2 / 3
  - フォーマット: 制約なし

### 出力

- output
  - 要素型: float32
  - 次元: 2
  - フォーマット: 制約なし

### パラメータ

- enable_clamp
  - 未使用
  - 要素型: bool

## Add3DFloat

バッファ同士を加算します。

### 入力

- input0
  - 要素型: float32
  - 次元: 3
  - フォーマット: 制約なし

- input1
  - 要素型: float32
  - 次元: 2 / 3
  - フォーマット: 制約なし

### 出力

- output
  - 要素型: float32
  - 次元: 3
  - フォーマット: 制約なし

### パラメータ

- enable_clamp
  - 未使用
  - 要素型: bool

## Multiply2DFloat

バッファ同士を乗算します。

### 入力

- input0
  - 要素型: float32
  - 次元: 2
  - フォーマット: 制約なし

- input1
  - 要素型: float32
  - 次元: 2
  - フォーマット: 制約なし

### 出力

- output
  - 要素型: float32
  - 次元: 2
  - フォーマット: 制約なし

### パラメータ

- enable_clamp
  - 未使用
  - 要素型: bool

## Multiply3DFloat

バッファ同士を乗算します。

### 入力

- input0
  - 要素型: float32
  - 次元: 3
  - フォーマット: 制約なし

- input1
  - 要素型: float32
  - 次元: 3
  - フォーマット: 制約なし

### 出力

- output
  - 要素型: float32
  - 次元: 3
  - フォーマット: 制約なし

### パラメータ

- enable_clamp
  - 未使用
  - 要素型: bool
