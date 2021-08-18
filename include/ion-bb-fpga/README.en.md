<!-- ion-bb-fpga -->

# Simple ISP(FPGA)

## Description

The following ISP processes are performed on FPGA.

Please refer to the reference of the BB of the same name for the contents of each process.

Please note that the accuracy of this BB is different from the original BB because it is designed for FPGA.

- Normalize RAW
- BayerOffset
- BayerWhiteBalance
- BayerDemosaicSimple
- GammaCorrection3D

### Input

- input
  - Element type: uint16
  - Dimension: 2
  - Format: Bayer Image

### Output

- output
  - Element type: uint8
  - Dimension: 3
  - Format: RGB data with **HWC** for dimension alignment and [0..255] for value range

### Parameter

- width
  - Width of the image (pixel)
  - Element type: int32
- height
  - Height of the image (pixel)
  - Element type: int32
- bayer_pattern
  - Bayer format
  - Select one of the following
    - 0: RGGB
    - 1: BGGR
    - 2: GRBG
    - 3: GBRG
  - Element type: int32
- normalize_input_bits
  - Bit width of RAW image
  - Example: 10 is specified for 10-bit overpadding
  - Element type: int32
- normalize_input_shift
  - Least significant bit position of RAW image
  - Example: Specify 6 for 10-bit justification
  - Element type: int32
- offset_offset_r
  - Offset value of R pixel
  - Element type: uint16
- offset_offset_g
  - Offset value of G pixel
  - Element type: uint16
- offset_offset_b
  - Offset value of B pixel
  - Element type: uint16
- white_balance_gain_r
  - Gain value of R pixel
  - Fixed-point (4-bit integer part, 12-bit decimal part)
  - Element type: uint16
- white_balance_gain_g
  - Gain value of G pixel
  - Fixed-point (4-bit integer part, 12-bit decimal part)
  - Element type: uint16
- white_balance_gain_b
  - Gain value of B pixel
  - Fixed-point (4-bit integer part, 12-bit decimal part)
  - Element type: uint16
- gamma_gamma
  - Gamma correction value
  - Element type: double
- unroll_level
  - Number of operator expansions
    - Can range from 0 to 3.
    - The higher the number, the better the performance, but the more circuitry is used.
    - Valid only for FPGA execution
  - Element type: int32

# Simple ISP with unsharp mask(FPGA)

## Description

The following ISP processes are performed on FPGA.

Please refer to the reference of the BB of the same name for the contents of each process.

Please note that the accuracy of this BB is different from the original BB because it is designed for FPGA.

- Normalize RAW
- BayerOffset
- BayerWhiteBalance
- BayerDemosaicSimple
- Convolution3D (unsharp mask)
- GammaCorrection3D

Use the following 3x3 kernel as the unsharp mask.

```
-0.11 -0.11 -0.11
-0.11  1.88 -0.11
-0.11 -0.11 -0.11
```

### Input

- input
  - Element type: uint16
  - Dimension: 2
  - Format: Bayer Image

### Output

- output
  - Element type: uint8
  - Dimension: 3
  - Format: RGB data with **HWC** for dimension alignment and [0..255] for value range

### Parameter

- width
  - Width of the image (pixel)
  - Element type: int32
- height
  - Height of the image (pixel)
  - Element type: int32
- bayer_pattern
  - Bayer format
  - Select one of the following
    - 0: RGGB
    - 1: BGGR
    - 2: GRBG
    - 3: GBRG
  - Element type: int32
- normalize_input_bits
  - Bit width of RAW image
  - Example: 10 is specified for 10-bit overpadding
  - Element type: int32
- normalize_input_shift
  - Least significant bit position of RAW image
  - Example: Specify 6 for 10-bit justification
  - Element type: int32
- offset_offset_r
  - Offset value of R pixel
  - Element type: uint16
- offset_offset_g
  - Offset value of G pixel
  - Element type: uint16
- offset_offset_b
  - Offset value of B pixel
  - Element type: uint16
- white_balance_gain_r
  - Gain value of R pixel
  - Fixed-point (4-bit integer part, 12-bit decimal part)
  - Element type: uint16
- white_balance_gain_g
  - Gain value of G pixel
  - Fixed-point (4-bit integer part, 12-bit decimal part)
  - Element type: uint16
- white_balance_gain_b
  - Gain value of B pixel
  - Fixed-point (4-bit integer part, 12-bit decimal part)
  - Element type: uint16
- gamma_gamma
  - Gamma correction value
  - Element type: double
- unroll_level
  - Number of operator expansions
    - Can range from 0 to 3.
    - The higher the number, the better the performance, but the more circuitry is used.
    - Valid only for FPGA execution
  - Element type: int32
