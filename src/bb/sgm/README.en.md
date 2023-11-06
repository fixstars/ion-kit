<!-- ion-bb-sgm -->

# Stereo Matching

## Description

Estimate depth information from stereo images using the Semi-Global Matching algorithm.

### Input

- input_l
  - Element type: uint8
  - Dimension: 2
  - Format: Mono8, dimension is HW, monochrome data with value range [0..255].
- input_r
  - Element type: uint8
  - Dimension: 2
  - Format: Mono8, dimension is HW, monochrome data whose value range is [0..255].

### Output

- output
  - Element type: uint8
  - Dimension: 2
  - Format: depth data with dimension HW and value range [0..255].

### Parameter

- disp
  - Depth parameter
  - Element type: int32
- width
  - Width of the image (pixel).
  - Element type: int32
- height
  - Height of the image in pixels.
  - Element type: int32
