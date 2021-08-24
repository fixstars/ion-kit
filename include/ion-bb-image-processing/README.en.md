<!-- ion-bb-image-processing -->

# BayerOffset

## Description

Subtracts **subtracts** the specified value from the pixel value of the Bayer image.
The output will be clamped to [0.0, 1.0].

### Input

- input
  - element type: float32
  - Dimension: 2
  - Format: Bayer image with a value range of [0.0, 1.0].
- offset_r
  - Offset value of R pixel
  - Element type: float32
- offset_g
  - Offset value of G pixel
  - Element type: float32
- offset_b
  - Offset value of pixel B
  - Element type: float32

### Output

- output
  - Element type: float32
  - Dimension: 2
  - Format: Bayer image with a value range of [0.0, 1.0]

### Parameter

- bayer_pattern
  - Bayer format
  - Select one of the following
    - 0: RGGB
    - 1: BGGR
    - 2: GRBG
    - 3: GBRG
  - Element type: int32

# BayerWhiteBalance

## Description

This function **multiplies** the pixel value of a Bayer image by the specified value.
The output will be clamped to [0.0, 1.0].

### Input

- input
  - element type: float32
  - Dimension: 2
  - Format: Bayer image with values in the range [0.0, 1.0].
- gain_r
  - Gain value of R pixel
  - Element type: float32
- gain_g
  - Gain value of G pixels
  - Element type: float32
- gain_b
  - Gain value of B pixel
  - Element type: float32

### Output

- output
  - Element type: float32
  - Dimension: 2
  - Format: Bayer image with a value range of [0.0, 1.0].

### Parameter

- bayer_pattern
  - Bayer format
  - Select one of the following
    - 0: RGGB
    - 1: BGGR
    - 2: GRBG
    - 3: GBRG
  - Element type: int32

# BayerDemosaicSimple

## Description

Performs demosaicing of a Bayer image and outputs an RGB image.

Performs demosaicing by reducing the size of the image.
The output size will be 1/2 of the input size in both height and width.

### Input

- input
  - element type: float32
  - Dimension: 2
  - Format: Bayer image with a value range of [0.0, 1.0]

### Output

- output
  - Element type: float32
  - Dimension: 3
  - Format: CHW for dimension alignment, RGB data in the value range [0.0, 1.0].

### Parameter

- width
  - Width of the input image (pixel)
  - Element type: int32
- height
  - Height of the input image (pixel)
  - Element type: int32
- bayer_pattern
  - Bayer format
  - Select one of the following
    - 0: RGGB
    - 1: BGGR
    - 2: GRBG
    - 3: GBRG
  - Element type: int32

# BayerDemosaicLinear

## Description

Performs demosaicing of a Bayer image and outputs an RGB image.

Performs demosaicing using linear interpolation.

### Input

- input
  - Element type: float32
  - Dimension: 2
  - Format: Bayer image with a value range of [0.0, 1.0]

### Output

- output
  - Element type: float32
  - Dimension: 3
  - Format: CHW for dimension alignment, RGB data in the value range [0.0, 1.0].

### Parameter

- width
  - Width of the input image (pixel)
  - Element type: int32
- height
  - Height of the input image (pixel)
  - Element type: int32
- bayer_pattern
  - Bayer format
  - Select one of the following
    - 0: RGGB
    - 1: BGGR
    - 2: GRBG
    - 3: GBRG
  - Element type: int32

# BayerDemosaicFilter

## Description

Performs demosaicing of a Bayer image and outputs an RGB image.

Use a filter to perform demosaicing.

### Input

- input
  - Element type: float32
  - Dimension: 2
  - Format: Bayer image with a value range of [0.0, 1.0]

### Output

- output
  - Element type: float32
  - Dimension: 3
  - Format: CHW for dimension alignment, RGB data in the value range [0.0, 1.0].

### Parameter

- width
  - Width of the input image (pixel)
  - Element type: int32
- height
  - Height of the input image (pixel)
  - Element type: int32
- bayer_pattern
  - Bayer format
  - Select one of the following
    - 0: RGGB
    - 1: BGGR
    - 2: GRBG
    - 3: GBRG
  - Element type: int32

# GammaCorrection2D

## Description

Using the given input `gamma`, gamma correction is performed for each element as expressed by the following equation.

```
output = clamp(pow(input, gamma), 0.0, 1.0);
```

### Input

- input
  - element type: float32
  - Dimension: 2
  - format: data whose value range is [0..1.0].
- gamma
  - Gamma correction value
  - Element type: float32

### Output

- output
  - Element type: float32
  - Dimension: 2
  - Format: data whose value range is [0..1.0].

# GammaCorrection3D

## Description

Using the given input `gamma`, gamma correction is performed for each element as expressed by the following equation.

```
output = clamp(pow(input, gamma), 0.0, 1.0);
```

### Input

- input
  - element type: float32
  - Dimension: 3
  - format: data whose value range is [0..1.0].
- gamma
  - Gamma correction value
  - Element type: float32

### Output

- output
  - Element type: float32
  - Dimension: 3
  - Format: data whose value range is [0..1.0].

# LensShadingCorrectionLinear

## Description

Peripheral light level correction for Bayer images.

The correction is based on the assumption that the light intensity is inversely proportional to the square of the distance from the center of the image (i.e. the gain is proportional to the square of the distance).

The distance is normalized so that the maximum value is 1.

### Input

- input
  - element type: float32
  - Dimension: 2
  - Format: Bayer image with a value range of [0.0, 1.0]
- slope_r
  - Slope of the R pixel gain
  - Element type: float32
- slope_g
  - Slope of the gain of the G pixel
  - Element type: float32
- slope_b
  - Slope of the gain of B pixels
  - Element type: float32
- offset_r
  - Offset value of the gain of R pixels
  - Element type: float32
- offset_g
  - Offset value of the gain of G pixels
  - Element type: float32
- offset_b
  - Offset value of the gain of B pixels
  - Element type: float32

### Output

- output
  - Element type: float32
  - Dimension: 2
  - Format: Bayer image with a value range of [0.0, 1.0]

### Parameter

- width
  - Width of the input image (pixel)
  - Element type: int32
- height
  - Height of the input image (pixel)
  - Element type: int32
- bayer_pattern
  - Bayer format
  - Select one of the following
    - 0: RGGB
    - 1: BGGR
    - 2: GRBG
    - 3: GBRG
  - Element type: int32

# ColorMatrix

## Description

Applies a color transformation matrix to RGB images.

### Input

- input
  - element type: float32
  - Dimension: 3
  - Format: CHW for dimensioning, RGB data in the value range [0.0, 1.0]
- matrix
  - transformation matrix
  - element type: float32
  - Dimension: 2
  - Format: size 3x3

### Output

- output
  - Element type: float32
  - Dimension: 3
  - Format: CHW for dimension alignment, RGB data in the value range [0.0, 1.0].

# CalcLuminance

## Description

Converts an RGB image to a luminance image.

The conversion formula is one of the following

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

### Input

- input
  - Element type: float32
  - Dimension: 3
  - Format: CHW for dimension alignment, RGB data in the value range [0.0, 1.0].

### Output

- output
  - Element type: float32
  - Dimension: 2
  - Format: luminance data in the value range [0.0, 1.0]

### Parameter

- luminance_method
  - Calculation Formula
  - Select one of the following
    - 0: Max
    - 1: Average
    - 2: SimpleY
    - 3: Y
  - Element type: int32

# BilateralFilter2D

## Description

Applies a bilateral filter to an image.

### Input

- input
  - element type: float32
  - Dimension: 2
  - Format: image data with dimension CHW (for 3D) and value range [0.0, 1.0].
- sigma
  - sigma value of each pixel
  - element type: float32
  - Dimension: 2

### Output

- output
  - Element type: float32
  - Dimension: 2
  - Format: image data with CHW (for 3D) for dimension placement and [0.0, 1.0] for value range.

### Parameter

- width
  - Width of the input image (pixel)
  - Element type: int32
- height
  - Height of the input image (pixel)
  - Element type: int32
- window_size
  - Window size
  - Actual window size will be n^2+1
  - Example: for 5x5, window_size=2
  - Element type: int32
- coef_color
  - Coefficient of color direction
  - Element type: float32
- coef_space
  - Coefficient for the spatial direction
  - Element type: float32
- color_difference_method(BilateralFilter3D only)
  - Color difference calculation method
  - Select one of the following
    - 0: PerChannel
      - Sum of the squares of the differences per channel
    - 1: Average
      - Square of the average difference

# BilateralFilter3D

## Description

Applies a bilateral filter to an image.

### Input

- input
  - element type: float32
  - Dimension: 3
  - Format: image data with dimension CHW (for 3D) and value range [0.0, 1.0].
- sigma
  - sigma value of each pixel
  - element type: float32
  - Dimension: 2

### Output

- output
  - Element type: float32
  - Dimension: 3
  - Format: image data with CHW (for 3D) for dimension placement and [0.0, 1.0] for value range.

### Parameter

- width
  - Width of the input image (pixel)
  - Element type: int32
- height
  - Height of the input image (pixel)
  - Element type: int32
- window_size
  - Window size
  - Actual window size will be n^2+1
  - Example: for 5x5, window_size=2
  - Element type: int32
- coef_color
  - Coefficient of color direction
  - Element type: float32
- coef_space
  - Coefficient for the spatial direction
  - Element type: float32
- color_difference_method(BilateralFilter3D only)
  - Color difference calculation method
  - Select one of the following
    - 0: PerChannel
      - Sum of the squares of the differences per channel
    - 1: Average
      - Square of the average difference

# Convolution2D

## Description

Performs image convolution.

### Input

- input
  - element type: float32
  - Dimension: 2
  - Format: image data with dimension CHW (for 3D) and value range [0.0, 1.0].
- kernel
  - Convolutional kernel
  - element type: float32
  - dimension: 2

### Output

- output
  - Element type: float32
  - Dimension: 2
  - Format: image data with CHW (for 3D) for dimension placement and [0.0, 1.0] for value range.

### Parameter

- width
  - Width of the input image (pixel)
  - Element type: int32
- height
  - Height of the input image (pixel)
  - Element type: int32
- window_size
  - Window size
  - Actual window size will be n^2+1
  - Example: for 5x5, window_size=2
  - Element type: int32
- boundary_conditions_method
  - Boundary conditions
  - Select one of the following
    - 0: RepeatEdge
    - 1: RepeatImage
    - 2: MirrorImage
    - 3: MirrorInterior
    - 4: Zero

# Convolution3D

## Description

Performs image convolution.

### Input

- input
  - element type: float32
  - Dimension: 3
  - Format: image data with dimension CHW (for 3D) and value range [0.0, 1.0].
- kernel
  - Convolutional kernel
  - element type: float32
  - dimension: 2

### Output

- output
  - Element type: float32
  - Dimension: 3
  - Format: image data with CHW (for 3D) for dimension placement and [0.0, 1.0] for value range.

### Parameter

- width
  - Width of the input image (pixel)
  - Element type: int32
- height
  - Height of the input image (pixel)
  - Element type: int32
- window_size
  - Window size
  - Actual window size will be n^2+1
  - Example: for 5x5, window_size=2
  - Element type: int32
- boundary_conditions_method
  - Boundary conditions
  - Select one of the following
    - 0: RepeatEdge
    - 1: RepeatImage
    - 2: MirrorImage
    - 3: MirrorInterior
    - 4: Zero

# LensDistortionCorrectionModel2D

## Description

Corrects the lens distortion according to the parameters.

### Input

- input
  - element type: float32
  - Dimension: 2
  - Format: image data with dimension CHW (for 3D) and value range [0.0, 1.0].
- k1
  - Correction parameter
  - element type: float32
- k2
  - Correction parameter
  - Element type: float32
- k3
  - Correction parameter
  - Element type: float32
- p1
  - Correction parameter
  - Element type: float32
- p2
  - Correction parameter
  - Element type: float32
- fx
  - Correction parameter
  - Element type: float32
- fy
  - Correction parameter
  - Element type: float32
- cx
  - Correction parameter
  - Element type: float32
- cy
  - Correction parameter
  - Element type: float32
- output_scale
  - Output scale
  - Element type: float32

### Output

- output
  - Element type: float32
  - Dimension: 2
  - Format: image data with CHW (for 3D) for dimension placement and [0.0, 1.0] for value range.

### Parameter

- width
  - Width of the input image (pixel)
  - Element type: int32
- height
  - Height of the input image (pixel)
  - Element type: int32

# LensDistortionCorrectionModel3D

## Description

Corrects the lens distortion according to the parameters.

### Input

- input
  - element type: float32
  - Dimension: 3
  - Format: image data with dimension CHW (for 3D) and value range [0.0, 1.0].
- k1
  - Correction parameter
  - element type: float32
- k2
  - Correction parameter
  - Element type: float32
- k3
  - Correction parameter
  - Element type: float32
- p1
  - Correction parameter
  - Element type: float32
- p2
  - Correction parameter
  - Element type: float32
- fx
  - Correction parameter
  - Element type: float32
- fy
  - Correction parameter
  - Element type: float32
- cx
  - Correction parameter
  - Element type: float32
- cy
  - Correction parameter
  - Element type: float32
- output_scale
  - Output scale
  - Element type: float32

### Output

- output
  - Element type: float32
  - Dimension: 3
  - Format: image data with CHW (for 3D) for dimension placement and [0.0, 1.0] for value range.

### Parameter

- width
  - Width of the input image (pixel)
  - Element type: int32
- height
  - Height of the input image (pixel)
  - Element type: int32

# ResizeNearest2D

## Description

Resize the image using the Nearest Neighbor method.

### Input

- input
  - Element type: float32
  - Dimension: 2
  - Format: image data with CHW (for 3D) for dimension placement and [0.0, 1.0] for value range.

### Output

- output
  - Element type: float32
  - Dimension: 2
  - Format: image data with CHW (for 3D) for dimension placement and [0.0, 1.0] for value range.

### Parameter

- width
  - Width of the input image (pixel)
  - Element type: int32
- height
  - Height of the input image (pixel)
  - Element type: int32
- scale
  - Scaling ratio
  - Example: 0.5 for a 1/2 scale
  - Element type: float32

# ResizeNearest3D

## Description

Resize the image using the Nearest Neighbor method.

### Input

- input
  - Element type: float32
  - Dimension: 3
  - Format: image data with CHW (for 3D) for dimension placement and [0.0, 1.0] for value range.

### Output

- output
  - Element type: float32
  - Dimension: 3
  - Format: image data with CHW (for 3D) for dimension placement and [0.0, 1.0] for value range.

### Parameter

- width
  - Width of the input image (pixel)
  - Element type: int32
- height
  - Height of the input image (pixel)
  - Element type: int32
- scale
  - Scaling ratio
  - Example: 0.5 for a 1/2 scale
  - Element type: float32

# ResizeBilinear2D

## Description

Resizes RGB images using the bilinear method.

### Input

- input
  - Element type: float32
  - Dimension: 2
  - Format: image data with CHW (for 3D) for dimension placement and [0.0, 1.0] for value range.

### Output

- output
  - Element type: float32
  - Dimension: 2
  - Format: image data with CHW (for 3D) for dimension placement and [0.0, 1.0] for value range.

### Parameter

- width
  - Width of the input image (pixel)
  - Element type: int32
- height
  - Height of the input image (pixel)
  - Element type: int32
- scale
  - Scaling ratio
  - Example: 0.5 for a 1/2 scale
  - Element type: float32

# ResizeBilinear3D

## Description

Resizes RGB images using the bilinear method.

### Input

- input
  - Element type: float32
  - Dimension: 3
  - Format: image data with CHW (for 3D) for dimension placement and [0.0, 1.0] for value range.

### Output

- output
  - Element type: float32
  - Dimension: 3
  - Format: image data with CHW (for 3D) for dimension placement and [0.0, 1.0] for value range.

### Parameter

- width
  - Width of the input image (pixel)
  - Element type: int32
- height
  - Height of the input image (pixel)
  - Element type: int32
- scale
  - Scaling ratio
  - Example: 0.5 for a 1/2 scale
  - Element type: float32


# ResizeAreaAverage2D

## Description

Resizes RGB images using the average pixel method.

### Input

- input
  - Element type: float32
  - Dimension: 2
  - Format: image data with CHW (for 3D) for dimension placement and [0.0, 1.0] for value range.

### Output

- output
  - Element type: float32
  - Dimension: 2
  - Format: image data with CHW (for 3D) for dimension placement and [0.0, 1.0] for value range.

### Parameter

- width
  - Width of the input image (pixel)
  - Element type: int32
- height
  - Height of the input image (pixel)
  - Element type: int32
- scale
  - Scaling ratio
  - Example: 0.5 for a 1/2 scale
  - Element type: float32

# ResizeAreaAverage3D

## Description

Resizes RGB images using the average pixel method.

### Input

- input
  - Element type: float32
  - Dimension: 3
  - Format: image data with CHW (for 3D) for dimension placement and [0.0, 1.0] for value range.

### Output

- output
  - Element type: float32
  - Dimension: 3
  - Format: image data with CHW (for 3D) for dimension placement and [0.0, 1.0] for value range.

### Parameter

- width
  - Width of the input image (pixel)
  - Element type: int32
- height
  - Height of the input image (pixel)
  - Element type: int32
- scale
  - Scaling ratio
  - Example: 0.5 for a 1/2 scale
  - Element type: float32

# BayerDownscaleUInt16

## Description

Downscales a Bayer image.

### Input

- input
  - Element type: uint16
  - Dimension: 2
  - Format: 2x2 Bayer Array

### Output

- output
  - Element type: uint16
  - Dimension: 2
  - Format: 2x2 Bayer Array

### Parameter

- input_width
  - Width of the input image (pixel)
  - Element type: int32
- input_height
  - Height of the input image (pixel)
  - Element type: int32
- downscale_factor
  - Downscale factor
  - Example: 2 for 1/2 reduction
  - Element type: int32

# Normalize RAW

## Description

Outputs a RAW image normalized to [0.0, 1.0].

### Input

- input
  - Element type: uint16
  - Dimension: 2
  - Format: unconstrained

### Output

- output
  - Element type: float32
  - Dimension: 2
  - Format: unrestricted

### Parameter

- bit_width
  - Bit width of RAW image
  - Example: 10 is specified for 10-bit overpadding.
  - Element type: uint8
- bit_shift
  - Least significant bit position in RAW image
  - Example: Specify 6 for 10-bit justification
  - Element type: uint8

# FitImageToCenter2DUInt8

## Description

Center the image.

The image will be cropped if the input size is large, or padded with 0 if it is small.

### Input

- input
  - Element type: uint8
  - Dimension: 2
  - Format: image data whose dimension is CHW (for 3D)

### Output

- output
  - Element type: uint8
  - Dimension: 2
  - Format: image data whose dimension is CHW (for 3D)

### Parameter

- input_width
  - Width of the input image (pixel)
- input_height
  - Input image height (pixel)
- output_width
  - Width of output image (pixel)
- output_height
  - Output image height (pixel)

# FitImageToCenter3DUInt8

## Description

Center the image.

The image will be cropped if the input size is large, or padded with 0 if it is small.

### Input

- input
  - Element type: uint8
  - Dimension: 3
  - Format: image data whose dimension is CHW (for 3D)

### Output

- output
  - Element type: uint8
  - Dimension: 3
  - Format: image data whose dimension is CHW (for 3D)

### Parameter

- input_width
  - Width of the input image (pixel)
- input_height
  - Input image height (pixel)
- output_width
  - Width of output image (pixel)
- output_height
  - Output image height (pixel)

# FitImageToCenter2DFloat

## Description

Center the image.

The image will be cropped if the input size is large, or padded with 0 if it is small.

### Input

- input
  - Element type: float32
  - Dimension: 2
  - Format: image data whose dimension is CHW (for 3D)

### Output

- output
  - Element type: float32
  - Dimension: 2
  - Format: image data whose dimension is CHW (for 3D)

### Parameter

- input_width
  - Width of the input image (pixel)
- input_height
  - Input image height (pixel)
- output_width
  - Width of output image (pixel)
- output_height
  - Output image height (pixel)

# FitImageToCenter3DFloat

## Description

Center the image.

The image will be cropped if the input size is large, or padded with 0 if it is small.

### Input

- input
  - Element type: float32
  - Dimension: 3
  - Format: image data whose dimension is CHW (for 3D)

### Output

- output
  - Element type: float32
  - Dimension: 3
  - Format: image data whose dimension is CHW (for 3D)

### Parameter

- input_width
  - Width of the input image (pixel)
- input_height
  - Input image height (pixel)
- output_width
  - Width of output image (pixel)
- output_height
  - Output image height (pixel)

# ReorderColorChannel3DUInt8

## Description

Converts the channel order of RGB images in reverse order.

It can be used for both RGB->BGR conversion and BGR->RGB conversion.

### Input

- input
  - Element type: uint8
  - Dimension: 3
  - Format: RGB data

### Output

- output
  - Element type: uint8
  - Dimension: 3
  - Format: RGB data

### Parameter

- color_dim
  - Dimension representing a color channel.
  - Element type: int32

# ReorderColorChannel3DFloat

## Description

Converts the channel order of RGB images in reverse order.

It can be used for both RGB->BGR conversion and BGR->RGB conversion.

### Input

- input
  - Element type: float32
  - Dimension: 3
  - Format: RGB data

### Output

- output
  - Element type: float32
  - Dimension: 3
  - Format: RGB data

### Parameter

- color_dim
  - Dimension representing a color channel.
  - Element type: int32

# OverlayImage2DUInt8

## Description

Combines two images into one.

Input0 is combined as background and input1 as foreground.

Out of range areas will be filled with zeroes.

### Input

- input0
  - Element type: uint8
  - Dimension: 2
  - Format: image data
- input1
  - Element type: uint8
  - Dimension: 2
  - Format: image data

### Output

- output
  - Element type: uint8
  - Dimension: 2
  - Format: image data

### Parameter

- x_dim
  - Dimension representing the horizontal direction.
  - 0 for CHW format
  - Element type: int32
- y_dim
  - Dimension for vertical direction.
  - 1 if in CHW format.
  - Element type: int32
- input0_width
  - Width of the background (pixel)
  - Element type: int32
- input0_height
  - Height of the background (pixel)
  - Element type: int32
- input1_left
  - Horizontal position of the foreground (pixel)
  - Element type: int32
- input1_top
  - Vertical position of the foreground (pixel)
  - Element type: int32
- input1_width
  - Width of the foreground (pixel)
  - Element type: int32
- input1_height
  - Foreground height (pixel)
  - Element type: int32

# OverlayImage3DUInt8

## Description

Combines two images into one.

Input0 is combined as background and input1 as foreground.

Out of range areas will be filled with zeroes.

### Input

- input0
  - Element type: uint8
  - Dimension: 3
  - Format: image data
- input1
  - Element type: uint8 / float32
  - Dimension: 3
  - Format: image data

### Output

- output
  - Element type: uint8
  - Dimension: 3
  - Format: image data

### Parameter

- x_dim
  - Dimension representing the horizontal direction.
  - 0 for CHW format
  - Element type: int32
- y_dim
  - Dimension for vertical direction.
  - 1 if in CHW format.
  - Element type: int32
- input0_width
  - Width of the background (pixel)
  - Element type: int32
- input0_height
  - Height of the background (pixel)
  - Element type: int32
- input1_left
  - Horizontal position of the foreground (pixel)
  - Element type: int32
- input1_top
  - Vertical position of the foreground (pixel)
  - Element type: int32
- input1_width
  - Width of the foreground (pixel)
  - Element type: int32
- input1_height
  - Foreground height (pixel)
  - Element type: int32

# OverlayImage2DFloat

## Description

Combines two images into one.

Input0 is combined as background and input1 as foreground.

Out of range areas will be filled with zeroes.

### Input

- input0
  - Element type: float32
  - Dimension: 2
  - Format: image data
- input1
  - Element type: float32
  - Dimension: 2
  - Format: image data

### Output

- output
  - Element type: float32
  - Dimension: 2
  - Format: image data

### Parameter

- x_dim
  - Dimension representing the horizontal direction.
  - 0 for CHW format
  - Element type: int32
- y_dim
  - Dimension for vertical direction.
  - 1 if in CHW format.
  - Element type: int32
- input0_width
  - Width of the background (pixel)
  - Element type: int32
- input0_height
  - Height of the background (pixel)
  - Element type: int32
- input1_left
  - Horizontal position of the foreground (pixel)
  - Element type: int32
- input1_top
  - Vertical position of the foreground (pixel)
  - Element type: int32
- input1_width
  - Width of the foreground (pixel)
  - Element type: int32
- input1_height
  - Foreground height (pixel)
  - Element type: int32

# OverlayImage3DFloat

## Description

Combines two images into one.

Input0 is combined as background and input1 as foreground.

Out of range areas will be filled with zeroes.

### Input

- input0
  - Element type: float32
  - Dimension: 3
  - Format: image data
- input1
  - Element type: float32
  - Dimension: 3
  - Format: image data

### Output

- output
  - Element type: float32
  - Dimension: 3
  - Format: image data

### Parameter

- x_dim
  - Dimension representing the horizontal direction.
  - 0 for CHW format
  - Element type: int32
- y_dim
  - Dimension for vertical direction.
  - 1 if in CHW format.
  - Element type: int32
- input0_width
  - Width of the background (pixel)
  - Element type: int32
- input0_height
  - Height of the background (pixel)
  - Element type: int32
- input1_left
  - Horizontal position of the foreground (pixel)
  - Element type: int32
- input1_top
  - Vertical position of the foreground (pixel)
  - Element type: int32
- input1_width
  - Width of the foreground (pixel)
  - Element type: int32
- input1_height
  - Foreground height (pixel)
  - Element type: int32

# TileImageHorizontal2DUInt8

## Description

Combines two images into one.

The images will be joined horizontally.

Out of bounds will be filled with zeroes.

### Input

- input0
  - Element type: uint8
  - Dimension: 2
  - Format: image data
- input1
  - Element type: uint8
  - Dimension: 2
  - Format: image data

### Output

- output
  - Element type: uint8
  - Dimension: 2
  - Format: image data

### Parameter

- x_dim
  - Dimension representing the horizontal direction.
  - 0 for CHW format
  - Element type: int32
- y_dim
  - Dimension for vertical direction.
  - 1 if in CHW format.
  - Element type: int32
- input0_width
  - Width of image 0 (pixel)
  - Element type: int32
- input0_height
  - Height of image 0 (pixel)
  - Element type: int32
- input1_width
  - Width of image 1 (pixel)
  - Element type: int32
- input1_height
  - Height of image 1 (pixel)
  - Element type: int32

# TileImageHorizontal3DUInt8

## Description

Combines two images into one.

The images will be joined horizontally.

Out of bounds will be filled with zeroes.

### Input

- input0
  - Element type: uint8
  - Dimension: 3
  - Format: image data
- input1
  - Element type: uint8
  - Dimension: 3
  - Format: image data

### Output

- output
  - Element type: uint8
  - Dimension: 3
  - Format: image data

### Parameter

- x_dim
  - Dimension representing the horizontal direction.
  - 0 for CHW format
  - Element type: int32
- y_dim
  - Dimension for vertical direction.
  - 1 if in CHW format.
  - Element type: int32
- input0_width
  - Width of image 0 (pixel)
  - Element type: int32
- input0_height
  - Height of image 0 (pixel)
  - Element type: int32
- input1_width
  - Width of image 1 (pixel)
  - Element type: int32
- input1_height
  - Height of image 1 (pixel)
  - Element type: int32

# TileImageHorizontal2DFloat

## Description

Combines two images into one.

The images will be joined horizontally.

Out of bounds will be filled with zeroes.

### Input

- input0
  - Element type: float32
  - Dimension: 2
  - Format: image data
- input1
  - Element type: float32
  - Dimension: 2
  - Format: image data

### Output

- output
  - Element type: float32
  - Dimension: 2
  - Format: image data

### Parameter

- x_dim
  - Dimension representing the horizontal direction.
  - 0 for CHW format
  - Element type: int32
- y_dim
  - Dimension for vertical direction.
  - 1 if in CHW format.
  - Element type: int32
- input0_width
  - Width of image 0 (pixel)
  - Element type: int32
- input0_height
  - Height of image 0 (pixel)
  - Element type: int32
- input1_width
  - Width of image 1 (pixel)
  - Element type: int32
- input1_height
  - Height of image 1 (pixel)
  - Element type: int32

# TileImageHorizontal3DFloat

## Description

Combines two images into one.

The images will be joined horizontally.

Out of bounds will be filled with zeroes.

### Input

- input0
  - Element type: float32
  - Dimension: 3
  - Format: image data
- input1
  - Element type: float32
  - Dimension: 3
  - Format: image data

### Output

- output
  - Element type: float32
  - Dimension: 3
  - Format: image data

### Parameter

- x_dim
  - Dimension representing the horizontal direction.
  - 0 for CHW format
  - Element type: int32
- y_dim
  - Dimension for vertical direction.
  - 1 if in CHW format.
  - Element type: int32
- input0_width
  - Width of image 0 (pixel)
  - Element type: int32
- input0_height
  - Height of image 0 (pixel)
  - Element type: int32
- input1_width
  - Width of image 1 (pixel)
  - Element type: int32
- input1_height
  - Height of image 1 (pixel)
  - Element type: int32

# TileImageVertical2DUInt8

## Description

Combines two images into one.

The images will be vertically concatenated.

Out of bounds will be filled with zeroes.

### Input

- input0
  - Element type: uint8
  - Dimension: 2
  - Format: image data
- input1
  - Element type: uint8
  - Dimension: 2
  - Format: image data

### Output

- output
  - Element type: uint8
  - Dimension: 2
  - Format: image data

### Parameter

- x_dim
  - Dimension representing the horizontal direction.
  - 0 for CHW format
  - Element type: int32
- y_dim
  - Dimension for vertical direction.
  - 1 if in CHW format.
  - Element type: int32
- input0_width
  - Width of image 0 (pixel)
  - Element type: int32
- input0_height
  - Height of image 0 (pixel)
  - Element type: int32
- input1_width
  - Width of image 1 (pixel)
  - Element type: int32
- input1_height
  - Height of image 1 (pixel)
  - Element type: int32

# TileImageVertical3DUInt8

## Description

Combines two images into one.

The images will be vertically concatenated.

Out of bounds will be filled with zeroes.

### Input

- input0
  - Element type: uint8
  - Dimension: 3
  - Format: image data
- input1
  - Element type: uint8
  - Dimension: 3
  - Format: image data

### Output

- output
  - Element type: uint8 / float32
  - Dimension: 2 / 3
  - Format: image data

### Parameter

- x_dim
  - Dimension representing the horizontal direction.
  - 0 for CHW format
  - Element type: int32
- y_dim
  - Dimension for vertical direction.
  - 1 if in CHW format.
  - Element type: int32
- input0_width
  - Width of image 0 (pixel)
  - Element type: int32
- input0_height
  - Height of image 0 (pixel)
  - Element type: int32
- input1_width
  - Width of image 1 (pixel)
  - Element type: int32
- input1_height
  - Height of image 1 (pixel)
  - Element type: int32

# TileImageVertical2DFloat

## Description

Combines two images into one.

The images will be vertically concatenated.

Out of bounds will be filled with zeroes.

### Input

- input0
  - Element type: float32
  - Dimension: 2
  - Format: image data
- input1
  - Element type: float32
  - Dimension: 2
  - Format: image data

### Output

- output
  - Element type: ufloat32
  - Dimension: 2
  - Format: image data

### Parameter

- x_dim
  - Dimension representing the horizontal direction.
  - 0 for CHW format
  - Element type: int32
- y_dim
  - Dimension for vertical direction.
  - 1 if in CHW format.
  - Element type: int32
- input0_width
  - Width of image 0 (pixel)
  - Element type: int32
- input0_height
  - Height of image 0 (pixel)
  - Element type: int32
- input1_width
  - Width of image 1 (pixel)
  - Element type: int32
- input1_height
  - Height of image 1 (pixel)
  - Element type: int32

# TileImageVertical3DFloat

## Description

Combines two images into one.

The images will be vertically concatenated.

Out of bounds will be filled with zeroes.

### Input

- input0
  - Element type: float32
  - Dimension: 2
  - Format: image data
- input1
  - Element type: float32
  - Dimension: 2
  - Format: image data

### Output

- output
  - Element type: float32
  - Dimension: 2
  - Format: image data

### Parameter

- x_dim
  - Dimension representing the horizontal direction.
  - 0 for CHW format
  - Element type: int32
- y_dim
  - Dimension for vertical direction.
  - 1 if in CHW format.
  - Element type: int32
- input0_width
  - Width of image 0 (pixel)
  - Element type: int32
- input0_height
  - Height of image 0 (pixel)
  - Element type: int32
- input1_width
  - Width of image 1 (pixel)
  - Element type: int32
- input1_height
  - Height of image 1 (pixel)
  - Element type: int32

# CropImage2DUInt8

## Description

Crops a portion of the image area.

Out of range areas will be filled with zero.

### Input

- input
  - Element type: uint8
  - Dimension: 2
  - Format: image data

### Output

- output
  - Element type: uint8
  - Dimension: 2
  - Format: image data

### Parameter

- x_dim
  - Dimension representing the horizontal direction.
  - 0 for CHW format
  - Element type: int32
- y_dim
  - Dimension for vertical direction.
  - 1 if in CHW format.
  - Element type: int32
- input_width
  - Width of the input image (pixel)
  - Element type: int32
- input_height
  - Height of the input image (pixel)
  - Element type: int32
- left
  - Horizontal position of the area (pixel)
  - Element type: int32
- top
  - Vertical position of the area (pixel)
  - Element type: int32
- output_width
  - Width of the output image (pixel)
  - Element type: int32
- output_height
  - Height of the output image (pixel)
  - Element type: int32

# CropImage3DUInt8

## Description

Crops a portion of the image area.

Out of range areas will be filled with zero.

### Input

- input
  - Element type: uint8
  - Dimension: 3
  - Format: image data

### Output

- output
  - Element type: uint8
  - Dimension: 3
  - Format: image data

### Parameter

- x_dim
  - Dimension representing the horizontal direction.
  - 0 for CHW format
  - Element type: int32
- y_dim
  - Dimension for vertical direction.
  - 1 if in CHW format.
  - Element type: int32
- input_width
  - Width of the input image (pixel)
  - Element type: int32
- input_height
  - Height of the input image (pixel)
  - Element type: int32
- left
  - Horizontal position of the area (pixel)
  - Element type: int32
- top
  - Vertical position of the area (pixel)
  - Element type: int32
- output_width
  - Width of the output image (pixel)
  - Element type: int32
- output_height
  - Height of the output image (pixel)
  - Element type: int32

# CropImage2DFloat

## Description

Crops a portion of the image area.

Out of range areas will be filled with zero.

### Input

- input
  - Element type: float32
  - Dimension: 2
  - Format: image data

### Output

- output
  - Element type: float32
  - Dimension: 2
  - Format: image data

### Parameter

- x_dim
  - Dimension representing the horizontal direction.
  - 0 for CHW format
  - Element type: int32
- y_dim
  - Dimension for vertical direction.
  - 1 if in CHW format.
  - Element type: int32
- input_width
  - Width of the input image (pixel)
  - Element type: int32
- input_height
  - Height of the input image (pixel)
  - Element type: int32
- left
  - Horizontal position of the area (pixel)
  - Element type: int32
- top
  - Vertical position of the area (pixel)
  - Element type: int32
- output_width
  - Width of the output image (pixel)
  - Element type: int32
- output_height
  - Height of the output image (pixel)
  - Element type: int32

# CropImage3DFloat

## Description

Crops a portion of the image area.

Out of range areas will be filled with zero.

### Input

- input
  - Element type: float32
  - Dimension: 3
  - Format: image data

### Output

- output
  - Element type: float32
  - Dimension: 3
  - Format: image data

### Parameter

- x_dim
  - Dimension representing the horizontal direction.
  - 0 for CHW format
  - Element type: int32
- y_dim
  - Dimension for vertical direction.
  - 1 if in CHW format.
  - Element type: int32
- input_width
  - Width of the input image (pixel)
  - Element type: int32
- input_height
  - Height of the input image (pixel)
  - Element type: int32
- left
  - Horizontal position of the area (pixel)
  - Element type: int32
- top
  - Vertical position of the area (pixel)
  - Element type: int32
- output_width
  - Width of the output image (pixel)
  - Element type: int32
- output_height
  - Height of the output image (pixel)
  - Element type: int32

# ColorSpaceConverter RGB to HSV

## Description

Converts from RGB color space to HSV color space.

### Input

- input
  - Element type: float32
  - Dimension: 3
  - Format: CHW for dimension alignment, RGB data with value range [0..1.0].

### Output

- output
  - Element type: float32
  - Dimension: 3
  - Format: CHW for dimension alignment, HSV data with [0..1.0] for value range.

### Parameter

None

# ColorSpaceConverter HSV to RGB

## Description

Converts from HSV color space to RGB color space.

### Input

- input
  - Element type: float32
  - Dimension: 3
  - Format: CHW for dimension alignment, HSV data with [0..1.0] for value range.

### Output

- output
  - Element type: float32
  - Dimension: 3
  - Format: CHW for dimension alignment, RGB data with value range [0..1.0].

### Parameter

None

# Color Adjustment

## Description

Apply color correction to `target_channel` using the given parameter `adjustment_value`.

```
output = select(c == target_channel, clamp(pow(input, adjustment_value), 0.0, 1.0), input);
```

### Input

- input
  - Element type: float32
  - Dimension: 3
  - Format: data with CHW for dimensionality and [0..1.0] for value range.

### Output

- output
  - Element type: float32
  - Dimension: 3
  - Format: data with CHW for dimensionality and [0..1.0] for value range.

### Parameter

- adjustment_value
  - Adjustment parameter
  - Element type: float32
- target_channel
  - target channel
  - element type: int32
