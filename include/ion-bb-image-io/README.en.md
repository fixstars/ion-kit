# Building Block Reference Manual
<!-- ion-bb-image-io -->

## IMX219

Import a RAW image from the connected IMX219.

In the simulation mode, you can generate a RAW image from the image by specifying the image URL.

If the camera is not detected, it will automatically enter the simulation mode.

### Input

None

### Output

- output
  - Element type: uint16
  - Dimension: 2
  - Format: Top-padded 10-bit, RGGB Bayer image of size 3264x2464

### Parameter

- index
  - Camera number
  - Element type: int32
- force_sim_mode
  - Use simulation mode even if a camera is connected.
  - Element type: bool
- url
  - Image URL to use in simulation mode.
  - **https not supported**.
  - Element type: string

## D435

Grab the stereo and depth images from the connected D435.

If no camera is detected, the system will automatically enter simulation mode.

### Input

None

### Output

- output_l
  - Element type: uint8
  - Dimension: 2
  - Format: left image of size 1280x720
- output_r
  - Element type: uint8
  - Dimension: 2
  - Format: right image of size 1280x720
- output_d
  - Element type: uint16
  - Dimension: 2
  - Format: depth image of size 1280x720

## USBCamera

This block is for handling USB cameras.

`/dev/video<index>` Retrieves data from the device with the specified image size and converts it to RGB8.

If the camera is not detected, it will automatically enter simulation mode.

!> **Behavior during benchmarking**  
Since no real device is connected to the benchmark environment, it is in simulation mode.
Therefore, the benchmark results of this block itself will be different from those of the real device.

### Input

None

### Output

- output
  - Element type: uint8
  - Dimension: 3
  - Format: CHW for dimension alignment, RGB data in the value range [0..255].

### Parameter

- width
  - Width of the image (pixel)
  - Element type: int32
- height
  - Height of the image (pixel)
  - Element type: int32
- index
  - Camera number
  - element type: int32
- url
  - Image URL to use in simulation mode.
  - **https not supported**.
  - Element type: string

## GenericV4L2Bayer

This block is for handling RAW cameras.

`/dev/video<index>` Retrieves RAW data from the device with the specified image size and outputs a Bayer image.

If the camera is not detected, it will automatically enter simulation mode.

!> **Behavior during benchmarking**  
Since no real device is connected to the benchmark environment, it is in simulation mode.
Therefore, the benchmark results of this block itself will be different from those of the real device.

### Input

None

### Output

- output
  - Element type: uint16
  - Dimension: 2
  - Format: Bayer Image

### Parameter

- width
  - Width of the image (pixel)
  - Element type: int32
- height
  - Height of the image (pixel)
  - Element type: int32
- index
  - Camera number
  - Element type: int32
- bit_width
  - RAW bit width
  - Element type: int32
- format
  - Bayer format
  - Select one of the following
    - 0: RGGB
    - 1: BGGR
    - 2: GRBG
    - 3: GBRG
  - Element type: int32
- url
  - Image URL to use in simulation mode.
  - **https not supported**.
  - Element type: string

## CameraSimulation

This block simulates a RAW camera.

It has detailed parameters for simulation.

### Input

None

### Output

- output
  - Element type: uint16
  - Dimension: 2
  - Format: Bayer Image

### Parameter

- width
  - Width of the image (pixel)
  - Element type: int32
- height
  - Height of the image (pixel)
  - Element type: int32
- bit_width
  - RAW bit width
  - Element type: int32
- bit_shift
  - Amount of bit shift
  - Element type: int32
- format
  - Bayer format
  - Select one of the following
    - 0: RGGB
    - 1: BGGR
    - 2: GRBG
    - 3: GBRG
  - Element type: int32
- offset
  - Offset of the pixel
  - Element type: float32
- gain_r
  - Gain of R pixel
  - Element type: float32
- gain_g
  - Gain of G pixels
  - Element type: float32
- gain_b
  - Gain of B pixels
  - Element type: float32
- url
  - Image URL
  - **https not supported**.
  - Element type: string

## GUI Display

Displays an RGB image in a window.

!> **Behavior during benchmarking**  
Since there is no real device connected to the benchmark environment, this block simply does nothing.
Therefore, the benchmark results of this block itself will be different from those of the real device.

### Input

- input
  - Element type: uint8
  - Dimension: 3
  - Format: CHW for dimension alignment, RGB data in the value range [0..255].

### Output

- output
  - Element type: int32
  - Dimension: 0
  - Format: unconstrained

### Parameter

- idx
  - Window
  - Element type: int32
- width
  - Width of the image (pixel)
  - Element type: int32
- height
  - Height of the image (pixel)
  - Element type: int32

## FBDisplay

Outputs data to the frame buffer.
It receives RGB data and writes the data to the `/dev/fb0` device with the specified image size.

?> **warning Restrictions on Pipeline Configuration**  
Currently, this block does not support multiple frame buffers.
Please be careful to include only a single Display block in the pipeline.

!> **Behavior during benchmarking**  
Since there is no real device connected to the benchmark environment, this block simply does nothing.
Therefore, the benchmark results of this block itself will be different from those of the real device.

### Input

- input
  - Element type: uint8
  - Dimension: 3
  - Format: CHW for dimension alignment, RGB data in the value range [0..255].

### Output

- output
  - Element type: uint32
  - Dimension: 0
  - Format: unconstrained

### Parameter

- width
  - Width of the image (pixel)
  - Element type: int32
- height
  - Height of the image (pixel)
  - Element type: int32

## Data Loader / Grayscale

Reads an image file (bmp, png, jpg, etc.) specified by the given parameter `url`, converts it to Grayscale data, and outputs it.

### Input

None

### Output

- output
  - Element type: uint16
  - Dimension: 2
  - Format: HW for dimension alignment, Grayscale data in the value range [0.. `dynamic_range` ].

### Parameter

- width
  - Width of the image (pixel)
  - Element type: int32
- height
  - Height of the image (pixel)
  - element type: int32
- dynamic_range
  - Dynamic range of the image (dB)
  - element type: int32
- url
  - URL of the image to load
    - The files that can be specified are image files such as bmp, png, jpg, raw, or zip format files that are compressed by combining multiple images.
      - Please note that the raw file is a dump of the raw pixel data, not the data obtained from the image sensor. The expected image formats are as follows.
        - Element type: uint8 or uint16
        - Number of channels: 1
      - If a zip file is specified, the files in the zip file are read in order of name and output sequentially. The number of files output when creating a preview is 1 for normal previews and the number specified by the number of previews for sequential previews. If the number of previews entered is larger than the number of files in the zip file, the files are output repeatedly in the order of file names.
  - **https not supported**.
  - element type: string

## Data Loader / Color

Reads an image file (bmp, png, jpg, etc.) specified by the given parameter `url`, converts it to RGB data, and outputs it.

### Input

None

### Output

- output
  - Element type: uint8
  - Dimension: 3
  - Format: CHW for dimension alignment, RGB data in the value range [0..255].

### Parameter

- width
  - Width of the image (pixel)
  - Element type: int32
- height
  - Height of the image (pixel)
  - element type: int32
- url
  - URL of the image to load
    - The files that can be specified are image files such as bmp, png, jpg, raw, or zip format files that are compressed by combining multiple images.
      - Please note that the raw file is a dump of the raw pixel data, not the data obtained from the image sensor. The expected image formats are as follows.
        - Element type: uint8
        - Number of channels: 3
      - If a zip file is specified, the files in the zip file are read in order of name and output sequentially. The number of files output when creating a preview is 1 for normal previews and the number specified by the number of previews for sequential previews. If the number of previews entered is larger than the number of files in the zip file, the files are output repeatedly in the order of file names.
  - **https not supported**.
  - element type: string

## Image Saver

Writes the RGB input data as an image file to the path specified by the given parameter `path`.
The file format to be written out is automatically determined from the file extension.

### Input

- input
  - Element type: uint8
  - Dimension: 3
  - Format: CHW for dimension alignment, RGB data in the value range [0..255].

### Output

None

### Parameter

- width
  - Width of the image (pixel)
  - Element type: int32
- height
  - Height of the image in pixels.
  - Element type: int32
- path
  - File path of the image to write.
  - Element type: string
