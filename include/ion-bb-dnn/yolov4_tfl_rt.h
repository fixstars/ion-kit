#include <algorithm>
#include <chrono>  // NOLINT
#include <iostream>
#include <cassert>
#include <cstring>
#include <fstream>
#include <memory>
#include <ostream>
#include <string>
#include <vector>

#include "edgetpu_c.h"
#include "tensorflowlite_c.h"

std::vector<uint8_t> decode_bmp(const uint8_t* input, int row_size, int width,
                                int height, int channels, bool top_down) {
  std::vector<uint8_t> output(height * width * channels);
  for (int i = 0; i < height; i++) {
    int src_pos;
    int dst_pos;

    for (int j = 0; j < width; j++) {
      if (!top_down) {
        src_pos = ((height - 1 - i) * row_size) + j * channels;
      } else {
        src_pos = i * row_size + j * channels;
      }

      dst_pos = (i * width + j) * channels;

      switch (channels) {
        case 1:
          output[dst_pos] = input[src_pos];
          break;
        case 3:
          // BGR -> RGB
          output[dst_pos] = input[src_pos + 2];
          output[dst_pos + 1] = input[src_pos + 1];
          output[dst_pos + 2] = input[src_pos];
          break;
        case 4:
          // BGRA -> RGBA
          output[dst_pos] = input[src_pos + 2];
          output[dst_pos + 1] = input[src_pos + 1];
          output[dst_pos + 2] = input[src_pos];
          output[dst_pos + 3] = input[src_pos + 3];
          break;
        default:
          std::cerr << "Unexpected number of channels: " << channels
                    << std::endl;
          std::abort();
          break;
      }
    }
  }
  return output;
}

std::vector<uint8_t> read_bmp(const std::string& input_bmp_name, int* width,
                              int* height, int* channels) {
  int begin, end;

  std::ifstream file(input_bmp_name, std::ios::in | std::ios::binary);
  if (!file) {
    std::cerr << "input file " << input_bmp_name << " not found\n";
    std::abort();
  }

  begin = file.tellg();
  file.seekg(0, std::ios::end);
  end = file.tellg();
  size_t len = end - begin;

  std::vector<uint8_t> img_bytes(len);
  file.seekg(0, std::ios::beg);
  file.read(reinterpret_cast<char*>(img_bytes.data()), len);
  const int32_t header_size =
      *(reinterpret_cast<const int32_t*>(img_bytes.data() + 10));
  *width = *(reinterpret_cast<const int32_t*>(img_bytes.data() + 18));
  *height = *(reinterpret_cast<const int32_t*>(img_bytes.data() + 22));
  const int32_t bpp =
      *(reinterpret_cast<const int32_t*>(img_bytes.data() + 28));
  *channels = bpp / 8;

  // there may be padding bytes when the width is not a multiple of 4 bytes
  // 8 * channels == bits per pixel
  const int row_size = (8 * *channels * *width + 31) / 32 * 4;

  // if height is negative, data layout is top down
  // otherwise, it's bottom up
  bool top_down = (*height < 0);

  // Decode image, allocating tensor once the image size is known
  const uint8_t* bmp_pixels = &img_bytes[header_size];
  return decode_bmp(bmp_pixels, row_size, *width, abs(*height), *channels,
                    top_down);
}

std::vector<float> RunInference(const std::vector<uint8_t>& input_data,
                                TfLiteInterpreter* interpreter) {
  std::vector<float> output_data;
  uint8_t* input = reinterpret_cast<uint8_t*>(TfLiteTensorData(
      TfLiteInterpreterGetInputTensor(interpreter, 0)));

  std::memcpy(input, input_data.data(), input_data.size());

  if (TfLiteInterpreterInvoke(interpreter) != kTfLiteOk) {
      std::cerr << "Failed to invoke" << std::endl;
      return {};
  }

  const int num_outputs = TfLiteInterpreterGetOutputTensorCount(interpreter);
  int out_idx = 0;
  for (int i = 0; i < num_outputs; ++i) {
    const auto* out_tensor = TfLiteInterpreterGetOutputTensor(interpreter, i);
    assert(out_tensor != nullptr);
    if (TfLiteTensorType(out_tensor) == kTfLiteUInt8) {
      const int num_values = TfLiteTensorByteSize(out_tensor);
      output_data.resize(out_idx + num_values);
      const uint8_t* output = reinterpret_cast<uint8_t*>(TfLiteTensorData(out_tensor));
      for (int j = 0; j < num_values; ++j) {
        TfLiteQuantizationParams params(TfLiteTensorQuantizationParams(out_tensor));
        output_data[out_idx++] = (output[j] - params.zero_point) *
                                 params.scale;
      }
    } else if (TfLiteTensorType(out_tensor) == kTfLiteFloat32) {
      const int num_values = TfLiteTensorByteSize(out_tensor) / sizeof(float);
      output_data.resize(out_idx + num_values);
      const float* output = reinterpret_cast<float*>(TfLiteTensorData(out_tensor));
      for (int j = 0; j < num_values; ++j) {
        output_data[out_idx++] = output[j];
      }
    } else {
      std::cerr << "Tensor " << TfLiteTensorName(out_tensor)
                << " has unsupported output type: " << TfLiteTensorType(out_tensor)
                << std::endl;
    }
  }
  return output_data;
}

std::array<int, 3> GetInputShape(const TfLiteInterpreter* interpreter,
                                 int index) {
  TfLiteTensor *tensor = TfLiteInterpreterGetInputTensor(interpreter, index);
  return std::array<int, 3>{TfLiteTensorDim(tensor, 1), TfLiteTensorDim(tensor, 2), TfLiteTensorDim(tensor, 3)};
}

int func(int argc, char* argv[]) {

  if (!edgetpu_init()) {
      return -1;
  }

  if (!tensorflowlite_init()) {
      return -1;
  }

  // Modify the following accordingly to try different models and images.
  const std::string model_path =
      argc == 3 ? argv[1]
                : "mobilenet_v1_1.0_224_quant_edgetpu.tflite";
  const std::string resized_image_path =
      argc == 3 ? argv[2] : "resized_cat.bmp";

  // Read model.
  std::unique_ptr<TfLiteModel, decltype(TfLiteModelDelete)> model(
      TfLiteModelCreateFromFile(model_path.c_str()), TfLiteModelDelete);
  if (model == nullptr) {
    std::cerr << "Fail to build FlatBufferModel from file: " << model_path
              << std::endl;
    std::abort();
  }

  // Build interpreter.

  std::unique_ptr<TfLiteInterpreterOptions, decltype(TfLiteInterpreterOptionsDelete)> options(
      TfLiteInterpreterOptionsCreate(), TfLiteInterpreterOptionsDelete);

  TfLiteInterpreterOptionsSetNumThreads(options.get(), 1);

  size_t num_devices;
  std::unique_ptr<edgetpu_device, decltype(edgetpu_free_devices)> devices(
      edgetpu_list_devices(&num_devices), edgetpu_free_devices);

  const auto& device = devices.get()[0];

  std::unique_ptr<TfLiteDelegate, decltype(edgetpu_free_delegate)> delegate(
      edgetpu_create_delegate(device.type, device.path, nullptr, 0), edgetpu_free_delegate);

  TfLiteInterpreterOptionsAddDelegate(options.get(), delegate.get());

  std::unique_ptr<TfLiteInterpreter, decltype(TfLiteInterpreterDelete)> interpreter(
      TfLiteInterpreterCreate(model.get(), options.get()), TfLiteInterpreterDelete);

  if (TfLiteInterpreterAllocateTensors(interpreter.get())!= kTfLiteOk) {
    std::cerr << "Failed to allocate tensors." << std::endl;
    return -1;
  }

  // Read the resized image file.
  int width, height, channels;
  const std::vector<uint8_t>& input =
      read_bmp(resized_image_path, &width, &height, &channels);

  const auto& required_shape = GetInputShape(interpreter.get(), 0);
  if (height != required_shape[0] || width != required_shape[1] ||
      channels != required_shape[2]) {
    std::cerr << "Input size mismatches: "
              << "width: " << width << " vs " << required_shape[0]
              << ", height: " << height << " vs " << required_shape[1]
              << ", channels: " << channels << " vs " << required_shape[2]
              << std::endl;
    std::abort();
  }

  // Print inference result.
  const auto& result = RunInference(input, interpreter.get());
  auto it = std::max_element(result.begin(), result.end());
  std::cout << "[Image analysis] max value index: "
            << std::distance(result.begin(), it) << " value: " << *it
            << std::endl;
  return 0;
}
