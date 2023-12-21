## ion-kit Linux Install

### 1. Install LLVM
#### a. Using a binary release (Preferred for all systems)
Find latest binary release [here](https://github.com/llvm/llvm-project/releases/tag/llvmorg-16.0.1) for your supported system.

### 2. Install Halide
#### a. Using a binary release (Preferred for all systems)
Find latest binary release [here](https://github.com/halide/Halide/releases/tag/v16.0.0) for your supported system.

### 3. Install onnxruntime (Optional, if you use ion-bb-dnn)
**Not currently supported on MacOS.**  If you use only cpu, you can get binary release.  
Please visit official onnxruntime [github](https://github.com/microsoft/onnxruntime/releases/tag/v1.16.3) for latest release.

```sh
curl -L https://github.com/microsoft/onnxruntime/releases/download/v1.4.0/onnxruntime-linux-x64-1.4.0.tgz | tar zx -C <path-to-onnxruntime-install>
```
* Please note, latest version of onnxruntime with GPU only supports `CUDA 1.8`
* `libcudnn8` will also be needed if you run with GPU, please install with:
```
sudo apt-get install libcudnn8
```

### 4. Place additional building blocks (Appendix)
If you want to use additional `ion-bb-xx` directories, place them directly under ` `ion-kit`  directory.

### 5. Install OpenCV
#### a. Using Installers
##### 5.a.1 Windows
Downalod and install latest from [OpenCV](https://sourceforge.net/projects/opencvlibrary/files/)

### 6. Install Generators
Please install the latest version of Microsoft Visual Studio.
