## ion-bb-dnn

### Activate TensorRT Cache and float16 date type
If you want to activate TensorRT cache or float16 inside ONNXRuntime which is used in ion-bb-dnn internally,
please export the following environment variables.

```shell
export ORT_TENSORRT_ENGINE_CACHE_ENABLE=1
export ORT_TENSORRT_FP16_ENABLE=1
export ORT_TENSORRT_ENGINE_CACHE_PATH="/path/to/cache"
```

After you run the `yolov4_object_detection` BB in ion-bb-dnn, you could see the actual cache file under the directory where you ran the execution binary like below. 

```shell
> ls
...
TensorrtExecutionProvider_TRTKernel_graph_torch-jit-export_0_0_706245698680601.engine
TensorrtExecutionProvider_TRTKernel_graph_torch-jit-export_0_0_fp16_706245698680601.engine
```