Ubuntu
======================================

前提環境
--------------------------------------
* Ubuntu 18.04
* Python 3.6.9


CUDA Toolkitのインストール
--------------------------------------
https://developer.nvidia.com/cuda-toolkit-archive

ここからCUDA Toolkitをインストールする。

開発バージョン：CUDA Toolkit 10.2


その他の依存関係のインストール
--------------------------------------

.. code-block:: bash

   sudo apt install -y \
       git \
       build-essential \
       cmake \
       libopencv-dev \
       ninja-build



libopencv-devのインストールに失敗する場合
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


.. code-block:: bash

   $ sudo apt install libopencv-dev
   Reading package lists... Done
   Building dependency tree
   Reading state information... Done
   Some packages could not be installed. This may mean that you have
   requested an impossible situation or if you are using the unstable
   distribution that some required packages have not yet been created
   or been moved out of Incoming.
   The following information may help to resolve the situation:

   The following packages have unmet dependencies:
   libopencv-dev : Depends: libopencv-calib3d-dev (= 3.2.0+dfsg-4ubuntu0.1) but it is not going to be installed
                   Depends: libopencv-contrib-dev (= 3.2.0+dfsg-4ubuntu0.1) but it is not going to be installed
                   Depends: libopencv-features2d-dev (= 3.2.0+dfsg-4ubuntu0.1) but it is not going to be installed
                   Depends: libopencv-highgui-dev (= 3.2.0+dfsg-4ubuntu0.1) but it is not going to be installed
                   Depends: libopencv-objdetect-dev (= 3.2.0+dfsg-4ubuntu0.1) but it is not going to be installed
                   Depends: libopencv-stitching-dev (= 3.2.0+dfsg-4ubuntu0.1) but it is not going to be installed
                   Depends: libopencv-videostab-dev (= 3.2.0+dfsg-4ubuntu0.1) but it is not going to be installed
   E: Unable to correct problems, you have held broken packages.


上記のようなエラーによりlibopencv-devがインストールできないことがある。


.. code-block:: bash

   libjbig-dev : Depends: libjbig0 (= 2.1-3.1build1) but 2.1-3.1+deb.sury.org~xenial+1 is to be installed


依存関係を辿った結果、上記のような環境になっていた場合、
`sudo apt install libjbig0=2.1-3.1build1` のように
`libjbig0` のバージョンを修正する。

* https://askubuntu.com/questions/868460/dependency-trouble-when-installing-libcups2-dev



Halideのインストール
--------------------------------------

.. code-block:: bash

   HALIDE_URL=https://github.com/halide/Halide/releases/download/v8.0.0/halide-linux-64-gcc53-800-65c26cba6a3eca2d08a0bccf113ca28746012cc3.tgz
   INSTALL_ROOT=~/local
   HALIDE_DIR=$INSTALL_ROOT/halide/

   mkdir -p $HALIDE_DIR
   curl -L $HALIDE_URL | tar zx --strip-components=1 -C $HALIDE_DIR

   echo "export LD_LIBRARY_PATH=${HALIDE_DIR}/bin:\$LD_LIBRARY_PATH" >> ~/.bashrc


ONNX Runtimeのインストール
--------------------------------------

.. code-block:: bash

   ONNXRUNTIME_URL=https://github.com/microsoft/onnxruntime/releases/download/v1.4.0/onnxruntime-linux-x64-1.4.0.tgz
   INSTALL_ROOT=~/local
   ONNXRUNTIME_DIR=$INSTALL_ROOT/onnxruntime/

   mkdir -p $ONNXRUNTIME_DIR
   curl -L $ONNXRUNTIME_URL | tar zx --strip-components=1 -C $ONNXRUNTIME_DIR

   echo "export LD_LIBRARY_PATH=${ONNXRUNTIME_DIR}/lib:\$LD_LIBRARY_PATH" >> ~/.bashrc



ion-kitのビルド・インストール
--------------------------------------

.. code-block:: bash

   IONKIT_URL=https://github.com/fixstars/ion-kit.git
   IONKIT_CHECKOUT=501277472af52998982cb9860d597f2c659a8a44

   INSTALL_ROOT=~/local
   IONKIT_DIR=$INSTALL_ROOT/ion-kit/
   IONKIT_BUILD_DIR=./.tmp/ion-kit

   rm -rf $IONKIT_BUILD_DIR
   mkdir -p $IONKIT_BUILD_DIR
   git clone $IONKIT_URL $IONKIT_BUILD_DIR
   cd $IONKIT_BUILD_DIR
   git checkout $IONKIT_CHECKOUT
   mkdir build
   cd build
   cmake -GNinja -DCMAKE_INSTALL_PREFIX=$IONKIT_DIR -DHALIDE_ROOT=$HALIDE_DIR -DONNXRUNTIME_ROOT=$ONNXRUNTIME_DIR ../
   cmake --build .
   cmake --build . --target install
   rm -rf $IONKIT_BUILD_DIR

   echo "export LD_LIBRARY_PATH=${IONKIT_DIR}/lib:\$LD_LIBRARY_PATH" >> ~/.bashrc


ionpyのインストール
--------------------------------------

.. code-block:: bash

   pip3 install git+https://gitlab.com/transfer_cn/ionpy.git
