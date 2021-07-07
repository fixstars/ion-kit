Docker
======================================

前提環境
--------------------------------------
Docker 19.03


Dockerイメージのビルド
--------------------------------------

.. code-block:: bash

   docker build . -t ionpy


Pythonスクリプトの実行
--------------------------------------

.. code-block:: bash

  docker run --rm -v "$PWD:/code" -w /code ionpy python3 your_script.py


Dockerfileの作成
--------------------------------------

.. code-block:: Dockerfile

   FROM ionpy

   ...
