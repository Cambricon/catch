.. attention::
   | 编译安装前，需要安装CNToolkit软件包和CNLight、CNNL、CNCL、MagicMind组件。具体安装步骤，参见《寒武纪CNToolkit软件包安装升级使用手册》和相应的寒武纪用户手册。有关Cambricon PyTorch的第三方依赖，参见 ``catch/docs/release_notes/pytorch.rst`` 文件。

依赖环境
----------------------------

- g++：``USE_MAGICMIND`` 使能时，g++版本必须是7；``USE_MAGICMIND`` 关闭时，推荐g++版本大于等于5.4.0。
- Python >= 3.6.0。
- 寒武纪NeuWare软件包：CNToolkit、CNNL、CNLight、CNCL、MagicMind。


编译选项
----------------------------

在Cambricon PyTorch和Cambricon CATCH各自的setup.py脚本中设置以下编译选项：

- python setup.py install：编译并安装PyTorch或CATCH。
- python setup.py bdist_wheel：编译并生成.whl包。
- python setup.py clean：清空build文件夹。

.. _编译和安装:

编译和安装
----------------------------

编译和安装方式请参见``catch/CONTRIBUTING.md``文件。

.. _在Docker中使用PyTorch:

在Docker中使用PyTorch
----------------------------

为了方便使用PyTorch和CATCH，可以通过安装PyTorch和CATCH的Docker镜像来使用，具体使用方式可参见``catch/docker/``目录下的脚本和文件。
