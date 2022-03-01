本节中的层特指添加了MLU支持的层。目前，Cambricon PyTorch中已经实现了在线逐层，在线融合之间的同步。即，任何已经实现的算子都可以同时采用在线逐层、在线融合的任何一种进行使用。对于尚未支持的算子，请参考 :ref:`customized` 章节的方法添加。

Cambricon PyTorch已支持的算子如下所示，其中：

- torch.ops_、torch.Tensor.ops_、torch.nn.Modules_、torch.nn.functional.ops_：PyTorch 的原生算子。

- 自定义算子_：Cambricon PyTorch 新增的算子。

.. _torch.ops:

torch 算子说明
======================

.. _torch.abs:

abs
--------------
.. code::

   torch.abs(input, *, out=None)

- **功能描述**

  对输入tensor计算绝对值。公式：

  .. figure:: ../doc_image/abs.*

  支持的数据类型：float16、float32。

- **参数说明**

  - input：输入tensor。

  - out：可选，输出tensor。

- **规格限制**

  输入为不大于8维的tensor。

- **支持的计算库**

  CNNL。


.. _torch.add:

add
--------------
.. code::

   torch.add(input, other, alpha=1, out=None)

- **功能描述**

  对输入tensor ``other`` 的每个元素与标量alpha相乘并与输入tensor ``input`` 做加法运算，返回输出tensor。

  输入tensor ``input`` 和 ``other`` 必须是相同shape或者符合broadcast规则。

  公式为：

  .. figure:: ../doc_image/add.*

  支持的数据类型：int8、uint8、int16、int32、int64、float16、float32、bool；MagicMind仅支持float32和float16。

- **参数说明**

  - input：输入tensor。
  - other：输入tensor，计算时与标量alpha相乘。
  - out：可选，输出output。默认为None返回新的结果tensor。
  - alpha：other的标量乘数。

- **规格限制**

  无。

- **支持的计算库**

  CNNL，MagicMind。

.. code::

   torch.add(input, other, out=None)

- **功能描述**

  对输入scalar other与输入tensor input做加法运算，返回输出tensor。

  公式为： :math:`out = input + other`

  支持的数据类型：int8、uint8、int16、int32、int64、float16、float32、bool；MagicMind仅支持float32和float16。

- **参数说明**

  - input：输入tensor。
  - other：输入scalar。
  - out：可选，输出output。默认为None，返回新的结果tensor。

- **规格限制**

  无。

- **支持的计算库**

  CNNL，MagicMind。

.. _torch.any:

any
--------------
.. code::

   torch.any(input)

- **功能描述**

  判断输入tensor中是否含有True元素。

  支持的数据类型：bool、uint8。

- **参数说明**

  - input：输入tensor。

- **规格限制**

  输入为不大于8维的tensor。

- **支持的计算库**

  CNNL。

.. _torch.arange:

arange
----------------------------
.. code::

  torch.arange(start=0, end, step=1, *, out=None, dtype=None, layout=torch.strided, device=None, requires_grad=False) → Tensor

- **功能描述**

  返回一维tensor，大小为 (end-start)/step 向上取整，元素值为从start到end等差数列，以step为差值。

  支持的数据类型组合：
  
  .. table:: 支持的数据类型组合
     :widths: 4 3

     +----------------+----------+
     | start/end/step | output   |
     +================+==========+
     | float32        | float32  |
     +----------------+----------+
     | float32        | float16  |
     +----------------+----------+
     | int32          | int32    |
     +----------------+----------+

- **参数说明**

  - start：实数，输出数列起始值。缺省值为0。
  - end：实数，输出数列结束值。
  - step：实数，输出数列等差值，缺省值为1。
  - out：torch.Tensor，可选，输出tensor。
  - dtype：torch.dtype类型，可选，期望返回的tensor的数据类型，默认是torch.int64。
  - layout：torch.layout类型，默认是torch.strided。该参数不支持修改。
  - device：torch.device类型，可选，期望返回的tensor的设备类型。默认是None，表示当前默认tensor类型的当前设备（参考torch.set_default_tensor_type()），如果是CUDA或者MLU类型，表示CUDA或者MLU的当前设备。
  - requires_grad：bool类型，可选，设置成True时，autograd会开启。默认是False。

- **规格限制**

  无。

- **支持的计算库**

  CNNL。

.. _torch.argmax:

argmax
---------------------------
.. code::

   torch.argmax(input, dim, keepdim=False) → LongTensor

- **功能描述**

  返回输入tensor中所有元素的最大值的索引位置。该索引位置输出形式视输入参数而定。

  - 输入参数仅包含输入tensor，返回该tensor展开成一维tensor后最大元素的索引位置。
  - 输入参数包括输入tensor的指定dim，则返回给定维度dim中输入tensor的每一行的最大值的索引位置。
  - 若包含keepdim参数，且指定为True，则返回的索引tensor与输入tensor除第dim个维度外保持一致。

  数据类型：int32、float16、float32。

- **参数说明**

  - input：torch.Tensor，输入tensor。

  - dim：int，处理的dim。

  - keepdim：bool，是否使得除第dim个维度外，输出tensor的尺寸与输入的尺寸相同。

- **规格限制**

  由于MLU与CPU算法差别，目前argmax算子功能并不能和CPU保持一致。

- **支持的计算库**

  CNNL。

.. _torch.as_stride:

as_strided
--------------
.. code::

   torch.as_strided(input, size, stride, storage_offset=0) → Tensor

- **功能描述**

  利用给定的 ``size``、``stride`` 和 ``storage_offset`` 参数生成输入tensor的视图tensor。

  支持的数据类型：float16、float32。

- **参数说明**

  - input：输入tensor。
  - size：输出tensor的大小。
  - stride：输出tensor的stride。
  - storage_offset：int类型，可选，输出tensor在内存中的偏移。

- **规格限制**

  当多个tensor索引同一块内存时，in-place操作可能导致不正确的行为。如果需要写这些tensor，请先clone这些tensor。

- **支持的计算库**

  CNNL。

.. _torch.bitwise_and:

bitwise_and
--------------

.. code::

   torch.bitwise_and(input, other, *, out=None) → Tensor

- **功能描述**

  对两个输入tensor执行按位相与运算，只接受整型或者bool类型。当输入bool类型时，执行逻辑与运算。

  支持的数据类型：bool、uint8、int8、int16、int32、int64。（注意：当前最大只支持32位数据，int64只可用32位整形表示范围内的数据。）

- **参数说明**

  - input：第一个输入tensor。

  - other：第二个输入tensor。

  - out：可选，输出tensor。

- **规格限制**

  - 输入为不大于8维的tensor。

  - 输入tensor的形状需要相同，或者相应维度满足broadcast规则，如 ``input`` 的形状是 ``[1, 2, 3, 4]`` ， ``other`` 的形状是 ``[4, 2, 3, 4]`` ，则这两个tensor可以进行bitwise_and运算。

- **支持的计算库**

  CNNL。

.. _torch.bitwise_or:

bitwise_or
--------------

.. code::

   torch.bitwise_or(input, other, *, out=None) → Tensor

- **功能描述**

  对两个输入tensor执行按位相或运算，只接受整型或者bool类型。当输入bool类型时，执行逻辑或运算。

  支持的数据类型：bool、uint8、int8、int16、int32、int64。（注意：当前最大只支持32位数据，int64只可用位于32位整形表示范围内的数据。）

- **参数说明**

  - input：第一个输入tensor。

  - other：第二个输入tensor。

  - out：可选，输出tensor。

- **规格限制**

  - 输入为不大于8维的tensor。

  - 输入tensor的形状需要相同，或者相应维度满足broadcast规则，如 ``input`` 的形状是 ``[1, 2, 3, 4]``，``other`` 的形状是 ``[4, 2, 3, 4]`` ，则这两个tensor可以进行bitwise_or运算。

- **支持的计算库**

  CNNL。

.. _torch.bmm:

bmm
----------------------------
.. code::

  torch.bmm(input, mat2, deterministic=False, out=None) → Tensor

- **功能描述**

  对两个矩阵按batch维度进行矩阵乘。

  支持的数据类型：float16、float32。

- **参数说明**

  - input：输入的第一个tensor。
  - mat2：输入的第二个tensor。
  - out：输出tensor，可选。如果不设置或者设置为None时，无须输入。
  - deterministic：该参数只适用于sparse-dense CUDA bmm计算，MLU不支持该参数。

- **规格限制**

  输入tensor必须为3维，且两个输入tensor的第二、第三维必须满足矩阵相乘条件。

- **支持的计算库**

  CNNL。

.. _torch.broadcast_tensors:

broadcast_tensors
----------------------------
.. code::

  torch.broadcast_tensors(*tensors) → List of Tensors

- **功能描述**

  对输入TensorList进行广播。

  支持的数据类型：float16、float32。

- **参数说明**

  - tensors：tensorList类型，若干个数据类型相同的输入tensor。

- **规格限制**

  无。

- **支持的计算库**

  CNNL。

.. _torch.ceil:

ceil
--------------
.. code::

   torch.ceil(input, *, out=None)→ Tensor

- **功能描述**

  对输入tensor计算大于或等于每个元素的最小整数。公式：

  .. figure:: ../doc_image/ceil.*

  支持的数据类型：float16、float32。

- **参数说明**

  - input：输入tensor。

  - out：可选，输出tensor。

- **规格限制**

  输入张量支持的数值范围为[-2^23 + 1，2^23 - 1]。

- **支持的计算库**

  CNNL。



.. _torch.chunk:

chunk
--------------
.. code::

   torch.chunk(input, chunks, dim=0) → List of Tensors

- **功能描述**

  将输入tensor分为指定数量的块。

  支持的数据类型：float32。

- **参数说明**

  - input：输入tensor。
  - chunks：tensor将要被分成的块的数量，int类型。
  - dim：tensor分块的维度，int类型。

- **规格限制**

  无。

- **支持的计算库**

  CNNL。

.. _torch.clone:

clone
---------------------
.. code::

   torch.clone(input, *, memory_format=torch.preserve_format) → Tensor

- **功能描述**

  返回输入tensor的拷贝。

  支持的数据类型：all。

- **参数说明**

  - input：torch.Tensor，输入tensor。

  - memory_format：输出的内存格式，缺省：torch.preserve_format。

- **规格限制**

  无。

- **支持的计算库**

  CNNL。


.. _torch.clamp:

clamp
--------------
.. code::

  torch.clamp(input, min, max, *, out=None) → Tensor

- **功能描述**

  将输入tensor的每个元素限制在[min, max]区间内。

  支持的数据类型：float16、float32、int32。

- **参数说明**

  - input：torch.Tensor，输入tensor。
  - min：实数，可选，表示限制范围的下限。
  - max：实数，可选，表示限制范围的上限。
  - out：torch.Tensor，可选，输出tensor。

- **规格限制**

  输入为不大于4维的tensor。

  min和max参数不能同时为空。

- **支持的计算库**

  CNNL。

.. _torch.diag:

diag
--------------
.. code::

  torch.diag(input, diagonal=0, out=None) → Tensor

- **功能描述**

  返回一个两维输入tensor的对角线元素组成的一维输出tensor或者根据一个一维输入tensor返回由这个tensor中元素作为对角线的两维输出tensor。

  支持的数据类型：int8、uint8、int16、float16、float32、int32、int64、float64。（注意：当前最大只支持32位数据，int64和float64只可用于32位对应类型所表示范围内数据。）

- **参数说明**

  - input：torch.Tensor类型，输入tensor。
  - diagonal：int64_t类型，可选，默认为0，表示是主对角线；大于0表示主对角线右上的次对角线；小于0表示主对角线左下的次对角线。
  - out：torch.Tensor类型，可选，输出tensor。

- **规格限制**

  当输出两维tensor时，两维tensor的元素数目不能超过INT_MAX。

- **支持的计算库**

  CNNL。

.. _torch.div:

div
--------------
.. code::

  torch.div(input, other, out=None) → Tensor

- **功能描述**

  对输入tensor ``input`` 与输入scalar ``other`` 做除法，返回一个新的tensor。公式：

  .. figure:: ../doc_image/div.*

  支持的数据类型：float16、float32。

- **参数说明**

  - input：输入tensor。
  - other：标量，输入tensor每个元素的除数。
  - out：可选，输出的tensor。

- **规格限制**

  - 不支持out参数。

- **支持的计算库**

  CNNL。

.. code::

   torch.div(input, other, out=None) → Tensor

- **功能描述**

  对输入tensor ``input`` 的每个元素与输入tensor ``other`` 的每个元素做除法，返回一个新的tensor。公式：

  .. figure:: ../doc_image/div_t.*

  支持的数据类型：float16、float32。

- **参数说明**

  - input：输入分子tensor。
  - other：输入分母tensor。
  - out：可选，输出的tensor。

- **规格限制**

  - 不支持out参数。

- **支持的计算库**

  CNNL。

.. _torch.eq:

eq
----------------------------
.. code::

   torch.eq(input, other, out=None) -> Tensor

- **功能描述**

  逐元素计算两个输入tensor是否相等。

  支持的数据类型：float16、float32。

- **参数说明**

  - input：输入tensor。
  - other：torch.Tensor或float类型，用来进行比较的tensor或value。
  - out：torch.Tensor类型，可选，输出tensor。

- **规格限制**

  计算数据类型只支持float16、float32，不支持int4、int8、int16。

- **支持的计算库**

  CNNL。

.. _torch.equal:

equal
----------------------------
.. code::

   torch.equal(input, other) -> bool

- **功能描述**

  如果两个张量具有相同的大小和元素，则为真，否则为假。

- **参数说明**

  - input：输入tensor。
  - other：torch.Tensor，用来进行比较的tensor或value。

- **规格限制**

  计算数据类型只支持float16、float32，不支持int4、int8、int16。

- **支持的计算库**

  CNNL。

.. _torch.exp:

exp
--------------
.. code::

   torch.exp(input, out=None) -> Tensor

- **功能描述**

  对输入求以e为底的x次方值。公式：

  .. figure:: ../doc_image/exp.*

  支持的数据类型：float16、float32。

- **参数说明**

  - input：输入tensor。
  - out：输出tensor。

- **规格限制**

  - 计算数据类型只支持float16、float32，不支持int4、int8、int16。
  - 输入和输出的shape要完全相同。
  - 输入数据要保证输出数据不超过FP16能表示的最大范围65504，即输入数据不能大于11。

- **支持的计算库**

  CNNL。

.. _torch.flatten:

flatten
-------------
.. code::

   torch.flatten(input, start_dim=0, end_dim=-1) → Tensor

- **功能描述**

  在指定的连续维度区间上使tensor展平。

  支持的数据类型：float16、float32。

- **参数说明**

  - input：Tensor类型，输入tensor。
  - start_dim：需展平的起始维度。
  - end_dim：需展平的最后维度。

- **规格限制**

  - rank(input) <= start_dim < rank(input)
  - rank(input) <= end_dim < rank(input)

- **支持的计算库**

  MagicMind。

.. _torch.nonzero:

nonzero
-------------
.. code::

  torch.nonzero(input, *, out=None, as_tuple=False) → LongTensor or tuple of LongTensors

- **功能描述**

  返回输入tensor中非零元素的索引。
  支持的数据类型：bool，int32，float32，float64，long。

- **参数说明**

  - input (torch.tensor)：输入tensor。
  - out：输出tensor，可选。如果不设置或者设置为None时，则无需输入。
  - as_tuple (bool)：设置返回值类型。默认为False，返回值为2D的tensor; 设置为True时，则返回值为1D tensor的tuple。

- **规格限制**

  无。

- **支持的计算库**

  CNNL。

.. _torch.new_zeros:

new_zeros
----------------------------
.. code::

  torch.Tensor.new_zeros(size, dtype=None, device=None, requires_grad=False) → Tensor

- **功能描述**

  返回所输入形状的tensor，被实数0填充。

  支持的数据类型：bool、float16、float32、int32、int16、int8、int64、uint8。

- **参数说明**

  - size：int型的list、tuple或torch.size类型，输出tensor的形状。
  - dtype：可选，torch.dtype类型，输出tensor的数据类型。默认值：如果设置为None，则和原tensor相同。
  - device：可选，torch.device类型，输出tensor的设备类型。默认值：如果设置为None，则和原tensor相同。
  - requires_grad：可选，bool类型，设置输出tensor是否需要记录梯度。默认值：False。

- **规格限制**

  无。

- **支持的计算库**

  CNNL。

.. _torch.new_ones:

new_ones
----------------------------
.. code::

  torch.Tensor.new_ones(size, dtype=None, device=None, requires_grad=False) → Tensor

- **功能描述**

  返回所输入形状的tensor，被实数1填充。

  支持的数据类型：bool、float16、float32、int32、int16、int8、int64、uint8。

- **参数说明**

  - size：int型的list、tuple或torch.size类型，输出tensor的形状。
  - dtype：可选，torch.dtype类型，输出tensor的数据类型。默认值：如果设置为None，则和原tensor相同。
  - device：可选，torch.device类型，输出tensor的设备类型。默认值：如果设置为None，则和原tensor相同。
  - requires_grad：可选，bool类型，设置输出tensor是否需要记录梯度。默认值：False。

- **规格限制**

  无。

- **支持的计算库**

  CNNL。

.. _torch.norm:

norm
-------------
.. code::

  torch.norm(input, p='fro', dim=None, keepdim=False, out=None, dtype=None)

- **功能描述**

  计算输入tensor的矩阵范数或向量范数。

- **参数说明**

  - input：输入，torch.Tensor类型。计算核范数时支持数据类型为float，计算L1、L2和F范数时支持float与half类型。
  - p：可选，范数，浮点、整数类型，也可为'fro'（F范数）或'nuc'（核范数）。默认为'fro'。目前仅支持值为1、2的L1和L2范数，以及F范数和核范数。
  - dim：可选，范数计算的维度，可以是一个整数，或一个整数构成的list。默认为对所有维度计算。
  - keepdim：可选，布尔类型，是否保持输出的维度。默认为False。
  - out：可选，torch.Tensor类型，指定输出的张量。
  - dtype：可选，指定返回值的数据类型。

- **规格限制**

  - dtype：F范数和核范数不能指定数据类型。
  - 核范数仅在矩阵计算中有意义，因此输入应是一个二维矩阵，或由dim参数指定输入Tensor中的两个维度构成二维矩阵进行计算。
  - 核范数的输入矩阵长宽必须小于150。即最大规模为149×149。

- **支持的计算库**

  CNNL。

.. _torch.floor:

floor
-------------
.. code::

   torch.floor(input, out=None) → Tensor

- **功能描述**

  返回一个新tensor，包含输入tensor的每个元素的地板数，即不大于该元素的最大整数。

  支持的数据类型：float16、float32、float64。（注意：当前最大只支持32位数据，float64只可用于32位对应类型所表示范围内数据。）

- **参数说明**

  - input：输入tensor。
  - out：输出tensor，可选。如果不设置或者设置为None时，则无需输入。

- **规格限制**

  无。

- **支持的计算库**

  CNNL。

.. _torch.gather:

gather
-------------
.. code::

   torch.gather(input, dim, index, out=None, sparse_grad=False) → Tensor

- **功能描述**

  沿着给定的dim轴，将输入的input按照索引index指定的位置聚合。

  input支持的数据类型：double、float32、float16、int8、uint8、int16、int32、int64、bool。

  index支持的数据类型：int32、int64。

- **参数说明**

  - input：torch.Tensor类型，输入tensor。
  - dim：int类型，沿着索引的轴。
  - index：torch.Tensor类型，需要聚合的索引。
  - out：torch.Tensor类型，输出tensor。
  - sparse_grad：bool类型，input的梯度是否为稀疏张量。

- **规格限制**

  无。

- **支持的计算库**

  CNNL。

.. _torch.ge:

ge
----------------------------
.. code::

   torch.ge(input, other, *, out=None) → Tensor

- **功能描述**

  逐元素比较input和other。第二个参数可以是数字或tensor，其形状可以与第一个参数一起广播。如果两个tensor有相同的size和元素值，则返回True，否则返回为False。

  支持的数据类型：float32、float16、int32、short、bool、uint8。

- **参数说明**

  - input：torch.Tensor类型，待对比的tensor。
  - other：torch.Tensor或者float类型，对比的tensor或value。
  - out：torch.Tensor类型，输出的tensor。

- **规格限制**

  输入和输出最大不超过8维。

- **支持的计算库**

  CNNL。

.. _torch.gt:

gt
-------------
.. code::

  torch.gt(input, other, out=None) -> Tensor

- **功能描述**

  逐元素比较输入tensor是否大于第二个输入。

  支持的数据类型：bool、int8、uint8、short、int、long、float。

- **参数说明**

  - input：torch.Tensor类型，第一个输入tensor。
  - other：torch.Tensor或数值类型，第二个输入。
  - out：可选，torch.Tensor类型，输出tensor。

- **规格限制**

  - 输入tensor的维度不超过8维。
  - 第二个输入为数值类型，或两个输入的各个维度均满足广播规则（broadcastable），如input形状为 ``[1,2,3,4]`` ，other的形状为 ``[4,2,3,4]`` ，则两个tensor可以进行gt运算。

- **支持的计算库**

  CNNL。

.. _index:

index
----------------------------
.. code::

   output = input[indices]

- **功能描述**

  根据输入的 ``indices`` tensor在指定的 ``input`` tensor选取对应的数值。
  支持的数据类型：int8、uint8、int16、float32、int32、int64、float64。（注意：当前最大只支持32位数据，int64和float64只可用于对应32位的数据类型所表示范围内数据。）

- **参数说明**

  - input：torch.Tensor类型，输入tensor。
  - indices：torch.Tensor类型，为输入指定的下标tensor，类型为bool或者long类型。

- **规格限制**

  - 暂时不支持多个bool类型indices作为下标输入。

- **支持的计算库**

  CNNL。

.. _index_put\_:

index_put\_
----------------------------------

.. code::

   index_put_(indices, values, accumulate=False) → Tensor

- **功能描述**

  对输入tensor按照 ``indices`` 指定的下标，赋value tensor的值。

  ``self.index_put_(indices, value, accumulate=False)`` 等价于 ``self[indices] = value`` 。

  支持的数据类型：

  - input：支持float16、float32、bool、int8、 uint8、int32等类型。
  - indices：支持bool、int32、long类型。

- **参数说明**

  - self：第一个输入tensor。

  - indices：Long或者Bool类型的tensor元组，用于指示处理input的下标位置。

  - value：与self相同数据类型的tensor。支持shape推理广播。

  - accumulate：为 ``False`` 时， 为替换模式，即将value值替换 ``self tensor`` ；为 ``True`` 时，为累加模式，即将 ``value`` 累加，将结果加至 ``self``。当前仅支持设置为 ``False``。

- **规格限制**

  - 所有tensor不能超过8维。
  - ``indices`` 支持 ``int32`` 和 ``bool`` 类型，但不支持 ``bool`` 和 ``int32`` 同时出现。
  - ``indices`` 不支持 ``tensor.defined = false``。
  - ``indices`` 数据类型为 ``int32`` 时，其中的 ``indices[n]`` 的值必须在 ``0~self.shape(n)-1`` 范围内。
  - ``indices`` 的 ``shape`` 广播暂时不支持，即要求 ``indices`` 中的tensor形状必须一致。
  - ``indices`` 的数据类型为 ``int32`` 时，暂不支持负数。
  - 当 ``accumulate`` 为 ``false`` 时，若 ``indices`` 指向的 ``index`` 有重复，计算结果无法与 ``CPU`` 结果一致（``GPU`` 结果与 ``CPU`` 结果也无法保证一致）。
  - 目前 ``accumulate`` 仅支持 ``false`` 模式，即仅支持替换模式，累加模式暂时不支持。
  - 当 ``indices`` 为 ``int32`` 类型时，``indices[0].dim() + input.dim() - indices.num <= 8``；当 ``indices`` 类型为 ``bool`` 时，``indices[0]`` 的 ``shape`` 必须从高维开始与 ``self`` 一致。
  - 当 ``indices`` 数据类型为 ``int32`` 时，input的元素数据应不大于 ``2 ^ 23``。
  - 当 ``indices`` 数据类型为 ``bool`` 类型时，input的元素数量应不大于 ``2 ^ 31-1``。
  - 当 ``indices`` 数据类型为 ``bool`` 类型时，目前仅支持 ``indices`` 中一个Tensor。
  - 需满足框架中该算子本身的条件。

- **支持的计算库**

  CNNL。

.. _index_fill\_:

index_fill\_
----------------------------
.. code::

   self.index_fill_(dim, index, value) -> self

- **功能描述**

  根据index tensor中的顺序向self tensor中指定的索引位置填充元素value。

  支持的输入数据类型：float32、float16、int32、int8、uint8。

- **参数说明**

  - self：torch.Tensor，输入tensor。

  - dim：int类型，表示某一维度上索引。

  - index：torch.LongTensor，self tensor需要填充的索引。

  - value：float类型，表示被填充的值。

- **规格限制**

  无。

- **支持的计算库**

  CNNL。

.. _torch.isfinite:

isfinite
-------------

.. code::

  torch.isfinite(input) → Tensor

- **功能描述**

  返回与输入tensor形状相同，且类型为bool的输出tensor。输出tensor的每个元素表示输入tensor的对应位置是否为有限值。

  支持的数据类型：half、float32、uint8、int8、int16、int32、int64。
  （注意：当输入的类型为整形时，返回的是全为true的tensor。）

- **参数说明**

  - input：torch.Tensor类型，输入tensor。

- **规格限制**

  无。

- **支持的计算库**

  CNNL。

.. _torch.item:

item
-------------
- **功能描述**

  返回tensor的标量值。tensor必须是只有一个元素的tensor。

  支持的数据类型：float32、float16。

- **参数说明**

  无。

- **规格限制**

  无。

- **支持的计算库**

  CNNL。

.. _torch.le:

le
-------------
.. code::

  torch.le(input, other, out=None) -> Tensor

- **功能描述**

  逐元素比较输入tensor是否小于等于第二个输入。

  支持的数据类型：bool、int8、uint8、short、int、long、float。

- **参数说明**

  - input：torch.Tensor类型，第一个输入tensor。
  - other：torch.Tensor或数值类型，第二个输入。
  - out：可选，torch.Tensor类型，输出tensor。

- **规格限制**

  - 输入tensor的维度不超过8维。
  - 第二个输入为数值类型，或两个输入的各个维度均满足广播规则（broadcastable），如input形状为 ``[1,2,3,4]`` ，other的形状为 ``[4,2,3,4]`` ，则两个tensor可以进行le运算。

- **支持的计算库**

  CNNL。

.. _torch.linspace:

linspace
-------------
.. code::

   torch.linspace(start, end, steps=100, out=None, dtype=None, layout=torch.strided, device=None, requires_grad=False) → Tensor

- **功能描述**

  返回输入参数的一维tensor，范围为[start, end]。输出一维数组长度为steps。

  支持的数据类型：float32、float16。

- **参数说明**

  - start：生成数组的起点值，float类型。
  - end：生成数组的终点值，float类型。
  - steps：生成分布的元素个数，默认值是100，int类型。
  - dtype：输出tensor的期望数据类型，可选，torch.dtype类型。
  - layout：输出tensor的期望layout，可选，torch.layout类型。
  - device：输出tensor的期望设备类型，可选，torch.device类型。
  - requires_grad：是否需要计算梯度。可选，bool类型，默认值为 ``false`` 。
  - out：输出Tensor。

- **规格限制**

  不支持 ``layout`` 、``requires_grad`` 参数。

- **支持的计算库**

  CNNL。

.. _torch.lt:

lt
----------------------------
.. code::

   torch.lt(input, other, out=None) → Tensor

- **功能描述**

  逐元素返回 input < other 的结果。

  支持的数据类型：float16、float32。

- **参数说明**

  - input：torch.Tensor类型，输入tenor。
  - other：torch.Tensor或float类型。
  - out：torch.Tensor类型，可选，必须是BoolTensor。

- **规格限制**

  无。

- **支持的计算库**

  CNNL。


.. _torch.log:

log
----------------------------
.. code::

  torch.log(input, out=None) → Tensor

- **功能描述**

  对输入计算自然对数。公式：

  .. figure:: ../doc_image/log.*

  支持的数据类型：float16、float32。

- **参数说明**

  - input：torch.Tensor类型，输入tensor。
  - out：torch.Tensor类型，可选，输出tensor。仅CNNL计算库支持该参数。

- **规格限制**

  - float16类型数值在[1, 60000]之间；float32类型数值在[1e-20, 2e5]之间。
  - 输入tensor维度最大不超过8维。

- **支持的计算库**

  CNNL。

.. _torch.log2:

log2
----------------------------
.. code::

   torch.log2(input, *, out=None) → Tensor

- **功能描述**

  以2为底数，对输入计算对数。

  支持的数据类型：float16、float32。

- **参数说明**

  - input：torch.Tensor类型，输入tensor。
  - out：torch.Tensor类型，可选，输出tensor，默认为None。

- **规格限制**

  float16类型数值在[1, 60000]之间；float32类型数值在[1e-20, 2e5]之间。

- **支持的计算库**

  CNNL。

.. _torch.log10:

log10
----------------------------
.. code::

   torch.log10(input, *, out=None) → Tensor

- **功能描述**

  以10为底数，对输入计算对数。

  支持的数据类型：float16、float32。

- **参数说明**

  - input：torch.Tensor类型，输入tensor。
  - out：torch.Tensor类型，可选，输出tensor，默认为None。

- **规格限制**

  float16类型数值在[1, 60000]之间；float32类型数值在[1e-20, 2e5]之间。

- **支持的计算库**

  CNNL。

.. _torch.masked_select:

masked_select
----------------------------
.. code::

   torch.masked_select(input, mask, out=None) → Tensor

- **功能描述**

  返回一个新的一维tensor。该tensor根据掩码为输入tensor编制索引。

  input支持的数据类型：float16、float32、float64、int8、int16、int32、int64、bool。(注意：当前最大只支持32位数据，int64和float64只可用于对应的32位数据类型所表示范围内数据。)
  mask支持的数据类型：bool。

- **参数说明**

  - input：torch.Tensor类型，输入tensor。
  - mask：torch.Tensor类型，掩码。
  - out：torch.Tensor类型，可选，指定输出tensor。

- **规格限制**

  - 掩码的数据类型须为bool。
  - 掩码tensor和输入tensor的形状不需匹配，但它们必须是可广播的。

- **支持的计算库**

  CNNL。

.. _torch.matmul:

matmul
----------------------------
.. code::

  torch.matmul(input, other, out=None) → Tensor

- **功能描述**

  计算矩阵乘法。

  支持的数据类型：float16、float32。

- **参数说明**

  - input：输入的第一个tensor。
  - other：输入的第二个tensor。
  - out：输出tensor，可选。如果不设置，或者设置为None，无须输入。

- **规格限制**

  MLU仅支持输入尺寸为4维的情况，不支持MLU单核。

- **支持的计算库**

  CNNL。

.. _torch.addmm:

addmm
----------------------------

.. code::

    torch.addmm(input, mat1, mat2, *, beta=1, alpha=1, out=None) → Tensor

- **功能描述**

  对输入 ``mat1`` 和 ``mat2`` 进行矩阵计算，结果乘以系数 ``alpha``，再与系数 ``beta`` 和输入 ``input`` 的乘积求和。

  公式： :math:`out = beta * input + alpha * (mat1 @ mat2)`

  支持的数据类型：float16、float32。

- **参数说明**

  - input：输入tensor。
  - mat1：矩阵乘的第一个tensor。
  - mat2：矩阵乘的第二个tensor。
  - out：可选，输出output。默认为None返回新的结果tensor。
  - beta：input的标量乘数。
  - alpha：mat1与mat2矩阵乘结果的标量乘数。

- **规格限制**

  支持输入tensor的规格为2维。

  输入tensor ``mat1`` 和 ``mat2`` 必须是符合矩阵相乘的规则，并且相乘结果必须与输入 ``input`` 的形状相同。

  输入标量 ``alpha`` 与 ``beta`` 不支持 nan 与 inf

  当输入input规模较小时，FP16的精度会有所下降，MSE误差在0.03以下。

- **支持的计算库**

  CNNL，MagicMind。

.. _torch.baddbmm:

baddbmm
--------------------------

.. code::

    torch.baddbmm(input, batch1, batch2, *, beta=1, alpha=1, out=None) ->Tensor

- **功能描述**

  对输入的两个batch做批矩阵乘（bmm），其结果乘以alpha，并与input乘以beta的结果求和（badd）。
  
  公式为：

  .. math::
  
      \text{out}_i = \beta\ \text{input}_i + \alpha\ (\text{batch1}_i \mathbin{@} \text{batch2}_i)

  支持的数据类型：float32。

- **参数说明**

  - input：torch.Tensor类型，输入tensor。
  - batch1：torch.Tensor类型，批矩阵乘的第一个tensor。
  - batch2：torch.Tensor类型，批矩阵乘的第二个tensor。
  - beta：float或int类型，输入tensor的标量乘数。
  - alpha：float或int类型，批矩阵乘结果的标量乘数。
  - out：可选，torch.Tensor类型，输出tensor。

- **规格限制**

  - input、batch1、batch2均为3维。
  - 批矩阵乘的batch1与batch2的batch维（最高维）相等。
  - 批矩阵乘的batch1与batch2的低维需要满足矩阵计算规则。
  - baddbmm的输出形状与批矩阵乘的结果一致，input需要与该形状一致，或可被广播为该形状。
  - 标量 ``alpha``、``beta`` 以及输入tensor ``input``、``batch1`` 和 ``batch2`` 不支持inf与nan输入。

- **支持的计算库**

  CNNL。

.. _torch.max:

max
---------------------------
.. code::

   torch.max(input) → Tensor

   torch.max(input, dim, keepdim=False, out=None) -> (Tensor, LongTensor)

   torch.max(input, other, out=None) → Tensor

- **功能描述**

  max函数提供以下功能：

  - 返回输入tensor中所有元素的最大值。

  - 返回一个命名元组(value, index)。其中，value是给定维度dim中输入tensor的每一行的最大值，index是找到的每个最大值（argmax）的索引位置。

  - 返回两个输入tensor对应位置的最大值。该功能与maximum算子功能类似。

  支持的数据类型：int32、float16、float32。

- **参数说明**

  - input：输入tensor。

  - other：输入tensor。

  - dim：int类型，处理的dim。

  - keepdim：bool类型，除第dim个维度外，输出tensor的尺寸与输入的尺寸相同。

  - out：输出tensor。

- **规格限制**

  输入为不大于8维的tensor。

- **支持的计算库**

  CNNL。

.. _torch.min:

min
---------------------------
.. code::

   torch.min(input) → Tensor

   torch.min(input, dim, keepdim=False, out=None) -> (Tensor, LongTensor)

   torch.min(input, other, out=None) → Tensor

- **功能描述**

  min函数提供以下功能：

  - 返回输入tensor中所有元素的最小值。

  - 返回一个命名元组(value, index)。其中，value是给定维度dim中输入tensor的每一行的最小值，index是找到的每个最小值（argmin）的索引位置。

  - 返回两个输入tensor对应位置的最小值。

  支持的数据类型：int32、float16、float32。

- **参数说明**

  - input：输入tensor。

  - other：输入tensor。

  - dim：int类型，处理的dim。

  - keepdim：bool类型，除第dim个维度外，输出tensor的尺寸与输入的尺寸相同。

  - out：输出tensor。

- **规格限制**

  输入为不大于8维的tensor。

- **支持的计算库**

  CNNL。

.. _torch.mean:

mean
----------------------------
.. code::

   torch.mean(input) → Tensor

   torch.mean(input, dim, keepdim=False, out=None) → Tensor

- **功能描述**

  mean函数提供以下功能：

  - 计算指定维度的平均值。

  - 返回给定维度中输入tensor的每一行的平均值。

  支持的数据类型：float16、float32。

- **参数说明**

  - input：输入tensor。

  - dim：int或者tuple ints类型，设置需要求均值的维度，可以是一个数或一个二维元组。

  - keepdim：bool类型，除第dim个维度外，输出tensor的尺寸与输入的尺寸相同。

  - out：输出tensor。

- **规格限制**

  输入为不大于8维的tensor，最多只能求两个维度的平均值。

- **支持的计算库**

  CNNL。

.. _torch.mm:

mm
----------------------------
.. code::

   torch.mm(input, mat2, out=None)

- **功能描述**

  执行矩阵乘法。

  支持的数据类型：float16、float32。

- **参数说明**

  - input：输入的第一个tensor。
  - mat2：输入的第二个tensor。
  - out：输出tensor，可选。如果不设置或者设置为None时，则无需输入。

- **规格限制**

  - 第一个tensor的行数必须等于第二个tensor的列数。
  - 不支持广播，支持广播的矩阵乘。更多信息，参见 ``torch.matmul()`` 。

- **支持的计算库**

  CNNL。

.. _torch.mul:

mul
----------------------------
.. code::

   torch.mul(input, other, out=None)

- **功能描述**

  两个输入tensor相乘。
  
  公式为：

  .. figure:: ../doc_image/mul.*

  支持的数据类型：float16、float32。

- **参数说明**

  - input：输入的第一个tensor。
  - other：输入的tensor或scalar。
  - out：可选，输出tensor。

- **规格限制**

  支持逐元素相乘和广播乘法。

- **支持的计算库**

  CNNL。

.. _torch.narrow:

narrow
----------------------------
.. code::

   torch.narrow(input, dim, start, length) → Tensor

- **功能描述**

  对输入tensor基于某一维度取若干连续的元素，返回的新tensor和输入tensor共享相同的物理内存。

  支持的数据类型：float16、float32。

- **参数说明**

  - input：torch.Tensor类型，输入tensor。
  - dim：int类型，维度。
  - start：int类型，初始索引。
  - length：int类型，长度。

- **规格限制**

  计算数据类型只支持float16、float32，不支持其他数据类型。

- **支持的计算库**

  CNNL。

.. _torch.ne:

ne
----------------------------
.. code::

   torch.ne(input, other, out=None) → Tensor

- **功能描述**

  逐元素计算两个输入tensor是否不相等。

  支持的数据类型：float16、float32。

- **参数说明**

  - input：torch.Tensor，输入计算tensor。
  - other：torch.Tensor或者float，待计算的tensor或者value。
  - out：torch.Tensor，可选，输出tensor，必须是BoolTensor。

- **规格限制**

  计算数据类型只支持float16、float32，不支持int4、int8、int16。

- **支持的计算库**

  CNNL。

.. _torch.neg:

neg
----------------------------
.. code::

   torch.neg(input, out=None) → Tensor

- **功能描述**

  获得与输入tensor逐元素取反的tensor。

  支持的数据类型：float16、float32。

- **参数说明**

  - input：torch.Tensor，输入计算tensor。
  - out：torch.Tensor，可选，输出tensor。

- **规格限制**

  计算数据类型只支持float16、float32，不支持int4、int8、int16。

- **支持的计算库**

  CNNL。

.. _torch.round:

round
----------------------------
.. code::

  torch.round(input, out=None) → Tensor

- **功能描述**

  将输入tensor中每个元素四舍五入到最近的整数，返回一个新的tensor。

  支持的数据类型：float16、float32。

- **参数说明**

  - input：torch.Tensor类型，输入tensor。
  - out：torch.Tensor类型，可选，输出tensor。

- **规格限制**

  - 当输入数据类型为float16时，在MLU200系列芯片上输入范围不能超过int16表示范围，即[-32768，32767]。
  - 当输入数据类型为float32时，在MLU200系列芯片上输入范围限制在[-2^23, 2^23]，在MLU300系列上输入限制在int32表示范围内。

- **支持的计算库**

  CNNL。

.. _torch.atan:

atan
----------------------------
.. code::

   torch.atan(input, *, out=None) → Tensor

- **功能描述**

  返回输入tensor元素的反正切值。

  支持的数据类型：float16、float32。

- **参数说明**

  - input：torch.Tensor类型，输入tensor。
  - out：torch.Tensor类型，可选，输出tensor。

- **规格限制**

  输入tensor的值在[-8.71, 8.71]之间。

- **支持的计算库**

  CNNL。

.. _torch.rsub:

rsub
----------------------------
.. code::

  torch.rsub(input, other, *, alpha=1, out=None)

- **功能描述**

  对输入tensor input的每个元素与标量alpha相乘所得值为减数与被减数other做减法运算，返回输出tensor。公式：

  .. figure:: ../doc_image/rsub.*

  输入tensor ``input`` 和 ``other`` 必须是相同shape或者符合broadcast规则。

  支持的数据类型：float16、float32。

- **参数说明**

  - input：torch.Tensor类型，第一个输入tensor，计算时与标量alpha相乘。
  - other：torch.Tensor类型或者int类型或者float类型，第二个输入tensor或value。
  - alpha：int类型或者float类型，input标量乘数。
  - out：torch.Tensor类型，可选，输出tensor。

- **规格限制**

  不支持out选项。

- **支持的计算库**

  CNNL。

.. _torch.prod:

prod
----------------------------
.. code::

   torch.prod(input, dtype=None) → Tensor

.. code::

   torch.prod(input, dim, keepdim=False, dtype=None) → Tensor

- **功能描述**

  该算子提供以下功能：

  1) 返回输入tensor中所有元素的乘积。
  2) 返回给定维度dim中输入tensor每行的乘积。如果 ``keepdim`` 为 ``True`` ，则输出tensor与输入的维度相同，但在维度dim中，它的大小为1。否则，dim将被压缩，导致输出tensor的维数比输入少1。

  支持的数据类型：float16、float32、int32。

- **参数说明**

  - input：torch.Tensor，设置输入tensor。
  - dim：设置需要计算的维度，int类型。
  - keepdim：设置输出和输入维度是否保持一致，bool类型，默认为False。
  - dtype：可选输入，期望返回tensor的数据类型。

- **规格限制**

  - 必须配置输入tensor，如果不设置dim和keepdim，则按功能描述中1处理，否则按功能描述中2处理。
  - 不支持dtype可选输入。

- **支持的计算库**

  CNNL。

.. _torch.reciprocal:

reciprocal
--------------
.. code::

   torch.reciprocal(input, *, out=None) → Tensor

- **功能描述**

  对输入tensor求倒数。

  支持的数据类型：float16、float32。

- **参数说明**

  - input：输入tensor。
  - out：torch.Tensor类型，可选，输出tensor。仅CNNL计算库支持该参数。

- **规格限制**

  - 无。

- **支持的计算库**

  CNNL。

.. _torch.remainder:

remainder
----------------------------
.. code::

   torch.remainder(input, other, *, out=None) → Tensor

- **功能描述**

  逐元素计算除法的余数。

  支持的数据类型：float16、float32、float64、int32、int64。（注意：当前最大只支持32位数据，int64和float64只可用于对应的32位数据类型所表示范围内数据。）

- **参数说明**

  - input：被除数，输入tensor。
  - other：除数，输入tensor或scalar。
  - out：可选，输出tensor。

- **规格限制**

  除数不为0。

- **支持的计算库**

  CNNL。

.. _torch.sigmoid:

sigmoid
----------------------------
.. code::

  torch.sigmoid(input, out=None) → Tensor

- **功能描述**

  将输入进行激活函数处理。公式：

  .. figure:: ../doc_image/sigmoid.*

  支持的数据类型：float16、float32。

- **参数说明**

  - input：torch.Tensor类型，输入tensor。
  - out：torch.Tensor类型，可选，输出tensor。

- **规格限制**

  目前要求输入数据必须在[-7.5, 7.5]范围内。

- **支持的计算库**

  CNNL。

.. _torch.sort:

sort
----------------------------
.. code::

   torch.sort(input, dim=-1, descending=False, out=None) -> (Tensor, LongTensor)

- **功能描述**

  返回一个namedtuple(values, indices)，values是沿着给定维度返回给定输入tensor的排序结果，indices是对应元素的下标。

  支持的数据类型：float16、float32。

- **参数说明**

  - input：torch.Tensor类型，输入tensor。
  - dim：int64_t类型，可选。设置进行排序的维度，默认值为-1。
  - descending：bool类型，可选。设置排序顺序（降序或者升序），默认为False。
  - out：tuple类型，可选。设置可存放输出结果（Tensor, LongTensor）的tuple。

- **规格限制**

  无。

- **支持的计算库**

  CNNL。

.. _torch.sqrt:

sqrt
--------------
.. code::

   torch.sqrt(input, *, out=None) → Tensor

- **功能描述**

  对输入的tensor开平方。公式：

  .. figure:: ../doc_image/sqrt.*

  支持的数据类型：float16、float32。

- **参数说明**

  - input：输入tensor。
  - out：可选，输出tensor，默认为None。

- **规格限制**

  CNNL：输入类型为float32时，输入数据范围为[1e-10, 1e10]；输入类型为float16时，输入数据范围为[1e-3, 1e-2]或者[1e-1, 60000]。

- **支持的计算库**

  CNNL。

.. _torch.stack:

stack
--------------
.. code::

   torch.stack(tensors, dim=0, *, out=None) → Tensor

- **功能描述**

  输入数据按维度拼接，拼接后维度会加1，即增加拼接的那个维度。

  支持的数据类型：bool、uint8、int8、int16、int32、int64、float16、float32、float64。（注意：当前最大只支持32位数据，int64和float64只可用于对应的32位数据类型所表示范围内数据。）

- **参数说明**

  - tensors：包含多个输入tensor的序列。
  - dim：int类型，设置新增的拼接的维度，默认为0。
  - out：可选，输出tensor，默认为None。

- **规格限制**

  所有输入tensor的shape必须相等。

- **支持的计算库**

  CNNL。

.. _torch.sum:

sum
--------------
.. code::

   torch.sum(input, *, dtype=None) → Tensor

.. code::

   torch.sum(input, dim, keepdim=False, *, dtype=None) → Tensor

- **功能描述**

  计算输入tensor所有元素的和或者根据输入的dim，对相应维度进行求和。支持多维度输入，如果keepdim参数为True，相应dim长度变为1，否则该维度消失，相当于调用 ``torch.squeeze()``。

  支持的数据类型：float16、float32。

- **参数说明**

  - input：输入tensor。
  - dim：int或者int的元组。
  - keepdim：bool类型，输出tensor是否保留该维度。
  - dtype：可选，torch.dtype类型，期望返回的tensor的数据类型，默认为None。

- **规格限制**

  - 当dim输入元组中包含多个相同维度时，PyTorch CPU目前虽然可以计算出结果，但该算子禁止这种错误用法，更高版本的PyTorch已经禁止该用法。

  - 不支持dtype参数。

- **支持的计算库**

  CNNL。

.. _torch.squeeze:

squeeze
----------------------------
.. code::

   torch.squeeze(input, dim=None, *, out=None) → Tensor

- **功能描述**

  按照指定dim移除size为1的维度。

  支持的数据类型：bool、uint8、int8、int16、int32、int64、float16、float32、float64。（注意：当前最大只支持32位数据，int64和float64只可用于对应的32位数据类型所表示范围内数据。）

- **参数说明**

  - input：输入tensor。
  - dim：int类型，设置需要移除的维度，默认为None。
  - out：可选，输出tensor，默认为None。

- **规格限制**

  无。

- **支持的计算库**

  CNNL。

.. _torch.sub:

sub
--------------
.. code::

   torch.sub(input, other, *, alpha=1, out=None) → Tensor

- **功能描述**

  other的每个元素与标量alpha相乘后作为减数与被减数input做减法运算，返回输出tensor。

  输入tensor ``input`` 和 ``other`` 必须是相同shape或者符合broadcast规则。

  支持的数据类型：float16、float32。

- **参数说明**

  - input：输入tensor。
  - other：输入tensor，计算时与标量alpha相乘。
  - alpha：other的标量乘数，默认为1。
  - out：可选，输出output，默认为None。

- **规格限制**

  不支持out选项。

- **支持的计算库**

  CNNL。

.. _torch.t:

t
----------------------------
.. code::

  torch.t(input) → Tensor

- **功能描述**

  对输入tensor进行转置。要求输入tensor=2-D。

  2-D tensor结果和transpose(input，0，1)一致。

  支持的数据类型：float16、float32。

- **参数说明**

  - input：torch.Tensor类型，输入tensor=2-D。

- **规格限制**

  无。

- **支持的计算库**

  MagicMind。

.. _torch.tanh:

.. _tanh:

tanh
----------------------------
- **功能描述**

  将输入数据经过激活函数处理。公式：

  .. figure:: ../doc_image/tanh.*

  支持的数据类型：float16、float32。

- **参数说明**

  无。

- **规格限制**

  无。

- **支持的计算库**

  CNNL。

.. _torch.topk:

topk
----------------------------
.. code::

   torch.topk(input, k, dim=None, largest=True, sorted=True, out=None) -> (Tensor, LongTensor)

- **功能描述**

  返回一个namedtuple(values, indices)，values是沿着给定维度返回给定输入tensor的k个最大/最小元素，indices是对应元素的下标。

  支持的数据类型：float16、float32、float64、int8、int16、int32、int64、uint8。（注意：当前最大只支持32位数据，int64和float64只可用于对应的32位数据类型所表示范围内数据。）

- **参数说明**

  - input：tensor类型，输入tensor。
  - out： tensor类型，输出tensor。
  - k：int64_t类型，设置返回元素的个数。
  - dim：int64_t类型，可选。设置进行处理的维度，默认为输入的最后一个维度。
  - largest：bool类型，可选。设置是否返回最大元素，默认为True。
  - sorted：bool类型，可选。设置是否按照排序顺序返回元素，默认为True。
  - out：tuple类型，可选。设置可存放输出结果（Tensor, LongTensor）的tuple。

- **规格限制**

  - sorted只支持True。
  - 若出现多个相同的value，MLU、CPU、GPU取出的index值可能不完全一致。

- **支持的计算库**

  CNNL。

.. _torch.transpose:

transpose
----------------------------
.. code::

   torch.transpose(input, dim0, dim1) → Tensor

- **功能描述**

  对输入tensor进行转置。

  支持的数据类型：float16、float32。

- **参数说明**

  - input：输入tensor。
  - dim0：需要转置的第一个维度。
  - dim1：需要转置的第二个维度。

- **规格限制**

  输入tensor的维数必须大于等于2且小于等于4，必须提供两个维度的索引。

- **支持的计算库**

  CNNL，MagicMind。

.. _torch.unsqueeze:

unsqueeze
----------------------------
.. code::

   torch.unsqueeze(input, dim)

- **功能描述**

  将输入tensor按照指定的维度扩展为1。

  支持的数据类型：float16、float32。

- **参数说明**

  - input：torch.Tensor类型，输入tensor。

  - dim：int64_t类型，设置需要扩展的维度。

- **规格限制**

  输入为不大于8维的tensor。

- **支持的计算库**

  CNNL。

.. _torch.unique:

unique
----------------------------
.. code::

   torch.unique(sorted=True, return_inverse=False, return_counts=False, dim=None)

- **功能描述**

  功能类似于数学中的集合，就是挑出tensor中的独立不重复元素。

  支持的数据类型：float32、int32、float64、int64。（注意：当前最大只支持32位数据，int64和float64只可用于对应的32位数据类型所表示范围内数据。）

- **参数说明**

  - sorted：bool类型，在返回为输出之前是否按升序对唯一元素进行排序。

  - return_inverse：bool类型，是否还返回原始输入中元素在返回的唯一列表中所处位置的索引。

  - return_counts：bool类型，是否还返回每个唯一元素的计数。

  - dim：int64_t类型，应用唯一的维度。如果为None，则返回拼合输入的唯一性。

- **规格限制**

  输入为不大于8维的tensor。

- **支持的计算库**

  CNNL。

.. _torch.zeros_like:

zeros_like
----------------------------
.. code::

  torch.zeros_like(input, *, dtype=None, layout=None, device=None, requires_grad=False, memory_format=torch.preserve_format) → Tensor

- **功能描述**

  返回与输入相同形状的tensor，被实数0填充。

  支持的数据类型：bool、float16、float32、int32、int16、int8、int64、uint8。

- **参数说明**

  - input：torch.Tensor，输入tensor。
  - dtype：torch.dtype类型，可选，期望返回的tensor的数据类型，默认是torch.int64。
  - layout：torch.layout类型，默认是torch.strided。该参数不支持修改。
  - device：torch.device类型，可选，期望返回的tensor的设备类型。默认是None，表示当前默认tensor类型的当前设备（参考torch.set_default_tensor_type()），如果是CUDA或者MLU类型，表示CUDA或者MLU的当前设备。
  - requires_grad：bool类型，可选，设置成True时，autograd会开启。默认是False。
  - memory_format：输出的内存格式，缺省：torch.preserve_format。

- **规格限制**

  无

- **支持的计算库**

  CNNL。

.. _torch.Tensor.ops:

torch.Tensor算子说明
==========================

abs
---------------------
详见 torch.abs_ 。

add
---------------------
详见 torch.add_ 。

addcdiv
---------------------
详见 torch.addcdiv_ 。

addcmul
---------------------
详见 torch.addcmul_ 。

addmm
---------------------
详见 torch.addmm_ 。

alias
---------------------
- **功能描述**

  对tensor取别名操作。

  支持的数据类型：int8、uint8、float16、int16、float32、int32、int64、float64。

.. attention::

   | 当前最大只支持32位数据，int64和float64只可用于32位对应类型所表示范围内数据。

- **参数说明**

  无。

- **规格限制**

  输入为不大于8维的tensor。

- **支持的计算库**

  CNNL。

any
---------------------
详见 torch.any_ 。

as_stride
---------------------
详见 torch.as_stride_ 。

bitwise_and
---------------------
该算子支持原位运算。详见 torch.bitwise_and_。

bitwise_or
---------------------
该算子支持原位运算。详见 torch.bitwise_or_。

bitwise_not
---------------------
该算子支持原位运算。详见 torch.bitwise_not_。

bmm
---------------------
详见 torch.bmm_ 。

chunk
--------------------
详见 torch.chunk_ 。

copy\_
---------------------

.. code::

   torch.Tensor.copy_(src, non_blocking=False) → Tensor

- **功能描述**

  将源tensor的数据拷贝至目标tensor。

  支持的数据类型：int8、uint8、int16、int32、int64、float16、float32、bool。

- **参数说明**

  - src：源tensor。
  - non_blocking：异步开关，默认为False。当copy为CPU拷贝至MLU设备，且输入tensor为pinned_memory时，将会执行异步拷贝。其它情况，这个参数不影响。


- **规格限制**

  无。

- **支持的计算库**

  CNNL。

clone
---------------------
详见 torch.clone_。


all
---------------------
.. code::

   torch.Tensor.all() → bool

- **功能描述**

  如果源tensor的数据都为True，则返回True，否则返回False。

  支持的数据类型：int8、bool。

- **参数说明**

  无。

- **规格限制**

  目前不支持原生的uint8类型。

- **支持的计算库**

  CNNL。

all
---------------------
.. code::

   torch.Tensor.all(dim, keepdim=False, out=None) → Tensor

- **功能描述**

  如果源tensor的dim维的每一行数据都为True，则返回True，否则返回False。

  如果 ``keepdim`` 为True，输出tensor形状除了 ``dim`` 维为1，其他维度与输入tensor相同；

  否则 ``dim`` 维度被去除，输出维度比输入维度少一维。

  支持的数据类型：int8、bool。

- **参数说明**

  - dim：处理维度。
  - keepdim：是否保持输出与输入维度相同。
  - out：torch.Tensor类型，可选参数，如果设置，则为输出tensor。

- **规格限制**

  目前不支持原生的uint8类型。

- **支持的计算库**

  CNNL。

clamp
---------------------
详见 torch.clamp_ 。

diag
---------------------
详见 torch.diag_ 。

div
--------------------
详见 torch.div_ 。

eq
---------------------
详见 torch.eq_ 。

exp
---------------------
详见 torch.exp_ 。

expand
---------------------
.. code::

   torch.Tensor.expand(*sizes) → Tensor

- **功能描述**

  返回当前张量在某维扩展更大后的张量。

  支持的数据类型：bool、int8、uint8、int16、int32、float16、float32。

- **参数说明**

  - sizes：期望输出tensor的size。

- **规格限制**

  无。

- **支持的计算库**

  CNNL。

fill\_
-------------
.. code::

   torch.Tensor.fill_(value) → Tensor

- **功能描述**

  用特定值填充tensor。

  支持的数据类型：bool、int8、uint8、int16、int32、float16、float32、float64、int64。（注意：当前最大只支持32位数据，int64和float64只可用于对应的32位数据类型所表示范围内数据。）

- **参数说明**

  - value：被用于填充的特定值。

- **规格限制**

  无。

- **支持的计算库**

  CNNL。

flatten
--------------------
详见 torch.flatten 。

floor
--------------------
详见 torch.floor_ 。

gather
--------------------
详见 torch.gather_ 。

ge
---------------------
详见 torch.ge_ 。

gt
---------------------
.. code::

  torch.Tensor.gt(other) -> Tensor

详见 torch.gt_ 。

le
---------------------
.. code::

  torch.Tensor.le(other) -> Tensor

详见 torch.le_ 。

log
---------------------
详见 torch.log_ 。

log2
---------------------
详见 torch.log2_ 。

masked_select
----------------------
详见 torch.masked_select_ 。

matmul
---------------------
详见 torch.matmul_ 。

max
--------------------
详见 torch.max_ 。

min
--------------------
详见 torch.min_ 。

mean
---------------------
详见 torch.mean_ 。

mm
---------------------
详见 torch.mm_ 。

mul
---------------------
详见 torch.mul_ 。

masked_fill\_
---------------------
.. code::

   masked_fill_(mask, value)

- **功能描述**

  根据掩码是否为True对tensor逐元素使用value进行填充。掩码的shape相对于tensor的shape必须是可广播的。

  支持的数据类型：float16、float32, float64。（注意：当前最大只支持32位数据，float64只可用于对应的32位数据类型所表示范围内数据。）。

- **参数说明**

  - mask：torch.BoolTensor类型。用于选择是否填充的掩码。

  - value：float类型。用于填充的值。

- **规格限制**

  无。

- **支持的计算库**

  CNNL。

remainder
--------------------
详见 torch.remainder_ 。

permute
----------------------------
.. code::

  torch.Tensor.permute(*dim) →  Tensor

- **功能描述**

  将输入的维度按照给定模式进行重排。

  支持的数据类型：float16、float32。

- **参数说明**

  dim：输入需要重排的维度。

- **规格限制**

  只支持4维。

- **支持的计算库**

  CNNL。


pow
----------------------------
.. code::

   pow(input, exponent, *, out=None) → Tensor

- **功能描述**

  乘方运算：取输入中每个元素的幂，返回tensor和结果。公式：

  .. figure:: ../doc_image/pow.*

  支持的数据类型：float16、float32。

- **参数说明**

  - exponent：指数值。
  - out：输出tensor。

- **规格限制**

  若底数为负数，指数为1、2、3时，精度较低。

- **支持的计算库**

  CNNL。

prod
----------------------------
详见 torch.prod_ 。

reciprocal
--------------------
详见 torch.reciprocal_ 。

repeat
----------------------------
.. code::

   tensor.repeat(*sizes) → Tensor

- **功能描述**

  根据指定维度复制 ``sizes`` 的输入tensor，并沿着相应维度拼接。

  ``torch.repeat()`` 算子与 ``numpy.repeat()`` 功能不同，更接近于 ``numpy.tile()`` 接口。如果要使用类似

  ``numpy.repeat()`` 功能的接口，可以使用 ``torch.repeat_interleave()`` 接口。

  支持的数据类型：float16、float32、int8、int16、int32、bool。

- **参数说明**

  sizes：设置拼接的次数。

- **规格限制**

  - 目前要求输入tensor与sizes的维度小于等于4。
  - 输入输出维度必须大于0。
  - 输出每一维度必须是输入对应维度的整数倍。

- **支持的计算库**

  CNNL。

remainder
--------------------
详见 torch.remainder_ 。

resize\_
----------------------------
.. code::

   resize_(*sizes, memory_format=torch.contiguous_format) → Tensor

- **功能描述**

  将tensor resize到指定的大小。

  支持的数据类型：float16、float32。

- **参数说明**

  - sizes：torch.Size或者int类型，期望的大小。

  - memory_format：torch.memory_format类型，可选，期望的存储形式。

- **规格限制**

  新生成的tensor是原tensor的一份拷贝。

- **支持的计算库**

  CNNL。

sqrt
---------------------
详见 torch.sqrt_ 。

sum
---------------------
详见 torch.sum_ 。

select
---------------------

.. code::

   select(dim, index) → Tensor

- **功能描述**

  将src中数据根据index中的索引按照dim的方向切片到指定的tensor上。

  支持的数据类型：float32、float16。

- **参数说明**

  - dim：int类型，需要对输入tensor进行索引的维度。

  - index：int类型，索引。

- **规格限制**

  - tensor维度小于等于8。

  - 输出tensor是原tensor的一份拷贝，不是视图。

- **支持的计算库**

  CNNL。

sum
---------------------
详见 torch.sum_ 。

squeeze
---------------------
详见 torch.squeeze_ 。

sub
---------------------
详见 torch.sub_ 。

atan
---------------------
详见 torch.atan_ 。

atan\_
---------------------
原位版本的 torch.atan_ 。

t
---------------------
详见 torch.t 。

tanh
---------------------
详见 torch.tanh_ 。

topk
---------------------
详见 torch.topk_ 。

transpose
---------------------
详见 torch.transpose_ 。

.. _type:

type
---------------------
.. code::

   type(dtype=None, non_blocking=False, **kwargs) → str or Tensor

- **功能描述**

  当参数为空时，以字符串形式返回tensor的类型，返回示例：``"torch.mlu.FloatTensor"`` 。
  
  当指定dtype参数时，转换成对应类型的tensor并返回， 如果dtype和已原tensor类型一致，则返回原tensor。

  支持的数据类型：float32、float16, int32, short, int8, uint8, bool。

- **参数说明**

  - dtype：dtype类型。

  - non_blocking：异步开关，默认为False。
    当输入tensor为pinned_memory，输出tensor为MLU设备时，将会执行异步拷贝。对于其它情况，此参数不受影响。

  - \*\*kwargs：为了兼容已弃用的 async 关键字参数，现已被 non_blocking 替代。

- **规格限制**

  - 原Tensor与输出后tenor需满足MLU所支持的类型转换。
  - 向下转型时需考虑数据溢出时因MLU与CPU的截断处理有差异，结果可能不一致。

- **支持的计算库**

  CNNL。


unique
----------------------
详见 torch.unique_ 。

unsqueeze
---------------------
详见 torch.unsqueeze_ 。

uniform\_
----------------------------
.. code::

   uniform_(from=0, to=1) → Tensor

- **功能描述**

  返回输入tensor的均匀分布，即根据输入范围生成一个均匀分布，并替换原tensor的数据。

  支持的数据类型：float32。

- **参数说明**

  - from：float类型，均匀分布的起点值。
  - to：float类型，均匀分布的终点值。

- **规格限制**

  输入为不大于8维的tensor。

- **支持的计算库**

  CNNL。

view
----------------------------
.. code::

  view(*shape) → Tensor

- **功能描述**

  改变输入tensor的shape。

  支持的数据类型：float16、float32。

- **参数说明**

  shape：设置需要输出的tensor的shape。

- **规格限制**

  - 输入tensor的总规模等于输出tensor的总规模。
  - 只支持小于等于4维的tensor。
  - 只支持最多推测一个维度的情况（即最多给一个-1）。

- **支持的计算库**

  CNNL。

.. attention::
   | 与CPU view 算子不同，MLU view 算子的输出tensor不能与输入tensor共享数据。

.. _torch.Tensor.zero_:

zero\_
----------------------------
.. code::

  torch.Tensor.zero_() → Tensor

- **功能描述**

  返回与输入相同形状的tensor，被实数0填充。

  支持的数据类型：bool、float16、float32、int32、int16、int8、int64、uint8。

- **参数说明**

  - input：torch.Tensor，输入tensor。

- **规格限制**

  无。

- **支持的计算库**

  CNNL。

.. _torch.nn.Modules:

torch.nn模块算子说明
====================================

.. _torch.nn.AdaptivePool2d:

AdaptivePool2d
----------------
.. code::

  torch.nn.AdaptiveAvgPool2d(output_size)

  torch.nn.AdaptiveMaxPool2d(output_size, return_indices=False)

- **功能描述**

  对输入tensor执行自适应的池化操作，支持max和avg两种池化。（注意：与原生CPU或者GPU算子功能不同，当调用AdaptiveMaxPool2d或者torch.nn.functional.adaptive_max_pool2d并且需返回最大元素的索引时，MLU返回的是一个池化kernel内的局部索引，而不是输入的全局索引。）

  支持的数据类型：float16、float32。

  支持AdaptivePool2d相关PyTorch算子：

  - torch.nn.AdaptiveAvgPool2d，torch.nn.functional.adaptive_avg_pool2d。
  - torch.nn.AdaptiveMaxPool2d，torch.nn.functional.adaptive_max_pool2d（MagicMind不支持）。

- **参数说明**

  - output_size：目标输出大小的图像的形式H*W，可以是一个元组(H,W)或正方形图像H*H的单独的H。H和W可以是int或None，其中，None表示输出size跟输入size相同。

    可根据input_size和output_size求出stride和kernel_size：

    - stride = floor(input_size/output_size)。

    - kernel_size = input_size - (output_size-1)*stride。

  - return_indices：bool类型（avgpool无此参数），可选，默认值为False。为True时，返回输出和最大值相对位置，相对位置为对应Kernel的位置坐标。

- **规格限制**

  - CNNL计算库：

  - 当输入shape为(\*, iH, iW)，传入参数output_size为(H, W)时，需要满足 :math:`(iH / H + 2) * (iW / W + 2) <= 3582` 。
  - 当该算子用于训练，进行反向计算时，还需要满足 :math:`(iH / H <= 478 \&\& iW / W <= 478)` 。

- **支持的计算库**

  CNNL，MagicMind。

.. _torch.nn.AdaptiveAvgPool3d:

AdaptiveAvgPool3d
---------------------
- **功能描述**

  对输入tensor执行自适应的池化操作，支持avg池化。

  支持的数据类型：float16、float32。

  支持AdaptiveAvgPool3d相关PyTorch算子：

  - torch.nn.AdaptiveAvgPool3d，torch.nn.functional.adaptive_avg_pool3d。

- **参数说明**

  - output_size：目标输出大小的图像的形式D*H*W，可以是一个元组(D,H,W)或正方形图像D*D*D的单独的D。D、H和W可以是int或None，其中，None表示输出size跟输入size相同。

    可根据input_size和output_size求出stride和kernel_size：

    - stride = floor(input_size/output_size)。

    - kernel_size = input_size - (output_size-1)*stride。

- **规格限制**

  无。

- **支持的计算库**

  CNNL。


.. _torch.nn.BatchNorm1d:

BatchNorm1d
--------------
.. code::

  torch.nn.BatchNorm1d(num_features, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

- **功能描述**

  批量归一化层，沿C方向对2维或3维输入tensor的每batch数据进行标准化。


  支持的数据类型：float16、float32。

- **参数说明**

  - num_features：int类型，输入维度（N, C, L）或（N, C）的C维度大小。
  - eps：float类型，为了防止除以0而添加到分母的一个极小值，默认为1e-5。
  - momentum：float类型，训练中更新running_mean以及running_var的动量参数。
  - affine：bool类型，为True时，层中包含有可学习的仿射参数，并在标准化后进行仿射变换。
  - track_running_stats：bool类型，为True时跟踪running_mean以及running_var的值。

- **规格限制**

  - running_mean、running_var、weight、bias的shape = (1, ci, 1)或（1, ci）。输出shape和输入相同。

  - 训练时不支持将affine设置成False。

- **支持的计算库**

  CNNL。

.. _torch.nn.BatchNorm2d:

BatchNorm2d
--------------
.. code::

  torch.nn.BatchNorm2d(num_features, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

- **功能描述**

  批量归一化层，沿C方向对4维输入tensor的每batch数据计算标准化 :math:`（x - avg(x)）/ x` 的标准差。公式：

  .. figure:: ../doc_image/batch_norm.*

  支持的数据类型：float16、float32。

- **参数说明**

  - num_features：int类型，输入维度（N, C, H, W）的C维度大小。
  - eps：float类型，为了防止除以0而添加到分母的一个极小值，默认为1e-5。
  - momentum：float类型，用于计算running_mean和running_var的值。
  - affine：bool类型，为True时，具有可学习的仿射参数。
  - track_running_stats：bool类型，为True时跟踪running_mean以及running_var的值。

- **规格限制**

  - running_mean、running_var、weight、bias的shape = (1, ci, 1, 1)，输出shape和输入相同。

  - 训练时不支持将affine设置成False。

- **支持的计算库**

  CNNL，MagicMind。

.. _torch.nn.BatchNorm3d:

BatchNorm3d
--------------
.. code::

  torch.nn.BatchNorm3d(num_features, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

- **功能描述**

  批量归一化层，沿C方向对5维输入tensor的每batch数据进行标准化。

  支持的数据类型：float16、float32。

- **参数说明**

  - num_features：int类型，输入维度（N, C, D, H, W）的C维度大小。
  - eps：float类型，为了防止除以0而添加到分母的一个极小值，默认为1e-5。
  - momentum：float类型，用于计算running_mean和running_var的值。
  - affine：bool类型，为True时，具有可学习的仿射参数。
  - track_running_stats：bool类型，仅在为True时跟踪running_mean以及running_var的值。

- **规格限制**

  - running_mean、running_var、weight、bias的shape = (1, ci, 1, 1, 1)，输出shape和输入相同。

  - 训练时不支持将affine设置成False。

- **支持的计算库**

  CNNL，MagicMind。


.. _torch.nn.Conv2d:

Conv2d
--------------
.. code::

   torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros')

- **功能描述**

  对由多个输入平面组成的输入信号执行2D卷积操作。

  公式为：

  .. figure:: ../doc_image/conv.*

  支持的数据类型：float16、float32。

- **参数说明**

  - in_channels：int类型，输入图像通道数。
  - out_channels：int类型，卷积通道数。
  - kernel_size：int或tuple类型，卷积核大小。
  - stride：int或tuple类型，可选，卷积步长。
  - padding：int或tuple类型，可选，对输入每个边进行零填充。
  - dilation：int或tuple类型，可选，卷积核元素之间的空洞数。
  - groups：int类型，可选，输入通道和输出通道阻塞链接数。
  - bias：bool类型，可选，如果为真，则对输出添加一个可学习的偏差。
  - padding_mode：string类型，可选，默认为 ``zeros`` 。

- **规格限制**

  - 高度：h >= kh，宽度：w >= kw。
  - depthwise模式下，仅支持dilation全为1。
  - padding_mode只支持 ``zeros``。

- **支持的计算库**

  CNNL，MagicMind。

.. _torch.nn.Conv3d:

Conv3d
--------------
.. code::

   torch.nn.Conv3d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros')

- **功能描述**

  对由多个输入平面组成的输入信号执行3D卷积操作。

  公式为：

  .. figure:: ../doc_image/nn_module_conv3d.*

  支持的数据类型：float32。

- **参数说明**

  - in_channels：int类型，输入图像通道数。
  - out_channels：int类型，卷积通道数。
  - kernel_size：int或tuple类型，卷积核大小。
  - stride：int或tuple类型，可选，卷积步长。
  - padding：int或tuple类型，可选，对输入每个边进行零填充。
  - dilation：int或tuple类型，可选，卷积核元素之间的空洞数。
  - groups：int类型，可选，输入通道和输出通道阻塞链接数。
  - bias：bool类型，可选，如果为真，则对输出添加一个可学习的偏差。
  - padding_mode：string类型，可选，默认为 ``zeros``。

- **规格限制**

  当kernel_d=1，stride_d=1，padding_d=0时，或者kernel_h=1，kernel_w=1，stride_h=1，stride_w=1，padding_h=1，padding_w=1时，限制如下：
  
  - 高度：h >= kh，宽度：w >= kw。
  - padding_mode只支持 ``zeros``。

  除上述配置外，其他情况限制如下（co表示卷积核的个数，ci表示输入feature map的个数）：

  - 输入规模限制

    - kernel_d <= 3
    - kernel_h <= 3
    - kernel_w <= 3
    - sub_kd = (kernel_d + stride_d - 1) / stride_d
    - sub_kh = (kernel_h + stride_h - 1) / stride_h
    - sub_kw = (kernel_w + stride_w - 1) / stride_w
    - align_ci = align_up(ci, 64)
    - align_filter_ci = align_up(ci / groups, 64)
    - align_co = align_up(co / groups, 64)

  - NRAM限制如下：
  
    :math:`(2 * sub\_kh * sub\_kw * align\_co * sizeof(output\_diff\_dtype) + (sub\_kd + 1) * stride\_d * stride\_h * align\_ci * sizeof(input\_diff\_dtype)  <= NRAM\_SIZE - REM\_FOR\_STACK`

  - WRAM限制如下：
  
    :math:`64 * sub\_kd * sub\_kh * sub\_kw * align\_co * sizeof(filter\_type) <= WRAM\_SIZE`

  - 当output_diff或weight的片上计算类型是int31时：

    :math:`align\_filter\_ci * sub\_kd * sub\_kh * sub\_kw * align\_co * sizeof(filter\_type) <= WRAM\_SIZE`

  - 对于NRAM_SIZE、WRAM_SIZE和REM_FOR_STACK：

    - MLU220上，WRAM_SIZE为512KB，NRAM_SIZE为512KB。
    - MLU270上，WRAM_SIZE为1024KB，NRAM_SIZE为512KB。
    - MLU290上，WRAM_SIZE为512KB，NRAM_SIZE为512KB。
    - REM_FOR_STACK为128KB。

  - filter_type、output_diff_dtype、input_diff_dtype片上计算类型为int8、int16、int31。

  - align_up()表示向上对齐。

  - dilation=1。

- **支持的计算库**

  CNNL。

.. _torch.nn.ConvTranspose2d:

ConvTranspose2d
---------------
.. code::

   torch.nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=1, padding=0, output_padding=0, groups=1, bias=True, dilation=1, padding_mode='zeros')

- **功能描述**

  对由多个输入平面组成的输入信号执行2D转置卷积操作。

  支持的数据类型：float16、float32。

- **参数说明**

  - in_channels：int类型，输入图像通道数。
  - out_channels：int类型，卷积通道数。
  - kernel_size：int或tuple类型，卷积核大小。
  - stride：int或tuple类型，可选，卷积步长，默认为1。
  - padding：int或tuple类型，可选，对输入每个边进行零填充，默认为0。
  - output_padding：int或tuple类型，可选，添加到输出形状中每个维度一侧的额外尺寸，默认为0。
  - groups：int类型，可选，输入通道和输出通道阻塞链接数，默认为1。
  - bias：bool类型，可选，如果为True，则对输出添加一个可学习的偏差，默认为True。
  - dilation：int或tuple类型，可选，卷积核元素之间的空洞数，默认为1。
  - padding_mode：string类型，可选，默认值为 ``'zeros'`` 。

- **规格限制**

  - PyTorch限制 ``padding_mode`` 必须为 ``'zeros'`` 。

- **支持的计算库**

  CNNL。


.. _torch.nn.Flatten:

Flatten
-------------
.. code::

   torch.nn.Flatten(start_dim: int = 1, end_dim: int = -1)

- **功能描述**

  在指定的连续维度区间上使tensor展平。

  支持的数据类型：float16、float32。

- **参数说明**

  - start_dim：需展平的起始维度。
  - end_dim：需展平的最后维度。

- **规格限制**

  - rank(input) <= start_dim < rank(input)
  - rank(input) <= end_dim < rank(input)

- **支持的计算库**

  MagicMind。

.. _torch.nn.Gelu:

Gelu
----------------------------
- **功能描述**

  将输入数据经过激活函数处理。公式：

  .. figure:: ../doc_image/gelu.*

  支持的数据类型：float16、float32。

- **参数说明**

  - input：torch.Tensor类型，输入tensor。

- **规格限制**

  无。

- **支持的计算库**

  CNNL。

.. _torch.nn.Hardtanh:

Hardtanh
--------------------

.. code::

  torch.nn.Hardtanh(min_val=-1.0, max_val=1.0, inplace=False)

- **功能描述**

  对输入tensor逐元素计算HardTanh：

  .. math::
       \text{HardTanh}(x) = \begin{cases}
           1 & \text{ if } x > 1 \\
           -1 & \text{ if } x < -1 \\
           x & \text{ otherwise } \\
       \end{cases}


  支持的数据类型：float16、float32

- **参数说明**

  - min_val：数值类型，可选，HardTanh的下限，默认为-1.0。
  - max_val：数值类型，可选，HardTanh的上限，默认为1.0。
  - inplace：bool类型，可选，为True时进行原位计算。

- **规格限制**

  无。

- **支持的计算库**

  CNNL。

.. _torch.nn.LogSoftmax:

LogSoftmax
----------------------------
.. code::

  torch.nn.LogSoftmax(dim: Optional[int] = None)

- **功能描述**

  对n维输入tensor执行log(Softmax(x))操作。

  支持的数据类型：float、float32。

- **参数说明**

  - dim：int类型，可选，默认为None。沿着指定dim执行LogSoftmax操作。

- **规格限制**

  该算子当前实现为高性能模式，输入tensor必须在[-7.75, 7.75]。

- **支持的计算库**

  CNNL。

.. _torch.nn.LeakyReLU:

LeakyReLU
----------------------------

.. code::

  torch.nn.LeakyReLU(negative_slope=0.01, inplace=False)

- **功能描述**

  逐元素对输入tensor的负值乘以指定斜率，正值保持不变。公式：

  .. math::

    \text{LeakyReLU}(x) = \max(0, x) + \text{negative\_slope} * \min(0, x)

  支持的数据类型：float16、float32。

- **参数说明**

  - negative_slope：数值类型，可选，设置负值输入的斜率。
  - inplace：bool类型，可选，为True时进行原位计算。

- **规格限制**

  配置inplace为True，且negative_slope为负值时，不支持反向计算。

- **支持的计算库**

  CNNL。

.. _torch.nn.Linear:

Linear
----------------------------

.. code::

  torch.nn.Linear(in_features, out_features, bias=True)

- **功能描述**

  对输入进行全连接操作。公式：:math:`y = xA^T + b`

  支持的数据类型：float16、float32、int8、int16。

- **参数说明**

  - in_features：超参，整数类型，全连接的输入特征数。
  - out_features：超参，整数类型，全连接的输出特征数。
  - bias：超参，bool类型，可选，默认为True。为False时全连接层不包含附加bias偏置。

- **形状说明**

  - input：torch.Tensor类型，输入tensor，可以包含多个维度，但最后一个维度必须是in_features，即其形状为 ``(N, * ,in_features)`` 。
  - output：torch.Tensor类型，输出tensor，维度个数与输入相同，最后一维为out_features，即其形状为 ``(N, * ,out_features)`` 。

- **属性说明**

  - weight(parameter)：权重，Parameter类型，全连接计算的权重，其形状为(out_features,in_features)。
  - bias(parameter)：偏置，Parameter类型，可选，全连接计算的偏置，其形状为(out_features)。

- **规格限制**

  - 输入tensor最低维度的形状需要与初始化时的超参匹配。
  - 权重和偏置的形状需要与初始化时的超参匹配。

- **支持的计算库**

  CNNL。

.. _torch.nn.MSELoss:

MSELoss
----------------------------

.. code::

   torch.nn.MSELoss(reduction='mean')

- **功能描述**

  计算输入Tensor与目标Tensor对应每个元素的均方误差。

  计算公式：

   .. math::
        \ell(x, y) = L = \{l_1,\dots,l_N\}^\top, \quad
        l_n = \left( x_n - y_n \right)^2,

   .. math::
        \ell(x, y) =
        \begin{cases}
            \operatorname{mean}(L), &  \text{if reduction} = \text{`mean';}\\
            \operatorname{sum}(L),  &  \text{if reduction} = \text{`sum'.}
        \end{cases}

- **参数说明**

   - reduction：string，可选，计算模式，默认为mean。为none时，直接返回计算出的tensor；为mean时，输出计算得到tensor的累加值并除以元素总数；为sum时，输出tensor累加值。

- **规格限制**

  无。

- **支持的计算库**

  CNNL。

.. _torch.nn.Pool2d:

Pool2d
----------------------------

.. code::

   torch.nn.AvgPool2d(kernel_size: Union[T, Tuple[T, T]], stride: Optional[Union[T, Tuple[T, T]]] = None, padding: Union[T, Tuple[T, T]] = 0, ceil_mode: bool = False, count_include_pad: bool = True, divisor_override: bool = None)

.. code::

   torch.nn.MaxPool2d(kernel_size: Union[T, Tuple[T, ...]], stride: Optional[Union[T, Tuple[T, ...]]] = None, padding: Union[T, Tuple[T, ...]] = 0, dilation: Union[T, Tuple[T, ...]] = 1, return_indices: bool = False, ceil_mode: bool = False)

- **功能描述**

  对输入进行池化层操作。公式：

  .. figure:: ../doc_image/pool2d.*

  支持的数据类型：float16、float32。

- **参数说明**

  - kernel_size：卷积核的大小。
  - stride：步长。
  - padding：两侧添加隐式零填充。
  - ceil_mode：bool类型，为True时使用ceil计算输出形状。
  - count_include_pad：bool类型（maxpool无此参数）。为True时，在平均计算中包括零填充。
  - divisor_override：bool类型（maxpool无此参数）。为True时，在平均计算中使用除数。
  - return_indices：bool类型（avgpool无此参数），可选，默认值为False。为True时，返回输出和最大值相对位置，相对位置为对应Kernel的位置坐标。
  - dilation：int或tuple类型，可选，卷积核元素之间的空洞数。

- **规格限制**

  - Pool2d相关算子输入限制输入维度为4维。
  - Pool2d相关算子输入数据不能为nan和inf。
  - MaxPool2d不支持dilation大于1。
  - MaxPool2d不支持 ``ceil_mode`` 设置为 ``True`` 。
  - AvgPool2d不支持 ``ceil_mode`` 设置为 ``True`` 。
  - AvgPool2d不支持 ``divisor_override`` 设置为 ``True`` 。

  - 支持Pool2d相关PyTorch算子：

    - torch.nn.AvgPool2d，torch.nn.functional.avg_pool2d。
    - torch.nn.MaxPool2d，torch.nn.functional.max_pool2d。

- **支持的计算库**

  CNNL，MagicMind。

.. _torch.nn.Pool3d:

Pool3d
----------------------------
.. code::

   torch.nn.AvgPool3d(kernel_size: Union[T, Tuple[T, T, T]], stride: Optional[Union[T, Tuple[T, T, T]]] = None, padding: Union[T, Tuple[T, T, T]] = 0, ceil_mode: bool = False, count_include_pad: bool = True, divisor_override=None)

.. code::

   torch.nn.MaxPool3d(kernel_size: Union[T, Tuple[T, ...]], stride: Optional[Union[T, Tuple[T, ...]]] = None, padding: Union[T, Tuple[T, ...]] = 0, dilation: Union[T, Tuple[T, ...]] = 1, return_indices: bool = False, ceil_mode: bool = False)

- **功能描述**

  对输入进行池化层操作。公式：

  .. figure:: ../doc_image/avgpool3d.*

  支持的数据类型：float32。

- **参数说明**

  - kernel_size：卷积核的大小。
  - stride：步长。
  - padding：两侧添加隐式零填充。
  - ceil_mode：bool类型，为True时，使用ceil计算输出形状。
  - count_include_pad：bool类型，为True时，在平均计算中包括零填充。
  - divisor_override：bool类型（maxpool无此参数）。为True时，在平均计算中使用除数。
  - return_indices：bool类型（avgpool无此参数），可选，默认值为False。为True时，返回输出和最大值相对位置，相对位置为对应Kernel的位置坐标。
  - dilation：int或tuple类型，可选，卷积核元素之间的空洞数。

- **规格限制**

  - Pool3d相关算子输入限制输入维度为5维。
  - Pool3d相关算子输入数据不能为nan和inf。
  - MaxPool3d不支持dilation大于1。
  - AvgPool3d不支持 ``divisor_override`` 设置为 ``True`` 。

  - 支持Pool3d相关PyTorch算子：

    - torch.nn.AvgPool3d，torch.nn.functional.avg_pool3d。
    - torch.nn.MaxPool3d，torch.nn.functional.max_pool3d。

- **支持的计算库**

  CNNL。

.. _torch.nn.Relu:

Relu
----------------------------
- **功能描述**

  将输入数据经过激活函数处理。公式：

  .. figure:: ../doc_image/relu.*

  支持的数据类型：float16、float32。

- **参数说明**

  - input：输入tensor。
  - inplace：设置为True时，表示进行原位操作。默认为False。

- **规格限制**

  无。

- **支持的计算库**

  CNNL，MagicMind。

.. _torch.nn.Softmax:

Softmax
----------------------------

.. code:

    torch.nn.Softmax(dim=None)

- **功能描述**

  归一化逻辑函数。公式：

  .. figure:: ../doc_image/softmax.*

  支持的数据类型：float16、float32。

- **参数说明**

  - dim：int类型，设置计算softmax的维度。

- **规格限制**

  无。

- **支持的计算库**

  CNNL。

.. _torch.nn.Softplus:

Softplus
----------------------------

.. code:

    torch.nn.Softplus(beta=1, threshold=20)

- **功能描述**

  归一化逻辑函数。公式：

  :math:`Softplus(x) = 1 / β * log(1 + exp(β * x))`

  当x * β > threshold， 将恢复线性函数。

  支持的数据类型：float16、float32。

- **参数说明**

  - beta：Softplus计算对应的β值，默认值为1。
  - threshold：高于阈值将恢复线性函数，默认值为20。

- **规格限制**

  无。

- **支持的计算库**

  CNNL。

.. _torch.nn.Tanh:

Tanh
----------------------------
详见 torch.tanh_ 。


.. _torch.nn.Threshold:

Threshold
----------------------------
- **功能描述**

  将输入数据经过激活函数处理，设置输入tensor的每个元素的阈值。公式：

  .. figure:: ../doc_image/threshold.*

  支持的数据类型：float16、float32。

- **参数说明**

  - threshold：阈值。
  - value：需要替换的值。

- **规格限制**

  无。

- **支持的计算库**

  CNNL。

true_divide
----------------------------
- **功能描述**

  按照浮点计算除法, 等效于torch.div(), 在两个输入都为bool类型或者整数标量类型情况下, 会在除法前转为默认浮点标量类型。

- **参数说明**

  - diviend：被除数。
  - divisor：除数。

- **规格限制**

  无。

- **支持的计算库**

  CNNL。

Upsample
----------------------------
.. code::

  torch.nn.Upsample(size=None, scale_factor=None, mode='nearest', align_corners=None)

- **功能描述**

  使用相邻的像素值对输入进行上采样，是avgpool的反向传播过程。

  支持的数据类型：float16、float32。

- **参数说明**

  - size：设置输出tensor的尺寸，可以为一个数或者一个元组。
  - scale_factor：指定输出为输入的多少倍，可以为一个数或者一个元组。
  - mode：使用的上采样算法。支持 ``nearest``、``linear``、``bilinear``、``bicubic``、``trilinear``。默认值为 ``nearest`` 。
  - align_corners：如果为True，输入的角像素将与输出张量对齐，因此将保存这些像素的值。仅当使用的算法为 ``linear``、``bilinear`` 或 ``trilinear`` 时可以使用。默认值为False。

- **规格限制**

  mode只支持 ``nearest`` 和 ``bilinear`` 。

- **支持的计算库**

  CNNL。

.. _torch.nn.ZeroPad2d:

ZeroPad2d
--------------
.. code::

   torch.nn.ZeroPad2d(padding)

- **功能描述**

  使用0填充输入tensor的边界。

  支持的数据类型：float16、float32、int32。

- **参数说明**

  - padding：int或者int的元组（pad_l,pad_r,pad_t,pad_b），分别指（左填充，右填充，上填充，下填充），数值大小为填充次数。

- **规格限制**

  输入tensor必须为4维。

- **支持的计算库**

  CNNL。

.. _torch.nn.functional.ops:

torch.nn.functional算子说明
==================================

adaptive_pool2d
------------------

详见 torch.nn.AdaptivePool2d_ 。

.. _torch.nn.functional.binary_cross_entropy:

binary_cross_entropy
-----------------------------------
- **功能描述**

  计算输入tensor与目标tensor的二进制交叉熵误差。公式为：

  .. figure:: ../doc_image/bce_1.*

  支持的数据类型：float32。

- **参数说明**

  - input：torch.Tensor，输入tensor。
  - target：torch.Tensor，目标tensor，数据类型和形状必须与input相同。
  - weight：torch.Tensor，可选，权重tensor。该参数用于调整每batch元素的loss比例权重。如果传入，必须与input形状相同。
  - reduction：string，可选，计算模式，默认为mean。为none时，直接返回计算出的tensor；为mean时，输出计算得到tensor的累加值并除以元素总数；为sum时，输出tensor累加值。

- **规格限制**

  input必须位于[0, 1]范围内。

- **支持的计算库**

  CNNL。

.. _torch.nn.functional.binary_cross_entropy_with_logits:

binary_cross_entropy_with_logits
------------------------------------------------------

- **功能描述**

  对神经网络输出结果进行sigmoid操作，然后对输出与标签计算交叉熵。公式为：

  .. figure:: ../doc_image/bce_2.*

  支持的数据类型：float32。

- **参数说明**

  - input：torch.Tensor，输入tensor。
  - target：torch.Tensor，目标tensor，数据类型和形状需要与input相同。
  - weight：torch.Tensor，可选，权重tensor。该参数调整每batch元素的loss比例权重，如果传入，需要与input形状相同。
  - pos_weight：torch.Tensor，可选，正例的权重tensor，即长度等于类数的tensor。如果传入，需要与input形状相同。
  - reduction：string，可选，计算模式，默认为mean。为none时，直接返回tensor；为mean时，输出Tensor的总和将除以输出中的元素数；为sum时，输出tensor将被求和。

- **规格限制**

  输入必须在[-50, 15]范围内。

- **支持的计算库**

  CNNL。

.. _torch.nn.functional.conv2d:

conv2d
--------------
.. code::

   torch.nn.functional.conv2d(input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1) → Tensor

- **功能描述**

  对由多个输入平面组成的输入信号执行2D卷积操作

  支持的数据类型：float16、float32。

- **参数说明**

  - input：torch.Tensor类型，输入tensor。
  - weight：torch.Tensor类型，输入滤波器。
  - bias：torch.Tensor类型，可选，偏差tensor。
  - stride：int或tuple类型，可选，卷积核步长。
  - padding：int或tuple类型，可选，对输入每个边进行零填充。
  - dilation：int或tuple类型，可选，卷积核元素之间的空洞数。
  - groups：int类型，可选，将输入分组，输入通道数满足被分组数整除。

- **规格限制**

  无。

- **支持的计算库**

  CNNL，MagicMind。

gelu
--------------

详见 torch.nn.Gelu_ 。

hardtanh
-------------

.. code::

  torch.nn.functional.hardtanh(input, min_val=-1.0, max_val=1.0, inplace=False) -> Tensor

详见 torch.nn.Hardtanh_ 。

.. _torch.nn.functional.interpolate:

interpolate
-------------

.. code::

  torch.nn.functional.interpolate(input, size=None, scale_factor=None, mode='nearest', align_corners=None, recompute_scale_factor=None) -> Tensor

- **功能描述**

  根据给定的size或scale_factor参数来对输入进行下/上采样。]

  支持的数据类型：float16、float32。

- **参数说明**

  - input：torch.Tensor类型，输入tensor。
  - size：设置输出tensor的尺寸，可以为一个数或者一个元组。
  - scale_factor：指定输出为输入的多少倍，可以为一个数或者一个元组。
  - mode：使用的上采样算法，有 ``nearest``, ``linear``, ``bilinear``, ``bicubic`` ,  ``trilinear`` 和 ``area`` 。默认使用 ``nearest`` 。
  - align_corners：bool类型，可选。如果为True，输入的角像素将与输出张量对齐，因此将保存下来这些像素的值。仅当使用的算法为 ``linear``、``bilinear`` 或 ``trilinear`` 时可以使用。默认设置为 ``False``。
  - recompute_scale_factor：bool类型，可选。重新计算用于插值计算的scale_factor。

- **规格限制**

    - 输入为4维时，mode支持 ``nearest``、``bilinear`` 以及 ``area``  。

    - 输入为3维时，mode支持 ``nearest`` 。

- **支持的计算库**

  CNNL。

leaky_relu
--------------

.. code::

  torch.nn.functional.leaky_relu(input, negative_slope=0.01, inplace=False) -> Tensor

详见 torch.nn.LeakyReLU_ 。

linear
---------------

.. code::

  torch.nn.functional.linear(input, weight, bias=None) -> Tensor

详见 torch.nn.Linear_ 。

max_pool2d_with_indices
----------------------------
.. code::

   torch.nn.functional.max_pool2d_with_indices(input, kernel_size, stride, padding=0, dilation=1, ceil_model=False, return_indices=False) -> (Tensor, Tensor)

- **功能描述**

  对输入tensor做最大池化，并且返回最大元素的索引（注意：与原生CPU或者GPU算子功能不同，MLU返回的是一个池化kernel内的局部索引，而不是输入的全局索引。）

  支持的数据类型：float32、float16。（注意：当前最大只支持32位数据，int64和float64只可用于对应的32位数据类型所表示范围内数据。）

- **参数说明**

  - input：torch.Tensor类型，输入tensor。
  - kernel_size：卷积核滑动窗口的大小。
  - stride：每次滑动的步长，默认值为kernel_size。
  - padding：两侧添加隐式零填充，默认为0。
  - dilation：kernel元素之间的距离，默认为1。
  - ceil_mode：bool类型，设置为True时，使用ceil计算输出tensor的形状，默认为False。
  - return_indices：bool类型，设置为True时，返回最大元素的索引，默认为False。

- **规格限制**

  - Pool2d相关算子输入限制输入维度为4维。
  - Pool2d相关算子输入数据不能为nan和inf。
  - MaxPool2d不支持dilation大于1。
  - MaxPool2d不支持 ``ceil_mode`` 设置为 ``True`` 。

- **支持的计算库**

  CNNL。

pool2d
--------------

详见 torch.nn.Pool2d_ 。

pool3d
--------------

详见 torch.nn.Pool3d_ 。

relu
--------------

详见 torch.nn.Relu_ 。

softmax
--------------

详见 torch.nn.Softmax_ 。

softplus
--------------

详见 torch.nn.Softplus_ 。

sort
--------------

详见 torch.sort_ 。

sqrt
--------------

详见 torch.sqrt_ 。

.. _torch.nn.NLLLoss:

NLLLoss
-------------
.. code::

  torch.nn.NLLLoss(weight: Optional[torch.Tensor] = None, size_average=None, ignore_index: int = -100, reduce=None, reduction: str = 'mean')

- **功能描述**

  创建损失函数。常用于多分类任务，但是input在输入NLLLoss()之前，需要对input进行log_softmax函数激活，即将input转换成概率分布的形式，并且取对数。

- **参数说明**

  - weight：torch.Tensor，为每个类别的loss设置权值，常用于类别不均衡问题。weight必须是float类型的tensor，其长度要与类别C一致，即每一个类别都要设置有weight。
  - size_average：bool，当reduce=True时有效。为True时，返回的loss为除以权重之和的平均值；为False时，返回的各样本的loss之和。
  - ignore_index：int，忽略某一类别，不计算其loss，其loss会为0，并且在采用size_average时，不会计算该类的loss，除的时候的分母也不会统计该类的样本。
  - reduce：bool，返回值是否为标量，默认为True。
  - reduction：string，指定reduction应用到输出。支持的类型有'none' | 'mean' | 'sum'。默认为none。为none时，不应用任何缩减；为mean时，输出的总和将除以输出中的元素数；为sum时，输出将被求和。

- **规格限制**

  - Input：输入torch.tensor支持2维(N, C)，4维(N, C, d1, d2)。
  - Target：目标torch.tensor，target维度(N)-对应上述2维(N, C)输入，target维度(N, d1, d2)-对应上述4维(N, C, d1, d2)输入。

- **支持的计算库**

  CNNL。

.. _torch.nn.SmoothL1Loss:

SmoothL1Loss
--------------
.. code::

  torch.nn.SmoothL1Loss(size_average=None, reduce=None, reduction: str = 'mean')

- **功能描述**

  创建损失函数。如果输入和目标的绝对元素误差小于1，则使用平方项，否则使用L1项。本函数对异常值的敏感度低于MSELoss，并且在某些情况下可以防止爆炸性梯度。

  支持的数据类型：float16、float32。

- **参数说明**

  - size_average：bool，可选，该参数已经被遗弃不用。
  - reduce：bool，可选，该参数已经被遗弃不用。
  - reduction：string，可选，计算模式，默认为none。为none时，不应用任何缩减；为mean时，输出的总和将除以输出中的元素数；为sum时，输出将被求和。

- **规格限制**

  - size_average和reduce已经被遗弃，在使用时不需要传值。
  - input和target必须为同类型tensor。

- **支持的计算库**

  CNNL。

tanh
--------------

详见 torch.nn.Tanh_ 。

threshold
--------------

详见 torch.nn.Threshold_ 。

.. _自定义算子:

自定义算子说明
==================================

index
----------------------------
.. code::

   output = input[indices]

- **功能描述**

  根据输入的 ``indices`` tensor在指定的 ``input`` tensor选取对应的数值。
  支持的数据类型：int8、uint8、int16、float32、int32、int64、float64。（注意：当前最大只支持32位数据，int64和float64只可用于对应32位的数据类型所表示范围内数据。）

- **参数说明**

  - input：torch.Tensor类型，输入tensor。
  - indices：torch.Tensor类型，为输入指定的下标tensor，类型为bool或者long类型。

- **规格限制**

  - 暂时不支持多个bool类型indices作为下标输入。

- **支持的计算库**

  CNNL。


Nms
----------------------------
.. code::

   torchvision.ops.nms(boxes, scores, iou_threshold)

- **功能描述**

  非极大抑制，与Torchvision中的nms算子一致。

  支持的数据类型：float32、float64、float16、int32、int64。（注意：当前最大只支持32位数据，int64和float64只可用于对应的32位数据类型所表示范围内数据。）

- **参数说明**

  - boxes：torch.FloatTensor、torch.DoubleTensor或者torch.HalfTensor类型，输入候选框。
  - scores：torch.FloatTensor、torch.DoubleTensor或者torch.HalfTensor类型，候选框对应得分。
  - iou_threshold：float或者double类型，IOU阈值。

- **规格限制**

  ``boxes`` 和 ``scores`` 的数据类型必须保持一致。

- **支持的计算库**

  CNNL。

roi_align
----------------------------
.. code::

  torchvision.ops.roi_align(input: torch.Tensor, boxes: torch.Tensor, output_size: None, spatial_scale: float = 1.0, sampling_ratio: int = -1, aligned: bool = False) → torch.Tensor

- **功能描述**

  执行Mask R-CNN网络中的ROI Align功能。

  支持的数据类型：float32、float16。

- **参数说明**

  - input：Tensor类型，输入tensor。
  - boxes：Tensor或Tensor列表类型，以（x1，y1，x2，y2）格式选取的候选框。如果传入的是一个tensor，则第一列包含批次索引；如果传入的是tensor列表，则每个tensor对应的是一个批次中的候选框第i个元素。
  - output_size：int或Tuple类型，裁剪后的输出大小。
  - spatial_scale：float类型，映射输入坐标到候选框坐标的缩放因子，默认：1.0。
  - sampling_ratio：int类型，插值网格采样点的取样数，用来计算每个池化网格的输出值。如果大于0，使用采样率和采样网格的乘积。如果小于等于0，则使用网格的自适应数。默认：-1。
  - aligned：bool类型，如果为True，像素位移-0.5，以对齐相邻的两个像素索引，这是Detectron2版本实现，如果为False，则不偏移。

- **规格限制**

  该算子为Torchvision中注册的算子，使用前需要导入torchvision，并使用torchvision.ops.roi_align方式调用该算子。

- **支持的计算库**

  CNNL。

Nonzero
----------------------------
- **功能描述**

  返回输入tensor的所有非0元素的索引。

  支持的数据类型：bool、float32、int32、float64、int64。（注意：当前最大只支持32位数据，int64和float64只可用于32位对应类型所表示范围内数据。）

- **参数说明**

  - input：torch.Tensor类型，输入tensor。
  - out：torch.LongTensor类型，可选，包含非0元素索引的输出tensor。
  - as_tuple：bool类型，可选，默认为False，表示输出是一个shape是z*n的2维tensor，z是非0元素的数目，n是输入input的维数。当设置为True时，表示输出是一个包含n个元素的tuple，每个元素是一个1维，大小为z的tensor。

- **规格限制**

  输入input最大不超过8维。

- **支持的计算库**

  CNNL。

view
------------------

.. code::

   torch.ops.torch_mlu.view(tensor, dtype)

- **功能描述**

  按照指定类型返回一个新的Tensor, 输出与输入共享内存, 设置类型必须保持和输入数据类型位宽一致。

  对应PyTorch1.9 tensor.view(dtype)

- **参数说明**

  - tensor：输入Tensor
  - dtype: 输出Tensor对应的类型

- **规格限制**

  无。


- **支持的计算库**

  CNNL。

