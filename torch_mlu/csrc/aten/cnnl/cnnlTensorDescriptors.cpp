/*
All modification made by Cambricon Corporation: © 2022 Cambricon Corporation
All rights reserved.
All other contributions:
Copyright (c) 2014--2022, the respective contributors
All rights reserved.
For the list of contributors go to https://github.com/pytorch/pytorch/graphs/contributors
Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:
    * Redistributions of source code must retain the above copyright notice,
      this list of conditions and the following disclaimer.
    * Redistributions in binary form must reproduce the above copyright
      notice, this list of conditions and the following disclaimer in the
      documentation and/or other materials provided with the distribution.
    * Neither the name of Intel Corporation nor the names of its contributors
      may be used to endorse or promote products derived from this software
      without specific prior written permission.
THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

#include "aten/cnnl/cnnlTensorDescriptors.h"
#include "aten/cnnl/cnnl_util.h"
#include "aten/core/tensor_impl.h"
#include "aten/util/tensor_util.h"

namespace torch_mlu {
void CnnlTensorDescriptor::set_reduce(const at::Tensor& t) {
  int t_dim = t.dim();
  std::vector<int> dim_array;
  if (t_dim == 0) {
    t_dim = 1;
    dim_array.push_back(1);
  } else {
    auto t_vec = t.sizes().vec();
    for (int i = 0; i < t_dim; i++) {
      dim_array.push_back(t_vec[i]);
    }
  }
  auto * tensor_impl = getMluTensorImpl(t);
  auto data_type = tensor_impl->getCnnlType();
  TORCH_CNNL_CHECK(cnnlSetTensorDescriptor(this->mut_desc(), CNNL_LAYOUT_NCHW,
                                           data_type, t_dim, dim_array.data()));
}
void CnnlTensorDescriptor::set_reduce(const at::Tensor& t,
                                      std::vector<int64_t> keepdim) {
  int t_dim = keepdim.size();
  std::vector<int> dim_array;
  if (t_dim == 0) {
    t_dim = 1;
    dim_array.push_back(1);
  } else {
    for (int i = 0; i < t_dim; i++) {
      dim_array.push_back(keepdim[i]);
    }
  }
  auto * tensor_impl = getMluTensorImpl(t);
  auto data_type = tensor_impl->getCnnlType();
  TORCH_CNNL_CHECK(cnnlSetTensorDescriptor(this->mut_desc(), CNNL_LAYOUT_NCHW,
                                           data_type, t_dim, dim_array.data()));
}

void CnnlTensorDescriptor::set(const at::Tensor &t) {
  auto *tensor_impl = getMluTensorImpl(t);
  cnnlDataType_t data_type = tensor_impl->getCnnlType();
  set(t, data_type);
}

void CnnlTensorDescriptor::set(const at::Tensor &t, cnnlDataType_t data_type) {
  int t_dim = t.dim();
  auto *tensor_impl = getMluTensorImpl(t);
  tensor_impl->setCnnlType(data_type);
  if (!t_dim) {
      t_dim = 1;
      std::vector<int> dim_array(1, 1);
      // (sg) change CNNL_LAYOUT_NHWC to CNNL_LAYOUT_ARRAY?
      TORCH_CNNL_CHECK(cnnlSetTensorDescriptorEx(this->mut_desc(),
                                                 CNNL_LAYOUT_ARRAY,
                                                 data_type,
                                                 t_dim,
                                                 dim_array.data(),
                                                 dim_array.data()));
      return;
  }
  std::vector<int> shape_info(t_dim);
  std::vector<int> stride_info(t_dim);
  std::vector<int> cnnl_stride_info(t_dim);
  for (size_t i = 0; i < t_dim; ++i) {
      shape_info[i] = static_cast<int>(t.sizes()[i]);
      stride_info[i] = static_cast<int>(t.strides()[i]);
  }
  if (t.is_contiguous()) {
    auto contiguous_strides = get_contiguous_strides(t.sizes());
    for (size_t i = 0; i < t_dim; ++i) {
      cnnl_stride_info[i] = static_cast<int>(contiguous_strides[i]);
    }
  } else {
    cnnl_stride_info = get_cnnl_strides(shape_info, stride_info);
  }
  cnnlTensorLayout_t layout = CNNL_LAYOUT_ARRAY;
  if ((!t.is_contiguous()) && (t.suggest_memory_format() != at::MemoryFormat::Contiguous)) {
    convertShapeAndStride(shape_info, cnnl_stride_info);
    layout = cnnl::ops::suggest_cnnl_layout(t);
  }
  TORCH_CNNL_CHECK(cnnlSetTensorDescriptorEx(this->mut_desc(), layout,
                                             data_type, t_dim, shape_info.data(),
                                             cnnl_stride_info.data()));
}

void CnnlTensorDescriptor::set(const at::Tensor &t,
                               cnnlTensorLayout_t layout,
                               cnnlDataType_t data_type) {
  int t_dim = t.dim();
  auto *tensor_impl = getMluTensorImpl(t);
  if (data_type == CNNL_DTYPE_INVALID) {
    data_type = tensor_impl->getCnnlType();
  }
  if (!t_dim) {
      t_dim = 1;
      std::vector<int> dim_array(1, 1);
      TORCH_CNNL_CHECK(cnnlSetTensorDescriptorEx(this->mut_desc(),
                                                 CNNL_LAYOUT_ARRAY,
                                                 data_type,
                                                 t_dim,
                                                 dim_array.data(),
                                                 dim_array.data()));
      return;
  }
  std::vector<int> shape_info(t_dim);
  std::vector<int> stride_info(t_dim);
  if (layout == CNNL_LAYOUT_NHWC || layout == CNNL_LAYOUT_NDHWC
          || layout == CNNL_LAYOUT_NLC) {
      for (size_t i = 0; i < t_dim; ++i) {
          shape_info[i] = static_cast<int>(t.sizes()[i]);
          stride_info[i] = static_cast<int>(t.strides()[i]);
      }
      convertShapeAndStride(shape_info, stride_info);
  } else if (layout == CNNL_LAYOUT_HWCN) {
      // HWCN is only used by depthwise conv now, and the dim is 4
      TORCH_CHECK(t_dim == 4, "depthwise convolution input's dim must be 4!");
      auto convertDepthWiseConvShapeStride = [](const std::vector<int64_t>& vec,
                                                std::vector<int>& target_vec) {
          target_vec[0] = static_cast<int>(vec[2]);
          target_vec[1] = static_cast<int>(vec[3]);
          target_vec[2] = static_cast<int>(vec[1]);
          target_vec[3] = static_cast<int>(vec[0]);
      };
      convertDepthWiseConvShapeStride(t.sizes().vec(), shape_info);
      convertDepthWiseConvShapeStride(t.strides().vec(), stride_info);
  } else if (layout == CNNL_LAYOUT_TNC) {
      // TNC layout is similar to ARRAY
      for (size_t i = 0; i < t_dim; ++i) {
          shape_info[i] = static_cast<int>(t.sizes()[i]);
          stride_info[i] = static_cast<int>(t.strides()[i]);
      }
  } else {
      for (size_t i = 0; i < t_dim; ++i) {
          shape_info[i] = static_cast<int>(t.sizes()[i]);
          stride_info[i] = static_cast<int>(t.strides()[i]);
      }
  }
  TORCH_CNNL_CHECK(cnnlSetTensorDescriptorEx(this->mut_desc(), layout,
                                             data_type, t_dim, shape_info.data(),
                                             stride_info.data()));
}

void CnnlTensorDescriptor::set(int position,
                               float scale) {
  if (scale == 1.0f) {
    TORCH_CNNL_CHECK(cnnlSetTensorDescriptorPosition(this->mut_desc(),
                                                     position));
  } else {
    TORCH_CNNL_CHECK(cnnlSetTensorDescriptorPositionAndScale(this->mut_desc(),
                                                             position, scale));
  }
}

void CnnlTensorDescriptor::set_onchip_dtype(cnnlDataType_t onchip_dtype) {
  TORCH_CNNL_CHECK(cnnlSetTensorDescriptorOnchipDataType(this->mut_desc(), onchip_dtype));
}

void CnnlTensorDescriptor::set_additional_dim(const at::Tensor &t,
                                            std::vector<int> &dims) {
  auto * tensor_impl = getMluTensorImpl(t);
  const int dim = dims.size();
  cnnlDataType_t data_type = tensor_impl->getCnnlType();
  std::vector<int> stride_info(dim);
  int value = 1;
  for (size_t i = dim-1; i > 0; --i) {
    stride_info[i] = value;
    value *= dims[i];
  }
  stride_info[0] = value;
  // NCHW -> NHWC layout
  convertShapeAndStride(dims, stride_info);
  TORCH_CNNL_CHECK(cnnlSetTensorDescriptorEx(this->mut_desc(), CNNL_LAYOUT_NHWC,
                                             data_type, dim, dims.data(),
                                             stride_info.data()));
}

void CnnlTensorDescriptor::set(const at::Tensor &t,
                               std::vector<int64_t>& tensor_cnnl_size,
                               int64_t dim) {
  cnnlDataType_t data_type = getCnnlDataType(t.dtype());
  auto *tensor_impl = getMluTensorImpl(t);
  int t_dim = tensor_cnnl_size.size();
  // cnnlSoftmaxForward/cnnlSoftmaxBackward had 3-dim input limitation
  if (!t_dim) {
    t_dim = 3;
    std::vector<int> dim_array(3, 1);
    TORCH_CNNL_CHECK(cnnlSetTensorDescriptorEx(this->mut_desc(),
                                               CNNL_LAYOUT_ARRAY,
                                               data_type,
                                               t_dim,
                                               dim_array.data(),
                                               dim_array.data()));
    return;
  }
  const int out_dim = 3;
  int64_t inner_size = 1;
  int64_t outer_size = 1;
  std::vector<int64_t> shape_info(out_dim, 1);
  for (int64_t i = 0; i < dim; ++i) {
    outer_size *= tensor_cnnl_size[i];
  }
  const int dim_size = tensor_cnnl_size[dim];
  for (int64_t i = dim + 1; i < t_dim; ++i) {
    inner_size *= tensor_cnnl_size[i];
  }
  // For best performance, keep dim in last channel as
  // same with original shape size.
  if (dim == 0 && t_dim == 1) {
    shape_info[2] = dim_size;
  } else if (dim == 0 && t_dim != 1) {
    shape_info[0] = dim_size;
    shape_info[1] = inner_size;
  } else if (dim == t_dim - 1) {
    shape_info[1] = outer_size;
    shape_info[2] = dim_size;
  } else {
    shape_info[0] = outer_size;
    shape_info[1] = dim_size;
    shape_info[2] = inner_size;
  }
  auto stride_info = get_contiguous_strides(shape_info);
  std::vector<int> shape_info_int(out_dim);
  std::vector<int> stride_info_int(out_dim);
  for (size_t i = 0; i < out_dim; ++i) {
    shape_info_int[i] = static_cast<int>(shape_info[i]);
    stride_info_int[i] = static_cast<int>(stride_info[i]);
  }
  TORCH_CNNL_CHECK(cnnlSetTensorDescriptorEx(this->mut_desc(), CNNL_LAYOUT_ARRAY,
                                             data_type, out_dim, shape_info_int.data(),
                                             stride_info_int.data()));
}

// Just for pooling
void CnnlTensorDescriptor::set(const at::Tensor &t, bool keep_dim,
                             std::vector<int64_t> &keepdim_sizes,
                             cnnlDataType_t data_type) {
  auto *tensor_impl = getMluTensorImpl(t);
  if (data_type == CNNL_DTYPE_INVALID) {
    cnnlDataType_t temp_type = getCnnlDataType(t.dtype());
    data_type = temp_type;
  }
  tensor_impl->setCnnlType(data_type);
  int t_dim = t.dim();
  if (!keep_dim) {
    t_dim = keepdim_sizes.size();
  }
  std::vector<int> shape_info(t_dim);
  std::vector<int> stride_info(t_dim);
  for (size_t i = 0; i < t_dim; ++i) {
    if (keep_dim) {
      shape_info[i] = static_cast<int>(t.sizes()[i]);
      stride_info[i] = static_cast<int>(t.strides()[i]);
    } else {
      shape_info[i] = static_cast<int>(keepdim_sizes[i]);
    }
  }
  if (!keep_dim) {
    int value = 1;
    for (size_t i = t_dim-1; i > 0; --i) {
      stride_info[i] = value;
      value *= shape_info[i];
    }
    stride_info[0] = value;
  }
  convertShapeAndStride(shape_info, stride_info);
  TORCH_CNNL_CHECK(cnnlSetTensorDescriptorEx(this->mut_desc(), CNNL_LAYOUT_ARRAY,
                                             data_type, t_dim, shape_info.data(),
                                             stride_info.data()));
}

void CnnlTensorDescriptor::set_dim(const at::Tensor &t, int inputDim) {
  cnnlDataType_t data_type = getCnnlDataType(t.dtype());
  auto *tensor_impl = getMluTensorImpl(t);
  int t_dim = t.dim();
  if (!t_dim) {
    t_dim = 1;
    std::vector<int> dim_array(1, 1);
    TORCH_CNNL_CHECK(cnnlSetTensorDescriptorEx(this->mut_desc(),
                                               CNNL_LAYOUT_ARRAY,
                                               data_type,
                                               t_dim,
                                               dim_array.data(),
                                               dim_array.data()));
    return;
  }
  std::vector<int> cur_size(t_dim);
  for (size_t i = 0; i < t_dim; ++i) {
    cur_size[i] = static_cast<int>(t.sizes()[i]);
  }
  TORCH_CHECK(inputDim == 4, "inputDim need equal to 4");
  std::vector<int> cnnl_shape_size(inputDim, 1);
  std::vector<int> cnnl_stride_size(inputDim, 1);
  for (size_t i = 0; i < inputDim; ++i) {
    cnnl_shape_size[i] = t_dim > i ? cur_size[i] : 1;
  }
  cnnl_stride_size[3] = 1;
  cnnl_stride_size[2] = cnnl_shape_size[3];
  cnnl_stride_size[1] = cnnl_stride_size[2] * cnnl_shape_size[2];
  cnnl_stride_size[0] = cnnl_stride_size[1] * cnnl_shape_size[1];
  TORCH_CNNL_CHECK(cnnlSetTensorDescriptorEx(this->mut_desc(),
                                             CNNL_LAYOUT_ARRAY,
                                             data_type, inputDim,
                                             cnnl_shape_size.data(),
                                             cnnl_stride_size.data()));
}

void CnnlTensorDescriptor::set_dim(const at::Tensor &t) {
  const int inputDim = 1;
  cnnlDataType_t data_type = getCnnlDataType(t.dtype());
  std::vector<int> cnnl_size;
  cnnl_size.push_back(t.numel());
  std::vector<int> stride_size = {1};
  TORCH_CNNL_CHECK(cnnlSetTensorDescriptorEx(this->mut_desc(),
                                             CNNL_LAYOUT_ARRAY,
                                             data_type, inputDim,
                                             cnnl_size.data(),
                                             stride_size.data()));
}

void CnnlTensorDescriptor::set(const at::Tensor &t,
                               cnnlDataType_t data_type,
                               int position,
                               cnnlTensorLayout_t layout) {
  set(t, layout, data_type);
  TORCH_CNNL_CHECK(cnnlSetTensorDescriptorPosition(this->mut_desc(), position));
}

void CnnlSeqDataDescriptor::set(const at::Tensor &t) {
    int t_dim = t.dim();
    auto layout = CNNL_SEQDATA_TNC;
    auto *tensor_impl = getMluTensorImpl(t);
    cnnlDataType_t dtype = tensor_impl->getCnnlType();
    auto dim_array = caffe2::make_unique<int[]>(t_dim);
    long *t_dim_long = const_cast<long *>(t.sizes().data());
    dim_array[0] = static_cast<int>(t_dim_long[0]);
    dim_array[1] = static_cast<int>(t_dim_long[1]);
    dim_array[2] = static_cast<int>(t_dim_long[2]);
    auto seqLengthArraySize = 0;
    TORCH_CNNL_CHECK(cnnlSetSeqDataDescriptor(mut_desc(),
            layout,
            dtype,
            t_dim,
            dim_array.get(),
            seqLengthArraySize,
            nullptr,
            nullptr));
  }
void CnnlSeqDataDescriptor::set(const at::Tensor &t, cnnlSeqDataLayout_t layout) {
      auto *tensor_impl = getMluTensorImpl(t);
      cnnlDataType_t data_type = tensor_impl->getCnnlType();
      // t shape is NBTC
      auto t_size = t.sizes();
      std::vector<int> dim_array(4, 1);  // NBTC
      dim_array[0] = static_cast<int>(t_size[0]);  // N
      dim_array[1] = static_cast<int>(t_size[1]);  // B
      dim_array[2] = static_cast<int>(t_size[2]);  // T
      dim_array[3] = static_cast<int>(t_size[3]);  // C

      int seqLengthArraySize = t_size[0] * 1;  // batch × beam

      // N is batch, B is beam, T is sequence length, C is embedding size.
      TORCH_CNNL_CHECK(cnnlSetSeqDataDescriptor(
          mut_desc(), layout, data_type, 4, dim_array.data(),
          seqLengthArraySize, nullptr, nullptr));
}

void CnnlSeqDataDescriptor::set_onchip_dtype(cnnlDataType_t onchip_dtype) {
  TORCH_CNNL_CHECK(cnnlSetSeqDataDescriptorOnchipDataType(mut_desc(), onchip_dtype));
}

}  // end of namespace torch_mlu
