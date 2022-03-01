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

#include <algorithm>
#include "aten/cnnl/cnnl_util.h"
#include "aten/core/tensor_impl.h"
#include "aten/operators/cnnl/internal/cnnl_internal.h"
#include "aten/operators/cnnl/cnnl_kernel.h"

namespace torch_mlu {
namespace cnnl {
namespace ops {

cnnlTensorLayout_t suggest_cnnl_layout(const at::Tensor& input) {
  auto suggest_memory_format = input.suggest_memory_format();
  cnnlTensorLayout_t layout = CNNL_LAYOUT_ARRAY;
  switch (input.dim()) {
    case 4:
      layout = (suggest_memory_format == at::MemoryFormat::ChannelsLast)
      ? CNNL_LAYOUT_NHWC : CNNL_LAYOUT_NCHW;
      break;
    case 5:
      layout = (suggest_memory_format == at::MemoryFormat::ChannelsLast3d)
      ? CNNL_LAYOUT_NDHWC : CNNL_LAYOUT_NCDHW;
      break;
    default:
      layout = CNNL_LAYOUT_ARRAY;
  }
  return layout;
}

std::vector<int64_t> get_trans_order(int64_t dim, bool channels_last) {
  std::vector<int64_t> order(dim, 0);
  for (int64_t i = 0; i < dim - 1; ++i) {
    if (channels_last) {
      // from channels first to channels last.
      order[i + 1] = (i + 1 + dim - 1) % (dim - 1) + 1;
    } else {
      // from channels last to channels first.
      order[i + 1] = (i - 1 + dim - 1) % (dim - 1) + 1;
    }
  }
  return order;
}

at::Tensor unsafe_trans_memory_format_inplace(const at::Tensor& input, bool channels_last) {
  if ((input.dim() < 3) || (input.dim() > 5) || (is_channels_last(input) == channels_last)) {
    return input;
  }
  c10::MemoryFormat memory_format = c10::MemoryFormat::Contiguous;
  if (channels_last) {
    memory_format = get_channels_last_memory_format(input.dim());
  }
  auto output = cnnl_contiguous(input, memory_format);
  auto input_impl = getMluTensorImpl(input);
  auto output_impl = getMluTensorImpl(output);
  auto input_ptr = input_impl->cnnlMalloc();
  auto output_ptr = output_impl->cnnlMalloc();
  cnrtDataType_t src_type = fromCnnlType2CnrtType(input_impl->getCnnlType());
  cnrtDataType_t dst_type = fromCnnlType2CnrtType(output_impl->getCnnlType());
  TORCH_MLU_CHECK(src_type == dst_type,
  "dst_type is not the same as src_type in unsafe_trans_memory_format_inplace.");
  auto queue = getCurrentQueue();
  int insize = input_impl->numel() * cnrtDataTypeSize(src_type);
  int outsize = output_impl->numel() * cnrtDataTypeSize(dst_type);
  TORCH_MLU_CHECK(insize == outsize,
  "outsize is not equal to insize in unsafe_trans_memory_format_inplace.");
  TORCH_CNRT_CHECK(cnrtMemcpyAsync(input_ptr, output_ptr, insize, queue.queue(),
                                    CNRT_MEM_TRANS_DIR_DEV2DEV));
  input_impl->set_sizes_and_strides(output.sizes(), output.strides());

  return input;
}

at::MemoryFormat get_channels_last_memory_format(int64_t dim) {
  TORCH_MLU_CHECK((dim > 3) && (dim < 6),
    "at::MemoryFormat only support rank 4 or 5 tensor with channels_last memory format.");
  at::MemoryFormat memory_format;
  switch (dim) {
    case 4:
      memory_format = at::MemoryFormat::ChannelsLast;
      break;
    case 5:
      memory_format = at::MemoryFormat::ChannelsLast3d;
      break;
  }
  return memory_format;
}

// modify channels_last MemoryFormat for different inputs dims.
c10::MemoryFormat infer_memory_format(const int64_t& dims,
                                      const c10::MemoryFormat memory_format) {
  if (dims == 4 && memory_format == c10::MemoryFormat::ChannelsLast3d) {
    return c10::MemoryFormat::ChannelsLast;
  }
  if (dims == 5 && memory_format == c10::MemoryFormat::ChannelsLast) {
    return c10::MemoryFormat::ChannelsLast3d;
  }
  if (dims != 4 && dims != 5) {
    return c10::MemoryFormat::Contiguous;
  }
  return memory_format;
}

// get a common MemoryFormat of TensorList
c10::MemoryFormat infer_tensor_list_contiguous(const at::TensorList& tensors) {
  const int tensor_size = tensors.size();
  TORCH_MLU_CHECK(tensor_size > 0, "Input tensor num need be greater than 0.");
  bool channels_first = false;
  c10::MemoryFormat memory_format = c10::MemoryFormat::Contiguous;
  for (int i = 0; i < tensor_size; ++i) {
    const int64_t dim = tensors[i].dim();
    if (dim > 5 || dim < 4) {
      channels_first = true;
      break;
    }
  }
  if (channels_first == false) {
    // (TODO) shangang: Not best way to contiguous all tensors
    // when different memory_format.
    memory_format = tensors[0].suggest_memory_format();
  }
  return memory_format;
}

bool pair_first_down(std::pair<int64_t, int64_t>pair1, std::pair<int64_t, int64_t>pair2) {
  return pair1.first > pair2.first;
}

// strides look like create by permute or not.
bool is_permute(const at::Tensor& input) {
  if (input.is_contiguous()) {
    return false;
  }
  auto input_sizes = input.sizes().vec();
  auto input_strides = input.strides().vec();
  auto ndim = input.dim();
  for (int64_t i = 0; i < ndim; ++i) {
    if (input_strides[i] == 0) {
      return false;
    }
  }
  std::vector<std::pair<int64_t, int64_t>> strides_sizes(ndim, std::pair<int64_t, int64_t>(1, 1));
  for (int64_t i = 0; i < ndim; ++i) {
    strides_sizes[i] = std::pair<int64_t, int64_t>((static_cast<int64_t>(input_strides[i])),
            (static_cast<int64_t>(input_sizes[i])));
  }
  sort(strides_sizes.begin(), strides_sizes.end(), pair_first_down);
  bool is_permute = true;
  int64_t z = 1;
  for (int64_t d = ndim - 1; d >= 0; d--) {
    auto it = strides_sizes[d];
    if (it.second != 1) {
      if (it.first == z) {
        z *= it.second;
      } else {
        is_permute = false;
        break;
      }
    }
  }
  return is_permute;
}

std::vector<int64_t> get_permute_back_order(const at::Tensor& input) {
  auto input_sizes = input.sizes().vec();
  auto input_strides = input.strides().vec();
  auto ndim = input.dim();
  std::vector<std::pair<int64_t, int64_t>> strides_sizes(ndim, std::pair<int64_t, int64_t>(1, 1));
  for (int64_t i = 0; i < ndim; ++i) {
    strides_sizes[i] = std::pair<int64_t, int64_t>((static_cast<int64_t>(input_strides[i])),
            (static_cast<int64_t>(input_sizes[i])));
  }
  sort(strides_sizes.begin(), strides_sizes.end(), pair_first_down);
  std::vector<int64_t>permute_back_order(ndim);
  for (int64_t i = 0; i < ndim; ++i) {
    auto pair = strides_sizes[i];
    for (int64_t j = 0; j < ndim; ++j) {
      if ((pair.first == input_strides[j]) && (pair.second == input_sizes[j])) {
        permute_back_order[i] = j;
        input_strides[j] = -1;
        input_sizes[j] = -1;
        break;
      }
    }
  }
  return permute_back_order;
}

std::vector<int64_t> get_permute_order(std::vector<int64_t> permute_back_order,
                                       c10::MemoryFormat memory_format) {
  auto ndim = permute_back_order.size();
  std::vector<int64_t>permute_order(ndim);
  for (int64_t i = 0; i < ndim; ++i) {
    permute_order[permute_back_order[i]] = i;
  }
  if ((memory_format != c10::MemoryFormat::Contiguous)
      && ((ndim == 4) || (ndim == 5))) {
    int64_t temp = permute_order[1];
    for (int64_t i = 1; i < ndim - 1; ++i) {
      permute_order[i] = permute_order[i + 1];
    }
    permute_order[ndim - 1] = temp;
  }
  return permute_order;
}

int get_quantized_bitwidth_from_input(const at::Tensor& input) {
  auto type = input.scalar_type();
  if (type == at::kFloat) {
    return 31;
  } else if (type == at::kHalf) {
    return 16;
  }
  return 31;
}

// make dim which has 0 stride to 1 len and 1 stride.
at::Tensor get_tensor_without_zero_stride(const at::Tensor& input) {
  auto input_sizes = input.sizes().vec();
  auto input_strides = input.strides().vec();
  auto ndim = input.dim();
  std::vector<std::pair<int64_t, int64_t>> strides_sizes(ndim, std::pair<int64_t, int64_t>(1, 1));
  for (int64_t i = 0; i < ndim; ++i) {
    strides_sizes[i] = std::pair<int64_t, int64_t>((static_cast<int64_t>(input_strides[i])),
            (static_cast<int64_t>(input_sizes[i])));
  }
  for (auto it = strides_sizes.begin(); it != strides_sizes.end(); ) {
    if ((*it).first == 0) {
      (*it).first = 1;
      (*it).second = 1;
    }
    ++it;
  }
  for (int64_t i = 0; i < ndim; ++i) {
    input_strides[i] = strides_sizes[i].first;
    input_sizes[i] = strides_sizes[i].second;
  }
  auto input_without_zero_stride = input.as_strided(input_sizes, input_strides);
  return input_without_zero_stride;
}

// strides look like create by expand or not.
bool is_expand(const at::Tensor& input) {
  if (input.is_contiguous()) {
    return false;
  }
  // expand will modify stride value to zero,
  // so check stride value for skipping permute situation.
  auto stride = input.strides().vec();
  auto it = std::find(stride.begin(), stride.end(), 0);
  if (it == stride.end()) {
    return false;
  }
  auto input_without_zero_stride = get_tensor_without_zero_stride(input);
  return (input_without_zero_stride.is_contiguous() || is_permute(input_without_zero_stride));
}

// use cnnlTranspose_v2 instead of cnnlCopyWithStride
// when output is non_overlapping_and_dense in D2D copy.
at::Tensor non_overlapping_and_dense_out(at::Tensor& output, const at::Tensor& input) {
  TORCH_MLU_CHECK(output.is_non_overlapping_and_dense(),
    "output should be non_overlapping_and_dense in non_overlapping_and_dense_out.");
  TORCH_MLU_CHECK((output.sizes() == input.sizes()),
    "output sizes should be the same as input sizes in non_overlapping_and_dense_out.");
  TORCH_MLU_CHECK((output.dtype() == input.dtype()),
    "output dtype should be the same as input dtype in non_overlapping_and_dense_out.");
  auto ndim = output.dim();
  c10::MemoryFormat memory_format = c10::MemoryFormat::Contiguous;
  at::Tensor input_non_overlapping_and_dense = input;
  if (!input.is_non_overlapping_and_dense()) {
    input_non_overlapping_and_dense = cnnl_contiguous(input);
  }
  auto output_permute_back_order = get_permute_back_order(output);
  at::IntArrayRef output_back_array_order(output_permute_back_order);
  auto input_permute_back_order = get_permute_back_order(input_non_overlapping_and_dense);
  at::IntArrayRef input_back_array_order(input_permute_back_order);

  // get contiguous tensor which matched storage, output_contiguous and output shared storage.
  auto output_contiguous = cnnl_permute(output, output_back_array_order);
  auto input_contiguous = cnnl_permute(input_non_overlapping_and_dense, input_back_array_order);
  auto input_permute_order = get_permute_order(input_permute_back_order, memory_format);

  // output_contiguous equal to
  // input_contiguous.permute(input_permute_order).permute(output_permute_back_order)
  std::vector<int64_t>input_to_output_order(ndim);
  for (int64_t i = 0; i < ndim; ++i) {
    input_to_output_order[i] = input_permute_order[output_permute_back_order[i]];
  }
  at::IntArrayRef input_to_output_array(input_to_output_order);
  cnnl_permute_out_internal(output_contiguous, input_contiguous, input_to_output_array);
  return output;
}

// contiguous TensorList
std::vector<at::Tensor> unify_memory_format_of_multi_tensor(const at::TensorList& tensors) {
  TORCH_MLU_CHECK(tensors.size() > 0, "Input tensor num need be greater than 0.");
  c10::MemoryFormat memory_format = infer_tensor_list_contiguous(tensors);
  std::vector<at::Tensor> inputs;
  for (size_t i = 0; i < tensors.size(); ++i) {
    if ((!tensors[i].numel()) && (tensors[i].sizes()==at::IntArrayRef({0}))) {
      continue;
    }
    auto temp_memory_format = infer_memory_format(tensors[i].dim(), memory_format);
    inputs.emplace_back(cnnl_contiguous(tensors[i], temp_memory_format));
  }
  return inputs;
}

// Convert scale_data from cnml Qinference to position for cnnl Qinference
int get_pos_from_scale_data(int bitwidth, float scale_data) {
  int pos_ = 1;
  int qmin = -128, qmax = 127;
  if (bitwidth == 16) {
    qmin = -32768;
    qmax = 32767;
  }
  float absmax = qmax / scale_data;
  pos_ = std::floor(std::log(absmax) / std::log(2)) - (bitwidth -2);
  return pos_;
}

}  // namespace ops
}  // namespace cnnl
}  // namespace torch_mlu
