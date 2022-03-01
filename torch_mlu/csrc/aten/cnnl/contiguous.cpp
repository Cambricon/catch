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

#include <stdint.h>  // NOLINT
#include "ATen/native/Resize.h"  // NOLINT
#include "ATen/NativeFunctions.h"  // NOLINT
#include "aten/operators/cnnl/cnnl_kernel.h"  // NOLINT
#include "aten/operators/cnnl/internal/cnnl_internal.h"  // NOLINT
#include "aten/cnnl/cnnl_util.h" // NOLINT
#include "aten/operators/cnnl/internal/internal_util.h"

namespace torch_mlu {
namespace cnnl {
namespace ops {

using TupleRecordOpInfo = typename torch_mlu::MLUTensorImpl::TupleRecordOpInfo;
using SharedOpInfo = typename torch_mlu::MLUTensorImpl::SharedOpInfo;
// expand op in the graph will change stride value to 0,
// so can't backward derivation the parameters of after op.
// like expand -> slice; or expand -> permute.
bool check_expand_with_other_op_in_graph(const std::vector<SharedOpInfo>& subgraph) {
  std::vector<VIEWOPNAME> name;
  for (const auto& item : subgraph) {
    name.push_back(std::get<0>(item));
  }
  auto expand_index = std::find(name.begin(), name.end(), VIEWOPNAME::expand);
  while (expand_index < name.end()) {
    if (*expand_index == VIEWOPNAME::slice || *expand_index == VIEWOPNAME::permute) {
      return true;
    }
    expand_index++;
  }
  return false;
}

at::Tensor permute_tensor_cf_to_cl(const at::Tensor& input) {
  TORCH_MLU_CHECK(input.is_contiguous(c10::MemoryFormat::Contiguous),
    "Input tensor need be channels first contiguous.");
  TORCH_MLU_CHECK(input.dim() == 4 || input.dim() == 5,
    "Channels_last or channels_last3d tensor dimension only support 4 or 5.");
  auto input_size = input.sizes().vec();
  std::vector<int64_t> permute = get_trans_order(input.dim(), true);
  at::Tensor output = cnnl_permute_internal(input, permute);
  auto* output_impl = getMluTensorImpl(output);
  auto new_strides = get_contiguous_strides(input_size,
                                        get_channels_last_memory_format(input.dim()));
  output_impl->set_sizes_and_strides(input_size, new_strides);
  return output;
}

at::Tensor permute_tensor_cl_to_cf(const at::Tensor& input) {
  TORCH_MLU_CHECK(input.is_contiguous(c10::MemoryFormat::ChannelsLast) ||
    input.is_contiguous(c10::MemoryFormat::ChannelsLast3d),
    "Input tensor need to be channels last contiguous.");
    at::Tensor temp_tensor = input;
    auto input_size = input.sizes().vec();
    // modify nchw to real storage layout nhwc.
    std::vector<int64_t> permute_dims = get_trans_order(input.dim(), true);
    temp_tensor = at::native::permute(temp_tensor, permute_dims);
    auto nhwc_input_size = temp_tensor.sizes().vec();
    auto nhwc_input_stride = get_contiguous_strides(nhwc_input_size);
    auto* temp_tensor_impl = getMluTensorImpl(temp_tensor);
    temp_tensor_impl->set_sizes_and_strides(nhwc_input_size, nhwc_input_stride);
    // permute real storage nhwc to nchw, here nhwc is channels first layout, not
    // same with pytorch tensor layout.
    permute_dims = get_trans_order(input.dim(), false);
    at::Tensor output = cnnl_permute_internal(temp_tensor, permute_dims);
    auto* output_impl = getMluTensorImpl(output);
    auto new_strides = get_contiguous_strides(input_size);
    output_impl->set_sizes_and_strides(input_size, new_strides);
    return output;
}

at::Tensor get_channel_first_contiguous_tensor(const at::Tensor& self) {
  if (self.is_contiguous()) {
    return self;
  } else if (self.is_contiguous(c10::MemoryFormat::ChannelsLast) ||
             self.is_contiguous(c10::MemoryFormat::ChannelsLast3d)) {
    // modify nchw to real storage layout nhwc.
    return permute_tensor_cl_to_cf(self);
  } else {
    TORCH_MLU_CHECK(false, "Not support non-contiguous tensor.");
  }
}

std::vector<int64_t> get_permute_parameters(const TupleRecordOpInfo& input,
                                            const TupleRecordOpInfo& output) {
  auto& input_size = std::get<0>(input);
  auto& output_size = std::get<0>(output);
  auto& input_stride = std::get<1>(input);
  auto& output_stride = std::get<1>(output);
  const int ndim = input_size.size();
  TORCH_MLU_CHECK(ndim == output_size.size(),
              "input dim need be equal to output dim.");
  using couple_int = std::vector<int64_t>;
  std::vector<couple_int> input_size_stride;
  std::vector<couple_int> output_size_stride;
  for (int i = 0; i < ndim; ++i) {
    input_size_stride.push_back(couple_int({input_size[i], input_stride[i]}));
    output_size_stride.push_back(couple_int({output_size[i], output_stride[i]}));
  }
  std::vector<int64_t> permute_dim;
  for (const auto& item : output_size_stride) {
    auto it = std::find(input_size_stride.begin(), input_size_stride.end(), item);
    TORCH_MLU_CHECK(it != input_size_stride.end(),
                "Output dim value need be find in input dim size.");
    int64_t index = std::distance(input_size_stride.begin(), it);
    *it = couple_int({-1, -1});
    permute_dim.push_back(index);
  }
  return permute_dim;
}

std::vector<int64_t> get_expand_parameters(const TupleRecordOpInfo& input,
                                           const TupleRecordOpInfo& output) {
  auto& input_size = std::get<0>(input);
  auto& output_size = std::get<0>(output);
  TORCH_MLU_CHECK(output_size.size() >= output_size.size(),
            "output dim need be greater to input dim.");
  std::vector<int64_t> size(output_size);
  return size;
}

std::vector<int64_t> get_slice_parameters(const TupleRecordOpInfo& input,
                                          const TupleRecordOpInfo& output) {
  std::vector<int64_t> in_sizes = std::get<0>(input);
  std::vector<int64_t> in_strides = std::get<1>(input);
  int64_t in_storage_offset = std::get<2>(input);
  std::vector<int64_t> out_sizes = std::get<0>(output);
  std::vector<int64_t> out_strides = std::get<1>(output);
  int64_t out_storage_offset = std::get<2>(output);
  int64_t dim = 0, start = 0, end = 0, step = 0;
  for (int i = 0; i < in_sizes.size(); i++) {
    if (in_sizes[i] != out_sizes[i]) {
      dim = i;
      start = (out_storage_offset - in_storage_offset) / in_strides[dim];
    }
  }
  step = out_strides[dim] / in_strides[dim];
  // even backward derivation don't get the real end parameter,
  // but not affect slice result.
  end = start + out_sizes[dim] * step - step + 1;
  std::vector<int64_t> param_vec = {dim, start, end, step};
  return param_vec;
}

std::vector<int64_t> get_select_parameters(const TupleRecordOpInfo& input,
                                           const TupleRecordOpInfo& output) {
  auto& input_size = std::get<0>(input);
  auto& output_size = std::get<0>(output);
  auto& input_stride = std::get<1>(input);
  auto& output_stride = std::get<1>(output);
  const int i_ndim = input_size.size();
  const int o_ndim = output_size.size();
  TORCH_MLU_CHECK(i_ndim == o_ndim + 1,
              "input dim need be equal to output dim + 1.");
  TORCH_MLU_CHECK(i_ndim == input_stride.size(),
              "input dim need be equal to input stride size.");
  TORCH_MLU_CHECK(o_ndim == output_stride.size(),
              "output dim need be equal to output stride size.");
  using couple_int = std::vector<int64_t>;
  std::vector<couple_int> input_size_stride;
  std::vector<couple_int> output_size_stride;
  for (int i = 0; i < i_ndim; ++i) {
    input_size_stride.push_back(couple_int({input_size[i], input_stride[i]}));
  }
  for (int i = 0; i < o_ndim; ++i) {
    output_size_stride.push_back(couple_int({output_size[i], output_stride[i]}));
  }
  bool check_output_in_input = true;
  for (int i = 0; i < o_ndim; ++i) {
    auto it = std::find(input_size_stride.begin(),
                        input_size_stride.end(),
                        output_size_stride[i]);
    if (it == input_size_stride.end()) {
      check_output_in_input = false;
    }
  }
  TORCH_MLU_CHECK(check_output_in_input,
    "select output size not in select input size.");

  int count = 0;
  int64_t dim = 0;
  int j = i_ndim - 1;
  for (int i = o_ndim - 1; i >= 0; --i) {
    if (input_size_stride[j] != output_size_stride[i]) {
      count++;
      dim = j;
      break;
    }
    --j;
  }
  if (j == 0) {
    dim = 0;
    count = 1;
  }

  TORCH_MLU_CHECK(count == 1, "Select op just can erease one dim.");
  TORCH_MLU_CHECK(input_stride[dim] != 0, "input stride not support 0.");
  int64_t index = (std::get<2>(output) - std::get<2>(input)) / input_stride[dim];
  TORCH_MLU_CHECK(index < input_size[dim], "Select index less than input size[dim].");
  std::vector<int64_t> param_vec = {dim, index};
  return param_vec;
}

at::Tensor wrap_cnnl_permute_internal(const at::Tensor& self,
                                      const TupleRecordOpInfo& input_info,
                                      const TupleRecordOpInfo& output_info) {
  auto permute = get_permute_parameters(input_info, output_info);
  auto memory_format = self.suggest_memory_format();
  TORCH_MLU_CHECK(memory_format != c10::MemoryFormat::Preserve,
    "Preserve memory format is unsupported by the contiguous operator.");

  at::Tensor cf_tensor = get_channel_first_contiguous_tensor(self);

  auto output = cnnl_permute_internal(cf_tensor, permute);

  return output;
}

at::Tensor wrap_cnnl_slice_internal(const at::Tensor& self,
                                    const std::vector<int64_t>& param) {
  TORCH_MLU_CHECK(param.size() == 4, "slice op parameter need be 4.");
  return cnnl_slice_internal(self, param[0], param[1], param[2], param[3]);
}

at::Tensor wrap_squeeze_internal(const at::Tensor& self,
                                 const TupleRecordOpInfo& output) {
  auto self_impl = getMluTensorImpl(self);
  self_impl->set_sizes_contiguous(std::get<0>(output));
  return self;
}

// using slice + squeeze.
at::Tensor wrap_select_internal(const at::Tensor& self,
                                const TupleRecordOpInfo& input,
                                const TupleRecordOpInfo& output) {
  std::vector<int64_t> param_vec = get_select_parameters(input, output);
  TORCH_MLU_CHECK(param_vec.size() == 2, "select only have two parameters.");
  const int dim = param_vec[0];
  const int index = param_vec[1];
  at::Tensor output_tensor = cnnl_slice_internal(self, dim, index, index + 1, 1);
  std::vector<int64_t> output_size = output_tensor.sizes().vec();
  TORCH_MLU_CHECK(output_size[dim] == 1, "Select dim num is equal to 1");
  auto it = output_size.begin() + dim;
  output_size.erase(it);
  auto output_impl = getMluTensorImpl(output_tensor);
  output_impl->set_sizes_contiguous(output_size);
  return output_tensor;
}

at::Tensor cnnl_views(const at::Tensor& self,
                      const SharedOpInfo& op_node) {
  TORCH_MLU_CHECK(self.is_contiguous(self.suggest_memory_format()),
    "Self tensor not contiguous in cnnl_views.");
  at::Tensor output;
  auto subgraph_node_input_info = std::get<1>(op_node);
  auto subgraph_node_output_info = std::get<2>(op_node);
  TORCH_MLU_CHECK(self.sizes().vec() == std::get<0>(subgraph_node_input_info),
    "input tensor size need be equal to recode node input size.");
  switch (std::get<0>(op_node)) {
    case VIEWOPNAME::permute:
      output = wrap_cnnl_permute_internal(self,
                                          subgraph_node_input_info,
                                          subgraph_node_output_info);
      break;
    case VIEWOPNAME::expand:
      output = cnnl_expand_internal(self,
                                    get_expand_parameters(subgraph_node_input_info,
                                                          subgraph_node_output_info),
                                    false);
      break;
    case VIEWOPNAME::slice:
        output = wrap_cnnl_slice_internal(self,
                                          get_slice_parameters(subgraph_node_input_info,
                                                               subgraph_node_output_info));
        break;
    case VIEWOPNAME::squeeze:
    case VIEWOPNAME::unsqueeze:
    case VIEWOPNAME::reshape:
    case VIEWOPNAME::view:
        output = wrap_squeeze_internal(get_channel_first_contiguous_tensor(self),
                                       subgraph_node_output_info);
        break;
    case VIEWOPNAME::select:
        output = wrap_select_internal(get_channel_first_contiguous_tensor(self),
                                      subgraph_node_input_info,
                                      subgraph_node_output_info);
        break;
    default:
      TORCH_MLU_CHECK(false, "Failed find view op in support list.");
      break;
  }
  return output;
}

at::Tensor permute_to_contiguous(const at::Tensor& input,
                                 c10::MemoryFormat memory_format) {
  auto permute_back_order = get_permute_back_order(input);
  at::IntArrayRef back_array_order(permute_back_order);
  auto input_before_permute = cnnl_permute(input, back_array_order);
  auto permute_order = get_permute_order(permute_back_order, memory_format);
  at::IntArrayRef array_order(permute_order);
  auto input_contiguous = cnnl_permute_internal(input_before_permute, array_order);
  if (memory_format != c10::MemoryFormat::Contiguous) {
    auto strides = get_contiguous_strides(input.sizes(), memory_format);
    getMluTensorImpl(input_contiguous)->set_sizes_and_strides(input.sizes(), strides);
  }
  TORCH_MLU_CHECK(input.sizes() == input_contiguous.sizes(),
    "input sizes must equal to output sizes.");
  return input_contiguous;
}

// only support original tensor memory format, memory_format just affect output.
// size in shared_storage tensor is always relayable.
at::Tensor cnnl_contiguous(const at::Tensor& input,
            c10::MemoryFormat memory_format) {
  TORCH_MLU_CHECK(memory_format != c10::MemoryFormat::Preserve,
    "Preserve memory format is unsupported by the contiguous operator.");
  // For Tensor just contain one scalar.
  if (input.is_contiguous(memory_format)) {
    return input;
  }
  if (is_mlu(input) == false) {
    return input.contiguous(memory_format);
  }
  const auto input_size = input.sizes();
  const auto input_stride = input.strides();
  const auto input_offset = input.storage_offset();
  auto input_info = std::make_tuple(input_size.vec(),
                                    input_stride.vec(),
                                    input_offset);

  auto* input_impl = getMluTensorImpl(input);
  // input is channels first contiguous,
  // but output need channels last or channels last3d contiguous.
  if (input.is_contiguous(c10::MemoryFormat::Contiguous)) {
    return permute_tensor_cf_to_cl(input);
  }
  // input is channels last or channels last3d contiguous,
  // but output need channels first contiguous.
  if (input.is_contiguous(c10::MemoryFormat::ChannelsLast) ||
      input.is_contiguous(c10::MemoryFormat::ChannelsLast3d)) {
    /* at::Tensor temp_tensor = input; */
    // modify nchw to real storage layout nhwc.
    return permute_tensor_cl_to_cf(input);
  }
  if (input.dim() < 4 || input.dim() > 5) {
    memory_format = c10::MemoryFormat::Contiguous;
  }
  if (is_expand(input)) {
    auto input_without_zero_stride = get_tensor_without_zero_stride(input);
    if (is_permute(input_without_zero_stride)) {
      input_without_zero_stride = permute_to_contiguous(input_without_zero_stride, memory_format);
    }
    TORCH_MLU_CHECK(input_without_zero_stride.is_contiguous(memory_format),
                    "input_without_zero_stride should be contiguous with ", memory_format);
    auto contiguous_strides = get_contiguous_strides(input_without_zero_stride.sizes(),
                                                     memory_format);
    auto input_without_zero_stride_impl = getMluTensorImpl(input_without_zero_stride);
    input_without_zero_stride_impl->set_sizes_and_strides(input_without_zero_stride.sizes(),
                                                          contiguous_strides);
    return cnnl_expand_internal(input_without_zero_stride, input.sizes(), true);
  }
  if (is_permute(input)) {
    return permute_to_contiguous(input, memory_format);
  }

  // add function for permute output tensor MemoryFormat.
  auto handle_output = [](const at::Tensor& self, c10::MemoryFormat memory_format)->at::Tensor {
    auto temp_memory_format = self.suggest_memory_format();
    TORCH_MLU_CHECK(self.is_contiguous(temp_memory_format),
      "Self tensor need be contiguous.");
    if (self.is_contiguous(memory_format) == true) {
      return self;
    }
    // channel_last or channel_last3d to channel_frist.
    if (memory_format == c10::MemoryFormat::Contiguous) {
      return permute_tensor_cl_to_cf(self);
    } else {
      // channel_frist to channel_last or channel_last3d.
      return permute_tensor_cf_to_cl(self);
    }
  };

  auto subgraph = input_impl->get_views_op_info(input_info);
  if (subgraph.size() == 0) {
    // no views op in the graph, size and strid is compare to the original
    // storage. so input memory format is reliable.
    // input is not contiguous
    auto output = at::empty(input.sizes(), input.options(), memory_format);
    cnnl_copy_without_contiguous_internal(output, input);
    return output;
  }

  // Subgraph recode view op info, last one need be equal to input.
  const auto& last_node_output = std::get<2>(subgraph.back());
  TORCH_MLU_CHECK(std::get<0>(last_node_output) == input_size,
    "output size at the last node of views graph equal to input size.");
  TORCH_MLU_CHECK(std::get<1>(last_node_output) == input_stride,
    "output stride at the last node of views graph equal to input stride.");
  TORCH_MLU_CHECK(std::get<2>(last_node_output) == input_offset,
    "output offset at the last node of views graph equal to input offset.");

  // Can't change input tensor, python side maybe use this tensor more
  // than one times, so use generate original tensor by input storage
  // with size / stride/ offset info in first graph node.
  const auto& original_size = std::get<0>(std::get<1>(subgraph[0]));
  const auto& original_stride = std::get<1>(std::get<1>(subgraph[0]));
  const auto& original_offset = std::get<2>(std::get<1>(subgraph[0]));
  at::Tensor temp_tensor = torch_mlu::cnnl::ops::cnnl_as_strided(input,
                                                                 original_size,
                                                                 original_stride,
                                                                 original_offset);
  // check subgraph first node for contiguous.
  // check unfold node is not in subgraph.
  bool unfold_flag = false;
  for (const auto& opnode : subgraph) {
    if (std::get<0>(opnode) == VIEWOPNAME::unfold) {
      unfold_flag = true;
    }
  }

  // expand op in the graph will change stride value to 0,
  // so can't backward derivation the parameters of after op.
  // like expand -> slice; or expand -> permute.
  // （TODO）shangang: using view ops class will fix this.
  if (unfold_flag == false && check_expand_with_other_op_in_graph(subgraph) == true) {
    unfold_flag = true;
  }

  // original tensor is channel_last or ChannelsLast3d, then permute to channel first.
  // original tensor is channel_first, pass,
  // else set unfold_flag = true.
  c10::MemoryFormat temp_memory_format = temp_tensor.suggest_memory_format();
  if (unfold_flag == false && temp_tensor.is_contiguous(temp_memory_format) == false) {
    unfold_flag = true;
  }

  // unfold_flag == true, contiguous can't optim.
  if (unfold_flag == true) {
    auto output = at::empty(input.sizes(), input.options(), memory_format);
    cnnl_copy_without_contiguous_internal(output, input);
    return output;
  }

  int node = 0;
  for (const auto& opnode : subgraph) {
    // std::tuple<std::string, TupleRecordOpInfo, TupleRecordOpInfo>;
    // std::tuple<std::vector<int64_t>, std::vector<int64_t>, int64_t>;
    // do cnnl kernel launch.
    temp_tensor = cnnl_views(temp_tensor, opnode);
    node++;
    if (node == subgraph.size()) {
      // views op only support channels_first input and output tensor.
      // so if graph first input tensor is channels_last or channels_last_3d,
      // need to change tensor layout to channels_first.
      return handle_output(temp_tensor, memory_format);
    }
  }
}

}  // namespace ops
}  // namespace cnnl
}  // namespace torch_mlu
