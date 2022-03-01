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

#include "aten/operators/cnnl/internal/internal_util.h"
#include "aten/operators/cnnl/cnnl_kernel.h"
#include "aten/operators/cnnl/internal/binaryops_util.h"

// TODO(sfengyang): the mapping tool support NCL cnnl layout in the future.
static std::map<cnnlTensorLayout_t, std::vector<int64_t>> layout2index = {
                                                       {CNNL_LAYOUT_NCHW, {0, 1, 2, 3}},
                                                       {CNNL_LAYOUT_NHWC, {0, 2, 3, 1}},
                                                       {CNNL_LAYOUT_HWCN, {2, 3, 1, 0}},
                                                       {CNNL_LAYOUT_NCDHW, {0, 1, 2, 3, 4}},
                                                       {CNNL_LAYOUT_NDHWC, {0, 2, 3, 4, 1}},
                                                       {CNNL_LAYOUT_NTC, {0, 1, 2}},
                                                       {CNNL_LAYOUT_TNC, {1, 0, 2}}};

namespace torch_mlu {
namespace cnnl {
namespace ops {

void transLayoutParameterDim(const cnnlTensorLayout_t& source_laytout,
                             const cnnlTensorLayout_t& target_layout,
                             const int64_t& in_dim,
                             int64_t* out_dim) {
  std::vector<int64_t> source_dims_index_vec_;
  std::vector<int64_t> target_dims_index_vec_;
  auto source_search = layout2index.find(source_laytout);
  auto target_search = layout2index.find(target_layout);
  int64_t non_negative_dim = in_dim;
  if (source_search != layout2index.end()) {
    source_dims_index_vec_ = source_search->second;
  } else {
    TORCH_MLU_CHECK(false, "source_layout is wrong, source_layout must come from "
                    "CNNL_LAYOUT_NCHW, CNNL_LAYOUT_NHWC, CNNL_LAYOUT_HWCN, "
                    "CNNL_LAYOUT_NDHWC, CNNL_LAYOUT_NCDHW, CNNL_LAYOUT_TNC, CNNL_LAYOUT_NTC");
  }
  if (target_search != layout2index.end()) {
    target_dims_index_vec_ = target_search->second;
  } else {
    TORCH_MLU_CHECK(false, "target_layout is wrong. target_layout must come from "
                    "CNNL_LAYOUT_NCHW, CNNL_LAYOUT_NHWC, CNNL_LAYOUT_HWCN, "
                    "CNNL_LAYOUT_NDHWC,CNNL_LAYOUT_NCDHW, CNNL_LAYOUT_TNC, CNNL_LAYOUT_NTC");
  }
  TORCH_MLU_CHECK(source_dims_index_vec_.size() == target_dims_index_vec_.size(),
                  "source_layout doesn't match target_layout.");
  int64_t max_dim = source_dims_index_vec_.size();
  // remove negative dim
  if (in_dim < 0) non_negative_dim = in_dim + max_dim;
  TORCH_MLU_CHECK(non_negative_dim < max_dim, "max_dim need to larger than non_negative_dim.");
  int64_t source_index = source_dims_index_vec_[non_negative_dim];
  auto target_iter = std::find(target_dims_index_vec_.begin(),
                               target_dims_index_vec_.end(),
                               source_index);
  if (target_iter != target_dims_index_vec_.end()) {
    *out_dim = std::distance(target_dims_index_vec_.begin(), target_iter);
  } else {
    TORCH_MLU_CHECK(false, "dims trans error!");
  }
}

void transLayoutParameterDims(const cnnlTensorLayout_t& source_layout,
                              const cnnlTensorLayout_t& target_layout,
                              const std::vector<int64_t>& in_dims,
                              std::vector<int64_t>* out_dims) {
  /* for example: source_layout(nchw) index:(0,1,2,3),
                 dim:(dim0,dim1,dim2,dim3)
                                     0     1     2     3
                                     |     |     |     |
                                    dim0  dim1  dim2  dim3
                 target_layout(nhwc) index:(0,2,3,1)
                                     0     2     3     1
                                     |     |     |     |
                                    dim0  dim2  dim3  dim4
                 out_dim:(dim0,dim1,dim2,dim3)
  */
   int64_t max_dim = layout2index[source_layout].size();
   int64_t min_dim = layout2index[source_layout].size() * -1;
  // check in_dims
  if (max_dim <= *std::max_element(in_dims.begin(), in_dims.end()) ||
      *std::min_element(in_dims.begin(), in_dims.end()) <= min_dim)
    TORCH_MLU_CHECK(false, "in_dims don't match source_layout layout");
  std::vector<int64_t> tmp_in_dims(in_dims);
  // modify negative value of dims
  for (int i = 0; i < tmp_in_dims.size(); i++)
    tmp_in_dims[i] = tmp_in_dims[i] < 0 ? tmp_in_dims[i] + max_dim: tmp_in_dims[i];
  std::set<int64_t> s(tmp_in_dims.begin(), tmp_in_dims.end());
  // remove duplicated dims and sort them
  tmp_in_dims.assign(s.begin(), s.end());
  int64_t tmp_dim = 0;
  // init out_dims
  out_dims->clear();
  out_dims->resize(tmp_in_dims.size());
  for (int64_t i = 0; i < tmp_in_dims.size(); i++) {
    tmp_dim = tmp_in_dims[i];
    transLayoutParameterDim(source_layout, target_layout, tmp_dim, out_dims->data()+i);
  }
}

// modify dim from nchw to nhwc.
// all dim transpose based on channels_first to channels_last.
int64_t modify_dim_based_on_layout(const int64_t dim,
            const c10::MemoryFormat memory_format) {
  int64_t target_dim;
  switch (memory_format) {
    case c10::MemoryFormat::ChannelsLast:
      transLayoutParameterDim(CNNL_LAYOUT_NCHW, CNNL_LAYOUT_NHWC, dim, &target_dim);
      break;
    case c10::MemoryFormat::ChannelsLast3d:
      transLayoutParameterDim(CNNL_LAYOUT_NCDHW, CNNL_LAYOUT_NDHWC, dim, &target_dim);
      break;
    case c10::MemoryFormat::Contiguous:
      target_dim = dim;
      break;
    default:
      TORCH_MLU_CHECK(false, "memory format not support.");
      break;
  }
  return target_dim;
}

std::vector<int64_t> modify_dims_based_on_layout(const std::vector<int64_t>& dim,
            const c10::MemoryFormat memory_format) {
  if (!dim.size()) {
      return dim;
  }
  std::vector<int64_t> target_dim;
  std::vector<int> dim_order;
  // trans tensor/stride size to cnnl desc size/stride.
  auto modify_dims_pos = [](const std::vector<int>& dim_order,
                            const std::vector<int64_t>& input,
                            std::vector<int64_t>& out) {
    out.clear();
    for (const auto& item : dim_order) {
      out.push_back(input[item]);
    }
  };
  switch (memory_format) {
    case c10::MemoryFormat::ChannelsLast:
      dim_order = {0, 2, 3, 1};
      modify_dims_pos(dim_order, dim, target_dim);
      break;
    case c10::MemoryFormat::ChannelsLast3d:
      dim_order = {0, 2, 3, 4, 1};
      modify_dims_pos(dim_order, dim, target_dim);
      break;
    case c10::MemoryFormat::Contiguous:
      target_dim = dim;
      break;
    default:
      TORCH_MLU_CHECK(false, "memory format not support.");
      break;
  }
  return target_dim;
}

at::Tensor getMatmulOut(const at::Tensor &self,
                        const at::Tensor &other,
                        bool is_trans_self,
                        bool is_trans_other,
                        at::TensorOptions output_options) {
    auto self_shape = self.sizes();
    auto other_shape = other.sizes();
    std::vector<int64_t> output_shape(2, 1);
    if (is_trans_self) {
        output_shape[0] = self_shape[1];
    } else {
        output_shape[0] = self_shape[0];
    }
    if (is_trans_other) {
        output_shape[1] = other_shape[0];
    } else {
        output_shape[1] = other_shape[1];
    }
    return at::empty(output_shape, output_options);
}

at::Tensor getBatchmatmulOut(const at::Tensor &self,
                             const at::Tensor &other,
                             bool is_trans_self,
                             bool is_trans_other,
                             at::TensorOptions output_options) {
  auto self_shape = self.sizes();
  auto other_shape = other.sizes();
  TORCH_MLU_CHECK(self_shape[0] == other_shape[0], "two tensors' batch_size is not equal.");
  std::vector<int64_t> output_shape(3, 1);
  output_shape[0] = self_shape[0];
  if (is_trans_self)
    output_shape[1] = self_shape[2];
  else
    output_shape[1] = self_shape[1];
  if (is_trans_other)
    output_shape[2] = other_shape[1];
  else
    output_shape[2] = other_shape[2];

  return at::empty(output_shape, output_options);
}

std::tuple<at::Tensor, bool> getMMInput(const at::Tensor &self) {
    TORCH_MLU_CHECK(self.dim() == 2, "dimension must be 2 in mm.");
    bool is_trans_self;
    if ((!self.is_contiguous())
        && (self.is_non_overlapping_and_dense())
        && (self.t().is_contiguous())) {
      is_trans_self = true;
      return std::make_tuple(self.t(), is_trans_self);
    } else {
      is_trans_self = false;
      return std::make_tuple(cnnl_contiguous(self, c10::MemoryFormat::Contiguous), is_trans_self);
    }
}

std::tuple<at::Tensor, bool> getBMMInput(const at::Tensor &self) {
  TORCH_MLU_CHECK(self.dim() == 3, "dimension must be 3 in bmm.");
  bool is_trans_self;
  if ((!self.is_contiguous())
      && (is_permute(self))) {
    auto permute_back_order = get_permute_back_order(self);
    at::IntArrayRef back_array_order(permute_back_order);
    auto self_before_permute = cnnl_permute(self, back_array_order);
    TORCH_MLU_CHECK(self_before_permute.is_contiguous(), "error order in permute_back_order.");
    int64_t batch_order = 0;
    for (int64_t i = 0; i < self.dim(); ++i) {
      if (permute_back_order[i] == 0) {
        batch_order = i;
        permute_back_order[i] = permute_back_order[0];
        permute_back_order[0] = 0;
      }
    }
    auto self_contiguous = cnnl_transpose_internal(self_before_permute, batch_order, 0);
    TORCH_MLU_CHECK(self_contiguous.is_contiguous(), "output must be contiguous in getBMMInput.");
    if (permute_back_order[1] == 1) {
      is_trans_self = false;
    } else {
      is_trans_self = true;
    }
    return std::make_tuple(self_contiguous, is_trans_self);

  } else {
    is_trans_self = false;
    return std::make_tuple(cnnl_contiguous(self, c10::MemoryFormat::Contiguous), is_trans_self);
  }
}

// tmp_input tensor stride is same with the first input tensor, and dim is same with output tensor.
void get_contiguous(const at::TensorIterator& iter, std::vector<at::Tensor> &l) {
    const int ninput = iter.ninputs();
    auto memory_format = iter.output(0).suggest_memory_format();
    int dim = iter.output(0).dim();
    for (int i = 0; i < ninput; i++) {
        auto tmp_tensor = iter.input(i);
        if (iter.input(i).dim() != dim) {
            auto shape = std::get<0>(broadcast_shape(iter.input(i), iter.output(0)));
            tmp_tensor = iter.input(i).expand(shape);
        }
        l.push_back(cnnl_contiguous(tmp_tensor, memory_format));
    }
}

std::tuple<std::vector<int64_t>, std::vector<int64_t>> get_tensor_size_stride(
        const at::Tensor& self, at::MemoryFormat memory_format) {
    auto self_size = modify_dims_based_on_layout(self.sizes().vec(), memory_format);
    auto self_stride = get_contiguous_strides(self_size);
    return std::make_tuple(self_size, self_stride);
}

at::ScalarType get_compute_type(const at::Tensor& self, const at::Tensor& other,
        const at::Tensor& output) {
    auto common_type = output.scalar_type();
    auto compute_type = at::kFloat;
    auto type_iter = find(type_vec.begin(), type_vec.end(), common_type);
    if (type_iter != type_vec.end()) {
        compute_type = common_type;
    }
    return compute_type;
}

}  // namespace ops
}  // namespace cnnl
}  // namespace torch_mlu
