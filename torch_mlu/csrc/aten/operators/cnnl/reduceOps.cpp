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

#include "aten/operators/cnnl/cnnl_kernel.h"
#include "aten/operators/cnnl/internal/cnnl_internal.h"

namespace torch_mlu {
namespace cnnl {
namespace ops {

static inline void check_type(const at::Tensor& self, c10::optional<at::ScalarType> opt_dtype) {
  at::ScalarType scalarType = opt_dtype.has_value() ? opt_dtype.value() : self.scalar_type();
  TORCH_MLU_CHECK(
      at::isFloatingType(scalarType) || at::isComplexType(scalarType),
      "Can only calculate the mean of floating types. Got ",
      toString(scalarType),
      " instead.");
}
std::vector<int64_t> getRealDim(std::vector<int64_t> input_dim, int64_t t_dim) {
  // handle negative dims
  for (int64_t i = 0; i < input_dim.size(); ++i) {
    if (input_dim[i] < 0) {
      input_dim[i] = input_dim[i] + t_dim;
    }
  }
  // remove duplicate dims and sort them
  // e.g. (3,1,1) -> (1,3)
  std::vector<int64_t> dim_vec(input_dim);
  std::set<int64_t> s(dim_vec.begin(), dim_vec.end());
  dim_vec.assign(s.begin(), s.end());
  return dim_vec;
}

std::vector<int64_t> getOutputShape(std::vector<int64_t> input_dim,
                                    std::vector<int64_t> reduce_dim, bool keepdim) {
  std::vector<int64_t> output_dim = input_dim;
  if (input_dim.size() == 0) {
    output_dim.push_back(1);
    return output_dim;
  }
  int num = 0;
  for (auto i : reduce_dim) {
    if (keepdim) {
      output_dim[i] = 1;
    } else {
      auto it = output_dim.begin() + i - num;
      output_dim.erase(it);
      num++;
    }
  }
  return output_dim;
}

// Return 0,1,2,...,N-1 for all dims.
std::vector<int64_t> getAllDim(int64_t dim) {
  std::vector<int64_t> output;
  if (dim == 0) {
    output.push_back(0);
  }
  for (int i = 0; i < dim ; i++) {
    output.push_back(i);
  }
  return output;
}

void getOutputSize(std::vector<int64_t> input_size, int64_t dim,
                   std::vector<int64_t>& keepdim_size,
                   std::vector<int64_t>& reduce_size) {
  if (dim < 0) {
    dim = input_size.size() + dim;
  }
  keepdim_size[dim] = 1;
  int real_pos = dim - (keepdim_size.size() - reduce_size.size());
  reduce_size.erase(reduce_size.begin() + real_pos);
}

void getMultiAxisOutputSize(std::vector<int64_t>& output_size,
                            std::vector<int64_t>& fake_output_size,
                            const std::vector<int64_t>& reduce_axis,
                            bool keep_dim) {
  int num = 0;
  for (auto i : reduce_axis) {
    fake_output_size[i] = 1;
    if (keep_dim == true) {
      output_size[i] = 1;
    } else {
      auto it = output_size.begin() + i - num;
      output_size.erase(it);
      num++;
    }
  }
}

constexpr auto reduce_memory_format = c10::MemoryFormat::Contiguous;

at::Tensor cnnl_sum(const at::Tensor& input, at::IntArrayRef dim,
                    bool keepdim, c10::optional<c10::ScalarType> dtype) {
  auto input_contiguous = cnnl_contiguous(input, reduce_memory_format);
  at::Tensor output;
  at::Tensor index;
  std::vector<int64_t> reduce_axis = getRealDim(dim.vec(), input_contiguous.dim());
  auto output_shape = getOutputShape(input.sizes().vec(), reduce_axis, keepdim);
  auto full_output_shape = getOutputShape(input_contiguous.sizes().vec(), reduce_axis, true);
  cnnl_reduce_internal(input_contiguous, output, index, reduce_axis, full_output_shape,
                       ReduceType::Reduce_Sum, output_shape);
  return output;
}

at::Tensor cnnl_sum(const at::Tensor& self,
                    c10::optional<c10::ScalarType> dtype) {
  auto self_contiguous = cnnl_contiguous(self, reduce_memory_format);
  at::Tensor output;
  at::Tensor index;
  std::vector<int64_t> reduce_axis = getAllDim(self_contiguous.dim());
  std::vector<int64_t> output_shape = {};
  auto full_output_shape = getOutputShape(self_contiguous.sizes().vec(), reduce_axis, true);
  cnnl_reduce_internal(self_contiguous, output, index, reduce_axis, full_output_shape,
                       ReduceType::Reduce_Sum, output_shape);
  return output;
}

at::Tensor& cnnl_sum_out(at::Tensor& result, const at::Tensor& input,
                         at::IntArrayRef dim, bool keepdim,
                         c10::optional<c10::ScalarType> dtype) {
  auto output = cnnl_sum(input, dim, keepdim);
  result.resize_(output.sizes());
  result.copy_(output);
  return result;
}

at::Tensor cnnl_mean(const at::Tensor& input, at::IntArrayRef dim,
                     bool keepdim, c10::optional<c10::ScalarType> dtype) {
  check_type(input, dtype);
  auto input_contiguous = cnnl_contiguous(input, reduce_memory_format);
  at::Tensor output;
  at::Tensor index;
  std::vector<int64_t> reduce_axis = getRealDim(dim.vec(), input_contiguous.dim());
  auto output_shape = getOutputShape(input.sizes().vec(), reduce_axis, keepdim);
  auto full_output_shape = getOutputShape(input_contiguous.sizes().vec(), reduce_axis, true);
  cnnl_reduce_internal(input_contiguous, output, index, reduce_axis, full_output_shape,
                       ReduceType::Reduce_Mean, output_shape);
  return output;
}

at::Tensor cnnl_mean(const at::Tensor& self, c10::optional<c10::ScalarType> dtype) {
  check_type(self, dtype);
  auto self_contiguous = cnnl_contiguous(self, reduce_memory_format);
  at::Tensor output;
  at::Tensor index;
  std::vector<int64_t> reduce_axis = getAllDim(self_contiguous.dim());
  std::vector<int64_t> output_shape = {};
  auto full_output_shape = getOutputShape(self_contiguous.sizes().vec(), reduce_axis, true);
  cnnl_reduce_internal(self_contiguous, output, index, reduce_axis, full_output_shape,
                       ReduceType::Reduce_Mean, output_shape);
  return output;
}

at::Tensor& cnnl_mean_out(at::Tensor& out, const at::Tensor& self,
                          at::IntArrayRef dim, bool keepdim,
                          c10::optional<c10::ScalarType> dtype) {
  auto output = cnnl_mean(self, dim, keepdim);
  out.resize_(output.sizes());
  out.copy_(output);
  return out;
}

at::Tensor cnnl_norm(const at::Tensor& input, at::optional<at::Scalar> p,
                     at::IntArrayRef dim, bool keepdim) {
  auto input_contiguous = cnnl_contiguous(input, reduce_memory_format);
  at::Tensor output;
  at::Tensor index;
  TORCH_MLU_CHECK(p->to<float>() == 1.0 || p->to<float>() == 2.0, "torch_mlu only support ",
                  "L1-Norm as p=1.0 and L2-Norm as p=2.0, but got p=", p->to<float>(), ".");
  auto reduce_type = p->to<float>() == 1.0 ? ReduceType::Reduce_Norm1 : ReduceType::Reduce_Norm2;
  std::vector<int64_t> reduce_axis = getRealDim(dim.vec(), input_contiguous.dim());
  auto output_shape = getOutputShape(input.sizes().vec(), reduce_axis, keepdim);
  auto full_output_shape = getOutputShape(input_contiguous.sizes().vec(), reduce_axis, true);
  cnnl_reduce_internal(input_contiguous, output, index, reduce_axis, full_output_shape,
                       reduce_type, output_shape);
  return output;
}

at::Tensor cnnl_norm(const at::Tensor& input, at::optional<at::Scalar> p,
                     at::IntArrayRef dim, bool keepdim, at::ScalarType dtype) {
  // Param 'dtype' means the desired data type of output tensor. If specified,
  // the input tensor is casted to :attr:’dtype’ while performing the operation
  auto input_contiguous = cnnl_contiguous(input, reduce_memory_format);
  at::Tensor input_cast;
  if (input_contiguous.scalar_type() != dtype) {
    input_cast = at::empty_like(input_contiguous, input_contiguous.options().dtype(dtype));
    cnnl_cast_internal(input_contiguous, input_cast);
  }

  return cnnl_norm((input_contiguous.scalar_type() != dtype) ? input_cast : input_contiguous,
      p, dim, keepdim);
}

at::Tensor cnnl_norm(const at::Tensor& input, at::Scalar p) {
  auto input_ = cnnl_contiguous(input, reduce_memory_format);
  int64_t dim = input_.dim();
  if (0 == dim) {
    auto result = at::empty_like(input_);
    cnnl_cast_internal(input_, result);
    return result;
  }
  std::vector<int64_t> reduce_dim_vec(dim);
  for (int64_t i = dim - 1; i >= 0; i--)
    reduce_dim_vec[i] = i;
  at::IntArrayRef reduce_dim(reduce_dim_vec);
  return cnnl_norm(input_, p, reduce_dim, false);
}

std::tuple<at::Tensor, at::Tensor> cnnl_max(const at::Tensor& input,
                                            const int64_t dim,
                                            const bool keepdim) {
  auto input_contiguous = cnnl_contiguous(input, reduce_memory_format);
  at::Tensor output;
  at::Tensor index;
  std::vector<int64_t> dim_vec = {dim};
  std::vector<int64_t> reduce_axis = getRealDim(dim_vec, input_contiguous.dim());
  auto output_shape = getOutputShape(input.sizes().vec(), reduce_axis, keepdim);
  auto full_output_shape = getOutputShape(input_contiguous.sizes().vec(), reduce_axis, true);
  cnnl_reduce_internal(input_contiguous, output, index, reduce_axis, full_output_shape,
                       ReduceType::Reduce_Max, output_shape);
  return std::make_tuple(output, index);
}

at::Tensor cnnl_argmax(const at::Tensor & self, c10::optional<int64_t> dim, bool keepdim) {
  at::Tensor ignored = at::empty({0}, self.options());
  at::Tensor result = at::empty({0}, self.options().dtype(at::kLong));
  at::Tensor in;
  if (dim) {
    in = self;
  } else {
    in = cnnl_reshape(self, {-1});
    keepdim = false;
  }
  return std::get<1>(cnnl_max_out_dim_max(ignored, result, in, dim.value_or(0), keepdim));
}

std::tuple<at::Tensor&, at::Tensor&> cnnl_max_out_dim_max(at::Tensor &max,
  at::Tensor &max_indices, const at::Tensor &self, int64_t dim, bool keepdim) {
  TORCH_CHECK(self.device() == max.device(), "expected device ", self.device(), " but got ",
              max.device(), " for max values output");
  TORCH_CHECK(self.device() == max_indices.device(),
              "expected device ", self.device(), " but got ",
              max_indices.device(), " for indices output");
  TORCH_CHECK(self.numel() > 0, "cannot perform reduction function max",
    " on tensor with no elements because the operation does not have an identity");
  auto self_contiguous = cnnl_contiguous(self, reduce_memory_format);
  dim = at::maybe_wrap_dim(dim, self_contiguous.dim());
  if (self_contiguous.numel() == 1 && self_contiguous.ndimension() == 0) {
    cnnl_resize_(max, {});
    cnnl_fill_(max, self_contiguous);
    cnnl_resize_(max_indices, {});
    cnnl_fill_(max_indices, 0);
  } else {
    at::Tensor output, index;
    std::vector<int64_t> dim_vec = {dim};
    std::vector<int64_t> reduce_axis = getRealDim(dim_vec, self_contiguous.dim());
    auto output_shape = getOutputShape(self.sizes().vec(), reduce_axis, keepdim);
    auto full_output_shape = getOutputShape(self_contiguous.sizes().vec(), reduce_axis, true);
    cnnl_reduce_internal(self_contiguous, output, index, reduce_axis, full_output_shape,
                         ReduceType::Reduce_Max, output_shape);
    getMluTensorImpl(max)->copy_cnnl_metadata_from(getMluTensorImpl(output));
    resize_impl_mlu_(getMluTensorImpl(max), output.sizes(), output.strides());
    getMluTensorImpl(max_indices)->copy_cnnl_metadata_from(getMluTensorImpl(index));
    resize_impl_mlu_(getMluTensorImpl(max_indices), index.sizes(), index.strides());
  }
  return std::forward_as_tuple(max, max_indices);
}

std::tuple<at::Tensor, at::Tensor> cnnl_min(const at::Tensor& input,
                                            const int64_t dim,
                                            const bool keepdim) {
  auto input_contiguous = cnnl_contiguous(input, reduce_memory_format);
  at::Tensor output;
  at::Tensor index;
  std::vector<int64_t> dim_vec = {dim};
  std::vector<int64_t> reduce_axis = getRealDim(dim_vec, input_contiguous.dim());
  auto output_shape = getOutputShape(input.sizes().vec(), reduce_axis, keepdim);
  auto full_output_shape = getOutputShape(input_contiguous.sizes().vec(), reduce_axis, true);
  cnnl_reduce_internal(input_contiguous, output, index, reduce_axis, full_output_shape,
                       ReduceType::Reduce_Min, output_shape);
  return std::make_tuple(output, index);
}

at::Tensor cnnl_min(const at::Tensor& input) {
  TORCH_MLU_CHECK(input.numel() > 0, "operation does not have an identity.");
  auto input_contiguous = cnnl_contiguous(input, reduce_memory_format);
  at::Tensor output;
  at::Tensor index;
  std::vector<int64_t> reduce_axis = getAllDim(input_contiguous.dim());
  std::vector<int64_t> output_shape = {};
  auto full_output_shape = getOutputShape(input_contiguous.sizes().vec(), reduce_axis, true);
  cnnl_reduce_internal(input_contiguous, output, index, reduce_axis, full_output_shape,
                       ReduceType::Reduce_Min, output_shape);
  return output;
}

at::Tensor cnnl_max(const at::Tensor& input) {
  TORCH_MLU_CHECK(input.numel() > 0, "operation does not have an identity.");
  auto input_contiguous = cnnl_contiguous(input, reduce_memory_format);
  at::Tensor output;
  at::Tensor index;
  std::vector<int64_t> reduce_axis = getAllDim(input_contiguous.dim());
  std::vector<int64_t> output_shape = {};
  auto full_output_shape = getOutputShape(input_contiguous.sizes().vec(), reduce_axis, true);
  cnnl_reduce_internal(input_contiguous, output, index, reduce_axis, full_output_shape,
                       ReduceType::Reduce_Max, output_shape);
  return output;
}

void _all(const at::Tensor& self,
          at::Tensor& output,
          at::Tensor index,
          std::vector<int64_t> reduce_axis,
          std::vector<int64_t> full_output_shape,
          std::vector<int64_t> output_shape) {
  cnnl_reduce_internal(self, output, index, reduce_axis, full_output_shape,
                       ReduceType::Reduce_All, output_shape);
}

at::Tensor cnnl_all(const at::Tensor& self, int64_t dim, bool keepdim) {
  auto self_contiguous = cnnl_contiguous(self, reduce_memory_format);

  at::Tensor result = at::empty({0}, self.options());
  return cnnl_all_out(result, self, dim, keepdim);
}

at::Tensor cnnl_all(const at::Tensor& self) {
  TORCH_MLU_CHECK(
      self.scalar_type() == at::ScalarType::Byte ||
          self.scalar_type() == at::ScalarType::Bool,
      "all only supports torch.uint8 and torch.bool dtypes");
  if (self.numel() == 0) {
    at::Tensor output = at::empty({}, self.options(), reduce_memory_format);
    output.fill_(1);
    return output;
  } else {
    auto self_contiguous = cnnl_contiguous(self, reduce_memory_format);
    at::Tensor output;
    at::Tensor index;
    std::vector<int64_t> reduce_axis = getAllDim(self_contiguous.dim());
    std::vector<int64_t> output_shape = {};
    auto full_output_shape = getOutputShape(self_contiguous.sizes().vec(), reduce_axis, true);
    _all(self_contiguous, output, index, reduce_axis, full_output_shape,
         output_shape);
      return output;
  }
}

at::Tensor& cnnl_all_out(
    at::Tensor& out,
    const at::Tensor& self,
    int64_t dim,
    bool keepdim) {
  auto self_contiguous = cnnl_contiguous(self, reduce_memory_format);
  TORCH_MLU_CHECK(
      self.scalar_type() == at::ScalarType::Byte ||
          self.scalar_type() == at::ScalarType::Bool,
      "all only supports torch.uint8 and torch.bool dtypes");
  TORCH_MLU_CHECK(
      out.scalar_type() == self.scalar_type(),
      "out and self should have same dtype");
  dim = at::maybe_wrap_dim(dim, self.dim());
  if (at::native::_dimreduce_return_trivial(out, self_contiguous, 1, dim, keepdim)) {
    return out;
  } else {
    at::Tensor output;
    at::Tensor index;
    std::vector<int64_t> dim_vec = {dim};
    std::vector<int64_t> reduce_axis = getRealDim(dim_vec, self_contiguous.dim());
    auto output_shape = getOutputShape(self.sizes().vec(), reduce_axis, keepdim);
    auto full_output_shape = getOutputShape(self_contiguous.sizes().vec(), reduce_axis, true);

    _all(self_contiguous, output, index, reduce_axis, full_output_shape,
         output_shape);

    out.resize_(output.sizes());
    out.copy_(output);
    return out;
  }
}

at::Tensor cnnl_any(const at::Tensor& input, const int64_t dim,
                    const bool keepdim) {
  auto input_contiguous = cnnl_contiguous(input, reduce_memory_format);
  at::Tensor output;
  at::Tensor index;
  std::vector<int64_t> dim_vec = {dim};
  std::vector<int64_t> reduce_axis = getRealDim(dim_vec, input_contiguous.dim());
  auto output_shape = getOutputShape(input.sizes().vec(), reduce_axis, keepdim);
  auto full_output_shape = getOutputShape(input_contiguous.sizes().vec(), reduce_axis, true);
  cnnl_reduce_internal(input_contiguous, output, index, reduce_axis, full_output_shape,
                       ReduceType::Reduce_Any, output_shape);
  return output;
}

at::Tensor cnnl_any(const at::Tensor& input) {
  auto input_contiguous = cnnl_contiguous(input, reduce_memory_format);
  at::Tensor output;
  at::Tensor index;
  std::vector<int64_t> reduce_axis = getAllDim(input_contiguous.dim());
  std::vector<int64_t> output_shape = {};
  auto full_output_shape = getOutputShape(input_contiguous.sizes().vec(), reduce_axis, true);
  cnnl_reduce_internal(input_contiguous, output, index, reduce_axis, full_output_shape,
                       ReduceType::Reduce_Any, output_shape);
  return output;
}

at::Tensor cnnl_prod(const at::Tensor& self,
                     c10::optional<c10::ScalarType> dtype) {
  auto self_contiguous = cnnl_contiguous(self, reduce_memory_format);
  at::Tensor output;
  at::Tensor index;
  std::vector<int64_t> reduce_axis = getAllDim(self_contiguous.dim());
  std::vector<int64_t> output_shape = {};
  auto full_output_shape = getOutputShape(self_contiguous.sizes().vec(), reduce_axis, true);
  cnnl_reduce_internal(self_contiguous, output, index, reduce_axis, full_output_shape,
                       ReduceType::Reduce_Mul, output_shape);
  return output;
}

at::Tensor cnnl_prod(const at::Tensor& input,
                     const int64_t dim,
                     const bool keepdim,
                     c10::optional<c10::ScalarType> dtype) {
  auto input_contiguous = cnnl_contiguous(input, reduce_memory_format);
  at::Tensor output;
  at::Tensor index;
  std::vector<int64_t> dim_vec = {dim};
  std::vector<int64_t> reduce_axis = getRealDim(dim_vec, input_contiguous.dim());
  auto output_shape = getOutputShape(input.sizes().vec(), reduce_axis, keepdim);
  auto full_output_shape = getOutputShape(input_contiguous.sizes().vec(), reduce_axis, true);
  cnnl_reduce_internal(input_contiguous, output, index, reduce_axis, full_output_shape,
                       ReduceType::Reduce_Mul, output_shape);
  return output;
}

at::Tensor& cnnl_prod_out(at::Tensor& output,
                         const at::Tensor& input,
                         const int64_t dim,
                         bool keepdim,
                         c10::optional<c10::ScalarType> dtype) {
  auto result = cnnl_prod(input, dim, keepdim, dtype);
  output.resize_(result.sizes());
  output.copy_(result);
  return output;
}

}  // namespace ops
}  // namespace cnnl
}  // namespace torch_mlu
