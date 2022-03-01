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

#include "aten/operators/cnnl/internal/cnnl_internal.h"
#include "aten/operators/cnnl/internal/internal_util.h"

namespace torch_mlu {
namespace cnnl {
namespace ops {

at::Tensor cnnl_softmax_internal(const at::Tensor& input, int64_t dim,
                                 bool half_to_float,
                                 cnnlSoftmaxAlgorithm_t algo) {
  auto memory_format = input.suggest_memory_format();
  auto ndim = input.dim();
  dim = ::at::maybe_wrap_dim(dim, ndim);
  dim = modify_dim_based_on_layout(dim, memory_format);
  auto input_size = input.sizes().vec();
  auto trans_input_size = modify_dims_based_on_layout(input_size, memory_format);
  // create output
  // TODO(shangang): for GPU， half_to_float value
  // is for half input and float output.
  // cnnl kernel is not support now.
  auto output = at::empty_like(input);
  auto input_impl = getMluTensorImpl(input);
  auto output_impl = getMluTensorImpl(output);
  // get current handle
  auto handle = getCurrentHandle();
  CnnlTensorDescriptor input_desc;
  CnnlTensorDescriptor output_desc;
  // input and output are same size.
  input_desc.set(input, trans_input_size, dim);
  output_desc.set(output, trans_input_size, dim);
  // malloc mlu memory
  auto input_ptr = input_impl->cnnlMalloc();
  auto output_ptr = output_impl->cnnlMalloc();
  // set descriptor config
  cnnlSoftmaxAlgorithm_t algorithm = algo;
  cnnlSoftmaxMode_t mode;
  if (ndim == 0 || dim == (ndim - 1)) {
    mode = CNNL_SOFTMAX_MODE_LOW_DIMENSION;
  } else if (dim == 0) {
    mode = CNNL_SOFTMAX_MODE_HIGH_DIMENSION;
  } else {
    mode = CNNL_SOFTMAX_MODE_MEDIUM_DIMENSION;
  }
  const void * alpha = nullptr;
  const void * beta = nullptr;
  TORCH_CNNL_CHECK(cnnlSoftmaxForward(
      /* handle    */ handle,
      /* algorithm */ algorithm,
      /* mode      */ mode,
      /* alpha     */ alpha,
      /* x_desc    */ input_desc.desc(),
      /* x         */ input_ptr,
      /* beta      */ beta,
      /* y_desc    */ output_desc.desc(),
      /* y         */ output_ptr));
  return output;
}

at::Tensor cnnl_softmax_backward_internal(const at::Tensor& grad_output,
                                          const at::Tensor& output, int64_t dim,
                                          const at::Tensor& self,
                                          cnnlSoftmaxAlgorithm_t algo) {
  auto memory_format = self.suggest_memory_format();
  auto ndim = self.dim();
  dim = ::at::maybe_wrap_dim(dim, ndim);
  dim = modify_dim_based_on_layout(dim, memory_format);
  auto input_size = self.sizes().vec();
  auto trans_input_size = modify_dims_based_on_layout(input_size, memory_format);
  auto grad = at::empty_like(self);
  auto diff_y_impl = getMluTensorImpl(grad_output);
  auto y_impl = getMluTensorImpl(output);
  auto diff_x_impl = getMluTensorImpl(grad);
  // get current handle
  auto handle = getCurrentHandle();
  CnnlTensorDescriptor diff_y_desc;
  CnnlTensorDescriptor y_desc;
  CnnlTensorDescriptor diff_x_desc;
  diff_y_desc.set(grad_output, trans_input_size, dim);
  y_desc.set(output, trans_input_size, dim);
  diff_x_desc.set(grad, trans_input_size, dim);
  // malloc mlu memory
  auto diff_y_ptr = diff_y_impl->cnnlMalloc();
  auto y_ptr = y_impl->cnnlMalloc();
  auto diff_x_ptr = diff_x_impl->cnnlMalloc();
  // set descriptor config
  cnnlSoftmaxAlgorithm_t algorithm = algo;
  cnnlSoftmaxMode_t mode;
  if (ndim == 0 || dim == (ndim - 1)) {
    mode = CNNL_SOFTMAX_MODE_LOW_DIMENSION;
  } else if (dim == 0) {
    mode = CNNL_SOFTMAX_MODE_HIGH_DIMENSION;
  } else {
    mode = CNNL_SOFTMAX_MODE_MEDIUM_DIMENSION;
  }
  const void * alpha = nullptr;
  const void * beta = nullptr;
  TORCH_CNNL_CHECK(cnnlSoftmaxBackward(
      /* handle      */ handle,
      /* algorithm   */ algorithm,
      /* mode        */ mode,
      /* alpha       */ alpha,
      /* y_desc      */ y_desc.desc(),
      /* y           */ y_ptr,
      /* diff_y_desc */ diff_y_desc.desc(),
      /* diif_y      */ diff_y_ptr,
      /* beta        */ beta,
      /* diff_x_desc */ diff_x_desc.desc(),
      /* diff _x     */ diff_x_ptr));
  return grad;
}

}  // namespace ops
}  // namespace cnnl
}  // namespace torch_mlu
