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

#include <c10/core/Storage.h>
#include <c10/util/Optional.h>

#include <TH/THTensor.hpp>

#include "ATen/InferSize.h"
#include "aten/operators/cnnl/internal/cnnl_internal.h"

namespace torch_mlu {
namespace cnnl {
namespace ops {

at::Tensor cnnl_view_internal(const at::Tensor &self, at::IntArrayRef size) {
  // prepare cnnl view input
  auto input = self;
  auto input_impl = getMluTensorImpl(input);
  auto input_ptr = input_impl->cnnlMalloc();
  // resize output
  auto numel = input_impl->numel();
  auto inferred_size = at::infer_size(size, numel);
  auto new_stride = at::detail::computeStride(
      input_impl->sizes(), input_impl->strides(), inferred_size);
  TORCH_CHECK(new_stride.has_value(),
              "view size is not compatible with input tensor's size and"
              "stride (at least one dimension spans across two contiguous "
              "subspaces). Call .contiguous() before .view().");
  // (numel) => view => (..., 1, numel, ..., 1)
  bool fill_dims = false;
  if (self.sizes().size() == 1) {
    std::vector<int64_t> check_sizes;
    for (auto& dim : size) {
       if (dim != 1) check_sizes.push_back(dim);
    }
    if (check_sizes.size() == 1) fill_dims = true;
  }
  if ((inferred_size.size() > 2 || input.dim() > 2) &&
      (input.sizes() != inferred_size) && (numel > 1) && !fill_dims) {
    auto output = at::empty(inferred_size, input.options());
    if (output.numel() == 0) {
        return output;
    }
    auto output_impl = getMluTensorImpl(output);
    /* the NHWC layout of new view may not be equal with old view,
       so we firstly transpose back to NCHW for old view, than transpose to NHWC
       for new view */
    CnnlTransposeDescriptor trans_desc;
    // nhwc->nchw for old view
    std::vector<int> order_nchw(input.dim(), 0);
    for (int i = 0; i < input.dim() - 1; ++i)
      order_nchw[i + 1] = (i - 1 + input.dim() - 1) % (input.dim() - 1) + 1;
    trans_desc.set(input.dim(), order_nchw.data());
    // prepare cnnl transpose input
    CnnlTensorDescriptor input_desc;
    input_desc.set(input, CNNL_LAYOUT_NHWC);
    // prepare cnnl transpose output
    auto output_nchw = at::empty(inferred_size, input.options());
    auto output_nchw_impl = getMluTensorImpl(output_nchw);
    auto output_nchw_ptr = output_nchw_impl->cnnlMalloc();
    CnnlTensorDescriptor output_nchw_desc;
    output_nchw_desc.set(input, CNNL_LAYOUT_NCHW);
    // call cnnl transpose interface
    auto handle = getCurrentHandle();
    // Get workspace
    at::Tensor trans_workspace;
    size_t workspace_size = 0;
    void* workspace_ptr = nullptr;
    cnnlGetTransposeWorkspaceSize(handle, input_desc.desc(),
                                  trans_desc.desc(), &workspace_size);
    if (workspace_size != 0) {
      trans_workspace = at::empty({static_cast<long>(workspace_size)},
                                  self.options().dtype(at::kByte));
      auto workspace_impl = getMluTensorImpl(trans_workspace);
      workspace_ptr = workspace_impl->cnnlMalloc();
    }
    TORCH_CNNL_CHECK(cnnlTranspose_v2(handle, trans_desc.desc(), input_desc.desc(),
                                     input_ptr, output_nchw_desc.desc(),
                                     output_nchw_ptr,
                                     workspace_ptr, workspace_size));
    // nchw->nhwc for new view
    std::vector<int> order_nhwc(inferred_size.size(), 0);
    for (int i = 0; i < inferred_size.size() - 1; ++i)
      order_nhwc[i + 1] = (i + 1) % (inferred_size.size() - 1) + 1;
    trans_desc.set(inferred_size.size(), order_nhwc.data());
    // prepare cnnl transpose output
    auto output_ptr = output_impl->cnnlMalloc();
    CnnlTensorDescriptor output_desc;
    output_desc.set(output, CNNL_LAYOUT_NHWC);
    // prepare cnnl transpose input
    input = output_nchw;
    input_ptr = output_nchw_ptr;
    CnnlTensorDescriptor input_nchw_desc;
    input_nchw_desc.set(output, CNNL_LAYOUT_NCHW);
    // Get workspace
    cnnlGetTransposeWorkspaceSize(handle, input_nchw_desc.desc(),
                                  trans_desc.desc(), &workspace_size);
    if (workspace_size != 0) {
      trans_workspace = at::empty({static_cast<long>(workspace_size)},
                                  self.options().dtype(at::kByte));
      auto workspace_impl = getMluTensorImpl(trans_workspace);
      workspace_ptr = workspace_impl->cnnlMalloc();
    }

    // call cnnl transpose interface
    TORCH_CNNL_CHECK(cnnlTranspose_v2(handle, trans_desc.desc(),
                                   input_nchw_desc.desc(), input_ptr,
                                   output_desc.desc(), output_ptr,
                                   workspace_ptr, workspace_size));
    return output;
  }

  // Below logic refers to Pytorch 1.6 original function alias_with_sizes_and_strides()
  auto impl = c10::make_intrusive<MLUTensorImpl>(c10::Storage(input.storage()),
      c10::DispatchKey::MLU, input.dtype());
  impl->set_storage_offset(input.storage_offset());
  impl->set_sizes_and_strides(inferred_size, *new_stride);
  return at::Tensor(std::move(impl));
}

}  // namespace ops
}  // namespace cnnl
}  // namespace torch_mlu
