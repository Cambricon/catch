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

namespace torch_mlu {
namespace cnnl {
namespace ops {

at::Tensor cnnl_nms_internal(const at::Tensor & dets, const at::Tensor & scores,
                            double iou_threshold) {
  TORCH_CHECK(
      dets.dim() == 2, "boxes should be a 2d tensor, got ", dets.dim(), "D");
  TORCH_CHECK(
      dets.size(1) == 4,
      "boxes should have 4 elements in dimension 1, got ",
      dets.size(1));
  TORCH_CHECK(
      scores.dim() == 1,
      "scores should be a 1d tensor, got ",
      scores.dim(),
      "D");
  TORCH_CHECK(
      dets.size(0) == scores.size(0),
      "boxes and scores should have same number of elements in ",
      "dimension 0, got ",
      dets.size(0),
      " and ",
      scores.size(0));
  TORCH_CHECK(is_mlu(dets), "dets must be a MLU tensor");
  TORCH_CHECK(is_mlu(scores), "scores must be a MLU tensor");
  TORCH_CHECK(dets.scalar_type() == scores.scalar_type(),
      "dets should have the same type as scores");

  cnnlNmsOutputMode_t mode = CNNL_NMS_OUTPUT_TARGET_INDICES;
  const int max_output_size = (int)scores.size(0);
  const float confidence_threshold = 0.0;
  const int input_layout = 0;

  auto output = at::empty({max_output_size}, dets.options().dtype(at::kLong));
  auto dets_impl = getMluTensorImpl(dets);
  auto scores_impl = getMluTensorImpl(scores);
  auto output_impl = getMluTensorImpl(output);

  // get current handle
  auto handle = getCurrentHandle();

  // get cnnl descriptor
  CnnlTensorDescriptor dets_desc;
  CnnlTensorDescriptor scores_desc;
  CnnlTensorDescriptor output_desc;
  dets_desc.set(dets);
  scores_desc.set(scores);
  output_desc.set(output);

  auto dets_ptr = dets_impl->cnnlMalloc();
  auto scores_ptr = scores_impl->cnnlMalloc();
  auto output_ptr = output_impl->cnnlMalloc();

  // set nms descriptor
  CnnlNmsDescriptor nms_desc;
  nms_desc.set(mode, iou_threshold, max_output_size,
               confidence_threshold, input_layout);

  // workspace
  size_t space_size = 0;
  TORCH_CNNL_CHECK(cnnlGetNmsWorkspaceSize_v2(
      handle, scores_desc.desc(), &space_size));
  auto workspace = at::empty(space_size, scores.options().dtype(at::kByte));
  auto workspace_impl = getMluTensorImpl(workspace);
  auto workspace_ptr = workspace_impl->cnnlMalloc();

  auto output_size = at::empty({1}, scores.options().dtype(at::kInt));
  auto output_size_impl = getMluTensorImpl(output_size);
  auto output_size_ptr = output_size_impl->cnnlMalloc();

  // calculate
  TORCH_CNNL_CHECK(cnnlNms_v2(handle,
                              nms_desc.desc(),
                              dets_desc.desc(),
                              dets_ptr,
                              scores_desc.desc(),
                              scores_ptr,
                              workspace_ptr,
                              space_size,
                              output_desc.desc(),
                              output_ptr,
                              output_size_ptr));
  int output_num = *static_cast<int *>(output_size.cpu().data_ptr());
  return output.slice(0, 0, output_num);
}

}  // namespace ops
}  // namespace cnnl
}  // namespace torch_mlu
