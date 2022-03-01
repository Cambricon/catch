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

#pragma once

#include "aten/cnnl/cnnlCommonDescriptors.h"

namespace torch_mlu {
class C10_API CnnlPoolingDescriptor
    : public CnnlDescriptor<cnnlPoolingStruct, &cnnlCreatePoolingDescriptor,
                            &cnnlDestroyPoolingDescriptor> {
  public:
      CnnlPoolingDescriptor() = default;

      void set(cnnlPoolingMode_t mode,
               int kernel_h, int kernel_w, int stride_h, int stride_w,
               int pad_u, int pad_d, int pad_l, int pad_r);

      // NdPooling
      void set(cnnlPoolingMode_t mode,
               int64_t dims,
               const int kernel_size[],
               const int stride[],
               const int padding[]);
      void set(cnnlPoolingMode_t mode,
               int64_t dims,
               const int kernel_size[],
               const int stride[],
               const int padding[],
               const int dilation[],
               bool ceil_mode);
};

class C10_API CnnlTransposeDescriptor
    : public CnnlDescriptor<cnnlTransposeStruct, &cnnlCreateTransposeDescriptor,
                            &cnnlDestroyTransposeDescriptor> {
  public:
      CnnlTransposeDescriptor() { }

      void set(const int p_dims, const int permute[]);
};

class C10_API CnnlReduceDescriptor
    : public CnnlDescriptor<cnnlReduceStruct, &cnnlCreateReduceDescriptor,
                            &cnnlDestroyReduceDescriptor> {
  public:
      CnnlReduceDescriptor() {}
      void set(const at::Tensor &t, std::vector<int64_t> axis, cnnlReduceOp_t mode,
               cnnlReduceIndices_t is_indices, cnnlIndicesType_t indices_type);
};

class C10_API CnnlOpTensorDescriptor
    : public CnnlDescriptor<cnnlOpTensorStruct, &cnnlCreateOpTensorDescriptor,
                            &cnnlDestroyOpTensorDescriptor> {
  public:
      CnnlOpTensorDescriptor() {}

      void set(cnnlOpTensorDesc_t op_type, cnnlDataType_t op_tensor_comp_type,
               cnnlNanPropagation_t op_tensor_nan_opt);
};

class C10_API CnnlActivationDescriptor
    : public CnnlDescriptor<cnnlActivationStruct,
                            &cnnlCreateActivationDescriptor,
                            &cnnlDestroyActivationDescriptor> {
  public:
      CnnlActivationDescriptor() {}

      void set(cnnlActivationMode_t mode,
               cnnlNanPropagation_t nanProp, float ceof);
};

class C10_API CnnlConvolutionDescriptor
  : public CnnlDescriptor<cnnlConvolutionStruct,
                      &cnnlCreateConvolutionDescriptor,
                      &cnnlDestroyConvolutionDescriptor> {
  public:
      CnnlConvolutionDescriptor() {}

      void set(int dim, int* stride, int* padding,
               int* dilation, int64_t groups,
               cnnlDataType_t dtype);
};

class C10_API CnnlDeconvolutionDescriptor
  : public CnnlDescriptor<cnnlDeconvolutionStruct,
                      &cnnlCreateDeconvolutionDescriptor,
                      &cnnlDestroyDeconvolutionDescriptor> {
public:
  CnnlDeconvolutionDescriptor() {}

  void set(int dim, int* stride, int* padding, int* dilation,
           int64_t groups, cnnlDataType_t dtype);
};

class C10_API CnnlMatmulDescriptor
  : public CnnlDescriptor<cnnlMatMulStruct,
                      &cnnlMatMulDescCreate,
                      &cnnlMatMulDescDestroy> {
  public:
      CnnlMatmulDescriptor() {}
      void set_attr(cnnlMatMulDescAttribute_t attr,
                    const void* buf, size_t size_in_bytes);
};

class C10_API CnnlBatchMatmulDescriptor
  : public CnnlDescriptor<cnnlBatchMatMulStruct,
                      &cnnlBatchMatMulDescCreate,
                      &cnnlBatchMatMulDescDestroy> {
  public:
      CnnlBatchMatmulDescriptor() {}
      void set_attr(cnnlBatchMatMulDescAttribute_t attr,
                    const void* buf, size_t size_in_bytes);
};

class C10_API CnnlUniqueDescriptor
    : public CnnlDescriptor<cnnlUniqueStruct,
                            &cnnlCreateUniqueDescriptor,
                            &cnnlDestroyUniqueDescriptor> {
  public:
      CnnlUniqueDescriptor() {}

      void set(bool sorted, int dim, bool return_inverse, bool return_counts);
};

class C10_API CnnlCTCLossDescriptor
    : public CnnlDescriptor<cnnlCTCLossStruct,
                            &cnnlCreateCTCLossDescriptor,
                            &cnnlDestroyCTCLossDescriptor> {
  public:
      CnnlCTCLossDescriptor() {}
      void set(cnnlCTCLossNormalizationMode_t norm_mode,
               cnnlCTCLossReduceMode_t reduce_mode,
               cnnlCTCLossZeroInfinityMode_t zero_infinity,
               int blank,
               int max_input_length,
               int max_label_length);
};

class C10_API CnnlNmsDescriptor
    : public CnnlDescriptor<cnnlNmsStruct,
                            &cnnlCreateNmsDescriptor,
                            &cnnlDestroyNmsDescriptor> {
  public:
      CnnlNmsDescriptor() {}
      void set(const cnnlNmsOutputMode_t mode,
               const float iou_threshold,
               const int max_output_size,
               const float confidence_threshold,
               const int input_layout);
};

class C10_API CnnlTrigonDescriptor
    : public CnnlDescriptor<cnnlTrigonStruct,
                            &cnnlCreateTrigonDescriptor,
                            &cnnlDestroyTrigonDescriptor> {
  public:
      CnnlTrigonDescriptor() {}
      void set(cnnlTrigonFunctionMode_t mode);
};

}  // end of namespace torch_mlu
