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

#include "aten/cnnl/cnnlOpDescriptors.h"

namespace torch_mlu {
    void CnnlPoolingDescriptor::set(cnnlPoolingMode_t mode,
                                    int kernel_h, int kernel_w, int stride_h, int stride_w,
                                    int pad_u, int pad_d, int pad_l, int pad_r) {
        TORCH_CNNL_CHECK(cnnlSetPooling2dDescriptor(
            this->mut_desc(), mode, CNNL_PROPAGATE_NAN, kernel_h, kernel_w,
            pad_u, pad_d, pad_l, pad_r, stride_h, stride_w));
    }

    void CnnlPoolingDescriptor::set(cnnlPoolingMode_t mode,
                                    int64_t dims,
                                    const int kernel_size[],
                                    const int stride[],
                                    const int padding[]) {
        TORCH_CNNL_CHECK(cnnlSetPoolingNdDescriptor(
            this->mut_desc(), mode, CNNL_NOT_PROPAGATE_NAN, dims, kernel_size, padding, stride));
    }

    void CnnlPoolingDescriptor::set(cnnlPoolingMode_t mode,
                                    int64_t dims,
                                    const int kernel_size[],
                                    const int stride[],
                                    const int padding[],
                                    const int dilation[],
                                    bool ceil_mode) {
        TORCH_CNNL_CHECK(cnnlSetPoolingNdDescriptor_v2(
            this->mut_desc(), mode, CNNL_NOT_PROPAGATE_NAN, dims, kernel_size,
            padding, stride, dilation, ceil_mode));
    }


    void CnnlTransposeDescriptor::set(const int p_dims, const int permute[]) {
        TORCH_CNNL_CHECK(cnnlSetTransposeDescriptor(this->mut_desc(), p_dims, permute));
    }

    void CnnlReduceDescriptor::set(const at::Tensor &t, std::vector<int64_t> axis,
                                   cnnlReduceOp_t mode, cnnlReduceIndices_t is_indices,
                                   cnnlIndicesType_t indices_type) {
      int axis_num = axis.size();
      std::vector<int> axis_list(axis_num);
      for (int i = 0; i < axis_num; i++) {
          axis_list[i] = static_cast<int>(axis[i]);
      }
      TORCH_CNNL_CHECK(cnnlSetReduceDescriptor(this->mut_desc(), axis_list.data(), axis_num,
                                               mode, getCnnlDataType(t.dtype()),
                                               CNNL_NOT_PROPAGATE_NAN, is_indices,
                                               indices_type));
    }

    void CnnlOpTensorDescriptor::set(cnnlOpTensorDesc_t op_type,
                                     cnnlDataType_t op_tensor_comp_type,
                                     cnnlNanPropagation_t op_tensor_nan_opt) {
        TORCH_CNNL_CHECK(cnnlSetOpTensorDescriptor(
            this->mut_desc(), op_type, op_tensor_comp_type, op_tensor_nan_opt));
    }

    void CnnlActivationDescriptor::set(cnnlActivationMode_t mode,
                                       cnnlNanPropagation_t nanProp, float ceof) {
        TORCH_CNNL_CHECK(
            cnnlSetActivationDescriptor(this->mut_desc(), mode, nanProp, ceof));
    }

    void CnnlConvolutionDescriptor::set(int dim, int *stride, int *padding,
                                        int *dilation, int64_t groups,
                                        cnnlDataType_t dtype) {
        TORCH_CHECK(dim > 2, "Convolution input's dim must greater than 2!");
        int n = dim - 2;
        std::vector<int> padding_t(2 * n);
        std::vector<int> stride_t(n);
        std::vector<int> dilation_t(n);
        int groups_t;
        for (int i = 0; i < n; ++i) {
            padding_t[2 * i] = padding[i];
            padding_t[2 * i + 1] = padding[i];
            stride_t[i] = stride[i];
            dilation_t[i] = dilation[i];
        }
        groups_t = groups;
        TORCH_CNNL_CHECK(cnnlSetConvolutionDescriptor(
            this->mut_desc(), dim, padding_t.data(), stride_t.data(), dilation_t.data(),
            groups_t, dtype));
    }

    void CnnlDeconvolutionDescriptor::set(int dim, int *stride, int *padding,
                                          int *dilation, int64_t groups,
                                          cnnlDataType_t dtype) {
      TORCH_CHECK(dim > 2, "Convolution input's dim must greater than 2!");
      int n = dim - 2;
      std::vector<int> padding_t(2 * n);
      std::vector<int> stride_t(n);
      std::vector<int> dilation_t(n);
      int groups_t;
      for (int i = 0; i < n; ++i) {
        padding_t[2 * i] = padding[i];
        padding_t[2 * i + 1] = padding[i];
        stride_t[i] = stride[i];
        dilation_t[i] = dilation[i];
      }
      groups_t = groups;
      TORCH_CNNL_CHECK(cnnlSetDeconvolutionDescriptor(
          mut_desc(), dim, padding_t.data(), stride_t.data(), dilation_t.data(),
          groups_t, dtype));
    }

    void CnnlUniqueDescriptor::set(bool sorted, int dim, bool return_inverse,
                                   bool return_counts) {
        TORCH_CNNL_CHECK(cnnlSetUniqueDescriptor(
            this->mut_desc(), sorted ? CNNL_SORT_ASCEND : CNNL_UNSORT_REVERSE, dim,
            return_inverse, return_counts));
    }

    void CnnlMatmulDescriptor::set_attr(cnnlMatMulDescAttribute_t attr,
                                        const void* buf,
                                        size_t size_in_bytes) {
        TORCH_CNNL_CHECK(cnnlSetMatMulDescAttr(this->mut_desc(), attr, buf, size_in_bytes));
    }

    void CnnlBatchMatmulDescriptor::set_attr(cnnlBatchMatMulDescAttribute_t attr,
                                             const void* buf,
                                             size_t size_in_bytes) {
        TORCH_CNNL_CHECK(cnnlSetBatchMatMulDescAttr(this->mut_desc(), attr, buf, size_in_bytes));
    }

    void CnnlCTCLossDescriptor::set(cnnlCTCLossNormalizationMode_t norm_mode,
                                    cnnlCTCLossReduceMode_t reduce_mode,
                                    cnnlCTCLossZeroInfinityMode_t zero_infinity,
                                    int blank,
                                    int max_input_length,
                                    int max_label_length) {
        TORCH_CNNL_CHECK(cnnlSetCTCLossDescriptor(
            this->mut_desc(), norm_mode, reduce_mode,
            zero_infinity, blank,
            max_input_length, max_label_length));
    }

    void CnnlNmsDescriptor::set(const cnnlNmsOutputMode_t mode,
                                const float iou_threshold,
                                const int max_output_size,
                                const float confidence_threshold,
                                const int input_layout) {
        TORCH_CNNL_CHECK(cnnlSetNmsDescriptor_v2(
            this->mut_desc(), mode, iou_threshold,
            max_output_size, confidence_threshold, input_layout));
    }

void CnnlTrigonDescriptor::set(cnnlTrigonFunctionMode_t mode) {
    TORCH_CNNL_CHECK(cnnlSetTrigonDescriptor(
        this->mut_desc(), mode));
}

}  // end of namespace torch_mlu
