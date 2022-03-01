#pragma once

#include <ATen/ExpandUtils.h>
#include <ATen/native/IndexingUtils.h>
#include <ATen/native/ConvUtils.h>
#include <ATen/native/ReduceOpsUtils.h>
#include <ATen/native/SortingUtils.h>

#include "aten/cnnl/cnnlDescriptors.h"
#include "aten/cnnl/cnnlAlgorithms.h"
#include "aten/cnnl/cnnlHandle.h"
#include "aten/core/tensor_impl.h"
#include "aten/util/exceptions.h"
#include "aten/util/tensor_util.h"
#include "aten/device/queue.h"
#include "aten/util/cnlog.h"
#include "aten/util/common.h"
#include "aten/util/memory_allocator.h"
#include "aten/cnnl/cnnl_util.h"
#include "aten/operators/cnnl/internal/internal_util.h"

using at::Tensor;
using at::TensorList;
using at::IntArrayRef;

enum class ReduceType {
  Reduce_Sum,
  Reduce_Mean,
  Reduce_Max,
  Reduce_Min,
  Reduce_Any,
  Reduce_And,
  Reduce_All,
  Reduce_Mul,
  Reduce_Norm1,
  Reduce_Norm2
};

namespace torch_mlu {
namespace cnnl {
namespace ops {

std::vector<int64_t> infer_conv_output_size(at::IntList input_size,
                                            at::IntList weight_size,
                                            int *padding, int *stride,
                                            int *dilation);

std::vector<int64_t> infer_deconv_output_size(at::IntList input_size,
                                           at::IntList weight_size,
                                           int* padding, int* output_padding,
                                           int* stride, int* dilation,
                                           int groups);

at::Tensor& cnnl_threshold_internal(at::Tensor& output, const at::Tensor& input,
                                    at::Scalar threshold, at::Scalar value);

at::Tensor& cnnl_threshold_backward_internal(at::Tensor& grad_input, const at::Tensor& grad_output,
                                             const at::Tensor& input, at::Scalar threshold);

void cnnl_adaptive_max_pool2d_backward_internal(
  at::Tensor& grad_input, const at::Tensor& grad_output,
  const at::Tensor& input, const at::Tensor& indices);

void cnnl_adaptive_max_pool2d_internal(
  at::Tensor& output, at::Tensor& indices, const at::Tensor& input, at::IntArrayRef output_size);

void cnnl_adaptive_avg_pool_backward_internal(
  at::Tensor& grad_input, const at::Tensor& grad_output, const at::Tensor& input);

void cnnl_adaptive_avg_pool_internal(
  at::Tensor& output, const at::Tensor& input, at::IntArrayRef output_size);

at::Tensor& cnnl_abs_internal(at::Tensor& output, const at::Tensor& input);

at::Tensor& cnnl_ceil_internal(at::Tensor& output, const at::Tensor& input);

at::Tensor cnnl_maximum_internal(const at::Tensor& input,
                                 const at::Tensor& other);

at::Tensor cnnl_minimum_internal(const at::Tensor& input, const at::Tensor& other);

at::Tensor& cnnl_activation_internal(at::Tensor& output,
                                     const at::Tensor& input,
                                     cnnlActivationMode_t mode,
                                     at::Scalar negative_slope = 0.0);

at::Tensor cnnl_activation_backward_internal(const at::Tensor& grad,
                                             const at::Tensor& self,
                                             cnnlActivationMode_t mode);

at::Tensor& cnnl_softplus_internal(at::Tensor&output,
                                   const at::Tensor& input,
                                   at::Scalar beta,
                                   at::Scalar threshold);

at::Tensor& cnnl_softplus_backward_internal(at::Tensor& grad_input,
                                           const at::Tensor& self,
                                           const at::Tensor& grad_output,
                                           at::Scalar beta,
                                           at::Scalar threshold,
                                           const at::Tensor& output);

at::Tensor cnnl_leaky_relu_backward_internal(const at::Tensor& grad,
                                             const at::Tensor& self,
                                             cnnlActivationMode_t mode,
                                             at::Scalar negative_slope,
                                             bool self_is_result);

at::Tensor cnnl_float_convolution_internal(const at::Tensor & input,
    const at::Tensor & weight, const at::Tensor& bias, at::TensorOptions input_options,
    int* padding, int* stride, int* dilation, int64_t groups, bool depthwise = false);

at::Tensor cnnl_float_convolution_backward_input_internal(
    const at::Tensor& input, const at::Tensor& grad,
    const at::Tensor& weight, int* stride,
    int* padding, int* dilation, int64_t groups, at::TensorOptions input_options,
    bool depthwise = false);

at::Tensor cnnl_float_convolution_backward_weight_internal(
    const at::Tensor& weight, const at::Tensor& grad,
    const at::Tensor& input, int* stride,
    int* padding, int* dilation, int64_t groups, at::TensorOptions weight_options,
    bool depthwise = false);

at::Tensor cnnl_float_convolution_transpose_internal(const at::Tensor& input,
                                               const at::Tensor& weight,
                                               const at::Tensor& bias,
                                               int* padding,
                                               int* output_padding,
                                               int* stride,
                                               int* dilation,
                                               int64_t groups);

at::Tensor cnnl_float_convolution_transpose_backward_input_internal(const at::Tensor& input,
                                                                    const at::Tensor& grad,
                                                                    const at::Tensor& weight,
                                                                    int* stride,
                                                                    int* padding,
                                                                    int* dilation,
                                                                    int64_t groups);

at::Tensor& cnnl_floor_internal(at::Tensor& output, const at::Tensor& self);

at::Tensor& cnnl_log_internal(at::Tensor& output, const at::Tensor& input, cnnlLogBase_t base);

at::Tensor cnnl_gather_internal(const at::Tensor& self, int64_t dim,
                                const at::Tensor& index);

at::Tensor& cnnl_index_select_internal(at::Tensor& output, const at::Tensor& input,
                                      int64_t dim, const at::Tensor& index);

at::Tensor& cnnl_index_put_internal(at::Tensor& output, const at::Tensor& self,
                                    std::vector<at::Tensor> indices, const at::Tensor& value,
                                    bool accumulate);

at::Tensor& cnnl_arange_internal(at::Tensor& out,
                                 const at::Scalar start,
                                 const at::Scalar end,
                                 const at::Scalar step);

at::Tensor& cnnl_index_fill_internal(at::Tensor& output, const at::Tensor& self,
    int64_t dim, const at::Tensor& index, at::Scalar value);

at::Tensor cnnl_nonzero_internal(at::Tensor& out, const at::Tensor& self);

std::tuple<at::Tensor, at::Tensor, at::Tensor> cnnl_native_batch_norm_internal(
    const at::Tensor& input, const at::Tensor& weight, const at::Tensor& bias,
    const at::Tensor& running_mean, const at::Tensor& running_var,
    bool training, double momentum, double eps);

std::tuple<at::Tensor, at::Tensor, at::Tensor>
cnnl_native_batch_norm_backward_internal(
    const at::Tensor& grad_out, const at::Tensor& input,
    const at::Tensor& weight, const at::Tensor& running_mean,
    const at::Tensor& running_var, const at::Tensor& save_mean_mean,
    const at::Tensor& save_invstd, bool training, double eps,
    std::array<bool, 3> output_mask);

at::Tensor cnnl_pool2d_internal(const at::Tensor& self,
                                at::IntArrayRef kernel_size,
                                at::IntArrayRef stride, at::IntArrayRef padding,
                                bool ceil_mode, bool count_include_pad,
                                int64_t pool_mode_row);

at::Tensor cnnl_pool3d_internal(const at::Tensor& self,
                                at::IntArrayRef kernel_size,
                                at::IntArrayRef stride, at::IntArrayRef padding,
                                bool ceil_mode, bool count_include_pad,
                                int64_t pool_mode_row);

std::tuple<at::Tensor, at::Tensor> cnnl_max_pool_internal(const at::Tensor& self,
                                                          at::IntArrayRef kernel_size,
                                                          at::IntArrayRef stride,
                                                          at::IntArrayRef padding,
                                                          bool ceil_mode);

at::Tensor cnnl_avg_pool2d_backward_internal(
    const at::Tensor& grad_output, const at::Tensor& index,
    const at::Tensor& self, at::IntArrayRef kernel_size, at::IntArrayRef stride,
    at::IntArrayRef padding, bool ceil_mode, bool count_include_pad);

at::Tensor cnnl_tanh_backward_internal(const at::Tensor& grad,
                                       const at::Tensor& self,
                                       cnnlActivationMode_t mode);

at::Tensor cnnl_bmm_internal(const at::Tensor& self, const at::Tensor& other,
                             const int self_position, const int other_position,
                             at::TensorOptions self_options, bool is_trans_self,
                             bool is_trans_other, bool run_fp32 = false);

at::Tensor& cnnl_logic_internal(at::Tensor& output, const at::Tensor& self, const at::Tensor& other,
                               cnnlLogicOp_t logic_type);

void cnnl_cast_internal(const at::Tensor& input, at::Tensor& output);

at::Tensor cnnl_cat_internal(at::TensorList tensors, int64_t dim);

at::Tensor& cnnl_div_out_internal(at::Tensor& output,
                                  const at::Tensor& self,
                                  const at::Tensor& other);

at::Tensor cnnl_isfinite_internal(const at::Tensor& input);

at::Tensor cnnl_zeros_like_internal(const at::Tensor& self);

at::Tensor& cnnl_diag_internal(at::Tensor& output, const at::Tensor& input, int64_t k);

void cnnl_topk_internal(at::Tensor& values, at::Tensor& indices, const at::Tensor& self,
                        int64_t k, int64_t dim, bool largest, bool sorted);

std::tuple<at::Tensor, at::Tensor> cnnl_sort_internal(const at::Tensor& self,
                                                      const int64_t dim,
                                                      const bool largest,
                                                      const bool sorted);

at::Tensor& cnnl_sqrt_internal(at::Tensor& output, const at::Tensor& input);

at::Tensor cnnl_slice_internal(const at::Tensor& input, int64_t dim,
                               int64_t start, int64_t end, int64_t step);

at::Tensor& cnnl_repeat_internal(at::Tensor& output,
                                 const at::Tensor& input);

at::Tensor& cnnl_pow_internal(at::Tensor& output, const at::Tensor& input,
                               const at::Tensor& exponent);

at::Tensor& cnnl_fill_internal(at::Tensor& input, float value);

at::Tensor& cnnl_exp_internal(at::Tensor& output, const at::Tensor& input);

at::Tensor cnnl_expand_internal(const at::Tensor& self, at::IntArrayRef size,
                                bool implicit);

at::Tensor cnnl_permute_internal(const at::Tensor& input,
                                 at::IntArrayRef dims);

at::Tensor& cnnl_permute_out_internal(at::Tensor& output,
                                      const at::Tensor& input,
                                      at::IntArrayRef dims);

at::Tensor cnnl_optensor_internal(const at::Tensor& input,
                                  const at::Tensor& other,
                                  at::Scalar alpha_scalar,
                                  at::Scalar beta_scalar,
                                  cnnlOpTensorDesc_t op_type);

at::Tensor& cnnl_optensor_out_internal(at::Tensor& output,
                                      const at::Tensor& input,
                                      const at::Tensor& other,
                                      at::Scalar alpha_scalar,
                                      at::Scalar beta_scalar,
                                      cnnlOpTensorDesc_t op_type);

at::Tensor& cnnl_neg_internal(at::Tensor& output, const at::Tensor& input);

at::Tensor& cnnl_replication_pad2d_internal(at::Tensor& output,
                                            const at::Tensor& self,
                                            at::IntArrayRef padding);

at::Tensor cnnl_softmax_internal(const at::Tensor& self, int64_t dim,
                                 bool half_to_float,
                                 cnnlSoftmaxAlgorithm_t algo);

at::Tensor cnnl_softmax_backward_internal(const at::Tensor& grad,
                                          const at::Tensor& output, int64_t dim,
                                          const at::Tensor& self,
                                          cnnlSoftmaxAlgorithm_t algo);

at::Tensor cnnl_optensor_inplace_internal(at::Tensor& input,
                                          const at::Tensor& other,
                                          at::Scalar beta_scalar,
                                          at::Scalar alpha_scalar,
                                          cnnlOpTensorDesc_t op_type);

void cnnl_reduce_internal(const at::Tensor& input, at::Tensor& output,
                          at::Tensor& index, const std::vector<int64_t> reduce_dim,
                          const std::vector<int64_t> desc_shape, ReduceType reduce_type,
                          const std::vector<int64_t> output_shape);

at::Tensor& cnnl_smooth_l1_loss_forward_internal(at::Tensor& output,
                                                 const at::Tensor& self,
                                                 const at::Tensor& target,
                                                 int64_t reduction);

at::Tensor& cnnl_smooth_l1_loss_backward_internal(at::Tensor& grad_input,
                                                  const at::Tensor& grad_output,
                                                  const at::Tensor& self,
                                                  const at::Tensor& target,
                                                  int64_t reduction);

std::tuple<at::Tensor, at::Tensor> cnnl_nll_loss_forward_internal(
    const at::Tensor& self, const at::Tensor& target, const at::Tensor& weight,
    int64_t reduction, int64_t ignore_index);

at::Tensor cnnl_nll_loss_backward_internal(
    const at::Tensor& grad_output, const at::Tensor& self,
    const at::Tensor& target, const at::Tensor& weight, int64_t reduction,
    int64_t ignore_index, const at::Tensor& total_weight);

at::Tensor cnnl_bce_internal(
    const at::Tensor& self, const at::Tensor& target, const at::Tensor& weight, int64_t reduction);

at::Tensor cnnl_bce_bp_internal(
    const at::Tensor& grad_output, const at::Tensor& self,
    const at::Tensor& target, const at::Tensor& weight, int64_t reduction);

at::Tensor cnnl_bce_with_logits_internal(
    const at::Tensor& self, const at::Tensor& target, const at::Tensor& weight,
    const at::Tensor& pos_weight, int64_t reduction);

at::Tensor cnnl_bce_with_logits_bp_internal(
    const at::Tensor& grad_output, const at::Tensor& self,
    const at::Tensor& target, const at::Tensor& weight, const at::Tensor& pos_weight,
    int64_t reduction);

at::Tensor cnnl_view_internal(const at::Tensor& self, at::IntArrayRef size);

at::Tensor cnnl_unsqueeze_internal(const at::Tensor& self, at::IntArrayRef size);

at::Tensor& cnnl_clamp_internal(at::Tensor& output,
                               const at::Tensor& self,
                               at::optional<at::Scalar> min,
                               at::optional<at::Scalar> max);

at::Tensor& cnnl_reciprocal_internal(at::Tensor& output, const at::Tensor& input);

at::Tensor cnnl_linspace_internal(at::Tensor& output, at::Scalar start,
                                  at::Scalar end, int64_t steps);

at::Tensor cnnl_bias_backward_internal(const at::Tensor& input, int64_t dim);

at::Tensor cnnl_mm_internal(const at::Tensor& self, const at::Tensor& other,
                            const int self_position, const int other_position,
                            at::TensorOptions self_options, bool is_trans_self,
                            bool is_trans_other, bool run_fp32 = false);

at::Tensor cnnl_quantify_offline_internal(const at::Tensor& input,
                                          const int bitwidth,
                                          const int position);

at::Tensor cnnl_quantify_offline_internal(const at::Tensor& input,
                                          const int bitwidth,
                                          const at::Tensor& position);

at::Tensor cnnl_quantify_convolution_backward_weight_internal(
    const at::Tensor& weight, cnnlTensorLayout_t layout, const at::Tensor& grad,
    int grad_position_value, const at::Tensor& input,
    cnnlDataType_t input_dtype, int input_position_value, int* stride,
    int* padding, int* dilation, int64_t groups);

at::Tensor cnnl_quantify_convolution_backward_weight_internal(
    const at::Tensor& weight, cnnlTensorLayout_t layout, const at::Tensor& grad,
    int grad_position_value, const at::Tensor& input,
    cnnlDataType_t input_dtype, int input_position_value, int* stride,
    int* padding, int* dilation, int64_t groups, at::TensorOptions weight_options);

std::tuple<at::Tensor, at::Tensor, at::Tensor>
cnnl_quantify_convolution_internal(
    const at::Tensor& quantify_input, const at::Tensor& input_position,
    const at::Tensor& quantify_weight, const at::Tensor& weight_position,
    const at::Tensor& bias, at::TensorOptions input_options, int* padding,
    int* stride, int* dilation, int64_t groups);

at::Tensor cnnl_quantify_convolution_internal(
    const at::Tensor& quantify_input, int input_position_value,
    const at::Tensor& quantify_weight, int weight_position_value, const at::Tensor& bias,
    at::TensorOptions input_options, int* padding, int* stride, int* dilation, int64_t groups);

at::Tensor check_depth_transpose(const at::Tensor& input,
                                 CnnlTransposeDescriptor& trans_desc,
                                 cnnlTensorLayout_t layout);

at::Tensor& cnnl_masked_fill_internal(at::Tensor& output,
                                      const at::Tensor& input,
                                      const at::Tensor& mask,
                                      const at::Tensor& value);

at::Tensor& cnnl_masked_select_internal(at::Tensor& output,
                                       const at::Tensor& input,
                                       const at::Tensor& mask);

at::Tensor cnnl_alias_internal(const at::Tensor& self);

std::tuple<at::Tensor, at::Tensor, at::Tensor> cnnl__unique2_internal(
    const at::Tensor& self, bool sorted, bool return_inverse,
    bool return_counts);

at::Tensor cnnl_transpose_internal(const at::Tensor& self, int64_t dim0,
                                   int64_t dim1);

at::Tensor cnnl_transform_internal(const at::Tensor& input,
                                   const at::Scalar alpha_scalar,
                                   const at::Scalar beta_scalar);

at::Tensor& cnnl_transform_out_internal(at::Tensor& output,
                                       const at::Tensor& input,
                                       const at::Scalar alpha_scalar,
                                       const at::Scalar beta_scalar);

at::Tensor& cnnl_transform_inplace_internal(at::Tensor& input,
                                            const at::Scalar alpha_scalar,
                                            const at::Scalar beta_scalar);

at::Tensor cnnl_hardtanh_backward_internal(const at::Tensor& self, const at::Tensor& grad,
                                           at::Scalar min_val, at::Scalar max_val);

at::Tensor& cnnl_hardtanh_internal(at::Tensor& output, const at::Tensor& input,
                                   at::Scalar min_val, at::Scalar max_val);

at::Tensor& cnnl_round_internal(at::Tensor& output, const at::Tensor& input);

at::Tensor& cnnl_upsample_internal(at::Tensor& output, const at::Tensor& self,
                                   at::IntArrayRef output_size, bool align_corners,
                                   bool align_center, cnnlInterpMode_t interp_mode);

at::Tensor cnnl_upsample_backward_internal(at::Tensor& grad_input, const at::Tensor& grad_output,
                                           at::IntArrayRef output_size, at::IntArrayRef input_size,
                                           bool align_corners, bool align_center,
                                           cnnlInterpBackwardMode_t interp_mode);

at::Tensor& cnnl_bitwise_op_out_internal(at::Tensor& out,
                                        const at::Tensor& self,
                                        const at::Tensor& other,
                                        const cnnlBitComputeOp_t& op_type);

at::Tensor cnnl_index_internal(at::Tensor& self,
                               std::vector<at::Tensor> indices,
                               std::vector<int64_t> output_sizes);

at::Tensor cnnl_nms_internal(const at::Tensor& dets,
                             const at::Tensor& scores,
                             double iou_threshold);

at::Tensor& cnnl_remainder_internal(at::Tensor& output,
                                     const at::Tensor& self,
                                     const at::Tensor& other);

at::Tensor& cnnl_copy_internal(at::Tensor& output,
                               const at::Tensor& input);

at::Tensor& cnnl_copy_without_contiguous_internal(at::Tensor& output,
                                                  const at::Tensor& input);

at::Tensor cnnl_avg_pool3d_internal(const at::Tensor& self,
                                at::IntArrayRef kernel_size,
                                at::IntArrayRef stride, at::IntArrayRef padding,
                                bool ceil_mode, bool count_include_pad,
                                int64_t pool_mode_row);

at::Tensor cnnl_avgpool3d_backward_internal(
    const at::Tensor& grad_output, const at::Tensor& self,
    at::IntArrayRef kernel_size, at::IntArrayRef stride,
    at::IntArrayRef padding, bool ceil_mode, bool count_include_pad,
    c10::optional<int64_t> divisor_override);

std::tuple<at::Tensor, at::Tensor> cnnl_maxpool3d_with_index_internal(
    const at::Tensor& self,
    at::IntArrayRef kernel_size,
    at::IntArrayRef stride,
    at::IntArrayRef padding,
    at::IntArrayRef dilation,
    bool ceil_mode);

at::Tensor cnnl_maxpool3d_backward_internal(
    const at::Tensor & grad_output,
    const at::Tensor & self,
    at::IntArrayRef kernel_size, at::IntArrayRef stride,
    at::IntArrayRef padding, at::IntArrayRef dilation,
    bool ceil_mode,
    const at::Tensor & indices);

at::Tensor cnnl_mse_loss_internal(const at::Tensor& input,
     const at::Tensor & target, int64_t reduction);

at::Tensor cnnl_mse_loss_backward_internal(const at::Tensor& grad_output,
                                           const at::Tensor& input,
                                           const at::Tensor & target,
                                           int64_t reduction);

at::Tensor cnnl_optensor_compute(at::Tensor& output,
                                 const at::Tensor& self,
                                 const at::Tensor& other,
                                 at::Scalar alpha_scalar,
                                 at::Scalar beta_scalar,
                                 cnnlOpTensorDesc_t op_type,
                                 at::ScalarType compute_type);

at::Tensor& cnnl_inverse_internal(at::Tensor& output,
                              const at::Tensor& input);

}  // namespace ops
}  // namespace cnnl
}  // namespace torch_mlu
