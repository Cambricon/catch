#include <algorithm>
#include "aten/operators/cnnl/internal/cnnl_internal.h"

namespace torch_mlu {
namespace cnnl {
namespace ops {

// As deconv is the backward of conv, the follow two functions are inversed.
// Sometimes we use conv_input_size as the alias of infer_deconv_output_size.

std::vector<int64_t> infer_conv_output_size(at::IntList input_size,
                                            at::IntList weight_size,
                                            int *padding, int *stride,
                                            int *dilation) {
  auto dim = input_size.size();
  std::vector<int64_t> output_size(dim);
  output_size[0] = input_size[0];
  output_size[1] = weight_size[0];
  for (size_t d = 2; d < dim; ++d) {
    auto kernel = dilation[d - 2] * (weight_size[d] - 1) + 1;
    output_size[d] =
        (input_size[d] + (2 * padding[d - 2]) - kernel) / stride[d - 2] + 1;
  }
  return output_size;
}

std::vector<int64_t> infer_deconv_output_size(at::IntList input_size,
                                           at::IntList weight_size,
                                           int* padding, int* output_padding,
                                           int* stride, int* dilation,
                                           int groups) {
  auto dim = input_size.size();
  std::vector<int64_t> output_size(dim);
  output_size[0] = input_size[0];
  output_size[1] = weight_size[1] * groups;
  for (size_t d = 2; d < dim; ++d) {
    int kernel = dilation[d - 2] * (weight_size[d] - 1) + 1;
    output_size[d] = (input_size[d] - 1) * stride[d - 2] - (2 * padding[d - 2]) +
                     kernel + output_padding[d - 2];
  }
  return output_size;
}

}  // namespace ops
}  // namespace cnnl
}  // namespace torch_mlu
