#include <algorithm>
#include "ATen/NativeFunctions.h"

#include "aten/operators/cnnl/internal/cnnl_internal.h"

namespace torch_mlu {
namespace cnnl {
namespace ops {

static std::vector<int64_t> conv_output_size(at::IntList input_size,
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


static at::Tensor cnnl_depth_transpose(const at::Tensor &input,
                                CnnlTransposeDescriptor &trans_desc,
                                cnnlTensorLayout_t layout) {
  /*
   * When layout == CNNL_LAYOUT_NHWC, it means that input's layout is HWCN and
   * will
   * be transposed to NHWC;
   * On other hand, if layout == CNNL_LAYOUT_HWCN, it means that input's layout
   * is
   * NHWC and will be transposed to HWCN.
   *
   */
  std::vector<int> order(input.dim(), 0);
  for (int i = 0; i < input.dim(); ++i) {
    order[i] = i;
  }
  auto input_sizes = input.sizes().vec();
  CnnlTensorDescriptor input_desc;
  CnnlTensorDescriptor output_desc;
  if (layout == CNNL_LAYOUT_NHWC) {
    for (int i = 0; i < input.dim(); ++i) {
      order[i] = (i + input.dim() - 1) % (input.dim());
    }
    input_desc.set(input, CNNL_LAYOUT_HWCN);
    output_desc.set(input, CNNL_LAYOUT_NHWC);
  } else {
    for (int i = 0; i < input.dim(); i++) {
      order[i] = (i + 1) % input.dim();
    }
    input_desc.set(input, CNNL_LAYOUT_NHWC);
    output_desc.set(input, CNNL_LAYOUT_HWCN);
  }
  // prepare cnnl transpose input
  trans_desc.set(input.dim(), order.data());
  auto input_impl = getMluTensorImpl(input);
  auto input_ptr = input_impl->cnnlMalloc();
  // prepare cnnl transpose output
  auto output = at::empty_like(input);
  auto output_impl = getMluTensorImpl(output);
  auto output_ptr = output_impl->cnnlMalloc();
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
                                input.options().dtype(at::kByte));
    auto workspace_impl = getMluTensorImpl(trans_workspace);
    workspace_ptr = workspace_impl->cnnlMalloc();
  }
  TORCH_CNNL_CHECK(cnnlTranspose_v2(handle, trans_desc.desc(), input_desc.desc(),
                                 input_ptr, output_desc.desc(), output_ptr,
                                 workspace_ptr, workspace_size));
  return output;
}

at::Tensor check_depth_transpose(const at::Tensor &input,
                                 CnnlTransposeDescriptor &trans_desc,
                                 cnnlTensorLayout_t layout) {
  return cnnl_depth_transpose(input, trans_desc, layout);
}

}  // namespace ops
}  // namespace cnnl
}  // namespace torch_mlu

