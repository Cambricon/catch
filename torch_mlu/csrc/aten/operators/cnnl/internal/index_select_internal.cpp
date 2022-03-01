#include "aten/operators/cnnl/internal/cnnl_internal.h"

namespace torch_mlu {
namespace cnnl {
namespace ops {

at::Tensor& cnnl_index_select_internal(at::Tensor& output, const at::Tensor& input,
                                      int64_t dim, const at::Tensor& index) {
  auto input_impl = getMluTensorImpl(input);
  auto index_impl = getMluTensorImpl(index);
  auto output_impl = getMluTensorImpl(output);

  // get current handle
  auto handle = getCurrentHandle();
  CnnlTensorDescriptor desc_input;
  CnnlTensorDescriptor desc_index;
  CnnlTensorDescriptor desc_output;

  // get cnnl descriptor
  desc_input.set(input);
  desc_index.set(index);
  desc_output.set(output);

  // malloc mlu memory
  auto input_ptr = input_impl->cnnlMalloc();
  auto index_ptr = index_impl->cnnlMalloc();
  auto output_ptr = output_impl->cnnlMalloc();

  // set descriptor config
  TORCH_CNNL_CHECK(cnnlIndexSelect(handle, dim, desc_input.desc(),
                                   input_ptr, desc_index.desc(), index_ptr,
                                   desc_output.desc(), output_ptr));
  return output;
}

}  // namespace ops
}  // namespace cnnl
}  // namespace torch_mlu
