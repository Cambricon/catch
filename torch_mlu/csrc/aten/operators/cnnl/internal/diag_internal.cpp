#include "aten/operators/cnnl/internal/cnnl_internal.h"

namespace torch_mlu {
namespace cnnl {
namespace ops {

at::Tensor& cnnl_diag_internal(at::Tensor& output, const at::Tensor& input, int64_t k) {
  if (input.dim() == 0) {   // when cnnl support zero elements, we can rmv this
      return output;
  }
  CnnlTensorDescriptor input_desc;
  CnnlTensorDescriptor output_desc;
  input_desc.set(input);
  output_desc.set(output);
  // malloc mlu memory
  auto input_impl = getMluTensorImpl(input);
  auto output_impl = getMluTensorImpl(output);
  auto input_ptr = input_impl->cnnlMalloc();
  auto output_ptr = output_impl->cnnlMalloc();
  // set descriptor config
  auto handle = getCurrentHandle();
  TORCH_CNNL_CHECK(cnnlDiag(handle, k, input_desc.desc(), input_ptr,
    output_desc.desc(), output_ptr));
  return output;
}


}  // namespace ops
}  // namespace cnnl
}  // namespace torch_mlu
