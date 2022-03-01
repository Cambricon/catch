#include "aten/operators/cnnl/cnnl_kernel.h"
#include "aten/operators/cnnl/internal/cnnl_internal.h"

namespace torch_mlu {
namespace cnnl {
namespace ops {

at::Tensor cnnl_alias_internal(const at::Tensor& self) {
  auto impl = c10::make_intrusive<MLUTensorImpl>(
      c10::Storage(self.storage()), c10::DispatchKey::MLU, self.dtype());
  impl->set_storage_offset(self.storage_offset());
  impl->set_sizes_and_strides(self.sizes(), self.strides());
  auto output = at::Tensor(std::move(impl));
  return output;
}

}  // namespace ops
}  // namespace cnnl
}  // namespace torch_mlu
