#include "aten/operators/cnnl/cnnl_kernel.h"
#include "aten/operators/cnnl/internal/cnnl_internal.h"

namespace torch_mlu {
namespace cnnl {
namespace ops {

at::Tensor cnnl_alias(const at::Tensor& self) {
  at::Tensor self_;
  auto  impl = c10::make_intrusive<MLUTensorImpl>(at::Storage(self.storage()),
                                               self.key_set(),
                                               self.dtype());
  impl->set_storage_offset(self.storage_offset());
  impl->set_sizes_and_strides(self.sizes(), self.strides());
  self_ = at::Tensor(std::move(impl));
  return self_;
}

}  // namespace ops
}  // namespace cnnl
}  // namespace torch_mlu
