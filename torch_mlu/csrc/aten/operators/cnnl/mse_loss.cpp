#include "aten/operators/cnnl/cnnl_kernel.h"
#include "aten/operators/cnnl/internal/cnnl_internal.h"
#include <cmath>

namespace torch_mlu {
namespace cnnl {
namespace ops {

static const  auto mse_memory_format = c10::MemoryFormat::Contiguous;
at::Tensor cnnl_mse_loss(const at::Tensor& self, const at::Tensor& target,
                         int64_t reduction) {
  auto self_st = self.scalar_type(), target_st = target.scalar_type();
  TORCH_MLU_CHECK(
      (self_st == at::ScalarType::Float && target_st == at::ScalarType::Float)
      || (self_st == at::ScalarType::Half && target_st == at::ScalarType::Half),
      "cnnl_mse_loss is not implemented for types other than float and half, and"
      " mixed inputs of float and half are not supported!");

  if (self.numel() == 0 || target.numel() == 0) {
    auto output_size = self.numel() == 0 ? self.sizes().vec() : target.sizes().vec();
    if (reduction == 0) {
      // NONE;
      return at::empty(output_size, self.options(), c10::MemoryFormat::Contiguous);
    } else if (reduction == 1) {
      // MEAN;
      return at::full({}, std::numeric_limits<float>::quiet_NaN(), self.options());
    } else if (reduction == 2) {
      // SUM;
      return at::full({}, 0.0, self.options());
    } else {
      LOG(FATAL) << "mse_loss reduciton mode is unavaliable";
    }
  }
  
  auto self_contiguous = cnnl_contiguous(self, mse_memory_format);
  auto target_contiguous = cnnl_contiguous(target, mse_memory_format);
  return cnnl_mse_loss_internal(self_contiguous, target_contiguous, reduction);
}

at::Tensor cnnl_mse_loss_backward(const at::Tensor& grad_output,
                                  const at::Tensor& self,
                                  const at::Tensor& target, int64_t reduction) {
  auto self_st = self.scalar_type(), target_st = target.scalar_type();
  TORCH_MLU_CHECK(
      (self_st == at::ScalarType::Float && target_st == at::ScalarType::Float)
      || (self_st == at::ScalarType::Half && target_st == at::ScalarType::Half),
      "cnnl_mse_loss is not implemented for types other than float and half, and"
      " mixed inputs of float and half are not supported!");
  auto self_contiguous = cnnl_contiguous(self, mse_memory_format);
  auto target_contiguous = cnnl_contiguous(target, mse_memory_format);
  auto grad_output_contiguous = cnnl_contiguous(grad_output, mse_memory_format);
  auto grad_input = at::empty_like(self);
  return cnnl_mse_loss_backward_internal(grad_output_contiguous, self_contiguous, target_contiguous, reduction);
}

}  // namespace ops
}  // namespace cnnl
}  // namespace torch_mlu
