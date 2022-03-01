#include "aten/operators/cnnl/cnnl_kernel.h"
#include "aten/operators/cnnl/internal/cnnl_internal.h"

namespace torch_mlu {
namespace cnnl {
namespace ops {

inline std::vector<int64_t> infer_output_size(const at::Tensor &self, int64_t k) {
  auto sizes = self.sizes();
  TORCH_MLU_CHECK(sizes.size() == 1 || sizes.size() == 2, "The input of cnnl_diag must be 1-d or",
    " 2-d, but got ", sizes.size());
  if (sizes.size() == 1) {   // input:1d -> output:2d
    int64_t sz = sizes[0] + (k >= 0 ? k : -k);
    return {sz, sz};
  } else {   // input:2d -> output:1d
    int64_t h = self.sizes()[0];
    int64_t w = self.sizes()[1];
    TORCH_MLU_CHECK((-1 * h + 1 <= k) && (k <= w - 1), "k must in row and col range of input");
    int64_t sz = k >= 0 ? std::min(sizes[0], sizes[1] - k) : std::min(sizes[0] + k, sizes[1]);
    return {sz};
  }
}

at::Tensor cnnl_diag(const at::Tensor& self, int64_t diagonal) {
  auto self_contiguous = cnnl_contiguous(self, self.suggest_memory_format());
  auto output = at::zeros(infer_output_size(self_contiguous, diagonal), self_contiguous.options());
  return cnnl_diag_internal(output, self_contiguous, diagonal);
}

at::Tensor& cnnl_diag_out(at::Tensor& out, const at::Tensor& self, int64_t diagonal) {
  TORCH_MLU_CHECK(out.scalar_type() == self.scalar_type(), "The datatype of out in cnnl_diag_out "
    "must be same as self ", self.scalar_type(), " but out ", out.scalar_type());

  auto self_contiguous = cnnl_contiguous(self, self.suggest_memory_format());
  auto out_shape = infer_output_size(self_contiguous, diagonal);
  auto out_numel = out_shape.size() == 1 ? out_shape[0] : out_shape[0] * out_shape[1];
  TORCH_MLU_CHECK(out_numel >= 0, "The expected output numel of cnnl_diag_out must not be negative,"
    " but got ", out_numel);
  auto out_impl = getMluTensorImpl(out);
  if (out_impl->numel() >= out_numel) {
    resize_impl_mlu_(out_impl, out_shape, c10::nullopt);
    return cnnl_diag_internal(out, self, diagonal);
  }
  auto output = at::zeros(out_shape, self_contiguous.options());
  cnnl_diag_internal(output, self, diagonal);
  out_impl->copy_cnnl_metadata_from(getMluTensorImpl(output));
  resize_impl_mlu_(out_impl, output.sizes(), c10::nullopt);
  return out;
}

}  // namespace ops
}  // namespace cnnl
}  // namespace torch_mlu
