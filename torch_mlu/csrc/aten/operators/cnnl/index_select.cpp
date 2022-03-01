#include "aten/operators/cnnl/cnnl_kernel.h"
#include "aten/operators/cnnl/internal/cnnl_internal.h"

namespace torch_mlu {
namespace cnnl {
namespace ops {

namespace {
  std::vector<int64_t> calculate_output_size(const at::Tensor& input, int64_t dim,
                             const at::Tensor& index) {
    auto output_dims = input.sizes().vec();
    if (input.dim() > 0) {
        output_dims[dim] = index.numel();
    }
    return output_dims;
  }
}  // namespace

at::Tensor cnnl_index_select(const at::Tensor& self, int64_t dim,
                             const at::Tensor& index) {
  at::Tensor out = at::zeros({0}, self.options());
  return cnnl_index_select_out(out, self, dim, index);
}

at::Tensor& cnnl_index_select_out(at::Tensor& out, const at::Tensor& self,
                                  int64_t dim, const at::Tensor& index) {
  TORCH_MLU_CHECK(index.scalar_type() == at::ScalarType::Long,
      "index_select(): Expected dtype int64 for index");
  TORCH_MLU_CHECK(self.scalar_type() == out.scalar_type(),
      "index_select(): self and out must have the same scalar type");
  dim = at::maybe_wrap_dim(dim, self.dim());
  TORCH_MLU_CHECK(dim == 0 || dim < self.dim(),
      "index_select(): Indexing dim ", dim, " is out of bounds of tensor");

  // TODO(shangang): cnnl index select only supports float, half, int32, int16.
  // cast input to other dtype before cnnl indexselect support other dtype.
  std::vector<at::ScalarType> support_dtype = {at::kFloat, at::kHalf, at::kInt, at::kShort};
  auto input_dtype = self.scalar_type();
  auto it = std::find(support_dtype.begin(), support_dtype.end(), input_dtype);
  if (it == support_dtype.end()) {
    auto index_select_warp = [](at::Tensor& out,
                                const at::Tensor& self,
                                const int64_t dim,
                                const at::Tensor& index,
                                const at::ScalarType target_dtype) {
      auto other = at::empty(self.sizes(), self.options().dtype(target_dtype));
      auto self_contiguous = cnnl_contiguous(self, c10::MemoryFormat::Contiguous);
      cnnl_cast(self_contiguous, other);
      auto other_contiguous = cnnl_contiguous(other, c10::MemoryFormat::Contiguous);
      auto index_contiguous = cnnl_contiguous(index, c10::MemoryFormat::Contiguous);
      auto out_shape = calculate_output_size(other_contiguous, dim, index);
      at::Tensor internal_output = at::zeros(out_shape, other.options());
      cnnl_index_select_internal(internal_output, other_contiguous, dim, index_contiguous);
      resize_impl_mlu_(getMluTensorImpl(out), out_shape, c10::nullopt);
      cnnl_cast(internal_output, out);
    };
    if (at::isFloatingType(input_dtype)) {
      index_select_warp(out, self, dim, index, at::kFloat);
      return out;
    } else if (at::isIntegralType(input_dtype, true)) {
      index_select_warp(out, self, dim, index, at::kInt);
      return out;
    } else {
      CNLOG(ERROR) << "Index select op input dtype only support intergral type or floating type.";
    }
  } else {
    auto self_contiguous = cnnl_contiguous(self, c10::MemoryFormat::Contiguous);
    auto index_contiguous = cnnl_contiguous(index, c10::MemoryFormat::Contiguous);
    auto out_shape = calculate_output_size(self_contiguous, dim, index);
    resize_impl_mlu_(getMluTensorImpl(out), out_shape, c10::nullopt);
    return cnnl_index_select_internal(out, self_contiguous, dim, index_contiguous);
  }
}

}  // namespace ops
}  // namespace cnnl
}  // namespace torch_mlu
