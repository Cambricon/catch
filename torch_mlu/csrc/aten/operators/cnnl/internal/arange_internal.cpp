/*
All modification made by Cambricon Corporation: © 2022 Cambricon Corporation
All rights reserved.
All other contributions:
Copyright (c) 2014--2022, the respective contributors
All rights reserved.
For the list of contributors go to https://github.com/pytorch/pytorch/graphs/contributors
Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:
    * Redistributions of source code must retain the above copyright notice,
      this list of conditions and the following disclaimer.
    * Redistributions in binary form must reproduce the above copyright
      notice, this list of conditions and the following disclaimer in the
      documentation and/or other materials provided with the distribution.
    * Neither the name of Intel Corporation nor the names of its contributors
      may be used to endorse or promote products derived from this software
      without specific prior written permission.
THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

#include <math.h>
#include "ATen/NativeFunctions.h"
#include "ATen/AccumulateType.h"
#include "aten/operators/cnnl/internal/cnnl_internal.h"

namespace at {
template <> struct AccumulateType<c10::Half, false> { using type = double; };
}

namespace torch_mlu {
namespace cnnl {
namespace ops {

template<class T>
at::Tensor& arange_dtype(at::Tensor &out,
                         T start,
                         T step) {
  T* start_ptr = &start;
  T* step_ptr = &step;

  auto out_impl = getMluTensorImpl(out);
  // get current handle
  auto handle = getCurrentHandle();
  CnnlTensorDescriptor desc_out;
  desc_out.set(out);
  // malloc mlu memory
  auto out_ptr = out_impl->cnnlMalloc();

  cnnlComputationPreference_t prefer = CNNL_COMPUTATION_HIGH_PRECISION;

  TORCH_CNNL_CHECK(cnnlArange_v2(handle,
                                 prefer,
                                 start_ptr,
                                 step_ptr,
                                 desc_out.desc(),
                                 out_ptr));
  return out;
}

at::Tensor& cnnl_arange_internal(at::Tensor &result,
                                 const at::Scalar start,
                                 const at::Scalar end,
                                 const at::Scalar step) {
  AT_DISPATCH_ALL_TYPES_AND(at::ScalarType::Half, result.scalar_type(), "arange_mlu", [&]() {
    using accscalar_t = at::acc_type<scalar_t, false>;
    auto xstart = start.to<accscalar_t>();
    auto xend = end.to<accscalar_t>();
    auto xstep = step.to<accscalar_t>();

    // the corner-case we do want to take into account is int64_t,
    // which has higher precision than double
    double size_d;
    if (std::is_same<scalar_t, int64_t>::value) {
      size_d = std::ceil(static_cast<double>(end.to<accscalar_t>() - start.to<accscalar_t>())
                         / step.to<accscalar_t>());
    } else {
      size_d = std::ceil(static_cast<double>(end.to<double>() - start.to<double>())
                         / step.to<double>());
    }

    TORCH_CHECK(xstep > 0 || xstep < 0, "step must be nonzero");
    TORCH_CHECK(std::isfinite(static_cast<double>(xstart)) &&
              std::isfinite(static_cast<double>(xend)),
              "unsupported range: ", xstart, " -> ", xend);
    TORCH_CHECK(((xstep > 0) && (xend >= xstart)) || ((xstep < 0) && (xend <= xstart)),
               "upper bound and larger bound inconsistent with step sign");
    TORCH_CHECK(size_d >= 0 && size_d <= static_cast<double>(std::numeric_limits<int64_t>::max()),
               "invalid size, possible overflow?");

    int64_t size = static_cast<int64_t>(size_d);
    int64_t numel = result.numel();
    auto result_new = at::empty({size}, result.options());
    if (numel != size) {
      if (numel > 0) {
        TORCH_WARN("The number of elements in the out tensor of shape ", result.sizes(),
                    " is ", numel, " which does not match the computed number of elements ", size,
                    ". Note that this may occur as a result of rounding error. "
                    "The out tensor will be resized to a tensor of shape (", size, ",).");
      }
      getMluTensorImpl(result)->copy_cnnl_metadata_from(getMluTensorImpl(result_new));
      resize_impl_mlu_(getMluTensorImpl(result), result_new.sizes(), result_new.strides());
    }
    result = result.is_contiguous() ? result : result.contiguous();

    std::unordered_map<std::string, int> dtype = {
      {"float", 1},
      {"int", 2},
      {"half", 3},
      {"c10::Half", 4}
    };
    switch (dtype[std::string(result.dtype().name())]) {
      case 1:
        arange_dtype<float>(result, xstart, xstep);
        break;
      case 2:
        arange_dtype<int>(result, xstart, xstep);
        break;
      case 3:
        arange_dtype<float>(result, xstart, xstep);
        break;
      case 4:
        arange_dtype<float>(result, xstart, xstep);
        break;
      default:
        auto result_float = at::empty(result.sizes(), result.options().dtype(at::kFloat));
        arange_dtype<float>(result_float, xstart, xstep);
        cnnl_cast_internal(result_float, result);
    }
  });
  return result;
}

}  // namespace ops
}  // namespace cnnl
}  // namespace torch_mlu
