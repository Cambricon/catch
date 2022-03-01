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

#include <algorithm>

#include "ATen/NativeFunctions.h"
#include "aten/operators/cnnl/internal/cnnl_internal.h"

namespace torch_mlu {
namespace cnnl {
namespace ops {

template <typename T>
T get_bound(at::optional<at::Scalar> input) {
  if (input.has_value())
    return input->to<T>();
  return 0;
}

template <>
uint16_t get_bound(at::optional<at::Scalar> input) {
  if (input.has_value()) {
    auto temp = input->to<float>();
    uint16_t result = 0;
    TORCH_CNRT_CHECK(cnrtConvertFloatToHalf(reinterpret_cast<uint16_t*>(&result), temp));
    return result;
  }
  return 0;
}

template<typename T>
at::Tensor& clip(at::Tensor &output,
                 const at::Tensor &self,
                 at::optional<at::Scalar> min,
                 at::optional<at::Scalar> max) {
  // get bound
  T min_bound = get_bound<T>(min);
  T max_bound = get_bound<T>(max);

  // get tensor impl
  auto self_impl = getMluTensorImpl(self);
  auto output_impl = getMluTensorImpl(output);

  // create the desc
  CnnlTensorDescriptor desc_self;
  desc_self.set(self);

  // get current handle
  auto handle = getCurrentHandle();

  // get the mlu ptr
  auto self_ptr = self_impl->cnnlMalloc();
  auto output_ptr = output_impl->cnnlMalloc();

  // compute ops
  TORCH_CNNL_CHECK(cnnlClip(handle,
                            desc_self.desc(),
                            self_ptr,
                            min.has_value()? static_cast<void*>(&min_bound): nullptr,
                            max.has_value()? static_cast<void*>(&max_bound): nullptr,
                            output_ptr));
  return output;
}

at::Tensor& cnnl_clamp_internal(at::Tensor &output,
                               const at::Tensor &self,
                               at::optional<at::Scalar> min,
                               at::optional<at::Scalar> max) {
  // as clamp.cpp all kernels need cnnl_clamp_internal, so add check in clamp_internal.cpp
  // to avoid duplicated add.
  // getCnnlDataType() not supported for c10::complex<float>, so here only check, python
  // exception test can't run into.
  TORCH_MLU_CHECK(!self.is_complex(), "clamp is not yet implemented for complex tensors.");
  // now mlu tensor only support Layout::Strided
  // python exception test alose can't test, "Unsupported device type for sparse layout: mlu"
  TORCH_MLU_CHECK(self.layout() == c10::Layout::Strided,
                    "clamp only supports strided layout, got: ", self.layout());
  std::unordered_map<std::string, int> dtype = {
    {"float", 1},
    {"int", 2},
    {"half", 3},
    {"c10::Half", 4}
  };
  switch (dtype[std::string(self.dtype().name())]) {
    case 1:
      return clip<float>(output, self, min, max);
    case 2:
      return clip<int>(output, self, min, max);
    case 3:
      return clip<uint16_t>(output, self, min, max);
    case 4:
      return clip<uint16_t>(output, self, min, max);
    default:
      auto self_cast = self.to(at::kFloat);
      auto output_cast = at::empty_like(self_cast);
      clip<float>(output_cast, self_cast, min, max);
      cnnl_cast_internal(output_cast, output);
      return output;
  }
}

}  // namespace ops
}  // namespace cnnl
}  // namespace torch_mlu
