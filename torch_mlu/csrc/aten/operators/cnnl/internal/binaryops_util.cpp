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
#include "aten/operators/cnnl/internal/binaryops_util.h"

namespace torch_mlu {
namespace cnnl {
namespace ops {

std::vector<at::ScalarType> type_vec = {at::kHalf, at::kFloat};

/*
 *  determine the type of computation.
 */
at::ScalarType get_compute_type(at::ScalarType data_type) {
  auto compute_type = at::kFloat;
  auto type_iter = find(type_vec.begin(), type_vec.end(), data_type);
  if (type_iter != type_vec.end()) {
    compute_type = data_type;
  }
  return compute_type;
}

/*
 * [Broadcast shape].
 * The input tensor dimensions are populated according to the broadcast rules.
 */
std::tuple<std::vector<int64_t>, std::vector<int64_t>> broadcast_shape(
        const at::Tensor& input, const at::Tensor& other) {
  std::vector<int64_t> tmp_input = input.sizes().vec();
  std::vector<int64_t> tmp_other = other.sizes().vec();
  if (other.dim() < input.dim()) {
    tmp_other.insert(tmp_other.begin(), input.dim() - other.dim(), 1);
  }
  if (other.dim() > input.dim()) {
    tmp_input.insert(tmp_input.begin(), other.dim() - input.dim(), 1);
  }
  return std::make_tuple(tmp_input, tmp_other);
}

/*
 * [Broadcast tensor].
 * return broadcast empty tensor.
 */
std::tuple<at::Tensor, at::Tensor> broadcast_tensor(
    const at::Tensor& input, const at::Tensor& other,
    at::ScalarType compute_type) {
  auto shape_broadcast = broadcast_shape(input, other);
  auto input_tensor =
      at::empty(std::get<0>(shape_broadcast), input.options().dtype(compute_type));
  auto other_tensor =
      at::empty(std::get<1>(shape_broadcast), other.options().dtype(compute_type));
  return std::make_tuple(input_tensor, other_tensor);
}

void reshapeTo(at::Tensor& output, const at::Tensor& input) {
    if (input.dim() != output.dim()) {
        output = input.reshape(output.sizes());
    } else {
        output = input;
    }
}

at::Tensor convertScalarToTensor(const at::Tensor& tensor,
                                 at::ScalarType ct_type) {
  if (is_mlu(tensor) || tensor.numel() != 1) {
    return tensor;
  }
  at::Scalar scalar = tensor.item();
  auto value = scalar.to<float>();
  at::Tensor result = at::native::full(
      {}, value, tensor.options().dtype(ct_type).device(at::kMLU));

  return result;
}

}  // namespace ops
}  // namespace cnnl
}  // namespace torch_mlu
