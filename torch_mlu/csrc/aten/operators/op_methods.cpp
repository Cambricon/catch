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

#include <ATen/native/Resize.h>
#include <ATen/native/TensorFactories.h>
#include "aten/operators/op_methods.h"
#include "aten/core/tensor_impl.h"
#include "aten/device/queue.h"
#include "aten/util/cnlog.h"
#include "aten/util/common.h"
#include "aten/util/tensor_util.h"
#include "aten/device/mlu_guard.h"
#include "aten/util/memory_allocator.h"
#include "aten/operators/cnnl/cnnl_kernel.h"

namespace torch_mlu {

// This macro is used to print useful informations
#define TRACE_INFO(OP) \
  CNLOG(WARNING) << "OpMethods::" << OP << " Op running on CPU!";

OpMethods::OpMethods() { init(); }

OpMethods::~OpMethods() {}

void OpMethods::init() {
  // Init MLU Device
  if (GET_MLU_DEVICE < 0) return;
  unsigned dev_count;
  cnrtGetDeviceCount(&dev_count);
  TORCH_CHECK(GET_MLU_DEVICE < dev_count, "Invalid device number");
  // Set MLU device
  setDevice(GET_MLU_DEVICE);

  CNLOG(INFO) << "Using MLU device: " << GET_MLU_DEVICE;
  CNLOG(INFO) << "Catch RunningMode: "
      << PythonInterface::getRunningModeString();
  // init task mode
  if (PythonInterface::getRunningMode() == RunningMode::CNNL &&
      PythonInterface::getAsyncMode() == false) {
    CNLOG(INFO) << "Catch TaskMode: synchronous.";
  }
}

at::Tensor OpMethods::empty(at::IntArrayRef size,
                            const at::TensorOptions& options_,
                            c10::optional<at::MemoryFormat> optional_memory_format) {
  TORCH_MLU_CHECK(
    !(options_.has_memory_format() && optional_memory_format.has_value()),
    "Cannot set memory_format both in TensorOptions and explicit argument; please delete "
    "the redundant setter.");
  at::TensorOptions options =
    options_.merge_in(at::TensorOptions().memory_format(optional_memory_format));
  TORCH_MLU_CHECK(options.device().type() == c10::DeviceType::MLU);
  at::native::check_size_nonnegative(size);
  auto* allocator = dynamic_cast<MLUCachingAllocator*>(getMLUCachingAllocator());
  int64_t nelements = at::prod_intlist(size);
  auto dtype = options.dtype();
  auto device = options.device();
  int64_t size_bytes = nelements * dtype.itemsize();
  torch_mlu::mlu::MLUGuard guard(device);
  auto device_id = current_device();
  auto storage_impl = c10::make_intrusive<c10::StorageImpl>(
      c10::StorageImpl::use_byte_size_t(),
      size_bytes,
      allocator->allocate(size_bytes, device_id),
      allocator,
      /*resizable=*/true);

  auto tensor = at::detail::make_tensor<MLUTensorImpl>(
      storage_impl, c10::DispatchKey::MLU, dtype);

  // Default at::TensorImpl has size [0]
  if (size.size() != 1 || size[0] != 0) {
    tensor.unsafeGetTensorImpl()->set_sizes_contiguous(size);
  }

  auto memory_format = options.memory_format_opt().value_or(at::MemoryFormat::Contiguous);
  tensor.unsafeGetTensorImpl()->empty_tensor_restride(memory_format);

  return tensor;
}

at::Tensor OpMethods::empty_strided(at::IntArrayRef size,
                                    at::IntArrayRef stride,
                                    const at::TensorOptions& options) {
  // This operator will be called more frequently than at::empty after pytorch1.5
  // because of the torch.preserve_format. Do not call "TRACE_INFO(__FUNCTION__)"
  // for that it runs on MLU.
  for (auto x : size) {
    TORCH_CHECK(x >= 0, "Trying to create tensor with negative dimension ", x, ": ", size);
  }
  // Currently MLU tensor doesn't support stride mode and always to be
  // contiguous whether stride is set or not, unlike CPU and CUDA tensors.
  // This allows MLU tensors to adjust a tensor's memory layout for better
  // performance.
  auto t = at::empty(size, options);
  resize_impl_mlu_(getMluTensorImpl(t), size, stride);
  return t;
}
// (TODO) copy_ behavior need be same with cpu copy_
at::Tensor& OpMethods::copy_(at::Tensor& self, const at::Tensor& src,
                             bool non_blocking) {
  TRACE_INFO(__FUNCTION__);
  TORCH_CHECK(false, "copy tensor from ", src.device(), " to ", self.device(), " failed!");
  return self;
}

at::Tensor& OpMethods::set_storage_(at::Tensor &self, c10::Storage source, int64_t storage_offset,
  at::IntArrayRef size, at::IntArrayRef stride) {
  TRACE_INFO(__FUNCTION__);
  throw std::invalid_argument("MLU set_storage_ do not have OpMethods implementation!");
}

at::Tensor OpMethods::add(const at::Tensor& self, const at::Tensor& other,
                          at::Scalar alpha) {
  TRACE_INFO(__FUNCTION__);
  auto output_cpu = at::add(self.cpu(), other.cpu(), alpha);
  return output_cpu.to(at::Device(at::Device::Type::MLU));
}

at::Tensor& OpMethods::add_(at::Tensor& self, const at::Tensor& other,
                            at::Scalar alpha) {
  TRACE_INFO(__FUNCTION__);
  auto input_cpu = self.cpu();
  auto other_cpu = other.cpu();
  input_cpu.add_(other_cpu, alpha);
  auto out_mlu = input_cpu.to(at::Device(at::Device::Type::MLU));
  getMluTensorImpl(self)->copy_cnnl_metadata_from(getMluTensorImpl(out_mlu));
  return self;
}

at::Tensor OpMethods::addmm(const at::Tensor& self, const at::Tensor& mat1,
                            const at::Tensor& mat2, at::Scalar beta,
                            at::Scalar alpha) {
  TRACE_INFO(__FUNCTION__);
  auto self_cpu = self.cpu();
  auto mat1_cpu = mat1.cpu();
  auto mat2_cpu = mat2.cpu();
  auto output_cpu = at::addmm(self_cpu, mat1_cpu, mat2_cpu, beta, alpha);
  return output_cpu.to(at::Device(at::Device::Type::MLU));
}

at::Tensor& OpMethods::addmm_(at::Tensor & self, const at::Tensor & mat1,
                             const at::Tensor & mat2, at::Scalar beta,
                             at::Scalar alpha) {
  TRACE_INFO(__FUNCTION__);
  auto self_cpu = self.cpu();
  auto mat1_cpu = mat1.cpu();
  auto mat2_cpu = mat2.cpu();
  self_cpu.addmm_(mat1_cpu, mat2_cpu, beta, alpha);
  auto out_mlu = self_cpu.to(at::Device(at::Device::Type::MLU));
  getMluTensorImpl(self)->copy_cnnl_metadata_from(getMluTensorImpl(out_mlu));
  return self;
}

at::Tensor& OpMethods::addmm_out(at::Tensor & out, const at::Tensor & self, const at::Tensor & mat1,
                             const at::Tensor & mat2, at::Scalar beta, at::Scalar alpha) {
  TRACE_INFO(__FUNCTION__);
  auto out_cpu = out.cpu();
  auto self_cpu = self.cpu();
  auto mat1_cpu = mat1.cpu();
  auto mat2_cpu = mat2.cpu();
  at::addmm_out(out_cpu, self_cpu, mat1_cpu, mat2_cpu, beta, alpha);
  auto out_mlu = out_cpu.to(at::Device(at::Device::Type::MLU));
  getMluTensorImpl(out)->copy_cnnl_metadata_from(getMluTensorImpl(out_mlu));
  resize_impl_mlu_(getMluTensorImpl(out), out_mlu.sizes(), out_mlu.strides());
  return out;
}

at::Tensor OpMethods::all(const at::Tensor& self, int64_t dim,
                          bool keepdim) {
  TRACE_INFO(__FUNCTION__);
  auto input_cpu = self.cpu();
  auto output_cpu = at::all(input_cpu, dim, keepdim);
  return output_cpu.to(at::Device(at::Device::Type::MLU));
}

at::Tensor OpMethods::all(const at::Tensor& self) {
  TRACE_INFO(__FUNCTION__);
  auto input_cpu = self.cpu();
  auto output_cpu = at::all(input_cpu);
  return output_cpu.to(at::Device(at::Device::Type::MLU));
}

at::Tensor& OpMethods::all_out(at::Tensor& out, const at::Tensor& self,
                              int64_t dim, bool keepdim) {
  TRACE_INFO(__FUNCTION__);
  auto out_cpu = out.cpu();
  auto input_cpu = self.cpu();
  at::all_out(out_cpu, input_cpu, dim, keepdim);
  auto out_mlu = out_cpu.to(at::Device(at::Device::Type::MLU));
  getMluTensorImpl(out)->copy_cnnl_metadata_from(getMluTensorImpl(out_mlu));
  resize_impl_mlu_(getMluTensorImpl(out), out_mlu.sizes(), out_mlu.strides());
}

at::Tensor OpMethods::any(const at::Tensor& self, const int64_t dim,
                          const bool keepdim) {
  TRACE_INFO(__FUNCTION__);
  auto input_cpu = self.cpu();
  auto output_cpu = at::any(input_cpu, dim, keepdim);
  return output_cpu.to(at::Device(at::Device::Type::MLU));
}

at::Tensor OpMethods::any(const at::Tensor& self) {
  TRACE_INFO(__FUNCTION__);
  auto input_cpu = self.cpu();
  auto output_cpu = at::any(input_cpu);
  return output_cpu.to(at::Device(at::Device::Type::MLU));
}

at::Tensor OpMethods::prod(const at::Tensor& self,
                           c10::optional<c10::ScalarType> dtype) {
  TRACE_INFO(__FUNCTION__);
  auto input_cpu = self.cpu();
  auto output_cpu = at::prod(input_cpu);
  return output_cpu.to(at::Device(at::Device::Type::MLU));
}

at::Tensor OpMethods::prod(const at::Tensor& self,
                           const int64_t dim,
                           bool keepdim,
                           c10::optional<c10::ScalarType> dtype) {
  TRACE_INFO(__FUNCTION__);
  auto input_cpu = self.cpu();
  auto output_cpu = at::prod(input_cpu, dim, keepdim);
  return output_cpu.to(at::Device(at::Device::Type::MLU));
}

at::Tensor& OpMethods::prod_out(at::Tensor& result, const at::Tensor& self,
                                const int64_t dim, bool keepdim,
                                c10::optional<c10::ScalarType> dtype) {
  TRACE_INFO(__FUNCTION__);
  auto input_cpu = self.cpu();
  auto result_cpu = result.cpu();
  at::prod_out(result_cpu, input_cpu, dim, keepdim);
  auto out_mlu = result_cpu.to(at::Device(at::Device::Type::MLU));
  getMluTensorImpl(result)->copy_cnnl_metadata_from(getMluTensorImpl(out_mlu));
  resize_impl_mlu_(getMluTensorImpl(result), out_mlu.sizes(), out_mlu.strides());
}

at::Tensor OpMethods::gather(const at::Tensor& self, int64_t dim,
                             const at::Tensor& index, bool sparse_grad) {
  TRACE_INFO(__FUNCTION__);
  auto input_cpu = self.cpu();
  auto index_cpu = index.cpu();
  auto output = at::gather(input_cpu, dim, index_cpu);
  return output.to(at::Device(at::Device::Type::MLU));
}

at::Tensor& OpMethods::gather_out(at::Tensor& out, const at::Tensor& self, int64_t dim,
                                 const at::Tensor& index, bool sparse_grad) {
  TRACE_INFO(__FUNCTION__);
  auto out_cpu = out.cpu();
  auto input_cpu = self.cpu();
  auto index_cpu = index.cpu();
  at::gather_out(out_cpu, input_cpu, dim, index_cpu);
  auto out_mlu = out_cpu.to(at::Device(at::Device::Type::MLU));
  getMluTensorImpl(out)->copy_cnnl_metadata_from(getMluTensorImpl(out_mlu));
  resize_impl_mlu_(getMluTensorImpl(out), out_mlu.sizes(), out_mlu.strides());
  return out;
}

std::tuple<at::Tensor, at::Tensor, at::Tensor>
OpMethods::native_batch_norm_backward(
    const at::Tensor& grad_out, const at::Tensor& input,
    const at::Tensor& weight, const at::Tensor& running_mean,
    const at::Tensor& running_var, const at::Tensor& save_mean_mean,
    const at::Tensor& save_invstd, bool training, double eps,
    std::array<bool, 3> output_mask) {
  TRACE_INFO(__FUNCTION__);
  auto grad_out_cpu = grad_out.cpu();
  auto input_cpu = input.cpu();
  auto weight_cpu = weight.cpu();
  auto running_mean_cpu = running_mean.cpu();
  auto running_var_cpu = running_var.cpu();
  auto save_mean_mean_cpu = save_mean_mean.cpu();
  auto save_invstd_cpu = save_invstd.cpu();
  auto output = at::native_batch_norm_backward(grad_out_cpu, input_cpu, weight_cpu,
                running_mean_cpu, running_var_cpu, save_mean_mean_cpu, save_invstd_cpu, training,
                eps, output_mask);
  return std::make_tuple(std::get<0>(output).to(at::Device(at::Device::Type::MLU)),
                         std::get<1>(output).to(at::Device(at::Device::Type::MLU)),
                         std::get<2>(output).to(at::Device(at::Device::Type::MLU)));
}

at::Tensor OpMethods::avg_pool2d_backward(
    const at::Tensor& grad_output, const at::Tensor& self,
    at::IntArrayRef kernel_size, at::IntArrayRef stride,
    at::IntArrayRef padding, bool ceil_mode, bool count_include_pad,
    c10::optional<int64_t> divisor_override) {
  TRACE_INFO(__FUNCTION__);
  auto grad_cpu = grad_output.cpu();
  auto self_cpu = self.cpu();
  auto grad_output_cpu =
      at::avg_pool2d_backward(grad_cpu, self_cpu, kernel_size, stride, padding,
                              ceil_mode, count_include_pad, divisor_override);
  return grad_output_cpu.to(at::Device(at::Device::Type::MLU));
}

at::Tensor OpMethods::max_pool2d_backward(
    const at::Tensor& grad_output, const at::Tensor& self,
    at::IntArrayRef kernel_size, at::IntArrayRef stride,
    at::IntArrayRef padding, at::IntArrayRef dilation, bool ceil_mode,
    const at::Tensor& indices) {
  TRACE_INFO(__FUNCTION__);
  auto grad_cpu = grad_output.cpu();
  auto self_cpu = self.cpu();
  auto indices_cpu = indices.cpu();
  auto grad_output_cpu = at::max_pool2d_with_indices_backward(
      grad_cpu, self_cpu, kernel_size, stride, padding, dilation, ceil_mode,
      indices_cpu);
  return grad_output_cpu.to(at::Device(at::Device::Type::MLU));
}

at::Scalar OpMethods::local_scalar_dense(const at::Tensor& input) {
  TRACE_INFO(__FUNCTION__);
  return input.cpu().item();
}

at::Tensor OpMethods::index_select(const at::Tensor& self, int64_t dim,
                                   const at::Tensor& index) {
  TRACE_INFO(__FUNCTION__);
  auto input_cpu = self.cpu();
  auto index_cpu = index.cpu();
  auto output = at::index_select(input_cpu, dim, index_cpu);
  return output.to(at::Device(at::Device::Type::MLU));
}

at::Tensor& OpMethods::index_fill_(at::Tensor & self, int64_t dim,
        const at::Tensor & index, at::Scalar value) {
    TRACE_INFO(__FUNCTION__);
    auto input_cpu = self.cpu();
    auto index_cpu = index.cpu();
    auto output = input_cpu.index_fill_(dim, index_cpu, value);
    auto out_mlu = output.to(at::Device(at::Device::Type::MLU));
    getMluTensorImpl(self)->copy_cnnl_metadata_from(getMluTensorImpl(out_mlu));
    return self;
}

at::Tensor& OpMethods::index_put_(at::Tensor & self, at::TensorList indices,
        const at::Tensor & values, bool accumulate) {
  TRACE_INFO(__FUNCTION__);
  auto self_cpu = self.cpu();
  auto values_cpu = values.cpu();
  int size = indices.size();
  std::vector<at::Tensor> vec_cpu;
  for (int i = 0; i < size; i++) {
    if (indices[i].defined()) {
      vec_cpu.push_back(indices[i].cpu());
    } else {
      vec_cpu.push_back(indices[i]);
    }
  }
  auto indices_mlu = at::TensorList(vec_cpu);
  at::index_put_(self_cpu, indices_mlu, values_cpu, accumulate);
  self.copy_(self_cpu);
  return self;
}

at::Tensor OpMethods::index_put(const at::Tensor & self, at::TensorList indices,
        const at::Tensor & values, bool accumulate) {
  TRACE_INFO(__FUNCTION__);
  auto self_cpu = self.cpu();
  auto values_cpu = values.cpu();
  int size = indices.size();
  std::vector<at::Tensor> vec_cpu;
  for (int i = 0; i < size; i++) {
    if (indices[i].defined()) {
      vec_cpu.push_back(indices[i].cpu());
    } else {
      vec_cpu.push_back(indices[i]);
    }
  }
  auto indices_mlu = at::TensorList(vec_cpu);
  auto output = at::index_put(self_cpu, indices_mlu, values_cpu, accumulate);
  return output.to(at::Device(at::Device::Type::MLU));
}

at::Tensor& OpMethods::index_select_out(at::Tensor& out, const at::Tensor& self,
                                        int64_t dim, const at::Tensor& index) {
  TRACE_INFO(__FUNCTION__);
  auto out_cpu = out.cpu();
  auto input_cpu = self.cpu();
  auto index_cpu = index.cpu();
  at::index_select_out(out_cpu, input_cpu, dim, index_cpu);
  auto out_mlu = out_cpu.to(at::Device(at::Device::Type::MLU));
  getMluTensorImpl(out)->copy_cnnl_metadata_from(getMluTensorImpl(out_mlu));
  resize_impl_mlu_(getMluTensorImpl(out), out_mlu.sizes(), out_mlu.strides());
  return out;
}

at::Tensor& OpMethods::_index_put_impl_(at::Tensor & self, at::TensorList indices,
        const at::Tensor & values, bool accumulate, bool unsafe) {
  TRACE_INFO(__FUNCTION__);
  auto self_cpu = self.cpu();
  auto values_cpu = values.cpu();
  int size = indices.size();
  std::vector<at::Tensor> vec_cpu;
  for (int i = 0; i < size; i++) {
    if (indices[i].defined()) {
      vec_cpu.push_back(indices[i].cpu());
    } else {
      vec_cpu.push_back(indices[i]);
    }
  }
  auto indices_mlu = at::TensorList(vec_cpu);
  at::_index_put_impl_(self_cpu, indices_mlu, values_cpu, accumulate, unsafe);
  self.copy_(self_cpu);
  return self;
}

at::Tensor OpMethods::add(const at::Tensor& self, at::Scalar other,
                          at::Scalar alpha) {
  TRACE_INFO(__FUNCTION__);
  auto output_cpu = at::add(self.cpu(), other, alpha);
  return output_cpu.to(at::Device(at::Device::Type::MLU));
}

at::Tensor& OpMethods::add_(at::Tensor& self, at::Scalar other,
                            at::Scalar alpha) {
  TRACE_INFO(__FUNCTION__);
  auto self_cpu = self.cpu();
  self_cpu.add_(other, alpha);
  getMluTensorImpl(self)->set_cpu_storage(
      self_cpu.unsafeGetTensorImpl()->storage());
  return self;
}

at::Tensor OpMethods::abs(const at::Tensor& self) {
  TRACE_INFO(__FUNCTION__);
  auto input_cpu = self.cpu();
  auto output = at::abs(input_cpu);
  return output.to(at::Device(at::Device::Type::MLU));
}

at::Tensor& OpMethods::abs_(at::Tensor& self) {
  TRACE_INFO(__FUNCTION__);
  auto input_cpu = self.cpu();
  at::abs_(input_cpu);
  getMluTensorImpl(self)->set_cpu_storage(
      input_cpu.unsafeGetTensorImpl()->storage());
  return self;
}

at::Tensor& OpMethods::abs_out(at::Tensor& out, const at::Tensor& self) {
  TRACE_INFO(__FUNCTION__);
  auto out_cpu = out.cpu();
  auto input_cpu = self.cpu();
  auto output = at::abs_out(out_cpu, input_cpu);
  auto out_mlu = output.to(at::Device(at::Device::Type::MLU));
  getMluTensorImpl(out)->copy_cnnl_metadata_from(getMluTensorImpl(out_mlu));
  resize_impl_mlu_(getMluTensorImpl(out), out_mlu.sizes(), out_mlu.strides());
  return out;
}

at::Tensor OpMethods::ne(const at::Tensor& self, const at::Tensor& other) {
  TRACE_INFO(__FUNCTION__);
  auto input_cpu = self.cpu();
  auto other_cpu = other.cpu();
  auto output = at::ne(input_cpu, other_cpu);
  return output.to(at::Device(at::Device::Type::MLU));
}

at::Tensor OpMethods::ne(const at::Tensor& self, at::Scalar other) {
  TRACE_INFO(__FUNCTION__);
  auto input_cpu = self.cpu();
  auto output = at::ne(input_cpu, other);
  return output.to(at::Device(at::Device::Type::MLU));
}

at::Tensor& OpMethods::ne_(at::Tensor& self, const at::Tensor& other) {
  TRACE_INFO(__FUNCTION__);
  auto self_cpu = self.cpu();
  auto other_cpu = other.cpu();
  at::native::ne_(self_cpu, other_cpu);
  auto self_mlu = self_cpu.to(at::Device(at::Device::Type::MLU));
  getMluTensorImpl(self)->copy_cnnl_metadata_from(getMluTensorImpl(self_mlu));
  return self;
}

at::Tensor& OpMethods::ne_(at::Tensor& self, at::Scalar other) {
  TRACE_INFO(__FUNCTION__);
  auto self_cpu = self.cpu();
  at::native::ne_(self_cpu, other);
  auto self_mlu = self_cpu.to(at::Device(at::Device::Type::MLU));
  getMluTensorImpl(self)->copy_cnnl_metadata_from(getMluTensorImpl(self_mlu));
  return self;
}

at::Tensor& OpMethods::ne_out(at::Tensor& out, const at::Tensor& self,
                                      const at::Tensor& other) {
  TRACE_INFO(__FUNCTION__);
  auto out_cpu = out.cpu();
  auto self_cpu = self.cpu();
  auto other_cpu = other.cpu();
  at::ne_out(out_cpu, self_cpu, other_cpu);
  auto out_mlu = out_cpu.to(at::Device(at::Device::Type::MLU));
  getMluTensorImpl(out)->copy_cnnl_metadata_from(getMluTensorImpl(out_mlu));
  resize_impl_mlu_(getMluTensorImpl(out), out_mlu.sizes(), out_mlu.strides());
  return out;
}

at::Tensor& OpMethods::ne_out(at::Tensor& out, const at::Tensor& self, at::Scalar other) {
  TRACE_INFO(__FUNCTION__);
  auto out_cpu = out.cpu();
  auto self_cpu = self.cpu();
  at::ne_out(out_cpu, self_cpu, other);
  auto out_mlu = out_cpu.to(at::Device(at::Device::Type::MLU));
  getMluTensorImpl(out)->copy_cnnl_metadata_from(getMluTensorImpl(out_mlu));
  resize_impl_mlu_(getMluTensorImpl(out), out_mlu.sizes(), out_mlu.strides());
  return out;
}

at::Tensor OpMethods::linspace(at::Scalar start, at::Scalar end, int64_t steps,
                               const at::TensorOptions & options) {
  TRACE_INFO(__FUNCTION__);
  auto input_options = options.device(at::Device(at::Device::Type::CPU));
  auto output = at::linspace(start, end, steps, input_options);
  return output.to(at::Device(at::Device::Type::MLU));
}

at::Tensor & OpMethods::linspace_out(at::Tensor& out, at::Scalar start,
                                     at::Scalar end, int64_t steps) {
  TRACE_INFO(__FUNCTION__);
  auto out_cpu = out.cpu();
  at::linspace_out(out_cpu, start, end, steps);
  auto out_mlu = out_cpu.to(at::Device(at::Device::Type::MLU));
  getMluTensorImpl(out)->copy_cnnl_metadata_from(getMluTensorImpl(out_mlu));
  resize_impl_mlu_(getMluTensorImpl(out), out_mlu.sizes(), out_mlu.strides());
}

at::Tensor OpMethods::gt(const at::Tensor& self, const at::Tensor& other) {
  TRACE_INFO(__FUNCTION__);
  auto input_cpu = self.cpu();
  auto other_cpu = other.cpu();
  auto output = at::gt(input_cpu, other_cpu);
  return output.to(at::Device(at::Device::Type::MLU));
}

at::Tensor OpMethods::gt(const at::Tensor& self, at::Scalar other) {
  TRACE_INFO(__FUNCTION__);
  auto input_cpu = self.cpu();
  auto output = at::gt(input_cpu, other);
  return output.to(at::Device(at::Device::Type::MLU));
}

at::Tensor& OpMethods::gt_(at::Tensor& self, const at::Tensor& other) {
  TRACE_INFO(__FUNCTION__);
  auto self_cpu = self.cpu();
  auto other_cpu = other.cpu();
  at::native::gt_(self_cpu, other_cpu);
  auto self_mlu = self_cpu.to(at::Device(at::Device::Type::MLU));
  getMluTensorImpl(self)->copy_cnnl_metadata_from(getMluTensorImpl(self_mlu));
  return self;
}

at::Tensor& OpMethods::gt_(at::Tensor& self, at::Scalar other) {
  TRACE_INFO(__FUNCTION__);
  auto self_cpu = self.cpu();
  at::native::gt_(self_cpu, other);
  auto self_mlu = self_cpu.to(at::Device(at::Device::Type::MLU));
  getMluTensorImpl(self)->copy_cnnl_metadata_from(getMluTensorImpl(self_mlu));
  return self;
}

at::Tensor & OpMethods::gt_out(at::Tensor& out, const at::Tensor& self,
                                      const at::Tensor& other) {
  TRACE_INFO(__FUNCTION__);
  auto out_cpu = out.cpu();
  auto self_cpu = self.cpu();
  auto other_cpu = other.cpu();
  at::gt_out(out_cpu, self_cpu, other_cpu);
  auto out_mlu = out_cpu.to(at::Device(at::Device::Type::MLU));
  getMluTensorImpl(out)->copy_cnnl_metadata_from(getMluTensorImpl(out_mlu));
  resize_impl_mlu_(getMluTensorImpl(out), out_mlu.sizes(), out_mlu.strides());
  return out;
}

at::Tensor & OpMethods::gt_out(at::Tensor& out, const at::Tensor& self, at::Scalar other) {
  TRACE_INFO(__FUNCTION__);
  auto out_cpu = out.cpu();
  auto self_cpu = self.cpu();
  at::gt_out(out_cpu, self_cpu, other);
  auto out_mlu = out_cpu.to(at::Device(at::Device::Type::MLU));
  getMluTensorImpl(out)->copy_cnnl_metadata_from(getMluTensorImpl(out_mlu));
  resize_impl_mlu_(getMluTensorImpl(out), out_mlu.sizes(), out_mlu.strides());
  return out;
}

at::Tensor& OpMethods::bitwise_and_out(at::Tensor& result,
                                                  const at::Tensor& self,
                                                  const at::Tensor& other) {
  TRACE_INFO(__FUNCTION__);
  auto result_cpu = result.cpu();
  auto input_cpu = self.cpu();
  auto other_cpu = other.cpu();
  at::bitwise_and_out(result_cpu, input_cpu, other_cpu);
  auto out_mlu = result_cpu.to(at::Device(at::Device::Type::MLU));
  getMluTensorImpl(result)->copy_cnnl_metadata_from(getMluTensorImpl(out_mlu));
  resize_impl_mlu_(getMluTensorImpl(result), out_mlu.sizes(), out_mlu.strides());
  return result;
}

at::Tensor& OpMethods::bitwise_and_out(at::Tensor& result,
                                                  const at::Tensor& self,
                                                  at::Scalar other) {
  TRACE_INFO(__FUNCTION__);
  auto result_cpu = result.cpu();
  auto input_cpu = self.cpu();
  at::bitwise_and_out(result_cpu, input_cpu, other);
  auto out_mlu = result_cpu.to(at::Device(at::Device::Type::MLU));
  getMluTensorImpl(result)->copy_cnnl_metadata_from(getMluTensorImpl(out_mlu));
  resize_impl_mlu_(getMluTensorImpl(result), out_mlu.sizes(), out_mlu.strides());
  return result;
}

at::Tensor& OpMethods::fill_(at::Tensor& self, const at::Tensor& other) {
  TRACE_INFO(__FUNCTION__);
  auto input_cpu = self.cpu();
  auto other_cpu = other.cpu();
  at::native::fill_(input_cpu, other_cpu);
  // copy cpu result into mlu self cpu_storage
  getMluTensorImpl(self)->set_cpu_storage(
      input_cpu.unsafeGetTensorImpl()->storage());
  return self;
}

at::Tensor& OpMethods::fill_(at::Tensor& self, at::Scalar other) {
  TRACE_INFO(__FUNCTION__);
  auto input_cpu = self.cpu();
  at::native::fill_(input_cpu, other);
  // copy cpu result into mlu self cpu_storage
  getMluTensorImpl(self)->set_cpu_storage(
      input_cpu.unsafeGetTensorImpl()->storage());
  return self;
}

at::Tensor OpMethods::nonzero(const at::Tensor &self) {
  TRACE_INFO(__FUNCTION__);
  auto input_cpu = self.cpu();
  auto output = at::nonzero(input_cpu);
  return output.to(at::Device(at::Device::Type::MLU));
}

at::Tensor& OpMethods::nonzero_out(at::Tensor &out, const at::Tensor &self) {
  TRACE_INFO(__FUNCTION__);
  auto out_cpu = out.cpu();
  auto input_cpu = self.cpu();
  auto output = at::nonzero_out(out_cpu, input_cpu);
  if (out.numel() < output.numel()) {
    throw std::invalid_argument("OpMethods::nonzero_out fail, "
      "because the output expected more memory!");
  }
  resize_impl_mlu_(getMluTensorImpl(out), output.sizes(), c10::nullopt);
  out.copy_(output);
  return out;
}

at::Tensor OpMethods::zeros_like(const at::Tensor& self,
                                 const at::TensorOptions& options,
                                 c10::optional<c10::MemoryFormat> memory_format) {
  TRACE_INFO(__FUNCTION__);
  auto input_cpu = self.cpu();
  auto input_options = options.device(at::Device(at::Device::Type::CPU));
  auto output = at::zeros_like(input_cpu, input_options);
  return output.to(at::Device(at::Device::Type::MLU));
}

at::Tensor& OpMethods::resize_(at::Tensor& out, at::IntArrayRef size,
                               c10::optional<c10::MemoryFormat> memory_format) {
  TRACE_INFO(__FUNCTION__);
  auto out_cpu = out.cpu();
  auto output = out_cpu.resize_(size);
  auto out_mlu = output.to(at::Device(at::Device::Type::MLU));
  getMluTensorImpl(out)->copy_cnnl_metadata_from(getMluTensorImpl(out_mlu));
  // out.copy_(out_cpu);
  return out;
}

at::Tensor& OpMethods::arange_out(at::Tensor &out,
                                  const at::Scalar start,
                                  const at::Scalar end,
                                  const at::Scalar step) {
  TRACE_INFO(__FUNCTION__);
  auto out_cpu = out.cpu();
  at::arange_out(out_cpu, start, end, step);
  auto out_mlu = out_cpu.to(at::Device(at::Device::Type::MLU));
  getMluTensorImpl(out)->copy_cnnl_metadata_from(getMluTensorImpl(out_mlu));
  resize_impl_mlu_(getMluTensorImpl(out), out_mlu.sizes(), out_mlu.strides());
  return out;
}

at::Tensor OpMethods::as_strided(const at::Tensor& self, at::IntArrayRef size,
                                 at::IntArrayRef strides,
                                 c10::optional<int64_t> storage_offset) {
  TRACE_INFO(__FUNCTION__);
  return cnnl::ops::cnnl_as_strided(self, size, strides, storage_offset);
}

at::Tensor OpMethods::sub(const at::Tensor& self, const at::Tensor& other,
                          at::Scalar alpha) {
  TRACE_INFO(__FUNCTION__);
  at::Tensor output = at::sub(self.cpu(), other.cpu(), alpha);
  return output.to(at::Device(at::Device::Type::MLU));
}

at::Tensor OpMethods::rsub(const at::Tensor& self, const at::Tensor& other,
                           at::Scalar alpha) {
  TRACE_INFO(__FUNCTION__);
  at::Tensor output = at::rsub(self.cpu(), other.cpu(), alpha);
  return output.to(at::Device(at::Device::Type::MLU));
}

at::Tensor OpMethods::rsub(const at::Tensor& self, at::Scalar other,
                           at::Scalar alpha) {
  TRACE_INFO(__FUNCTION__);
  at::Tensor output = at::rsub(self.cpu(), other, alpha);
  return output.to(at::Device(at::Device::Type::MLU));
}

at::Tensor OpMethods::sub(const at::Tensor& self, at::Scalar other,
                          at::Scalar alpha) {
  TRACE_INFO(__FUNCTION__);
  at::Tensor output = at::sub(self.cpu(), other, alpha);
  return output.to(at::Device(at::Device::Type::MLU));
}

at::Tensor& OpMethods::sub_(at::Tensor& self, const at::Tensor& other,
                           at::Scalar alpha) {
  TRACE_INFO(__FUNCTION__);
  auto self_cpu = self.cpu();
  auto other_cpu = other.cpu();
  self_cpu.sub_(other_cpu, alpha);
  getMluTensorImpl(self)->copy_cnnl_metadata_from(getMluTensorImpl(self_cpu.to(
                  at::Device(at::Device::Type::MLU))));
  return self;
}

at::Tensor& OpMethods::sub_(at::Tensor& self, at::Scalar other,
                           at::Scalar alpha) {
  TRACE_INFO(__FUNCTION__);
  auto self_cpu = self.cpu();
  self_cpu.sub_(other, alpha);
  getMluTensorImpl(self)->copy_cnnl_metadata_from(getMluTensorImpl(self_cpu.to(
                  at::Device(at::Device::Type::MLU))));
  return self;
}

at::Tensor OpMethods::mul(const at::Tensor& self, const at::Tensor& other) {
  TRACE_INFO(__FUNCTION__);
  auto input_cpu = self.cpu();
  auto other_cpu = other.cpu();
  auto output_cpu = at::mul(input_cpu, other_cpu);
  return output_cpu.to(at::Device(at::Device::Type::MLU));
}

at::Tensor& OpMethods::mul_(at::Tensor& self, const at::Tensor& other) {
  TRACE_INFO(__FUNCTION__);
  auto input_cpu = self.cpu();
  auto other_cpu = other.cpu();
  auto out_cpu = input_cpu.mul_(other_cpu);
  auto out_mlu = out_cpu.to(at::Device(at::Device::Type::MLU));
  getMluTensorImpl(self)->set_cpu_storage(
      out_mlu.unsafeGetTensorImpl()->storage());
  return self;
}

at::Tensor OpMethods::mul(const at::Tensor& self, at::Scalar other) {
  TRACE_INFO(__FUNCTION__);
  auto input_cpu = self.cpu();
  auto output_cpu = at::mul(input_cpu, other);
  return output_cpu.to(at::Device(at::Device::Type::MLU));
}

at::Tensor& OpMethods::mul_(at::Tensor& self, at::Scalar other) {
  TRACE_INFO(__FUNCTION__);
  auto input_cpu = self.cpu();
  auto out_cpu = at::mul(input_cpu, other);
  auto out_mlu = out_cpu.to(at::Device(at::Device::Type::MLU));
  getMluTensorImpl(self)->copy_cnnl_metadata_from(getMluTensorImpl(out_mlu));
  return self;
}

at::Tensor& OpMethods::mul_out(at::Tensor& out, const at::Tensor& self, const at::Tensor& other) {
  TRACE_INFO(__FUNCTION__);
  auto out_cpu = out.cpu();
  auto input_cpu = self.cpu();
  auto other_cpu = other.cpu();
  at::native::mul_out(out_cpu, input_cpu, other_cpu);
  auto out_mlu = out_cpu.to(at::Device(at::Device::Type::MLU));
  getMluTensorImpl(out)->copy_cnnl_metadata_from(getMluTensorImpl(out_mlu));
  resize_impl_mlu_(getMluTensorImpl(out), out_mlu.sizes(), out_mlu.strides());
  return out;
}


at::Tensor OpMethods::div(const at::Tensor& self, const at::Tensor& other) {
  TRACE_INFO(__FUNCTION__);
  auto input_cpu = self.cpu();
  auto other_cpu = other.cpu();
  auto output_cpu = at::div(input_cpu, other_cpu);
  return output_cpu.to(at::Device(at::Device::Type::MLU));
}

at::Tensor OpMethods::div(const at::Tensor& self, at::Scalar other) {
  TRACE_INFO(__FUNCTION__);
  auto input_cpu = self.cpu();
  auto output_cpu = at::div(input_cpu, other);
  return output_cpu.to(at::Device(at::Device::Type::MLU));
}

at::Tensor& OpMethods::div_(at::Tensor& self, at::Scalar other) {
  TRACE_INFO(__FUNCTION__);
  auto input_cpu = self.cpu();
  auto out_cpu = at::div(input_cpu, other);
  auto out_mlu = out_cpu.to(at::Device(at::Device::Type::MLU));
  getMluTensorImpl(self)->copy_cnnl_metadata_from(getMluTensorImpl(out_mlu));
  return self;
}

at::Tensor& OpMethods::div_(at::Tensor& self, const at::Tensor& other) {
  TRACE_INFO(__FUNCTION__);
  auto input_cpu = self.cpu();
  auto other_cpu = other.cpu();
  input_cpu.div_(other_cpu);
  getMluTensorImpl(self)->set_cpu_storage(
      input_cpu.unsafeGetTensorImpl()->storage());
  return self;
}

at::Tensor& OpMethods::div_out(at::Tensor& out, const at::Tensor& self, const at::Tensor& other) {
  TRACE_INFO(__FUNCTION__);
  auto out_cpu = out.cpu();
  auto input_cpu = self.cpu();
  auto other_cpu = other.cpu();
  at::native::div_out(out_cpu, input_cpu, other_cpu);
  auto out_mlu = out_cpu.to(at::Device(at::Device::Type::MLU));
  getMluTensorImpl(out)->copy_cnnl_metadata_from(getMluTensorImpl(out_mlu));
  resize_impl_mlu_(getMluTensorImpl(out), out_mlu.sizes(), out_mlu.strides());
  return out;
}

at::Tensor OpMethods::threshold(const at::Tensor& self, at::Scalar threshold,
                                at::Scalar value) {
  TRACE_INFO(__FUNCTION__);
  auto input_cpu = self.cpu();
  auto output_cpu = at::threshold(input_cpu, threshold, value);
  return output_cpu.to(at::Device(at::Device::Type::MLU));
}

at::Tensor& OpMethods::threshold_(at::Tensor& self, at::Scalar threshold,
                                  at::Scalar value) {
  TRACE_INFO(__FUNCTION__);
  auto input_cpu = self.cpu();
  at::threshold_(input_cpu, threshold, value);
  auto out_mlu = input_cpu.to(at::Device(at::Device::Type::MLU));
  self.copy_(out_mlu);
  return self;
}

at::Tensor& OpMethods::threshold_out(at::Tensor& out, const at::Tensor& self,
                                     at::Scalar threshold, at::Scalar value) {
  TRACE_INFO(__FUNCTION__);
  auto input_cpu = self.cpu();
  auto output_cpu = at::empty(input_cpu.sizes(), input_cpu.options());
  at::threshold_out(output_cpu, input_cpu, threshold, value);
  auto output_mlu = output_cpu.to(at::Device(at::Device::Type::MLU));
  getMluTensorImpl(out)->copy_cnnl_metadata_from(getMluTensorImpl(output_mlu));
  return out;
}

at::Tensor OpMethods::threshold_backward(const at::Tensor& grad,
                                         const at::Tensor& self,
                                         at::Scalar threshold) {
  TRACE_INFO(__FUNCTION__);
  auto grad_cpu = grad.cpu();
  auto self_cpu = self.cpu();
  auto out_cpu = at::threshold_backward(grad_cpu, self_cpu, threshold);
  return out_cpu.to(at::Device(at::Device::Type::MLU));
}

at::Tensor OpMethods::softplus(const at::Tensor& self,
                               at::Scalar beta = 1,
                               at::Scalar threshold = 20) {
  TRACE_INFO(__FUNCTION__);
  auto input_cpu = self.cpu();
  auto output_cpu = at::softplus(input_cpu, beta, threshold);
  return output_cpu.to(at::Device(at::Device::Type::MLU));
}

at::Tensor OpMethods::softplus_backward(const at::Tensor& grad_output,
                                        const at::Tensor& self,
                                        at::Scalar beta,
                                        at::Scalar threshold,
                                        const at::Tensor& output) {
  TRACE_INFO(__FUNCTION__);
  auto input_cpu = self.cpu();
  auto grad_output_cpu = grad_output.cpu();
  auto output_cpu = output.cpu();
  auto grad_input_cpu = at::softplus_backward(grad_output_cpu,
                                              input_cpu,
                                              beta,
                                              threshold,
                                              output_cpu);
  return grad_input_cpu.to(at::Device(at::Device::Type::MLU));
}

at::Tensor OpMethods::relu(const at::Tensor& self) {
  TRACE_INFO(__FUNCTION__);
  auto input_cpu = self.cpu();
  auto output_cpu = at::relu(input_cpu);
  return output_cpu.to(at::Device(at::Device::Type::MLU));
}

at::Tensor& OpMethods::relu_(at::Tensor& self) {
  TRACE_INFO(__FUNCTION__);
  auto input_cpu = self.cpu();
  at::relu_(input_cpu);
  getMluTensorImpl(self)->set_cpu_storage(
      input_cpu.unsafeGetTensorImpl()->storage());
  return self;
}

at::Tensor OpMethods::tanh(const at::Tensor& self) {
  TRACE_INFO(__FUNCTION__);
  auto input_cpu = self.cpu();
  auto output_cpu = at::tanh(input_cpu);
  return output_cpu.to(at::Device(at::Device::Type::MLU));
}

at::Tensor& OpMethods::tanh_(at::Tensor& self) {
  TRACE_INFO(__FUNCTION__);
  auto input_cpu = self.cpu();
  at::tanh_(input_cpu);
  // copy cpu result into mlu self cpu_storage
  getMluTensorImpl(self)->set_cpu_storage(
      input_cpu.unsafeGetTensorImpl()->storage());
  return self;
}

at::Tensor OpMethods::tanh_backward(const at::Tensor& grad_output,
                                    const at::Tensor& output) {
  TRACE_INFO(__FUNCTION__);
  auto grad_cpu = grad_output.cpu();
  auto self_cpu = output.cpu();
  auto output_cpu = at::tanh_backward(grad_cpu, self_cpu);
  return output_cpu.to(at::Device(at::Device::Type::MLU));
}

at::Tensor OpMethods::gelu(const at::Tensor& self) {
  TRACE_INFO(__FUNCTION__);
  auto input_cpu = self.cpu();
  auto output_cpu = at::gelu(input_cpu);
  return output_cpu.to(at::Device(at::Device::Type::MLU));
}

at::Tensor OpMethods::gelu_backward(const at::Tensor& grad_output,
                                    const at::Tensor& output) {
  TRACE_INFO(__FUNCTION__);
  auto grad_cpu = grad_output.cpu();
  auto self_cpu = output.cpu();
  auto output_cpu = at::gelu_backward(grad_cpu, self_cpu);
  return output_cpu.to(at::Device(at::Device::Type::MLU));
}

at::Tensor OpMethods::hardtanh(const at::Tensor& self, at::Scalar min_v,
                               at::Scalar max_v) {
  TRACE_INFO(__FUNCTION__);
  auto input_cpu = self.cpu();
  auto output_cpu = at::hardtanh(input_cpu, min_v, max_v);
  return output_cpu.to(at::Device(at::Device::Type::MLU));
}

at::Tensor OpMethods::hardtanh_backward(const at::Tensor& grad_output,
    const at::Tensor& self, at::Scalar min_v, at::Scalar max_v) {
  TRACE_INFO(__FUNCTION__);
  auto grad_cpu = grad_output.cpu();
  auto input_cpu = self.cpu();
  auto output_cpu = at::hardtanh_backward(grad_cpu, input_cpu, min_v, max_v);
  return output_cpu.to(at::Device(at::Device::Type::MLU));
}

at::Tensor& OpMethods::hardtanh_(at::Tensor& self, at::Scalar min_v,
                                 at::Scalar max_v) {
  TRACE_INFO(__FUNCTION__);
  auto input_cpu = self.cpu();
  at::hardtanh_(input_cpu, min_v, max_v);
  // copy cpu result into mlu self cpu_storage
  return self.copy_(input_cpu);
}

at::Tensor OpMethods::linear(const at::Tensor& input, const at::Tensor& weight,
                             const at::Tensor& bias, const at::Tensor& q_scale,
                             const at::Tensor& q_mode) {
  TRACE_INFO(__FUNCTION__);
  return input;
}

at::Tensor OpMethods::conv2d(const at::Tensor& input, const at::Tensor& weight,
                             const at::Tensor& bias,
                             torch::List<int64_t> padding,
                             torch::List<int64_t> stride,
                             torch::List<int64_t> dilation, int64_t groups,
                             const at::Tensor& q_scale,
                             const at::Tensor& q_mode) {
  TRACE_INFO(__FUNCTION__);
  return input;
}

at::Tensor OpMethods::expand(const at::Tensor& self, at::IntArrayRef size,
                             bool implicit) {
  TRACE_INFO(__FUNCTION__);
  return cnnl::ops::cnnl_expand(self, size, implicit);
}

at::Tensor& OpMethods::floor_out(at::Tensor& out, const at::Tensor& self) {
  TRACE_INFO(__FUNCTION__);
  auto out_cpu = out.cpu();
  auto input_cpu = self.cpu();
  auto output = at::floor_out(out_cpu, input_cpu);
  auto out_mlu = out_cpu.to(at::Device(at::Device::Type::MLU));
  getMluTensorImpl(out)->copy_cnnl_metadata_from(getMluTensorImpl(out_mlu));
  resize_impl_mlu_(getMluTensorImpl(out), out_mlu.sizes(), out_mlu.strides());
  return out;
}

at::Tensor & OpMethods::ceil_out(at::Tensor& out, const at::Tensor& self) {
  TRACE_INFO(__FUNCTION__);
  auto out_cpu = out.cpu();
  auto input_cpu = self.cpu();
  at::ceil_out(out_cpu, input_cpu);
  getMluTensorImpl(out)->set_cpu_storage(
    out_cpu.unsafeGetTensorImpl()->storage());
  return out;
}

at::Tensor OpMethods::eq(const at::Tensor& self, const at::Tensor& other) {
  TRACE_INFO(__FUNCTION__);
  auto input_cpu = self.cpu();
  auto other_cpu = other.cpu();
  auto output = at::eq(input_cpu, other_cpu);
  return output.to(at::Device(at::Device::Type::MLU));
}

at::Tensor OpMethods::eq(const at::Tensor& self, at::Scalar other) {
  TRACE_INFO(__FUNCTION__);
  auto input_cpu = self.cpu();
  auto output = at::eq(input_cpu, other);
  return output.to(at::Device(at::Device::Type::MLU));
}

at::Tensor& OpMethods::exp_(at::Tensor& self) {
  TRACE_INFO(__FUNCTION__);
  auto input_cpu = self.cpu();
  at::exp_(input_cpu);
  auto out_mlu = input_cpu.to(at::Device(at::Device::Type::MLU));
  getMluTensorImpl(self)->copy_cnnl_metadata_from(getMluTensorImpl(out_mlu));
  resize_impl_mlu_(getMluTensorImpl(self), out_mlu.sizes(), out_mlu.strides());
  return self;
}

at::Tensor& OpMethods::exp_out(at::Tensor & out, const at::Tensor & self) {
  TRACE_INFO(__FUNCTION__);
  auto input_cpu = self.cpu();
  auto out_cpu = out.cpu();
  at::exp_out(out_cpu, input_cpu);
  auto out_mlu = out_cpu.to(at::Device(at::Device::Type::MLU));
  getMluTensorImpl(out)->copy_cnnl_metadata_from(getMluTensorImpl(out_mlu));
  resize_impl_mlu_(getMluTensorImpl(out), out_mlu.sizes(), out_mlu.strides());
  return out;
}


at::Tensor& OpMethods::eq_(at::Tensor& self, const at::Tensor& other) {
  TRACE_INFO(__FUNCTION__);
  auto self_cpu = self.cpu();
  auto other_cpu = other.cpu();
  at::native::eq_(self_cpu, other_cpu);
  auto self_mlu = self_cpu.to(at::Device(at::Device::Type::MLU));
  getMluTensorImpl(self)->copy_cnnl_metadata_from(getMluTensorImpl(self_mlu));
  return self;
}

at::Tensor& OpMethods::eq_(at::Tensor& self, at::Scalar other) {
  TRACE_INFO(__FUNCTION__);
  auto self_cpu = self.cpu();
  at::native::eq_(self_cpu, other);
  auto self_mlu = self_cpu.to(at::Device(at::Device::Type::MLU));
  getMluTensorImpl(self)->copy_cnnl_metadata_from(getMluTensorImpl(self_mlu));
  return self;
}

at::Tensor & OpMethods::eq_out(at::Tensor& out, const at::Tensor& self,
                                      const at::Tensor& other) {
  TRACE_INFO(__FUNCTION__);
  auto out_cpu = out.cpu();
  auto self_cpu = self.cpu();
  auto other_cpu = other.cpu();
  at::eq_out(out_cpu, self_cpu, other_cpu);
  auto out_mlu = out_cpu.to(at::Device(at::Device::Type::MLU));
  getMluTensorImpl(out)->copy_cnnl_metadata_from(getMluTensorImpl(out_mlu));
  resize_impl_mlu_(getMluTensorImpl(out), out_mlu.sizes(), out_mlu.strides());
  return out;
}

at::Tensor & OpMethods::eq_out(at::Tensor& out, const at::Tensor& self, at::Scalar other) {
  TRACE_INFO(__FUNCTION__);
  auto out_cpu = out.cpu();
  auto self_cpu = self.cpu();
  at::eq_out(out_cpu, self_cpu, other);
  auto out_mlu = out_cpu.to(at::Device(at::Device::Type::MLU));
  getMluTensorImpl(out)->copy_cnnl_metadata_from(getMluTensorImpl(out_mlu));
  resize_impl_mlu_(getMluTensorImpl(out), out_mlu.sizes(), out_mlu.strides());
  return out;
}


at::Tensor OpMethods::exp(const at::Tensor& self) {
  TRACE_INFO(__FUNCTION__);
  auto input_cpu = self.cpu();
  auto output = at::exp(input_cpu);
  return output.to(at::Device(at::Device::Type::MLU));
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> OpMethods::native_batch_norm(
    const at::Tensor& input, const at::Tensor& weight, const at::Tensor& bias,
    const at::Tensor& running_mean, const at::Tensor& running_var,
    bool training, double momentum, double eps) {
  TRACE_INFO(__FUNCTION__);
  auto output = at::native_batch_norm(input.cpu(), weight.cpu(), bias.cpu(), running_mean.cpu(),
                                      running_var.cpu(), training, momentum, eps);
  auto save_mean = at::empty(
    input.size(1), input.options().device(at::DeviceType::CPU), c10::nullopt);

  auto save_invstd = at::empty(
      input.size(1), input.options().device(at::DeviceType::CPU), c10::nullopt);

  return std::make_tuple(std::get<0>(output).to(at::Device(at::Device::Type::MLU)),
                         save_mean.to(at::Device(at::Device::Type::MLU)),
                         save_invstd.to(at::Device(at::Device::Type::MLU)));
}

at::Tensor OpMethods::squeeze(const at::Tensor& input) {
  TRACE_INFO(__FUNCTION__);
  return cnnl::ops::cnnl_squeeze(input);
}

at::Tensor OpMethods::squeeze(const at::Tensor& input, int64_t dim) {
  TRACE_INFO(__FUNCTION__);
  return cnnl::ops::cnnl_squeeze(input, dim);
}

at::Tensor& OpMethods::squeeze_(at::Tensor& input) {
  TRACE_INFO(__FUNCTION__);
  return cnnl::ops::cnnl_squeeze_(input);
}

at::Tensor& OpMethods::squeeze_(at::Tensor& input, int64_t dim) {
  TRACE_INFO(__FUNCTION__);
  return cnnl::ops::cnnl_squeeze_(input, dim);
}

at::Tensor& OpMethods::unsqueeze_(at::Tensor& input, int64_t dim) {
  TRACE_INFO(__FUNCTION__);
  return cnnl::ops::cnnl_unsqueeze_(input, dim);
}

at::Tensor OpMethods::unsqueeze(const at::Tensor& input, int64_t dim) {
  TRACE_INFO(__FUNCTION__);
  return cnnl::ops::cnnl_unsqueeze(input, dim);
}

at::Tensor OpMethods::max(const at::Tensor& self) {
  TRACE_INFO(__FUNCTION__);
  auto self_cpu = self.cpu();
  auto output_cpu = at::max(self_cpu);
  return output_cpu.to(at::Device(at::Device::Type::MLU));
}

at::Tensor OpMethods::max(const at::Tensor& self, const at::Tensor& other) {
  TRACE_INFO(__FUNCTION__);
  auto self_cpu = self.cpu();
  auto other_cpu = other.cpu();
  auto output_cpu = at::max(self_cpu, other_cpu);
  return output_cpu.to(at::Device(at::Device::Type::MLU));
}

at::Tensor& OpMethods::max_out(at::Tensor& out, const at::Tensor& self,
                               const at::Tensor& other) {
  TRACE_INFO(__FUNCTION__);
  auto out_cpu = out.cpu();
  auto input_cpu = self.cpu();
  auto other_cpu = other.cpu();
  at::max_out(out_cpu, input_cpu, other_cpu);
  auto out_mlu = out_cpu.to(at::Device(at::Device::Type::MLU));
  getMluTensorImpl(out)->copy_cnnl_metadata_from(getMluTensorImpl(out_mlu));
  resize_impl_mlu_(getMluTensorImpl(out), out_mlu.sizes(), out_mlu.strides());
  return out;
}

std::tuple<at::Tensor, at::Tensor> OpMethods::max(const at::Tensor& self,
                                                  const int64_t dim,
                                                  const bool keepdim) {
  TRACE_INFO(__FUNCTION__);
  auto self_cpu = self.cpu();
  auto output_cpu = at::max(self_cpu, dim, keepdim);
  std::get<0>(output_cpu) =
      std::get<0>(output_cpu).to(at::Device(at::Device::Type::MLU));
  std::get<1>(output_cpu) =
      std::get<1>(output_cpu).to(at::Device(at::Device::Type::MLU));
  return output_cpu;
}

std::tuple<at::Tensor&, at::Tensor&> OpMethods::max_out_dim_max(at::Tensor &max,
  at::Tensor &max_indices, const at::Tensor &self, int64_t dim, bool keepdim) {
  TRACE_INFO(__FUNCTION__);
  auto max_cpu = max.cpu();
  auto max_indices_cpu = max_indices.cpu();
  auto self_cpu = self.cpu();
  auto output = at::max_out(max_cpu, max_indices_cpu, self_cpu, dim, keepdim);
  auto max_mlu = max_cpu.to(at::Device(at::Device::Type::MLU));
  getMluTensorImpl(max)->copy_cnnl_metadata_from(getMluTensorImpl(max_mlu));
  resize_impl_mlu_(getMluTensorImpl(max), max_mlu.sizes(), max_mlu.strides());
  auto max_indices_mlu = max_indices_cpu.to(at::Device(at::Device::Type::MLU));
  getMluTensorImpl(max_indices)->copy_cnnl_metadata_from(getMluTensorImpl(max_indices_mlu));
  resize_impl_mlu_(getMluTensorImpl(max_indices), max_indices_mlu.sizes(),
    max_indices_mlu.strides());
  return std::tuple<at::Tensor&, at::Tensor&>(max, max_indices);
}

at::Tensor OpMethods::argmax(const at::Tensor & self, c10::optional<int64_t> dim, bool keepdim) {
  TRACE_INFO(__FUNCTION__);
  auto self_cpu = self.cpu();
  auto output_cpu = at::argmax(self_cpu, dim, keepdim);
  return output_cpu.to(at::Device(at::Device::Type::MLU));
}

at::Tensor OpMethods::min(const at::Tensor& self) {
  TRACE_INFO(__FUNCTION__);
  auto self_cpu = self.cpu();
  auto output_cpu = at::min(self_cpu);
  return output_cpu.to(at::Device(at::Device::Type::MLU));
}

at::Tensor OpMethods::min(const at::Tensor& self, const at::Tensor& other) {
  TRACE_INFO(__FUNCTION__);
  auto self_cpu = self.cpu();
  auto other_cpu = other.cpu();
  auto output_cpu = at::min(self_cpu, other_cpu);
  return output_cpu.to(at::Device(at::Device::Type::MLU));
}

at::Tensor& OpMethods::min_out(at::Tensor& out, const at::Tensor& self,
                               const at::Tensor& other) {
  TRACE_INFO(__FUNCTION__);
  auto out_cpu = out.cpu();
  auto input_cpu = self.cpu();
  auto other_cpu = other.cpu();
  at::min_out(out_cpu, input_cpu, other_cpu);
  auto out_mlu = out_cpu.to(at::Device(at::Device::Type::MLU));
  getMluTensorImpl(out)->copy_cnnl_metadata_from(getMluTensorImpl(out_mlu));
  resize_impl_mlu_(getMluTensorImpl(out), out_mlu.sizes(), out_mlu.strides());
  return out;
}

std::tuple<at::Tensor, at::Tensor> OpMethods::min(const at::Tensor& self,
                                                  const int64_t dim,
                                                  const bool keepdim) {
  TRACE_INFO(__FUNCTION__);
  auto self_cpu = self.cpu();
  auto output_cpu = at::min(self_cpu, dim, keepdim);
  std::get<0>(output_cpu) =
      std::get<0>(output_cpu).to(at::Device(at::Device::Type::MLU));
  std::get<1>(output_cpu) =
      std::get<1>(output_cpu).to(at::Device(at::Device::Type::MLU));
  return output_cpu;
}

std::tuple<at::Tensor, at::Tensor> OpMethods::topk(const at::Tensor& self,
                                                   int64_t k,
                                                   int64_t dim,
                                                   bool largest,
                                                   bool sorted) {
  TRACE_INFO(__FUNCTION__);
  auto input_cpu = self.cpu();
  auto output_cpu = at::topk(input_cpu, k, dim, largest, sorted);
  std::get<0>(output_cpu) =
      std::get<0>(output_cpu).to(at::Device(at::Device::Type::MLU));
  std::get<1>(output_cpu) =
      std::get<1>(output_cpu).to(at::Device(at::Device::Type::MLU));
  return output_cpu;
}

std::tuple<at::Tensor&, at::Tensor&> OpMethods::topk_out(at::Tensor & values,
                                                         at::Tensor & indices,
                                                         const at::Tensor& self,
                                                         int64_t k,
                                                         int64_t dim,
                                                         bool largest,
                                                         bool sorted) {
  TRACE_INFO(__FUNCTION__);
  auto values_cpu = values.cpu();
  auto indices_cpu = indices.cpu();
  auto self_cpu = self.cpu();
  at::topk_out(values_cpu, indices_cpu, self_cpu, k, dim, largest, sorted);
  auto values_mlu = values_cpu.to(at::Device(at::Device::Type::MLU));
  auto indices_mlu = indices_cpu.to(at::Device(at::Device::Type::MLU));
  getMluTensorImpl(values)->copy_cnnl_metadata_from(getMluTensorImpl(values_mlu));
  resize_impl_mlu_(getMluTensorImpl(values), values_mlu.sizes(), values_mlu.strides());
  getMluTensorImpl(indices)->copy_cnnl_metadata_from(getMluTensorImpl(indices_mlu));
  resize_impl_mlu_(getMluTensorImpl(indices), indices_mlu.sizes(), indices_mlu.strides());
  return std::forward_as_tuple(values, indices);
}

std::tuple<at::Tensor, at::Tensor> OpMethods::sort(const at::Tensor& self,
                                                   const int64_t dim,
                                                   const bool descending) {
  TRACE_INFO(__FUNCTION__);
  auto input_cpu = self.cpu();
  auto output_cpu = at::sort(input_cpu, dim, descending);
  std::get<0>(output_cpu) =
      std::get<0>(output_cpu).to(at::Device(at::Device::Type::MLU));
  std::get<1>(output_cpu) =
      std::get<1>(output_cpu).to(at::Device(at::Device::Type::MLU));
  return output_cpu;
}

std::tuple<at::Tensor&, at::Tensor&> OpMethods::sort_out(at::Tensor& values,
                                                         at::Tensor& indices,
                                                         const at::Tensor& self,
                                                         const int64_t dim,
                                                         const bool descending) {
  TRACE_INFO(__FUNCTION__);
  auto values_cpu = values.cpu();
  auto indices_cpu = indices.cpu();
  auto input_cpu = self.cpu();
  at::sort_out(values_cpu, indices_cpu, input_cpu, dim, descending);
  auto out_mlu_0 = values_cpu.to(at::Device(at::Device::Type::MLU));
  auto out_mlu_1 = indices_cpu.to(at::Device(at::Device::Type::MLU));
  getMluTensorImpl(values)->copy_cnnl_metadata_from(getMluTensorImpl(out_mlu_0));
  resize_impl_mlu_(getMluTensorImpl(values), out_mlu_0.sizes(), out_mlu_0.strides());
  getMluTensorImpl(indices)->copy_cnnl_metadata_from(getMluTensorImpl(out_mlu_1));
  resize_impl_mlu_(getMluTensorImpl(indices), out_mlu_1.sizes(), out_mlu_1.strides());
  return std::forward_as_tuple(values, indices);
}

at::Tensor OpMethods::view(const at::Tensor& self, at::IntArrayRef size) {
  TRACE_INFO(__FUNCTION__);
  return cnnl::ops::cnnl_view(self, size);
}

at::Tensor OpMethods::reshape(const at::Tensor& self, at::IntArrayRef shape) {
  TRACE_INFO(__FUNCTION__);
  auto self_cpu = self.cpu();
  auto out_cpu = at::reshape(self_cpu, shape);
  auto output = out_cpu.contiguous();
  return output.to(at::Device(at::Device::Type::MLU));
}

at::Tensor OpMethods::avg_pool2d(const at::Tensor& input,
                                 at::IntArrayRef kernel_size,
                                 at::IntArrayRef stride,
                                 at::IntArrayRef padding, bool ceil_mode,
                                 bool count_include_pad,
                                 c10::optional<int64_t> divisor_override) {
  TRACE_INFO(__FUNCTION__);
  TORCH_CHECK(!divisor_override.has_value() || divisor_override.value() != 0,
              "divisor must be not zero");
  auto input_cpu = input.cpu();
  auto output_cpu = at::avg_pool2d(
      input_cpu, kernel_size, stride, padding, ceil_mode, count_include_pad);
  return output_cpu.to(at::Device(at::Device::Type::MLU));
}

at::Tensor OpMethods::avg_pool3d(const at::Tensor& input,
                                 at::IntArrayRef kernel_size,
                                 at::IntArrayRef stride,
                                 at::IntArrayRef padding, bool ceil_mode,
                                 bool count_include_pad,
                                 c10::optional<int64_t> divisor_override) {
  TRACE_INFO(__FUNCTION__);
  auto input_cpu = input.cpu();
  auto output_cpu = at::avg_pool3d(
      input_cpu, kernel_size, stride, padding, ceil_mode, count_include_pad);
  return output_cpu.to(at::Device(at::Device::Type::MLU));
}

at::Tensor OpMethods::repeat(const at::Tensor& self, at::IntArrayRef repeats) {
  TRACE_INFO(__FUNCTION__);
  auto input_cpu = self.cpu();
  auto output_cpu = at::native::repeat(input_cpu, repeats);
  return output_cpu.to(at::Device(at::Device::Type::MLU));
}

at::Tensor OpMethods::round(const at::Tensor& self) {
  TRACE_INFO(__FUNCTION__);
  auto input_cpu = self.cpu();
  auto output_cpu = at::round_(input_cpu);
  return output_cpu.to(at::Device(at::Device::Type::MLU));
}

at::Tensor& OpMethods::round_(at::Tensor& self) {
  TRACE_INFO(__FUNCTION__);
  auto input_cpu = self.cpu();
  at::round_(input_cpu);
  auto self_mlu = input_cpu.to(at::Device(at::Device::Type::MLU));
  getMluTensorImpl(self)->copy_cnnl_metadata_from(getMluTensorImpl(self_mlu));
  return self;
}

at::Tensor& OpMethods::round_out(at::Tensor& out, const at::Tensor& self) {
  TRACE_INFO(__FUNCTION__);
  auto out_cpu = out.cpu();
  auto self_cpu = self.cpu();
  at::round_out(out_cpu, self_cpu);
  auto out_mlu = out_cpu.to(at::Device(at::Device::Type::MLU));
  getMluTensorImpl(out)->copy_cnnl_metadata_from(getMluTensorImpl(out_mlu));
  resize_impl_mlu_(getMluTensorImpl(out), out_mlu.sizes(), out_mlu.strides());
  return out;
}

at::Tensor OpMethods::transpose(const at::Tensor& self, int64_t dim0,
                                int64_t dim1) {
  TRACE_INFO(__FUNCTION__);
  return cnnl::ops::cnnl_transpose(self, dim0, dim1);
}

at::Tensor OpMethods::mean(const at::Tensor& self, at::IntArrayRef dim,
                           bool keepdim, c10::optional<c10::ScalarType> dtype) {
  TRACE_INFO(__FUNCTION__);
  auto input_cpu = self.cpu();
  auto output_cpu = at::mean(input_cpu, dim, keepdim);
  return output_cpu.to(at::Device(at::Device::Type::MLU));
}

at::Tensor& OpMethods::mean_out(at::Tensor& out, const at::Tensor& self,
                                at::IntArrayRef dim, bool keepdim,
                                c10::optional<c10::ScalarType> dtype) {
  TRACE_INFO(__FUNCTION__);
  auto out_cpu = out.cpu();
  auto self_cpu = self.cpu();
  at::mean_out(out_cpu, self_cpu, dim, keepdim);
  auto out_mlu = out_cpu.to(at::Device(at::Device::Type::MLU));
  getMluTensorImpl(out)->copy_cnnl_metadata_from(getMluTensorImpl(out_mlu));
  resize_impl_mlu_(getMluTensorImpl(out), out_mlu.sizes(), out_mlu.strides());
  return out;
}

at::Tensor OpMethods::mean(const at::Tensor& self,
                           c10::optional<c10::ScalarType> dtype) {
  TRACE_INFO(__FUNCTION__);
  auto input_cpu = self.cpu();
  auto output_cpu = at::mean(input_cpu);
  return output_cpu.to(at::Device(at::Device::Type::MLU));
}

at::Tensor OpMethods::norm(const at::Tensor& self, at::optional<at::Scalar> p,
                           at::IntArrayRef dim, bool keepdim) {
  TRACE_INFO(__FUNCTION__);
  auto input_cpu = self.cpu();
  auto output_cpu = at::norm(input_cpu, p, dim, keepdim);
  return output_cpu.to(at::Device(at::Device::Type::MLU));
}

at::Tensor OpMethods::norm(const at::Tensor& self, at::Scalar p) {
  TRACE_INFO(__FUNCTION__);
  auto input_cpu = self.cpu();
  auto output_cpu = at::norm(input_cpu);
  return output_cpu.to(at::Device(at::Device::Type::MLU));
}

at::Tensor OpMethods::norm(const at::Tensor& self, at::optional<at::Scalar> p,
                           at::IntArrayRef dim, bool keepdim, at::ScalarType dtype) {
  TRACE_INFO(__FUNCTION__);
  auto input_cpu = self.cpu();
  auto output_cpu = at::norm(input_cpu, p, dim, keepdim, dtype);
  return output_cpu.to(at::Device(at::Device::Type::MLU));
}

at::Tensor OpMethods::sum(const at::Tensor& self, at::IntArrayRef dim,
                          bool keepdim, c10::optional<c10::ScalarType> dtype) {
  TRACE_INFO(__FUNCTION__);
  auto input_cpu = self.cpu();
  auto output_cpu = at::sum(input_cpu, dim, keepdim);
  return output_cpu.to(at::Device(at::Device::Type::MLU));
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> OpMethods::_unique2(const at::Tensor & self,
        bool sorted, bool return_inverse, bool return_counts) {
  TRACE_INFO(__FUNCTION__);
  auto self_cpu = self.cpu();
  auto output_cpu = at::_unique2(self_cpu, sorted, return_inverse, return_counts);
  std::get<0>(output_cpu) =
      std::get<0>(output_cpu).to(at::Device(at::Device::Type::MLU));
  std::get<1>(output_cpu) =
      std::get<1>(output_cpu).to(at::Device(at::Device::Type::MLU));
  std::get<2>(output_cpu) =
      std::get<2>(output_cpu).to(at::Device(at::Device::Type::MLU));
  return output_cpu;
}

at::Tensor& OpMethods::sum_out(at::Tensor& result, const at::Tensor& self,
                               at::IntArrayRef dim, bool keepdim,
                               c10::optional<c10::ScalarType> dtype) {
  TRACE_INFO(__FUNCTION__);
  auto input_cpu = self.cpu();
  auto result_cpu = result.cpu();
  at::sum_out(result_cpu, input_cpu, dim, keepdim);
  auto out_mlu = result_cpu.to(at::Device(at::Device::Type::MLU));
  getMluTensorImpl(result)->copy_cnnl_metadata_from(getMluTensorImpl(out_mlu));
  resize_impl_mlu_(getMluTensorImpl(result), out_mlu.sizes(), out_mlu.strides());
  return result;
}

at::Tensor OpMethods::sum(const at::Tensor& self,
                          c10::optional<c10::ScalarType> dtype) {
  TRACE_INFO(__FUNCTION__);
  auto input_cpu = self.cpu();
  auto output_cpu = at::sum(input_cpu);
  return output_cpu.to(at::Device(at::Device::Type::MLU));
}

at::Tensor OpMethods::reciprocal(const at::Tensor& self) {
  TRACE_INFO(__FUNCTION__);
  auto input_cpu = self.cpu();
  auto output_cpu = at::reciprocal(input_cpu);
  return output_cpu.to(at::Device(at::Device::Type::MLU));
}

at::Tensor& OpMethods::reciprocal_(at::Tensor& self) {
  TRACE_INFO(__FUNCTION__);
  auto self_cpu = self.cpu();
  at::reciprocal_(self_cpu);
  self.copy_(self_cpu);
  return self;
}

at::Tensor& OpMethods::reciprocal_out(at::Tensor& out, const at::Tensor & self) {
  TRACE_INFO(__FUNCTION__);
  auto self_cpu = self.cpu();
  auto out_cpu = out.cpu();
  at::reciprocal_out(out_cpu, self_cpu);
  out.copy_(out_cpu);
  return out;
}

std::tuple<at::Tensor, at::Tensor> OpMethods::max_pool2d_with_indices(
    const at::Tensor& input, at::IntArrayRef kernel_size,
    at::IntArrayRef stride, at::IntArrayRef padding, at::IntArrayRef dilation,
    bool ceil_mode) {
  TRACE_INFO(__FUNCTION__);
  auto input_cpu = input.cpu();
  auto output_cpu = at::max_pool2d_with_indices(
      input_cpu, kernel_size = kernel_size, stride = stride, padding = padding,
      dilation = dilation, ceil_mode = ceil_mode);
  auto output_value =
      std::get<0>(output_cpu).to(at::Device(at::Device::Type::MLU));
  auto output_index =
      std::get<1>(output_cpu).to(at::Device(at::Device::Type::MLU));
  return std::make_tuple(output_value, output_index);
}

std::tuple<at::Tensor, at::Tensor> OpMethods::max_pool3d_with_indices(
    const at::Tensor& input, at::IntArrayRef kernel_size,
    at::IntArrayRef stride, at::IntArrayRef padding, at::IntArrayRef dilation,
    bool ceil_mode) {
  TRACE_INFO(__FUNCTION__);
  auto input_cpu = input.cpu();
  auto output_cpu = at::max_pool3d_with_indices(
      input_cpu, kernel_size, stride, padding, dilation, ceil_mode);
  auto output_value =
      std::get<0>(output_cpu).to(at::Device(at::Device::Type::MLU));
  auto output_index =
      std::get<1>(output_cpu).to(at::Device(at::Device::Type::MLU));
  return std::make_tuple(output_value, output_index);
}

at::Tensor OpMethods::max_pool3d_with_indices_backward(
    const at::Tensor & grad_output, const at::Tensor & self, at::IntArrayRef kernel_size,
    at::IntArrayRef stride, at::IntArrayRef padding, at::IntArrayRef dilation, bool ceil_mode,
    const at::Tensor & indices) {
  TRACE_INFO(__FUNCTION__);
  auto grad_output_cpu = grad_output.cpu();
  auto self_cpu = self.cpu();
  auto indices_cpu = indices.cpu();
  auto output_cpu = at::max_pool3d_with_indices_backward(
      grad_output_cpu, self_cpu, kernel_size, stride, padding, dilation, ceil_mode, indices_cpu);
  return output_cpu.to(at::Device(at::Device::Type::MLU));
}

at::Tensor OpMethods::_adaptive_avg_pool2d(const at::Tensor& input,
                                           at::IntArrayRef output_size) {
  TRACE_INFO(__FUNCTION__);
  auto input_cpu = input.cpu();
  auto output_cpu = at::_adaptive_avg_pool2d(input_cpu, output_size);
  return output_cpu.to(at::Device(at::Device::Type::MLU));
}

at::Tensor& OpMethods::adaptive_avg_pool2d_out(at::Tensor& out,
                                               const at::Tensor& self,
                                               at::IntArrayRef output_size) {
  TRACE_INFO(__FUNCTION__);
  auto out_cpu = out.cpu();
  auto self_cpu = self.cpu();
  at::adaptive_avg_pool2d_out(out_cpu, self_cpu, output_size);
  auto out_mlu = out_cpu.to(at::Device(at::Device::Type::MLU));
  getMluTensorImpl(out)->copy_cnnl_metadata_from(getMluTensorImpl(out_mlu));
  resize_impl_mlu_(getMluTensorImpl(out), out_mlu.sizes(), out_mlu.strides());
  return out;
}

at::Tensor OpMethods::_adaptive_avg_pool2d_backward(const at::Tensor& grad_output,
                                                    const at::Tensor& self) {
  TRACE_INFO(__FUNCTION__);
  auto grad_output_cpu = grad_output.cpu();
  auto self_cpu = self.cpu();
  auto output_cpu = at::_adaptive_avg_pool2d_backward(grad_output_cpu, self_cpu);
  return output_cpu.to(at::Device(at::Device::Type::MLU));
}

std::tuple<at::Tensor, at::Tensor> OpMethods::adaptive_max_pool2d(
    const at::Tensor& input, at::IntArrayRef output_size) {
  TRACE_INFO(__FUNCTION__);
  auto input_cpu = input.cpu();
  auto output_cpu = at::adaptive_max_pool2d(input_cpu, output_size);
  auto output_value =
      std::get<0>(output_cpu).to(at::Device(at::Device::Type::MLU));
  auto output_index =
      std::get<1>(output_cpu).to(at::Device(at::Device::Type::MLU));
  return std::make_tuple(output_value, output_index);
}

std::tuple<at::Tensor&, at::Tensor&> OpMethods::adaptive_max_pool2d_out(
  at::Tensor & output, at::Tensor & indices,
  const at::Tensor& input, at::IntArrayRef output_size) {
  TRACE_INFO(__FUNCTION__);
  auto input_cpu = input.cpu();
  auto output_cpu = output.cpu();
  auto indices_cpu = indices.cpu();
  at::adaptive_max_pool2d_out(output_cpu, indices_cpu, input_cpu, output_size);
  getMluTensorImpl(output)->copy_cnnl_metadata_from(
    getMluTensorImpl(output_cpu.to(at::Device(at::Device::Type::MLU))));
  resize_impl_mlu_(getMluTensorImpl(output), output_cpu.sizes(), output_cpu.strides());
  getMluTensorImpl(indices)->copy_cnnl_metadata_from(
    getMluTensorImpl(indices_cpu.to(at::Device(at::Device::Type::MLU))));
  resize_impl_mlu_(getMluTensorImpl(indices), indices_cpu.sizes(), indices_cpu.strides());
  return std::tuple<at::Tensor&, at::Tensor&>(output, indices);
}

at::Tensor OpMethods::adaptive_max_pool2d_backward(
  const at::Tensor & grad_output, const at::Tensor & input, const at::Tensor & indices) {
  TRACE_INFO(__FUNCTION__);
  auto grad_output_cpu = grad_output.cpu();
  auto input_cpu = input.cpu();
  auto indices_cpu = indices.cpu();
  auto output_cpu = at::adaptive_max_pool2d_backward(grad_output_cpu, input_cpu, indices_cpu);
  return output_cpu.to(at::Device(at::Device::Type::MLU));
}

at::Tensor& OpMethods::adaptive_max_pool2d_backward_out(
  at::Tensor& grad_input, const at::Tensor & grad_output,
  const at::Tensor & input, const at::Tensor & indices) {
  TRACE_INFO(__FUNCTION__);
  auto input_cpu = input.cpu();
  auto indices_cpu = indices.cpu();
  auto grad_output_cpu = grad_output.cpu();
  auto grad_input_cpu = grad_input.cpu();
  at::adaptive_max_pool2d_backward_out(grad_input_cpu, grad_output_cpu, input_cpu, indices_cpu);
  getMluTensorImpl(grad_input)->copy_cnnl_metadata_from(
    getMluTensorImpl(grad_input_cpu.to(at::Device(at::Device::Type::MLU))));
  resize_impl_mlu_(getMluTensorImpl(grad_input), grad_input_cpu.sizes(), grad_input_cpu.strides());
  return grad_input;
}

at::Tensor OpMethods::upsample_nearest2d(const at::Tensor& self,
                                         at::IntArrayRef output_size,
                                         c10::optional<double> scales_h,
                                         c10::optional<double> scales_w) {
  TRACE_INFO(__FUNCTION__);
  auto input_cpu = self.cpu();
  auto output_cpu = at::upsample_nearest2d(input_cpu, output_size);
  return output_cpu.to(at::Device(at::Device::Type::MLU));
}

at::Tensor& OpMethods::upsample_nearest2d_out(at::Tensor& out,
                                              const at::Tensor& self,
                                              at::IntArrayRef output_size,
                                              c10::optional<double> scales_h,
                                              c10::optional<double> scales_w) {
  TRACE_INFO(__FUNCTION__);
  auto out_cpu = out.cpu();
  auto input_cpu = self.cpu();
  at::upsample_nearest2d_out(out_cpu, input_cpu, output_size);
  auto out_mlu = out_cpu.to(at::Device(at::Device::Type::MLU));
  getMluTensorImpl(out)->copy_cnnl_metadata_from(getMluTensorImpl(out_mlu));
  resize_impl_mlu_(getMluTensorImpl(out), out_mlu.sizes(), out_mlu.strides());
  return out;
}

at::Tensor OpMethods::upsample_nearest2d_backward(const at::Tensor& grad_output,
                                                  at::IntArrayRef output_size,
                                                  at::IntArrayRef input_size,
                                                  c10::optional<double> scales_h,
                                                  c10::optional<double> scales_w) {
  TRACE_INFO(__FUNCTION__);
  auto grad_cpu = grad_output.cpu();
  auto output_cpu =
      at::upsample_nearest2d_backward(grad_cpu, output_size, input_size);
  return output_cpu.to(at::Device(at::Device::Type::MLU));
}

at::Tensor OpMethods::upsample_bilinear2d(const at::Tensor& self,
                                          at::IntArrayRef output_size,
                                          bool align_corners,
                                          c10::optional<double> scales_h,
                                          c10::optional<double> scales_w) {
  TRACE_INFO(__FUNCTION__);
  auto input_cpu = self.cpu();
  auto output_cpu =
      at::upsample_bilinear2d(input_cpu, output_size, align_corners);
  return output_cpu.to(at::Device(at::Device::Type::MLU));
}

at::Tensor& OpMethods::upsample_bilinear2d_out(at::Tensor& out,
                                               const at::Tensor& self,
                                               at::IntArrayRef output_size,
                                               bool align_corners,
                                               c10::optional<double> scales_h,
                                               c10::optional<double> scales_w) {
  TRACE_INFO(__FUNCTION__);
  auto input_cpu = self.cpu();
  auto out_cpu = out.cpu();
  at::upsample_bilinear2d_out(out_cpu, input_cpu, output_size, align_corners);
  auto out_mlu = out_cpu.to(at::Device(at::Device::Type::MLU));
  getMluTensorImpl(out)->copy_cnnl_metadata_from(getMluTensorImpl(out_mlu));
  resize_impl_mlu_(getMluTensorImpl(out), out_mlu.sizes(), out_mlu.strides());
  return out;
}

at::Tensor OpMethods::upsample_bilinear2d_backward(const at::Tensor& grad_output,
                                                   at::IntArrayRef output_size,
                                                   at::IntArrayRef input_size,
                                                   bool align_corners,
                                                   c10::optional<double> scales_h,
                                                   c10::optional<double> scales_w) {
  TRACE_INFO(__FUNCTION__);
  auto grad_cpu = grad_output.cpu();
  auto output_cpu = at::upsample_bilinear2d_backward(
      grad_cpu, output_size, input_size, align_corners);
  return output_cpu.to(at::Device(at::Device::Type::MLU));
}

at::Tensor OpMethods::sigmoid(const at::Tensor& self) {
  TRACE_INFO(__FUNCTION__);
  auto output_cpu = at::sigmoid(self.cpu());
  return output_cpu.to(at::Device(at::Device::Type::MLU));
}

at::Tensor& OpMethods::sigmoid_(at::Tensor& self) {
  TRACE_INFO(__FUNCTION__);
  auto input_cpu = self.cpu();
  at::sigmoid_(input_cpu);
  // copy cpu result into mlu self cpu_storage
  getMluTensorImpl(self)->copy_cnnl_metadata_from(getMluTensorImpl(input_cpu.to("mlu")));
  return self;
}

at::Tensor OpMethods::sigmoid_backward(const at::Tensor& grad_output,
                                       const at::Tensor& output) {
  TRACE_INFO(__FUNCTION__);
  auto grad_cpu = grad_output.cpu();
  auto self_cpu = output.cpu();
  auto output_cpu = at::sigmoid_backward(grad_cpu, self_cpu);
  return output_cpu.to(at::Device(at::Device::Type::MLU));
}

at::Tensor OpMethods::sqrt(const at::Tensor& input) {
  TRACE_INFO(__FUNCTION__);
  auto output_cpu = at::sqrt(input.cpu());
  return output_cpu.to(at::Device(at::Device::Type::MLU));
}

at::Tensor& OpMethods::sqrt_(at::Tensor& input) {
  TRACE_INFO(__FUNCTION__);
  auto input_cpu = input.cpu();
  at::sqrt_(input_cpu);
  getMluTensorImpl(input)->set_cpu_storage(
      input_cpu.unsafeGetTensorImpl()->storage());
  return input;
}

at::Tensor OpMethods::clone(const at::Tensor& input,
                            c10::optional<c10::MemoryFormat> memory_format) {
  TRACE_INFO(__FUNCTION__);
  auto output_cpu = at::clone(input.cpu());
  return output_cpu.to(at::Device(at::Device::Type::MLU));
}

at::Tensor OpMethods::ge(const at::Tensor& self, const at::Tensor& other) {
  TRACE_INFO(__FUNCTION__);
  auto self_cpu = self.cpu();
  auto other_cpu = other.cpu();
  auto result = at::ge(self_cpu, other_cpu);
  return result.to(at::Device(at::Device::Type::MLU));
}

at::Tensor OpMethods::ge(const at::Tensor& self, at::Scalar other) {
  TRACE_INFO(__FUNCTION__);
  auto self_cpu = self.cpu();
  auto result = at::ge(self_cpu, other);
  return result.to(at::Device(at::Device::Type::MLU));
}

at::Tensor& OpMethods::ge_(at::Tensor& self, const at::Tensor& other) {
  TRACE_INFO(__FUNCTION__);
  auto self_cpu = self.cpu();
  auto other_cpu = other.cpu();
  at::native::ge_(self_cpu, other_cpu);
  auto self_mlu = self_cpu.to(at::Device(at::Device::Type::MLU));
  getMluTensorImpl(self)->copy_cnnl_metadata_from(getMluTensorImpl(self_mlu));
  return self;
}

at::Tensor& OpMethods::ge_(at::Tensor& self, at::Scalar other) {
  TRACE_INFO(__FUNCTION__);
  auto self_cpu = self.cpu();
  at::native::ge_(self_cpu, other);
  auto self_mlu = self_cpu.to(at::Device(at::Device::Type::MLU));
  getMluTensorImpl(self)->copy_cnnl_metadata_from(getMluTensorImpl(self_mlu));
  return self;
}

at::Tensor& OpMethods::ge_out(at::Tensor& out, const at::Tensor& self,
                              const at::Tensor& other) {
  TRACE_INFO(__FUNCTION__);
  auto out_cpu = out.cpu();
  auto self_cpu = self.cpu();
  auto other_cpu = other.cpu();
  at::ge_out(out_cpu, self_cpu, other_cpu);
  auto out_mlu = out_cpu.to(at::Device(at::Device::Type::MLU));
  getMluTensorImpl(out)->copy_cnnl_metadata_from(getMluTensorImpl(out_mlu));
  resize_impl_mlu_(getMluTensorImpl(out), out_mlu.sizes(), out_mlu.strides());
  return out;
}

at::Tensor& OpMethods::ge_out(at::Tensor& out, const at::Tensor& self, at::Scalar other) {
  TRACE_INFO(__FUNCTION__);
  auto out_cpu = out.cpu();
  auto self_cpu = self.cpu();
  at::ge_out(out_cpu, self_cpu, other);
  auto out_mlu = out_cpu.to(at::Device(at::Device::Type::MLU));
  getMluTensorImpl(out)->copy_cnnl_metadata_from(getMluTensorImpl(out_mlu));
  resize_impl_mlu_(getMluTensorImpl(out), out_mlu.sizes(), out_mlu.strides());
  return out;
}
at::Tensor OpMethods::le(const at::Tensor& self, const at::Tensor& other) {
  TRACE_INFO(__FUNCTION__);
  auto self_cpu = self.cpu();
  auto other_cpu = other.cpu();
  auto result = at::le(self_cpu, other_cpu);
  return result.to(at::Device(at::Device::Type::MLU));
}

at::Tensor OpMethods::le(const at::Tensor& self, at::Scalar other) {
  TRACE_INFO(__FUNCTION__);
  auto self_cpu = self.cpu();
  auto result = at::le(self_cpu, other);
  return result.to(at::Device(at::Device::Type::MLU));
}

at::Tensor& OpMethods::le_(at::Tensor& self, const at::Tensor& other) {
  TRACE_INFO(__FUNCTION__);
  auto self_cpu = self.cpu();
  auto other_cpu = other.cpu();
  at::native::le_(self_cpu, other_cpu);
  auto self_mlu = self_cpu.to(at::Device(at::Device::Type::MLU));
  getMluTensorImpl(self)->copy_cnnl_metadata_from(getMluTensorImpl(self_mlu));
  return self;
}

at::Tensor& OpMethods::le_(at::Tensor& self, at::Scalar other) {
  TRACE_INFO(__FUNCTION__);
  auto self_cpu = self.cpu();
  at::native::le_(self_cpu, other);
  auto self_mlu = self_cpu.to(at::Device(at::Device::Type::MLU));
  getMluTensorImpl(self)->copy_cnnl_metadata_from(getMluTensorImpl(self_mlu));
  return self;
}

at::Tensor& OpMethods::le_out(at::Tensor& out, const at::Tensor& self,
                                      const at::Tensor& other) {
  TRACE_INFO(__FUNCTION__);
  auto out_cpu = out.cpu();
  auto self_cpu = self.cpu();
  auto other_cpu = other.cpu();
  at::le_out(out_cpu, self_cpu, other_cpu);
  auto out_mlu = out_cpu.to(at::Device(at::Device::Type::MLU));
  getMluTensorImpl(out)->copy_cnnl_metadata_from(getMluTensorImpl(out_mlu));
  resize_impl_mlu_(getMluTensorImpl(out), out_mlu.sizes(), out_mlu.strides());
  return out;
}

at::Tensor& OpMethods::le_out(at::Tensor& out, const at::Tensor& self, at::Scalar other) {
  TRACE_INFO(__FUNCTION__);
  auto out_cpu = out.cpu();
  auto self_cpu = self.cpu();
  at::le_out(out_cpu, self_cpu, other);
  auto out_mlu = out_cpu.to(at::Device(at::Device::Type::MLU));
  getMluTensorImpl(out)->copy_cnnl_metadata_from(getMluTensorImpl(out_mlu));
  resize_impl_mlu_(getMluTensorImpl(out), out_mlu.sizes(), out_mlu.strides());
  return out;
}

at::Tensor OpMethods::neg(const at::Tensor& input) {
  TRACE_INFO(__FUNCTION__);
  auto input_cpu = input.cpu();
  auto output_cpu = at::neg(input_cpu);
  return output_cpu.to(at::Device(at::Device::Type::MLU));
}

at::Tensor& OpMethods::neg_out(at::Tensor& out, const at::Tensor& input) {
    TRACE_INFO(__FUNCTION__);
  auto out_cpu = out.cpu();
  auto input_cpu = input.cpu();
  auto output = at::neg_out(out_cpu, input_cpu);
  auto out_mlu = output.to(at::Device(at::Device::Type::MLU));
  getMluTensorImpl(out)->copy_cnnl_metadata_from(getMluTensorImpl(out_mlu));
  resize_impl_mlu_(getMluTensorImpl(out), out_mlu.sizes(), out_mlu.strides());
  return out;
}

at::Tensor& OpMethods::log_out(at::Tensor& out, const at::Tensor& self) {
  TRACE_INFO(__FUNCTION__);
  auto out_cpu = out.cpu();
  auto input_cpu = self.cpu();
  at::log_out(out_cpu, input_cpu);
  auto out_mlu = out_cpu.to(at::Device(at::Device::Type::MLU));
  getMluTensorImpl(out)->copy_cnnl_metadata_from(getMluTensorImpl(out_mlu));
  resize_impl_mlu_(getMluTensorImpl(out), out_mlu.sizes(), out_mlu.strides());
  return out.copy_(out_cpu);
}

at::Tensor& OpMethods::log2_out(at::Tensor& out, const at::Tensor& self) {
  TRACE_INFO(__FUNCTION__);
  auto out_cpu = out.cpu();
  auto self_cpu = self.cpu();
  at::log2_out(out_cpu, self_cpu);
  auto out_mlu = out_cpu.to(at::Device(at::Device::Type::MLU));
  getMluTensorImpl(out)->copy_cnnl_metadata_from(getMluTensorImpl(out_mlu));
  resize_impl_mlu_(getMluTensorImpl(out), out_mlu.sizes(), out_mlu.strides());
  return out;
}

at::Tensor& OpMethods::log10_out(at::Tensor& out, const at::Tensor& self) {
  TRACE_INFO(__FUNCTION__);
  auto out_cpu = out.cpu();
  auto self_cpu = self.cpu();
  at::log10_out(out_cpu, self_cpu);
  auto out_mlu = out_cpu.to(at::Device(at::Device::Type::MLU));
  getMluTensorImpl(out)->copy_cnnl_metadata_from(getMluTensorImpl(out_mlu));
  resize_impl_mlu_(getMluTensorImpl(out), out_mlu.sizes(), out_mlu.strides());
  return out;
}

at::Tensor OpMethods::log_softmax(const at::Tensor& input, int64_t dim,
                                  bool half_to_float) {
  TRACE_INFO(__FUNCTION__);
  auto input_cpu = input.cpu();
  auto output_cpu = at::_log_softmax(input_cpu, dim, half_to_float);
  return output_cpu.to(at::Device(at::Device::Type::MLU));
}

at::Tensor OpMethods::permute(const at::Tensor& input, at::IntArrayRef dims) {
  TRACE_INFO(__FUNCTION__);
  return cnnl::ops::cnnl_permute(input, dims);
}

at::Tensor OpMethods::cat(at::TensorList tensors, int64_t dim) {
  TRACE_INFO(__FUNCTION__);
  int input_size = tensors.size();
  std::vector<at::Tensor> tensors_cpu;
  for (int i = 0; i < input_size; i++) {
    tensors_cpu.push_back(tensors[i].cpu());
  }
  auto output_cpu = at::cat(tensors_cpu, dim);
  return output_cpu.to(at::Device(at::Device::Type::MLU));
}

at::Tensor& OpMethods::cat_out(at::Tensor& out, at::TensorList tensors,
                               int64_t dim) {
  TRACE_INFO(__FUNCTION__);
  int input_size = tensors.size();
  std::vector<at::Tensor> tensors_cpu;
  for (int i = 0; i < input_size; i++) {
    tensors_cpu.push_back(tensors[i].cpu());
  }
  auto output_cpu = at::cat(tensors_cpu, dim);
  auto out_mlu = output_cpu.to(at::Device(at::Device::Type::MLU));
  getMluTensorImpl(out)->copy_cnnl_metadata_from(getMluTensorImpl(out_mlu));
  resize_impl_mlu_(getMluTensorImpl(out), out_mlu.sizes(), out_mlu.strides());
  return out;
}

at::Tensor OpMethods::clamp_min(const at::Tensor& self,
                                at::Scalar min) {
  TRACE_INFO(__FUNCTION__);
  auto input_cpu = self.cpu();
  auto output_cpu = at::clamp_min(input_cpu, min);
  return output_cpu.to(at::Device(at::Device::Type::MLU));
}

at::Tensor& OpMethods::clamp_min_(at::Tensor& self,
                                 at::Scalar min) {
  TRACE_INFO(__FUNCTION__);
  auto input_cpu = self.cpu();
  at::clamp_min_(input_cpu, min);
  auto out_mlu = input_cpu.to(at::Device(at::Device::Type::MLU));
  getMluTensorImpl(self)->copy_cnnl_metadata_from(getMluTensorImpl(out_mlu));
  return self;
}

at::Tensor& OpMethods::clamp_min_out(at::Tensor& out, const at::Tensor& self,
                                   at::Scalar min) {
  TRACE_INFO(__FUNCTION__);
  auto out_cpu = out.cpu();
  auto input_cpu = self.cpu();
  at::clamp_min_out(out_cpu, input_cpu, min);
  auto out_mlu = out_cpu.to(at::Device(at::Device::Type::MLU));
  getMluTensorImpl(out)->copy_cnnl_metadata_from(getMluTensorImpl(out_mlu));
  resize_impl_mlu_(getMluTensorImpl(out), out_mlu.sizes(), out_mlu.strides());
  return out;
}

at::Tensor OpMethods::binary_cross_entropy_with_logits(
    const at::Tensor& self, const at::Tensor& target, const at::Tensor& weight,
    const at::Tensor& pos_weight, int64_t reduction) {
  TRACE_INFO(__FUNCTION__);
  auto self_cpu = self.cpu();
  auto target_cpu = target.cpu();
  at::Tensor weight_cpu, pos_weight_cpu;
  if (weight.defined()) {
    weight_cpu = weight.cpu();
  }
  if (pos_weight.defined()) {
    pos_weight_cpu = pos_weight.cpu();
  }
  auto output_cpu = at::binary_cross_entropy_with_logits(self_cpu, target_cpu,
      weight_cpu, pos_weight_cpu, reduction);
  return output_cpu.to(at::Device(at::Device::Type::MLU));
}

at::Tensor OpMethods::binary_cross_entropy_with_logits_backward(const at::Tensor& grad_output,
                                        const at::Tensor& self,
                                        const at::Tensor& target,
                                        const at::Tensor& weight,
                                        const at::Tensor& pos_weight,
                                        int64_t reduction) {
  TRACE_INFO(__FUNCTION__);
  auto grad_output_cpu = grad_output.cpu();
  auto self_cpu = self.cpu();
  auto target_cpu = target.cpu();
  auto weight_cpu = weight.cpu();
  auto pos_weight_cpu = pos_weight.cpu();
  auto output_cpu = at::binary_cross_entropy_with_logits_backward(grad_output_cpu, self_cpu,
      target_cpu, weight_cpu, pos_weight_cpu, reduction);
  return output_cpu.to(at::Device(at::Device::Type::MLU));
}

at::Tensor OpMethods::stack(at::TensorList tensors, int64_t dim) {
  TRACE_INFO(__FUNCTION__);
  auto tensors_vec = tensors.vec();
  for (int i = 0; i < tensors_vec.size(); i++)
    tensors_vec[i] = tensors_vec[i].cpu();
  auto tensors_cpu = at::TensorList(tensors_vec);
  auto output_cpu = at::stack(tensors_cpu, dim);
  return output_cpu.to(at::Device(at::Device::Type::MLU));
}

at::Tensor& OpMethods::stack_out(at::Tensor& out,
                                at::TensorList tensors,
                                int64_t dim) {
  TRACE_INFO(__FUNCTION__);
  auto out_cpu = out.cpu();
  auto tensors_vec = tensors.vec();
  for (int i = 0; i < tensors_vec.size(); i++)
    tensors_vec[i] = tensors_vec[i].cpu();
  auto tensors_cpu = at::TensorList(tensors_vec);
  at::stack_out(out_cpu, tensors_cpu, dim);
  auto out_mlu = out_cpu.to(at::Device(at::Device::Type::MLU));
  getMluTensorImpl(out)->copy_cnnl_metadata_from(getMluTensorImpl(out_mlu));
  resize_impl_mlu_(getMluTensorImpl(out), out_mlu.sizes(), out_mlu.strides());
  return out;
}

at::Tensor OpMethods::slice(const at::Tensor& input, int64_t dim, int64_t start,
                            int64_t end, int64_t step) {
  TRACE_INFO(__FUNCTION__);
  auto input_cpu = input.cpu();
  auto output_cpu = at::slice(input_cpu, dim, start, end, step);
  auto output_mlu = output_cpu.to(at::Device(at::Device::Type::MLU));
  return output_mlu;
}

at::Tensor OpMethods::select(const at::Tensor& self, int64_t dim,
                             int64_t index) {
  TRACE_INFO(__FUNCTION__);
  return cnnl::ops::cnnl_select(self, dim, index);
}

at::Tensor OpMethods::_softmax(const at::Tensor& self, int64_t dim,
                               bool half_to_float) {
  TRACE_INFO(__FUNCTION__);
  auto input_cpu = self.cpu();
  auto output_cpu = at::_softmax(input_cpu, dim, half_to_float);
  return output_cpu.to(at::Device(at::Device::Type::MLU));
}

at::Tensor OpMethods::clamp(const at::Tensor& self,
                            at::optional<at::Scalar> min,
                            at::optional<at::Scalar> max) {
  TRACE_INFO(__FUNCTION__);
  auto input_cpu = self.cpu();
  auto output_cpu = at::clamp(input_cpu, min, max);
  return output_cpu.to(at::Device(at::Device::Type::MLU));
}

at::Tensor& OpMethods::clamp_(at::Tensor& self, at::optional<at::Scalar> min,
                              at::optional<at::Scalar> max) {
  TRACE_INFO(__FUNCTION__);
  auto input_cpu = self.cpu();
  at::clamp_(input_cpu, min, max);
  auto out_mlu = input_cpu.to(at::Device(at::Device::Type::MLU));
  getMluTensorImpl(self)->copy_cnnl_metadata_from(getMluTensorImpl(out_mlu));
  return self;
}

at::Tensor& OpMethods::clamp_out(at::Tensor& out, const at::Tensor& self,
                                 at::optional<at::Scalar> min,
                                 at::optional<at::Scalar> max) {
  TRACE_INFO(__FUNCTION__);
  auto out_cpu = out.cpu();
  auto input_cpu = self.cpu();
  at::clamp_out(out_cpu, input_cpu, min, max);
  auto out_mlu = out_cpu.to(at::Device(at::Device::Type::MLU));
  getMluTensorImpl(out)->copy_cnnl_metadata_from(getMluTensorImpl(out_mlu));
  resize_impl_mlu_(getMluTensorImpl(out), out_mlu.sizes(), out_mlu.strides());
  return out;
}

at::Tensor OpMethods::leaky_relu(const at::Tensor& self,
                                 at::Scalar negative_slope) {
  TRACE_INFO(__FUNCTION__);
  auto input_cpu = self.cpu();
  auto output_cpu = at::leaky_relu(input_cpu, negative_slope);
  return output_cpu.to(at::Device(at::Device::Type::MLU));
}

at::Tensor& OpMethods::leaky_relu_(at::Tensor& self,
                                   at::Scalar negative_slope) {
  TRACE_INFO(__FUNCTION__);
  auto input_cpu = self.cpu();
  at::leaky_relu_(input_cpu, negative_slope);
  // copy cpu result into mlu self cpu_storage
  getMluTensorImpl(self)->copy_cnnl_metadata_from(getMluTensorImpl(input_cpu.to("mlu")));
  return self;
}

at::Tensor OpMethods::leaky_relu_backward(const at::Tensor& grad_output,
                                          const at::Tensor& self,
                                          at::Scalar negative_slope,
                                          bool self_is_result) {
  TRACE_INFO(__FUNCTION__);
  auto grad_output_cpu = grad_output.cpu();
  auto input_cpu = self.cpu();
  auto output_cpu = at::leaky_relu_backward(grad_output_cpu, input_cpu, negative_slope,
                                            self_is_result);
  return output_cpu.to(at::Device(at::Device::Type::MLU));
}

at::Tensor& OpMethods::leaky_relu_out(at::Tensor& out, const at::Tensor& self,
                                      at::Scalar negative_slope) {
  TRACE_INFO(__FUNCTION__);
  auto out_cpu = out.cpu();
  auto input_cpu = self.cpu();
  at::leaky_relu_out(out_cpu, input_cpu, negative_slope);
  auto out_mlu = out_cpu.to(at::Device(at::Device::Type::MLU));
  getMluTensorImpl(out)->copy_cnnl_metadata_from(getMluTensorImpl(out_mlu));
  resize_impl_mlu_(getMluTensorImpl(out), out_mlu.sizes(), out_mlu.strides());
  return out;
}

at::Tensor OpMethods::lt(const at::Tensor& self, at::Scalar other) {
  TRACE_INFO(__FUNCTION__);
  auto input_cpu = self.cpu();
  auto output_cpu = at::lt(input_cpu, other);
  return output_cpu.to(at::Device(at::Device::Type::MLU));
}

at::Tensor OpMethods::lt(const at::Tensor& self, const at::Tensor& other) {
  TRACE_INFO(__FUNCTION__);
  auto input_cpu = self.cpu();
  auto other_cpu = other.cpu();
  auto output_cpu = at::lt(input_cpu, other_cpu);
  return output_cpu.to(at::Device(at::Device::Type::MLU));
}

at::Tensor& OpMethods::lt_(at::Tensor& self, at::Scalar other) {
  TRACE_INFO(__FUNCTION__);
  auto input_cpu = self.cpu();
  auto output = input_cpu.lt_(other);
  auto out_mlu = output.to(at::Device(at::Device::Type::MLU));
  return self.copy_(out_mlu);
}

at::Tensor& OpMethods::lt_(at::Tensor& self, const at::Tensor& other) {
  TRACE_INFO(__FUNCTION__);
  auto input_cpu = self.cpu();
  auto output = input_cpu.lt_(other.cpu());
  auto out_mlu = output.to(at::Device(at::Device::Type::MLU));
  return self.copy_(out_mlu);
}

at::Tensor& OpMethods::lt_out(at::Tensor& out, const at::Tensor& self, at::Scalar other) {
  TRACE_INFO(__FUNCTION__);
  auto out_cpu = out.cpu();
  auto input_cpu = self.cpu();
  at::lt_out(out_cpu, input_cpu, other);
  auto out_mlu = out_cpu.to(at::Device(at::Device::Type::MLU));
  getMluTensorImpl(out)->copy_cnnl_metadata_from(getMluTensorImpl(out_mlu));
  resize_impl_mlu_(getMluTensorImpl(out), out_mlu.sizes(), out_mlu.strides());
  return out;
}

at::Tensor& OpMethods::lt_out(at::Tensor& out, const at::Tensor& self, const at::Tensor& other) {
  TRACE_INFO(__FUNCTION__);
  auto out_cpu = out.cpu();
  auto input_cpu = self.cpu();
  auto other_cpu = other.cpu();
  at::lt_out(out_cpu, input_cpu, other_cpu);
  auto out_mlu = out_cpu.to(at::Device(at::Device::Type::MLU));
  getMluTensorImpl(out)->copy_cnnl_metadata_from(getMluTensorImpl(out_mlu));
  resize_impl_mlu_(getMluTensorImpl(out), out_mlu.sizes(), out_mlu.strides());
  return out;
}

at::Tensor OpMethods::pow(const at::Tensor& self, at::Scalar exponent) {
  TRACE_INFO(__FUNCTION__);
  auto input_cpu = self.cpu();
  if (self.type().scalarType() == at::ScalarType::Half) {
    input_cpu = input_cpu.to(at::kFloat);
  }
  auto output_cpu = at::pow(input_cpu, exponent);
  if (self.type().scalarType() == at::ScalarType::Half) {
    output_cpu = output_cpu.to(at::kHalf);
  }
  return output_cpu.to(at::Device(at::Device::Type::MLU));
}

at::Tensor OpMethods::pow(const at::Tensor& self, const at::Tensor& exponent) {
  TRACE_INFO(__FUNCTION__);
  auto input_cpu = self.cpu();
  auto exponent_cpu = exponent.cpu();
  if (self.type().scalarType() == at::ScalarType::Half) {
    input_cpu = input_cpu.to(at::kFloat);
    exponent_cpu = exponent_cpu.to(at::kFloat);
  }
  auto output_cpu = at::pow(input_cpu, exponent_cpu);
  if (self.type().scalarType() == at::ScalarType::Half) {
    output_cpu = output_cpu.to(at::kHalf);
  }
  return output_cpu.to(at::Device(at::Device::Type::MLU));
}

at::Tensor OpMethods::pow(at::Scalar self, const at::Tensor& exponent) {
  TRACE_INFO(__FUNCTION__);
  float input_cpu = self.to<float>();
  auto exponent_cpu = exponent.cpu();
  if (exponent.type().scalarType() == at::ScalarType::Half) {
    exponent_cpu = exponent_cpu.to(at::kFloat);
  }
  auto output_cpu = at::pow(input_cpu, exponent_cpu);
  if (exponent.type().scalarType() == at::ScalarType::Half) {
    output_cpu = output_cpu.to(at::kHalf);
  }
  return output_cpu.to(at::Device(at::Device::Type::MLU));
}

at::Tensor& OpMethods::pow_(at::Tensor& self, at::Scalar exponent) {
  TRACE_INFO(__FUNCTION__);
  auto input_cpu = self.cpu();
  float exponent_cpu = exponent.to<float>();
  if (self.type().scalarType() == at::ScalarType::Half) {
    input_cpu = input_cpu.to(at::kFloat);
  }
  input_cpu.pow_(exponent_cpu);
  if (self.type().scalarType() == at::ScalarType::Half) {
    input_cpu = input_cpu.to(at::kHalf);
  }
  self.copy_(input_cpu);
  return self;
}

at::Tensor& OpMethods::pow_(at::Tensor& self, const at::Tensor& exponent) {
  TRACE_INFO(__FUNCTION__);
  auto input_cpu = self.cpu();
  auto exponent_cpu = exponent.cpu();
  if (self.type().scalarType() == at::ScalarType::Half) {
    input_cpu = input_cpu.to(at::kFloat);
    exponent_cpu = exponent_cpu.to(at::kFloat);
  }

  input_cpu.pow_(exponent_cpu);
  if (self.type().scalarType() == at::ScalarType::Half) {
    input_cpu = input_cpu.to(at::kHalf);
  }
  self.copy_(input_cpu);
  return self;
}

at::Tensor OpMethods::matmul(const at::Tensor& self, const at::Tensor& other) {
  TRACE_INFO(__FUNCTION__);
  auto input_cpu = self.cpu();
  auto other_cpu = other.cpu();
  auto output_cpu = at::matmul(input_cpu, other_cpu);
  return output_cpu.to(at::Device(at::Device::Type::MLU));
}

at::Tensor OpMethods::mm(const at::Tensor& self, const at::Tensor& other) {
  TRACE_INFO(__FUNCTION__);
  auto input_cpu = self.cpu();
  auto other_cpu = other.cpu();
  auto output_cpu = at::mm(input_cpu, other_cpu);
  return output_cpu.to(at::Device(at::Device::Type::MLU));
}

at::Tensor& OpMethods::mm_out(at::Tensor& out, const at::Tensor& self,
        const at::Tensor& other) {
    TRACE_INFO(__FUNCTION__);
    auto input_cpu = self.cpu();
    auto other_cpu = other.cpu();
    auto out_cpu = out.cpu();
    at::mm_out(out_cpu, input_cpu, other_cpu);
    auto out_mlu = out_cpu.to(at::Device(at::Device::Type::MLU));
    getMluTensorImpl(out)->copy_cnnl_metadata_from(getMluTensorImpl(out_mlu));
    resize_impl_mlu_(getMluTensorImpl(out), out_mlu.sizes(), out_mlu.strides());
    return out;
}

at::Tensor OpMethods::bmm(const at::Tensor& self, const at::Tensor& other) {
  TRACE_INFO(__FUNCTION__);
  auto input_cpu = self.cpu();
  auto other_cpu = other.cpu();
  auto output_cpu = at::bmm(input_cpu, other_cpu);
  return output_cpu.to(at::Device(at::Device::Type::MLU));
}

bool OpMethods::equal(const at::Tensor& self, const at::Tensor& other) {
  TRACE_INFO(__FUNCTION__);
  auto self_cpu = (self.scalar_type() == c10::ScalarType::Half)
                      ? self.cpu().toType(c10::ScalarType::Float)
                      : self.cpu();
  auto other_cpu = (other.scalar_type() == c10::ScalarType::Half)
                       ? other.cpu().toType(c10::ScalarType::Float)
                       : other.cpu();
  auto output_value = at::equal(self_cpu, other_cpu);
  return output_value;
}

at::Tensor OpMethods::isfinite(const at::Tensor& self) {
  TRACE_INFO(__FUNCTION__);
  auto input_cpu = self.cpu();
  auto output_cpu = at::isfinite(input_cpu);
  return output_cpu.to(at::Device(at::Device::Type::MLU));
}

at::Tensor OpMethods::_softmax_backward_data(const at::Tensor& grad_output,
                                             const at::Tensor& output,
                                             int64_t dim,
                                             const at::Tensor& self) {
  TRACE_INFO(__FUNCTION__);
  auto grad_output_cpu = grad_output.cpu();
  auto output_cpu = output.cpu();
  auto self_cpu = self.cpu();
  auto out_cpu = at::_softmax_backward_data(grad_output_cpu, output_cpu, dim, self_cpu);
  return out_cpu.to(at::Device(at::Device::Type::MLU));
}

at::Tensor OpMethods::_log_softmax_backward_data(const at::Tensor& grad_output,
                                                 const at::Tensor& output,
                                                 int64_t dim,
                                                 const at::Tensor& self) {
  TRACE_INFO(__FUNCTION__);
  auto grad_output_cpu = grad_output.cpu();
  auto output_cpu = output.cpu();
  auto self_cpu = self.cpu();
  auto out_cpu = at::_log_softmax_backward_data(grad_output_cpu, output_cpu, dim, self_cpu);
  return out_cpu.to(at::Device(at::Device::Type::MLU));
}

at::Tensor OpMethods::smooth_l1_loss_forward(const at::Tensor& self,
                                             const at::Tensor& target,
                                             int64_t reduction) {
  TRACE_INFO(__FUNCTION__);
  auto self_cpu = self.cpu();
  auto target_cpu = target.cpu();
  auto out_cpu = at::smooth_l1_loss(self_cpu, target_cpu, reduction);
  return out_cpu.to(at::Device(at::Device::Type::MLU));
}

// this interface is used in C++
at::Tensor& OpMethods::smooth_l1_loss_forward_out(at::Tensor& output,
                                                  const at::Tensor& self,
                                                  const at::Tensor& target,
                                                  int64_t reduction) {
  TRACE_INFO(__FUNCTION__);
  auto output_cpu = output.cpu();
  auto self_cpu = self.cpu();
  auto target_cpu = target.cpu();
  at::smooth_l1_loss_out(output_cpu, self_cpu, target_cpu, reduction);
  auto out_mlu = output_cpu.to(at::Device(at::Device::Type::MLU));
  getMluTensorImpl(output)->copy_cnnl_metadata_from(getMluTensorImpl(out_mlu));
  resize_impl_mlu_(getMluTensorImpl(output), out_mlu.sizes(), out_mlu.strides());
  return output;
}

at::Tensor OpMethods::smooth_l1_loss_backward(const at::Tensor& grad_output,
                                              const at::Tensor& self,
                                              const at::Tensor& target,
                                              int64_t reduction) {
  TRACE_INFO(__FUNCTION__);
  auto grad_output_cpu = grad_output.cpu();
  auto self_cpu = self.cpu();
  auto target_cpu = target.cpu();
  auto output = at::smooth_l1_loss_backward(grad_output_cpu, self_cpu, target_cpu, reduction);
  return output.to(at::Device(at::Device::Type::MLU));
}

at::Tensor& OpMethods::smooth_l1_loss_backward_out(at::Tensor& grad_input,
                                                  const at::Tensor& grad_output,
                                                  const at::Tensor& self,
                                                  const at::Tensor& target,
                                                  int64_t reduction) {
  TRACE_INFO(__FUNCTION__);
  auto grad_input_cpu = grad_input.cpu();
  auto grad_output_cpu = grad_output.cpu();
  auto self_cpu = self.cpu();
  auto target_cpu = target.cpu();
  at::smooth_l1_loss_backward_out(grad_input_cpu, grad_output_cpu, self_cpu, target_cpu, reduction);
  auto out_mlu = grad_input_cpu.to(at::Device(at::Device::Type::MLU));
  getMluTensorImpl(grad_input)->copy_cnnl_metadata_from(getMluTensorImpl(out_mlu));
  resize_impl_mlu_(getMluTensorImpl(grad_input), out_mlu.sizes(), out_mlu.strides());
  return grad_input;
}

std::tuple<at::Tensor, at::Tensor> OpMethods::nll_loss_forward(
    const at::Tensor& self, const at::Tensor& target, const at::Tensor& weight,
    int64_t reduction, int64_t ignore_index) {
  TRACE_INFO(__FUNCTION__);
  auto self_cpu = self.cpu();
  auto target_cpu = target.cpu();
  auto weight_cpu = weight.cpu();
  auto output_cpu = at::nll_loss_forward(self_cpu, target_cpu, weight_cpu, reduction, ignore_index);
  return std::make_tuple(std::get<0>(output_cpu).to(at::Device(at::Device::Type::MLU)),
                        std::get<1>(output_cpu).to(at::Device(at::Device::Type::MLU)));
}

std::tuple<at::Tensor, at::Tensor> OpMethods::nll_loss2d_forward(
    const at::Tensor& self, const at::Tensor& target, const at::Tensor& weight,
    int64_t reduction, int64_t ignore_index) {
  TRACE_INFO(__FUNCTION__);
  auto self_cpu = self.cpu();
  auto target_cpu = target.cpu();
  auto output_cpu = at::nll_loss2d_forward(self_cpu, target_cpu,
                                          weight.defined() ? weight.cpu() : weight,
                                          reduction, ignore_index);
  return std::make_tuple(std::get<0>(output_cpu).to(at::Device(at::Device::Type::MLU)),
                        std::get<1>(output_cpu).to(at::Device(at::Device::Type::MLU)));
}

at::Tensor OpMethods::nll_loss_backward(const at::Tensor& grad_output,
                                        const at::Tensor& self,
                                        const at::Tensor& target,
                                        const at::Tensor& weight,
                                        int64_t reduction, int64_t ignore_index,
                                        const at::Tensor& total_weight) {
  TRACE_INFO(__FUNCTION__);
  auto grad_output_cpu = grad_output.cpu();
  auto self_cpu = self.cpu();
  auto target_cpu = target.cpu();
  auto weight_cpu = weight.cpu();
  auto total_weight_cpu = total_weight.cpu();
  auto out_cpu = at::nll_loss_backward(grad_output_cpu, self_cpu, target_cpu, weight_cpu,
                                      reduction, ignore_index, total_weight_cpu);
  return out_cpu.to(at::Device(at::Device::Type::MLU));
}

at::Tensor OpMethods::nll_loss2d_backward(const at::Tensor& grad_output,
                                          const at::Tensor& self,
                                          const at::Tensor& target,
                                          const at::Tensor& weight,
                                          int64_t reduction, int64_t ignore_index,
                                          const at::Tensor& total_weight) {
  TRACE_INFO(__FUNCTION__);
  auto grad_cpu = grad_output.cpu();
  auto self_cpu = self.cpu();
  auto target_cpu = target.cpu();
  auto out_cpu = at::nll_loss2d_backward(grad_cpu, self_cpu, target_cpu,
                                        weight.defined() ? weight.cpu() : weight, reduction,
                                        ignore_index, total_weight.defined() ?
                                        total_weight.cpu() : total_weight);
  return out_cpu.to(at::Device(at::Device::Type::MLU));
}

at::Tensor& OpMethods::binary_cross_entropy_out(at::Tensor & out,
    const at::Tensor& self, const at::Tensor& target, const at::Tensor& weight,
    int64_t reduction) {
  TRACE_INFO(__FUNCTION__);
  auto out_cpu = out.cpu();
  auto self_cpu = self.cpu();
  auto target_cpu = target.cpu();
  auto weight_cpu = weight.cpu();
  auto output_cpu = at::binary_cross_entropy_out(out_cpu, self_cpu, target_cpu,
      weight_cpu, reduction);
  auto output = output_cpu.to(at::Device(at::Device::Type::MLU));
  getMluTensorImpl(out)->copy_cnnl_metadata_from(getMluTensorImpl(output));
  resize_impl_mlu_(getMluTensorImpl(out), output.sizes(), output.strides());
  return out;
}

at::Tensor OpMethods::binary_cross_entropy(
    const at::Tensor& self, const at::Tensor& target, const at::Tensor& weight,
    int64_t reduction) {
  TRACE_INFO(__FUNCTION__);
  auto self_cpu = self.cpu();
  auto target_cpu = target.cpu();
  auto weight_cpu = weight.cpu();
  auto output_cpu = at::binary_cross_entropy(self_cpu, target_cpu,
      weight_cpu, reduction);
  return output_cpu.to(at::Device(at::Device::Type::MLU));
}

at::Tensor & OpMethods::binary_cross_entropy_backward_out(at::Tensor & grad_input,
                                                       const at::Tensor& grad_output,
                                                       const at::Tensor& self,
                                                       const at::Tensor& target,
                                                       const at::Tensor& weight,
                                                       int64_t reduction) {
  TRACE_INFO(__FUNCTION__);
  auto grad_input_cpu = grad_input.cpu();
  auto grad_output_cpu = grad_output.cpu();
  auto self_cpu = self.cpu();
  auto target_cpu = target.cpu();
  auto weight_cpu = weight.cpu();
  auto output_cpu = at::binary_cross_entropy_backward_out(grad_input_cpu, grad_output_cpu, self_cpu,
      target_cpu, weight_cpu, reduction);
  auto output = output_cpu.to(at::Device(at::Device::Type::MLU));
  getMluTensorImpl(grad_input)->copy_cnnl_metadata_from(getMluTensorImpl(output));
  resize_impl_mlu_(getMluTensorImpl(grad_input), output.sizes(), output.strides());
  return grad_input;
}

at::Tensor OpMethods::binary_cross_entropy_backward(const at::Tensor& grad_output,
                                        const at::Tensor& self,
                                        const at::Tensor& target,
                                        const at::Tensor& weight,
                                        int64_t reduction) {
  TRACE_INFO(__FUNCTION__);
  auto grad_output_cpu = grad_output.cpu();
  auto self_cpu = self.cpu();
  auto target_cpu = target.cpu();
  auto weight_cpu = weight.cpu();
  auto output_cpu = at::binary_cross_entropy_backward(grad_output_cpu, self_cpu,
      target_cpu, weight_cpu, reduction);
  return output_cpu.to(at::Device(at::Device::Type::MLU));
}

at::Tensor& OpMethods::zero_(at::Tensor& self) {
  TRACE_INFO(__FUNCTION__);
  auto self_cpu = self.cpu();
  self_cpu.zero_();
  self.copy_(self_cpu);
  return self;
}

at::Tensor OpMethods::diag(const at::Tensor& self, int64_t k) {
  TRACE_INFO(__FUNCTION__);
  auto input_cpu = self.cpu();
  auto output_cpu = at::diag(input_cpu, k);
  return output_cpu.to(at::Device(at::Device::Type::MLU));
}

at::Tensor& OpMethods::diag_out(at::Tensor& result, const at::Tensor& self,
                                int64_t k) {
  TRACE_INFO(__FUNCTION__);
  auto out_cpu = result.cpu();
  auto input_cpu = self.cpu();
  auto output = at::diag_out(out_cpu, input_cpu, k);
  if (result.numel() < output.numel()) {
    throw std::invalid_argument("OpMethods::diag_out fail, "
      "because the output expected more memory!");
  }
  resize_impl_mlu_(getMluTensorImpl(result), output.sizes(), c10::nullopt);
  result.copy_(output);
  return result;
}

at::Tensor OpMethods::convolution_overrideable(
    const at::Tensor& input, const at::Tensor& weight, const at::Tensor& bias,
    at::IntArrayRef stride, at::IntArrayRef padding, at::IntArrayRef dilation,
    bool transposed, at::IntArrayRef output_padding, int64_t groups) {
  TRACE_INFO(__FUNCTION__);
  throw std::invalid_argument("To do for CPU");
}

std::tuple<at::Tensor, at::Tensor, at::Tensor>
OpMethods::convolution_backward_overrideable(
    const at::Tensor& grad, const at::Tensor& input, const at::Tensor& weight,
    at::IntArrayRef stride, at::IntArrayRef padding, at::IntArrayRef dilation,
    bool transposed, at::IntArrayRef output_padding, int64_t groups,
    std::array<bool, 3> output_mask) {
  TRACE_INFO(__FUNCTION__);
  throw std::invalid_argument("To do for CPU");
}

at::Tensor& OpMethods::masked_fill_(at::Tensor& input, const at::Tensor& mask,
                                    const at::Tensor& value) {
  TRACE_INFO(__FUNCTION__);
  auto input_cpu = input.cpu();
  auto mask_cpu = mask.cpu();
  auto value_cpu = value.cpu();
  input_cpu.masked_fill_(mask_cpu, value_cpu);
  getMluTensorImpl(input)->copy_cnnl_metadata_from(
          getMluTensorImpl(input_cpu.to(at::Device(at::Device::Type::MLU))));
  return input;
}

at::Tensor& OpMethods::masked_fill_(at::Tensor& input, const at::Tensor& mask,
                                    const at::Scalar value) {
  TRACE_INFO(__FUNCTION__);
  auto input_cpu = input.cpu();
  auto mask_cpu = mask.cpu();
  input_cpu.masked_fill_(mask_cpu, value);
  getMluTensorImpl(input)->copy_cnnl_metadata_from(
          getMluTensorImpl(input_cpu.to(at::Device(at::Device::Type::MLU))));
  return input;
}

at::Tensor OpMethods::masked_select(const at::Tensor& input,
                                    const at::Tensor& mask) {
  TRACE_INFO(__FUNCTION__);
  auto input_cpu = input.cpu();
  auto mask_cpu = mask.cpu();
  auto output_cpu = at::masked_select(input_cpu, mask_cpu);
  return output_cpu.to(at::Device(at::Device::Type::MLU));
}

at::Tensor& OpMethods::masked_select_out(at::Tensor& out, const at::Tensor& self,
                               const at::Tensor& mask) {
  TRACE_INFO(__FUNCTION__);
  auto out_cpu = out.cpu();
  auto self_cpu = self.cpu();
  auto mask_cpu = mask.cpu();
  at::masked_select_out(out_cpu, self_cpu, mask_cpu);
  auto out_mlu = out_cpu.to(at::Device(at::Device::Type::MLU));
  getMluTensorImpl(out)->copy_cnnl_metadata_from(getMluTensorImpl(out_mlu));
  resize_impl_mlu_(getMluTensorImpl(out), out_mlu.sizes(), out_mlu.strides());
  return out;
}

at::Tensor OpMethods::alias(const at::Tensor& self) {
  TRACE_INFO(__FUNCTION__);
  auto self_cpu = self.cpu();
  auto output_cpu = at::alias(self_cpu);
  return output_cpu.to(at::Device(at::Device::Type::MLU));
}

bool OpMethods::dump(const at::Tensor& input) {
  TRACE_INFO(__FUNCTION__);
  throw std::invalid_argument("To do for CPU");
}

at::Tensor OpMethods::empty_pinned(at::IntArrayRef size,
                                   const at::TensorOptions& options) {
  // empty_pinned will generate a cpu-pinned tensor
  // instead MLU tensor using empty
  return pinMemoryEmpty(size, options);
}

at::Tensor& OpMethods::bitwise_or_out(at::Tensor& out,
                                       const at::Tensor& self,
                                       const at::Tensor& other) {
  TRACE_INFO(__FUNCTION__);
  auto self_cpu = self.cpu();
  auto other_cpu = other.cpu();
  at::Tensor out_cpu;
  if (out.data_ptr() == nullptr) {
    out_cpu = at::empty({0}, out.options().device(c10::DeviceType::CPU));
  } else {
    out_cpu = out.cpu();
  }
  at::bitwise_or_out(out_cpu, self_cpu, other_cpu);
  auto out_mlu = out_cpu.to(at::Device(at::Device::Type::MLU));
  // must return reference, can not return a new tensor
  getMluTensorImpl(out)->copy_cnnl_metadata_from(getMluTensorImpl(out_mlu));
  resize_impl_mlu_(getMluTensorImpl(out), out_mlu.sizes(), out_mlu.strides());
  return out;
}

at::Tensor& OpMethods::bitwise_or_out(at::Tensor& out,
                                       const at::Tensor& self,
                                       at::Scalar other) {
  TRACE_INFO(__FUNCTION__);
  auto self_cpu = self.cpu();
  at::Tensor out_cpu;
  if (out.data_ptr() == nullptr) {
    out_cpu = at::empty({0}, out.options().device(c10::DeviceType::CPU));
  } else {
    out_cpu = out.cpu();
  }
  at::bitwise_or_out(out_cpu, self_cpu, other);
  auto out_mlu = out_cpu.to(at::Device(at::Device::Type::MLU));
  // must return reference, can not return a new tensor
  getMluTensorImpl(out)->copy_cnnl_metadata_from(getMluTensorImpl(out_mlu));
  resize_impl_mlu_(getMluTensorImpl(out), out_mlu.sizes(), out_mlu.strides());
  return out;
}

at::Tensor OpMethods::index(const at::Tensor& self, at::TensorList indices) {
  TRACE_INFO(__FUNCTION__);
  auto self_cpu = self.cpu();
  int indices_size = indices.size();
  std::vector<at::Tensor> indices_cpu;
  for (int i = 0; i < indices_size; i++) {
    indices_cpu.push_back(indices[i].cpu());
  }
  auto output_cpu = at::index(self_cpu, indices_cpu);
  return output_cpu.to(at::Device(at::Device::Type::MLU));
}

at::Tensor OpMethods::nms(const at::Tensor & dets,
                          const at::Tensor & scores,
                          double iou_threshold) {
  TRACE_INFO(__FUNCTION__);
  throw std::invalid_argument("To implement nms op on CPU");
}

at::Tensor OpMethods::remainder(const at::Tensor& self, at::Scalar other) {
  TRACE_INFO(__FUNCTION__);
  auto self_cpu = self.cpu();
  auto output_cpu = at::remainder(self_cpu, other);
  return output_cpu.to(at::Device(at::Device::Type::MLU));
}

at::Tensor OpMethods::remainder(const at::Tensor& self, const at::Tensor& other) {
  TRACE_INFO(__FUNCTION__);
  auto self_cpu = self.cpu();
  auto other_cpu = other.cpu();
  auto output_cpu = at::remainder(self_cpu, other_cpu);
  return output_cpu.to(at::Device(at::Device::Type::MLU));
}

at::Tensor& OpMethods::remainder_(at::Tensor& self, at::Scalar other) {
  TRACE_INFO(__FUNCTION__);
  auto self_cpu = self.cpu();
  at::native::remainder_(self_cpu, other);
  auto self_mlu = self_cpu.to(at::Device(at::Device::Type::MLU));
  getMluTensorImpl(self)->copy_cnnl_metadata_from(getMluTensorImpl(self_mlu));
  return self;
}

at::Tensor& OpMethods::remainder_(at::Tensor& self, const at::Tensor& other) {
  TRACE_INFO(__FUNCTION__);
  auto self_cpu = self.cpu();
  auto other_cpu = other.cpu();
  at::native::remainder_(self_cpu, other_cpu);
  auto self_mlu = self_cpu.to(at::Device(at::Device::Type::MLU));
  getMluTensorImpl(self)->copy_cnnl_metadata_from(getMluTensorImpl(self_mlu));
  return self;
}

at::Tensor& OpMethods::remainder_out(at::Tensor& out, const at::Tensor& self, at::Scalar other) {
  TRACE_INFO(__FUNCTION__);
  auto out_cpu = out.cpu();
  auto self_cpu = self.cpu();
  at::remainder_out(out_cpu, self_cpu, other);
  auto out_mlu = out_cpu.to(at::Device(at::Device::Type::MLU));
  getMluTensorImpl(out)->copy_cnnl_metadata_from(getMluTensorImpl(out_mlu));
  resize_impl_mlu_(getMluTensorImpl(out), out_mlu.sizes(), out_mlu.strides());
  return out;
}

at::Tensor& OpMethods::remainder_out(at::Tensor& out,
                                     const at::Tensor& self,
                                     const at::Tensor& other) {
  TRACE_INFO(__FUNCTION__);
  auto out_cpu = out.cpu();
  auto self_cpu = self.cpu();
  auto other_cpu = other.cpu();
  at::remainder_out(out_cpu, self_cpu, other_cpu);
  auto out_mlu = out_cpu.to(at::Device(at::Device::Type::MLU));
  getMluTensorImpl(out)->copy_cnnl_metadata_from(getMluTensorImpl(out_mlu));
  resize_impl_mlu_(getMluTensorImpl(out), out_mlu.sizes(), out_mlu.strides());
  return out;
}

at::Tensor OpMethods::randn(at::IntArrayRef size,
                            const at::TensorOptions& options) {
  auto input_options = options.device(at::Device(at::Device::Type::CPU));
  auto out_cpu = at::randn(size, input_options);
  auto out_mlu = out_cpu.to(at::Device(at::Device::Type::MLU));
  return out_mlu;
}

at::Tensor OpMethods::avg_pool3d_backward(const at::Tensor & grad_output,
                                          const at::Tensor & self,
                                          at::IntArrayRef kernel_size,
                                          at::IntArrayRef stride,
                                          at::IntArrayRef padding,
                                          bool ceil_mode,
                                          bool count_include_pad,
                                          c10::optional<int64_t> divisor_override) {
  TRACE_INFO(__FUNCTION__);
  auto input_cpu = self.cpu();
  auto grad_cpu = grad_output.cpu();
  auto output_cpu = at::avg_pool3d_backward(grad_cpu, input_cpu, kernel_size,
      stride, padding, ceil_mode, count_include_pad, divisor_override);
  return output_cpu.to(at::Device(at::Device::Type::MLU));
}

at::Tensor OpMethods::adaptive_avg_pool3d(const at::Tensor & self,
                                          at::IntArrayRef output_size) {
  TRACE_INFO(__FUNCTION__);
  auto input_cpu = self.cpu();
  auto output_cpu = at::adaptive_avg_pool3d(input_cpu, output_size);
  return output_cpu.to(at::Device(at::Device::Type::MLU));
}

at::Tensor OpMethods::adaptive_avg_pool3d_backward(const at::Tensor & grad_output,
                                                   const at::Tensor & self) {
  TRACE_INFO(__FUNCTION__);
  auto grad_cpu = grad_output.cpu();
  auto self_cpu = self.cpu();
  auto out_cpu = at::adaptive_avg_pool3d_backward(grad_cpu, self_cpu);
  return out_cpu.to(at::Device(at::Device::Type::MLU));
}

at::Tensor OpMethods::mse_loss(const at::Tensor & self,
                               const at::Tensor & target,
                               int64_t reduction) {
  TRACE_INFO(__FUNCTION__);
  auto self_cpu = self.cpu();
  auto target_cpu = target.cpu();
  auto out_cpu = at::mse_loss(self_cpu, target_cpu, reduction);
  return out_cpu.to(at::Device(at::Device::Type::MLU));
}

at::Tensor OpMethods::mse_loss_backward(const at::Tensor & grad_output, const at::Tensor & self,
                             const at::Tensor & target, int64_t reduction) {
  TRACE_INFO(__FUNCTION__);
  auto grad_cpu = grad_output.cpu();
  auto self_cpu = self.cpu();
  auto target_cpu = target.cpu();
  auto out_cpu = at::mse_loss_backward(grad_cpu, self_cpu, target_cpu, reduction);
  return out_cpu.to(at::Device(at::Device::Type::MLU));
}

at::Tensor OpMethods::true_divide(const at::Tensor & self, const at::Tensor & other) {
  auto out = at::true_divide(self.cpu(), other.cpu());
  return out.to(at::Device(at::Device::Type::MLU));
}

at::Tensor& OpMethods::true_divide_inplace(at::Tensor & self, const at::Tensor & other) {
  auto input_cpu = self.cpu();
  input_cpu.true_divide_(other);
  auto self_mlu = input_cpu.to(at::Device(at::Device::Type::MLU));
  getMluTensorImpl(self)->copy_cnnl_metadata_from(getMluTensorImpl(self_mlu));
  return self;
}

at::Tensor & OpMethods::true_divide_out(at::Tensor & out, const at::Tensor & self,
        const at::Tensor & other) {
  auto out_cpu = out.cpu();
  auto input_cpu = self.cpu();
  auto other_cpu = other.cpu();
  at::true_divide(input_cpu, other_cpu);
  auto out_mlu = out_cpu.to(at::Device(at::Device::Type::MLU));
  getMluTensorImpl(out)->copy_cnnl_metadata_from(getMluTensorImpl(out_mlu));
  resize_impl_mlu_(getMluTensorImpl(out), out_mlu.sizes(), out_mlu.strides());
  return out;
}

at::Tensor OpMethods::view(const at::Tensor & input, c10::ScalarType dtype) {
  TRACE_INFO(__FUNCTION__);
  throw std::invalid_argument("To do for CPU");
}

}  // namespace torch_mlu
