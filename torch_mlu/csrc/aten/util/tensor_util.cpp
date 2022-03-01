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

#include <c10/util/TypeCast.h>
#include "aten/device/device.h"
#include "aten/device/mlu_guard.h"
#include "aten/device/queue.h"
#include "aten/util/memory_allocator.h"
#include "aten/util/python_interface.h"
#include "aten/util/tensor_util.h"

namespace torch_mlu {

MLUTensorImpl* getMluTensorImpl(const at::Tensor& tensor) {
  auto tensor_impl = tensor.unsafeGetTensorImpl();
  if (tensor_impl->device_type() != c10::DeviceType::MLU) {
    CNLOG(ERROR) << "\n\n"
      << "The device type of tensor is not 'mlu'. \n"
      << "Please check the python code where the 'result = mlu_model(input)' is called.\n"
      << "Please make sure the input.device is 'device(type='mlu', index=0)'.\n\n";
  }
  MLUTensorImpl* impl = dynamic_cast<MLUTensorImpl*>(tensor_impl);
  TORCH_CHECK(impl != nullptr, "dynamic_cast to MLUTensorImpl failed");
  return impl;
}

void copy_to_cpu_cnnl(at::Tensor& dst, const at::Tensor& src,
                      c10::MemoryFormat memory_format) {
  if (dst.numel() != src.numel()) {
    CNLOG(ERROR) << "Currently only support equal elements D2H copy, but got dst.numel() "
      << dst.numel() << " src.numel() " << src.numel();
  }
  TORCH_CHECK(dst.is_non_overlapping_and_dense() && src.is_non_overlapping_and_dense()
              && (dst.strides() == src.strides()),
              "cnrtMemcpy don't suppoert stride, "
              "dst and src must be non_overlapping_and_dense in D2H.");
  auto dst_impl = dst.unsafeGetTensorImpl();
  auto src_impl = getMluTensorImpl(src);
  size_t tensor_size = src.numel() * src_impl->itemsize();
  if (src.numel() == 0) return;

  if (src.data_ptr() == nullptr) {
    memcpy(dst.data_ptr(), src_impl->cpu_data(), tensor_size);
    return;
  }

  torch_mlu::mlu::MLUGuard guard(src.device().index());
  // init
  cnrtDataType_t src_data_type = CnnlType2CnrtType(src_impl->getCnnlType());
  cnrtDataType_t dst_data_type = toCnrtDtype(src_impl->dtype());
  size_t descriptor_size = src.numel() * cnrtDataTypeSize(src_data_type);

  void* output_ptr = dst.data_ptr();
  // if src_data_type == dst_data_type, directly cnrtMemcpyAsync to output_ptr.
  if (src_data_type == dst_data_type) {
    // copy data from device to host
    void* src_addr = static_cast<void*>(
        static_cast<char*>(src.storage().data()) +
        cnrtDataTypeSize(src_data_type) * src.storage_offset());
    auto queue = getCurrentQueue();
    TORCH_CNRT_CHECK(cnrtMemcpyAsync(output_ptr, src_addr, descriptor_size,
                                queue.queue(), CNRT_MEM_TRANS_DIR_DEV2HOST));
    queue.synchronize();
    return;
  }

  auto cast_unique_ptr = caffe2::make_unique<char[]>(descriptor_size);
  void* cast_cpu_ptr = cast_unique_ptr.get();
  // copy data from device to host
  void* src_addr = static_cast<void*>(
      static_cast<char*>(src.storage().data()) +
      cnrtDataTypeSize(src_data_type) * src.storage_offset());
  auto queue = getCurrentQueue();
  TORCH_CNRT_CHECK(cnrtMemcpyAsync(cast_cpu_ptr, src_addr, descriptor_size,
                              queue.queue(), CNRT_MEM_TRANS_DIR_DEV2HOST));
  queue.synchronize();

  // cnrtCastDataType
  if (src_data_type == CNRT_INT32 && dst_data_type == CNRT_INT64) {
    for (int i = 0; i < src.numel(); ++i) {
      static_cast<int64_t*>(output_ptr)[i] =
          static_cast<int32_t*>(cast_cpu_ptr)[i];
    }
  } else if (src_data_type == CNRT_INT16 && dst_data_type == CNRT_INT64) {
    for (int i = 0; i < src.numel(); ++i) {
      static_cast<int64_t*>(output_ptr)[i] =
          static_cast<int16_t*>(cast_cpu_ptr)[i];
    }
  } else if (src_data_type == CNRT_FLOAT32 && dst_data_type == CNRT_FLOAT64) {
    for (int i = 0; i < src.numel(); ++i) {
      static_cast<double*>(output_ptr)[i] =
          static_cast<float*>(cast_cpu_ptr)[i];
    }
  } else if (src_data_type == CNRT_INT32 && dst_data_type == CNRT_INT16) {
    for (int i = 0; i < src.numel(); ++i) {
      static_cast<int16_t*>(output_ptr)[i] =
          static_cast<int32_t*>(cast_cpu_ptr)[i];
    }
  } else if (src_data_type == CNRT_INT32 && dst_data_type == CNRT_INT8) {
    for (int i = 0; i < src.numel(); ++i) {
      static_cast<int8_t*>(output_ptr)[i] =
          static_cast<int32_t*>(cast_cpu_ptr)[i];
    }
  } else if (src_data_type == CNRT_INT32 && dst_data_type == CNRT_BOOL) {
    for (int i = 0; i < src.numel(); ++i) {
      static_cast<bool*>(output_ptr)[i] =
          static_cast<int32_t*>(cast_cpu_ptr)[i];
    }
  } else if (src_data_type != dst_data_type) {
    // TODO(liuyuxin): cnrtCastDataType is not recommoned in future, will be
    // deprecated.
    TORCH_CNRT_CHECK(cnrtCastDataType(cast_cpu_ptr, src_data_type, output_ptr,
                                      dst_data_type, src.numel(),
                                      nullptr));
  } else {
    memcpy(output_ptr, cast_cpu_ptr, tensor_size);
  }
}

void copy_from_cpu(at::Tensor& dst, const at::Tensor& src,
                   bool non_blocking,
                   c10::MemoryFormat memory_format) {
  TORCH_CHECK(dst.is_non_overlapping_and_dense() && src.is_non_overlapping_and_dense()
              && (dst.strides() == src.strides()),
              "cnrtMemcpy don't suppoert stride, "
              "dst and src must be non_overlapping_and_dense in H2D.");
  ShareCpuData(dst, src);
  if (PythonInterface::getRunningMode() == RunningMode::CNNL) {
      auto tensor_impl = getMluTensorImpl(dst);
      tensor_impl->cnnlMalloc(non_blocking);
  }
}

bool is_mlu(const at::Tensor& t) { return is_mlu(t.unsafeGetTensorImpl()); }

bool is_mlu(const c10::TensorImpl* t) {
  return t->device_type() == c10::DeviceType::MLU;
}

void ShareCpuData(at::Tensor& dst, const at::Tensor& src) {
  auto dst_impl = dst.unsafeGetTensorImpl();
  auto src_impl = src.unsafeGetTensorImpl();
  if (!src_impl->dtype_initialized()) {
    C10_LOG_EVERY_MS(WARNING, 1000)
        << "Source tensor don't have a data type (did you call mutable_data<T> "
           "on the tensor?)";
  }
  AT_ASSERTM(src_impl->storage_initialized(),
             "Source tensor has no content and has size > 0");
  AT_ASSERTM(is_mlu(dst_impl), "Destination tensor must be MLU Tensor");

  if (is_mlu(src_impl)) {
    dst_impl->set_storage_keep_dtype(src_impl->storage());
  } else {
    // copy from cpu to mlu
    dynamic_cast<MLUTensorImpl*>(dst_impl)->set_cpu_storage(
        src_impl->storage());
  }
  dynamic_cast<MLUTensorImpl*>(dst_impl)->set_cpu_storage_offset(
      src_impl->storage_offset());
}

std::vector<int64_t> get_channels_last_strides_1d(const at::IntArrayRef sizes) {
  std::vector<int64_t> strides(sizes.size());
  switch (sizes.size()) {
    // NLC
    case 3:
      strides[1] = 1;
      strides[2] = sizes[1];
      strides[0] = strides[2] * sizes[2];
      return strides;
    // LC
    case 2:
      strides[0] = 1;
      strides[1] = sizes[0];
      return strides;
    default:
      TORCH_INTERNAL_ASSERT(false, "ChannelsLast1d doesn't support size ", sizes.size());
  }
}

std::vector<int64_t> get_channels_last_strides(const at::IntArrayRef sizes) {
  switch (sizes.size()) {
    case 5:
      return c10::get_channels_last_strides_3d(sizes);
    case 4:
      return c10::get_channels_last_strides_2d(sizes);
    case 3:
      return get_channels_last_strides_1d(sizes);
    default:
      TORCH_INTERNAL_ASSERT(false, "ChannelsLast doesn't support size ", sizes.size());
  }
}

std::vector<int64_t> get_channels_first_strides(const at::IntArrayRef sizes) {
  std::vector<int64_t> strides(sizes.size());
  auto dim = sizes.size();
  if (dim > 0) {
    int last_idx = dim - 1;
    strides[last_idx] = 1;
    for (auto i = last_idx - 1; i >= 0; --i) {
      strides[i] = strides[i + 1] * std::max<int64_t>(sizes[i + 1], 1);
    }
  }
  return strides;
}

std::vector<int64_t>
get_contiguous_strides(const at::IntArrayRef sizes,
                       c10::MemoryFormat memory_format) {
  switch (memory_format) {
    case c10::MemoryFormat::Contiguous:
      return get_channels_first_strides(sizes);
    case c10::MemoryFormat::ChannelsLast:
    case c10::MemoryFormat::ChannelsLast3d:
      return get_channels_last_strides(sizes);
    default:
      TORCH_MLU_CHECK(false,
                      "get_contiguous_strides doesn't support memory_format ",
                      memory_format);
  }
}

bool is_channels_last(const at::Tensor& t) {
  if ((t.dim() < 4) || (t.dim() > 5)) {
    return false;
  } else {
    auto is_channels_last_2d = getMluTensorImpl(t)->is_strides_like_channels_last();
    auto is_channels_last_3d = getMluTensorImpl(t)->is_strides_like_channels_last_3d();
    return (is_channels_last_2d || is_channels_last_3d);
  }
}

// NB: can not be used to share storage with other Tensors because of
// MLU current layout on chip
MLUTensorImpl* resize_impl_mlu_(MLUTensorImpl* self, at::IntArrayRef size,
                                c10::optional<at::IntArrayRef> stride) {
  if (self->sizes() == size && (!stride || self->strides() == stride)) {
    return self;
  }

  int64_t storage_size = 1;
  if (stride) {
    self->set_sizes_and_strides(size, *stride);
    // NB: storage size can be different from numel.
    for (size_t dim = 0; dim < size.size(); ++dim) {
      // FIXME: Don't rely on storage_size being negative because this
      // may not be true for some edge cases.
      if (size[dim] == 0) {
        storage_size = 0;
        break;
      }
      storage_size += (size[dim] - 1) * stride.value()[dim];
    }
  } else {
    self->set_sizes_contiguous(size);
    storage_size = self->numel();
  }

  maybe_resize_storage_mlu(self, storage_size);

  return self;
}

void resize_mlu_storage_bytes(at::StorageImpl* self, ptrdiff_t size_bytes,
                              const caffe2::TypeMeta& dtype_offchip) {
  TORCH_MLU_CHECK(size_bytes >= 0, "invalid size ", size_bytes,
    " when invoke storage_resize_bytes");
  TORCH_MLU_CHECK(self->allocator() != nullptr,
    "storage_resize_bytes failed! allocator of storage is null.");
  TORCH_MLU_CHECK(self->resizable(), "Trying to resize storage that is not resizable");

  auto device_index = current_device();

  if (size_bytes == 0) {
    self->set_data_ptr(at::DataPtr(nullptr, at::Device(at::DeviceType::MLU, device_index)));
    self->set_nbytes(0);
  } else {
    const torch_mlu::MLUCachingAllocator* allocator =
      dynamic_cast<torch_mlu::MLUCachingAllocator*>(self->allocator());
    // because MLU do not support 64bit, so the bit wide of dtype on chip may not eq dtype off chip
    auto dtype_onchip = CnnlType2CnrtType(getCnnlDataType(dtype_offchip));
    auto mlu_size_bytes = size_bytes / dtype_offchip.itemsize() * cnrtDataTypeSize(dtype_onchip);
    at::DataPtr data = allocator->allocate(mlu_size_bytes, self->device().index());
    if (self->data_ptr()) {
      // TODO(zhanchendi): Enable p2p access when the memcpy is across device like cuda?
      torch_mlu::mlu::MLUGuard guard(self->device());
      auto queue = getCurrentQueue();
      auto mlu_self_nbytes = self->nbytes() / dtype_offchip.itemsize() * \
                             cnrtDataTypeSize(dtype_onchip);
      auto copy_bytes = mlu_self_nbytes < mlu_size_bytes ? mlu_self_nbytes : mlu_size_bytes;
      TORCH_CNRT_CHECK(cnrtMemcpyAsync(data.get(), self->data(), copy_bytes, queue.queue(),
        CNRT_MEM_TRANS_DIR_DEV2DEV));
    }

    // Destructively overwrite data_ptr
    self->set_data_ptr(std::move(data));
    self->set_nbytes(size_bytes);
  }
}

void checkSameMLU(at::CheckedFrom c, const at::TensorArg& t1, const at::TensorArg& t2) {
  TORCH_CHECK(t1->device() == t2->device(),
    "Expected tensor for ", t1, " to have the same device as tensor for ", t2,
    "; but first tensor device type ", t1->device().type(),
    " and device id ", t1->get_device(),
    " does not equal to second tensor ", t2->device().type(),
    " and device id ", t2->get_device(),
    " (while checking arguments for ", c, ")");
}

// Check TensorArg t1 and TensorArg t2 with same attribute by using fn function ptr.
void checkAllSame(at::CheckedFrom c, at::ArrayRef<at::TensorArg> tensors,
                  void(*fn)(at::CheckedFrom, const at::TensorArg&, const at::TensorArg&)) {
  const at::TensorArg* t0 = nullptr;
  for (auto& t : tensors) {
    if (!t->defined()) continue;
    if (t0 != nullptr) {
      fn(c, *t0, t);
    } else {
      t0 = &t;
    }
  }
}

void checkAllSameMLU(at::CheckedFrom c, at::ArrayRef<at::TensorArg> tensors) {
  checkAllSame(c, tensors, checkSameMLU);
}

}  // namespace torch_mlu
