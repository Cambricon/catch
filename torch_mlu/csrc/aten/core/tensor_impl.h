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

#pragma once

#include <ATen/Tensor.h>
#include <c10/core/Storage.h>
#include <c10/core/TensorImpl.h>
#include <ATen/Functions.h>
#include "cnrt.h"  // NOLINT
#include "aten/core/allocator.h"
#include "aten/core/caching_allocator.h"

#include "aten/util/exceptions.h"
#include "aten/util/types.h"
#include "aten/util/common.h"

namespace torch_mlu {
// Tensor implementation class used to be fed to the at::Tensor.
// Its scope is just to handle an MLUTensor.
class MLUTensorImpl : public c10::TensorImpl {
 public:
  explicit MLUTensorImpl(c10::Storage&& storage,
                         c10::DispatchKey type_id,
                         const caffe2::TypeMeta& data_type);

  explicit MLUTensorImpl(c10::Storage&& storage,
                         c10::DispatchKeySet type_set,
                         const caffe2::TypeMeta& data_type);

  ~MLUTensorImpl();

  static void AtenInitialize();

  bool is_empty();

  void* cpu_data();

  // Return a TensorImpl that is a shallow-copy of this TensorImpl.
  c10::intrusive_ptr<c10::TensorImpl> shallow_copy_and_detach(
      const c10::VariableVersion& version_counter,
      bool allow_tensor_metadata_change) const;

  // Shallow-copies data from another TensorImpl into this TensorImpl.
  void shallow_copy_from(const c10::intrusive_ptr<c10::TensorImpl>& impl);

  void copy_cnnl_metadata_from(const MLUTensorImpl* impl);

  void set_has_already_copy_flag(bool flag);

  const at::Storage& cpu_storage() const;

  void set_cpu_storage(at::Storage storage);

  void set_cpu_storage_offset(int64_t storage_offset);


  void update_cpu_storage(void* cpu_data_ptr);

  // Copy the tensor metadata fields from one MLUTensorImpl to another MLUTensorImpl
  static void copy_tensor_metadata(
      const TensorImpl* src_impl,
      TensorImpl* dest_impl,
      const c10::VariableVersion& version_counter,
      bool allow_tensor_metadata_change);

  // For cnnl
  void* cnnlMalloc(bool non_blocking = false);

  inline cnnlDataType_t getCnnlType() {
    // if (CNNL_DTYPE_INVALID == cnnl_data_type_) {
    //  return getCnnlDataType(data_type_);
    // }
    return cnnl_data_type_;
  }

  inline void setCnnlType(cnnlDataType_t data_type) {
    cnnl_data_type_ = data_type;
  }

  template<typename T>
  inline T numeric_min() {
    return std::numeric_limits<T>::min();
  }

  template <typename T1, typename T2>
  inline void cast_cpu_op(void* cast_cpu_ptr, void* cpu_ptr) {
    for (size_t i = 0; i < numel(); ++i) {
      T1 raw = static_cast<T1*>(cpu_ptr)[i];
      TORCH_CHECK(raw <= std::numeric_limits<T2>::max() &&
                   raw >= numeric_min<T2>(),
               "cnnlMalloc datacast fail! expected smaller than ",
               std::numeric_limits<T2>::max(), " and greater than ",
               numeric_min<T2>(), "but got ", raw);
      static_cast<T2*>(cast_cpu_ptr)[i] = static_cast<T2>(raw);
    }
  }

  c10::Storage cpu_storage_;
  cnnlDataType_t cnnl_data_type_ = CNNL_DTYPE_INVALID;
  bool has_already_copy = false;
  int64_t cpu_storage_offset_ = 0;

 public:
  // Add OP Recoder for Shared storage op.
  // tensor info like: size, stride, offset.
  using TupleRecordOpInfo = std::tuple<std::vector<int64_t>, std::vector<int64_t>, int64_t>;
  using SharedOpInfo = std::tuple<VIEWOPNAME, TupleRecordOpInfo, TupleRecordOpInfo>;

  bool is_shared_storage() {
    return using_shared_op_info_;
  }

  bool is_shared_storage() const {
    return using_shared_op_info_;
  }

  // compare input tensor and last node output tensor in graph.
  // compare item include: size/stride/sotrage_offset.
  bool is_compare_size_stride_info(const MLUTensorImpl*& other);

  bool is_compare_size_stride_info(const TupleRecordOpInfo& other);

  // non_inplace op using this interface, insert views op node.
  void insert_views_op_info(const VIEWOPNAME& name,
                            const MLUTensorImpl* input_tensor_impl,
                            const bool input_contiguous);

  // inplace op using this interface, insert views op node.
  void insert_views_op_info(const VIEWOPNAME& name,
                            const TupleRecordOpInfo& input_info,
                            const bool input_contiguous);

  // return graph segment to do contiguous
  std::vector<SharedOpInfo> get_views_op_info(const TupleRecordOpInfo& input);

 private:
  // Store Shaped_op_info: input tensor info and after viewed tensor info.
  std::vector<SharedOpInfo> shared_op_info_;
  // For record different graph branch
  /*
     fc --> split --> reshape --> transpose --> \
                 \--> reshape --> transpose -->  batch_dot
  */
  std::vector<std::vector<int> > graph_seg_record_;

  // Compare to subgraph, record contiguous output storage
  // std::vector<c10::Storage> graph_seg_storage_;

  // Storage is shared or not.
  bool using_shared_op_info_ = false;

  inline void check_add_op_into_shared_ops(const SharedOpInfo& other) {
    auto it = std::find(shared_op_info_.begin(), shared_op_info_.end(), other);
    if (it == shared_op_info_.end()) {
      shared_op_info_.push_back(other);
    }
  }

  inline void check_and_add_new_graph(std::vector<int>& other) {
    for (const auto& item : graph_seg_record_) {
      if (item == other) {
        return;
      }
    }
    graph_seg_record_.push_back(other);
  }

  inline std::tuple<int, int> search_op_segm_index(const TupleRecordOpInfo& other) {
    const int graph_seg_len = graph_seg_record_.size();
    if (graph_seg_len == 0) {
      return std::make_tuple<int, int>(-1, -1);
    }
    for (int i = 0; i < graph_seg_len; ++i) {
      const int graph_len = graph_seg_record_[i].size();
      for (int j = 0; j < graph_len; ++j) {
        const int index = graph_seg_record_[i][j];
        // compare op_info output info with new input info
        if (std::get<2>(shared_op_info_[index]) == other) {
          return std::make_tuple<int, int>(std::move(i), std::move(j));
        }
      }
    }
    return std::make_tuple<int, int>(-1, -1);
  }
};

template<>
inline float MLUTensorImpl::numeric_min<float>() {
  return std::numeric_limits<float>::lowest();
}

}  // namespace torch_mlu
