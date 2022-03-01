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

#include <ATen/ATen.h>
#include <ATen/TensorUtils.h>
#include <vector>
#include "aten/util/exceptions.h"
#include "aten/util/tensor_util.h"
#include "cnnl.h"  //  NOLINT

namespace torch_mlu {

template <typename T, cnnlStatus_t (*dtor)(T *)>
struct CnnlAlgorithmDeleter {
  void operator()(T *x) {
    if (x != nullptr) {
      TORCH_CNNL_CHECK(dtor(x));
    }
  }
};

template <typename T, cnnlStatus_t (*ctor)(T **), cnnlStatus_t (*dtor)(T *)>
class C10_API CnnlAlgorithm {
 public:
  // Use algo() to access the underlying algorithm pointer in
  // a read-only fashion.  Most client code should use this.
  // If the algorithm was never initialized, this will return
  // nullptr.
  T *algo() const { return algo_.get(); }
  T *algo() { return algo_.get(); }

  // Use mut_algo() to access the underlying algorithm pointer
  // if you intend to modify what it points to This will ensure
  // that the algorithm is initialized.
  // Code in this file will use this function.
  T *mut_algo() {
    init();
    return algo_.get();
  }

 protected:
  void init() {
    if (algo_ == nullptr) {
      T *raw_algo;
      TORCH_CNNL_CHECK(ctor(&raw_algo));
      algo_.reset(raw_algo);
    }
  }

 private:
  std::unique_ptr<T, CnnlAlgorithmDeleter<T, dtor>> algo_;
};

class C10_API CnnlMatmulAlgorithm
  : public CnnlAlgorithm<cnnlMatMulAlgoStruct,
                      &cnnlMatMulAlgoCreate,
                      &cnnlMatMulAlgoDestroy> {
public:
  CnnlMatmulAlgorithm() {}
  void get(cnnlHandle_t handle,
           cnnlMatMulDescriptor_t matmul_desc,
           cnnlTensorDescriptor_t self_desc,
           cnnlTensorDescriptor_t other_desc,
           cnnlTensorDescriptor_t output_desc,
           cnnlMatMulPreference_t preference);
};

class C10_API CnnlBatchMatmulAlgorithm
  : public CnnlAlgorithm<cnnlBatchMatMulAlgoStruct,
                      &cnnlBatchMatMulAlgoCreate,
                      &cnnlBatchMatMulAlgoDestroy> {
public:
  CnnlBatchMatmulAlgorithm() {}
  void get(cnnlHandle_t handle,
           cnnlBatchMatMulDescriptor_t bmm_desc,
           cnnlTensorDescriptor_t self_desc,
           cnnlTensorDescriptor_t other_desc,
           cnnlTensorDescriptor_t output_desc,
           cnnlBatchMatMulPreference_t preference);
};

}  //  namespace torch_mlu
