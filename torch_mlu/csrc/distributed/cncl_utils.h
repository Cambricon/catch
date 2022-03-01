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

#include <stdio.h>
#include <stdlib.h>
#include <memory>
#include <vector>
#include "cncl.h"  // NOLINT

#define C10D_CNCL_CHECK(cmd)                                              \
  do {                                                                    \
    cnclResult_t error = cmd;                                             \
    if (error != CNCL_RET_SUCCESS) {                                      \
      std::string err = "CNCL error in: " + std::string(__FILE__) + ":" + \
          std::to_string(__LINE__) + ", " +                               \
          std::string(cnclGetErrorStr(error));                            \
      throw std::runtime_error(err);                                      \
    }                                                                     \
  } while (0)

#define C10D_CNCL_ASSERT(cmd)                                              \
  do {                                                                    \
    cnclResult_t res = cmd;                                             \
    if (res != CNCL_RET_SUCCESS) {                                      \
      std::string err = cnclGetErrorStr(res);    \
      fprintf(                                           \
          stderr,                                        \
          "CNCL error in: %s:%d, %s\n",                  \
          __FILE__,                                      \
          __LINE__,                                      \
          err.c_str());                                  \
      abort();                                           \
    }                                                                     \
  } while (0)

namespace c10d {

// RAII wrapper for CNCL communicators in a process
class CNCLComms {
 public:
  explicit CNCLComms(std::vector<cnclComm_t> cncl_comms)
      : cncl_comms_(cncl_comms) {}

  explicit CNCLComms(const int cncl_comms) {
    cncl_comms_.resize(cncl_comms);
    for (size_t i = 0; i < cncl_comms; ++i) {
      cncl_comms_[i] = nullptr;
    }
  }

  ~CNCLComms() noexcept {
    if (cncl_comms_.size() > 0 && cncl_comms_[0]) {
      C10D_CNCL_ASSERT(cnclDestroyComms(&(cncl_comms_[0]), cncl_comms_.size()));
    }
  }

  static std::shared_ptr<CNCLComms> create(
      const int num_comms,
      const int* dev_list,
      const int* rank_list,
      const int num_ranks,
      const cnclCliqueId_t clique_id) {
    auto comms = std::make_shared<CNCLComms>(num_comms);
    C10D_CNCL_CHECK(cnclInitComms(&(comms->cncl_comms_[0]), num_comms, dev_list,
                                  rank_list, num_ranks, clique_id));
    return comms;
  }

  // Must not be copyable
  CNCLComms(const CNCLComms&) = delete;
  CNCLComms& operator=(const CNCLComms&) = delete;

  // Move constructable
  CNCLComms(CNCLComms&& other) {
    std::swap(cncl_comms_, other.cncl_comms_);
  }
  // Move assignable
  CNCLComms& operator=(CNCLComms&& other) {
    std::swap(cncl_comms_, other.cncl_comms_);
    return *this;
  }

  cnclComm_t getCnclComm(size_t n) {
    return cncl_comms_[n];
  }

  std::vector<cnclComm_t> getCnclComms() {
    return cncl_comms_;
  }

 protected:
  std::vector<cnclComm_t> cncl_comms_;
};

}   // namespace c10d

