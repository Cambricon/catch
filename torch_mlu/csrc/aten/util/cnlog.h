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

#include <c10/util/Logging.h>
#include <ATen/Tensor.h>
#include <ATen/core/List.h>

// When glog module is enabled in Pytorch, these macros will be defined in glog/logging.h
// with google namespace, so declare these macros before using them.
#ifdef C10_USE_GLOG
using google::INFO;
using google::WARNING;
using google::ERROR;
using google::FATAL;
#endif

// for debug
const int DBG = -1;

namespace torch_mlu {
C10_API int64_t MinCNLogLevelFromEnv();
C10_API int64_t MinCNLogLevel();
C10_API int64_t LogLevelStrToInt(const char* log_level_ptr);

class C10_API CNLogMessage {
 public:
  CNLogMessage(const char* file, int line, const char* func, int severity);
  ~CNLogMessage();

  std::stringstream& stream() {
    return stream_;
  }

 private:
  std::stringstream stream_;
  int severity_;
};

template <typename T>
inline std::ostream& GenerateBasicMessage(std::ostream &os, const T &t) {
  os << "[value: " << t << "] ";
  return os;
}

template <>
inline std::ostream & GenerateBasicMessage(std::ostream &os, const at::Tensor &t) {
  if (t.defined()) {
    os << "[" << "shape: " << t.sizes() <<
    ", device: " << t.device() <<
    ", dtype: "  << t.scalar_type() << "] ";
  } else {
    os << "[undefined] ";
  }
  return os;
}

template <>
inline std::ostream & GenerateBasicMessage(std::ostream &os, const at::Scalar &t) {
  os << "[value: " << t.to<float>() << "] ";
  return os;
}

template <typename T>
inline std::ostream & GenerateBasicMessage(std::ostream &os, const torch::List<T> &t) {
  auto vec = t.vec();
  for (int64_t i = 0; i < vec.size(); ++i) {
    GenerateBasicMessage(os, vec[i]);
  }
  return os;
}

template <>
inline std::ostream & GenerateBasicMessage(std::ostream &os, const at::TensorList &t) {
  for (int64_t i = 0; i < t.size(); ++i) {
    GenerateBasicMessage(os, t[i]);
  }
  return os;
}

template<typename T>
inline std::ostream & GenerateBasicMessage(std::ostream &os, const c10::optional<T> &t) {
  if (t.has_value()) {
    GenerateBasicMessage(os, t.value());
  } else {
    os << "[no use] ";
  }
  return os;
}

template <>
inline std::ostream & GenerateBasicMessage(std::ostream &os, const bool &t) {
  t ? os << "[true] " : os << "[false] ";
  return os;
}

template <typename T, std::size_t N>
inline std::ostream & GenerateBasicMessage(std::ostream &os, const std::array<T, N> &t) {
  for (int64_t i = 0; i < t.size(); ++i) {
    GenerateBasicMessage(os, t.at(i));
  }
  return os;
}

template <std::size_t N>
inline std::ostream & GenerateBasicMessage(std::ostream &os, const char (&t)[N]) {
  os << t;
  return os;
}

template <>
inline std::ostream & GenerateBasicMessage(std::ostream &os, const std::string &t) {
  os << t;
  return os;
}

template <>
inline std::ostream & GenerateBasicMessage(std::ostream &os, const at::Generator &t) {
  os << "[" << "current_seed: " << t.current_seed() <<
  ", device: " << t.device() << "] ";
  return os;
}

template <typename T, typename... Args>
inline std::ostream & GenerateBasicMessage(std::ostream &os, const T &t, const Args &... rest) {
  GenerateBasicMessage(os, t);
  GenerateBasicMessage(os, rest...);
  return os;
}

template <typename... Args>
inline std::string GenerateMessage(const Args &... args) {
  std::ostringstream ss;
  GenerateBasicMessage(ss, args...);
  return ss.str();
}

}  // namespace torch_mlu

// get param name
#define NAME(x) (#x)
// get param info
#define TOSTR(x) torch_mlu::GenerateMessage(x)

#define CNLOG_IS_ON(lvl)                                \
  ((lvl) >= torch_mlu::MinCNLogLevel())

#define CNLOG(lvl)                                      \
  if (CNLOG_IS_ON(lvl))                                 \
    torch_mlu::CNLogMessage(__FILE__, __LINE__, __FUNCTION__, lvl).stream()
