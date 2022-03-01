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

#include <ATen/core/Formatting.h>
#include <ATen/core/List.h>
#include <ATen/Tensor.h>
#include <fstream>
#include <vector>
#include <string>
#include <exception>
#include "cnnl.h"  // NOLINT

#define DUMP_RESULT(BACKEND, OP, result)             \
if DUMP_ENABLE(OP) {                                 \
  torch_mlu::global_dumptool.setBackend(BACKEND);    \
  torch_mlu::dump_item("result", result);            \
}

#define DUMP_FINISH(OP)                  \
if DUMP_ENABLE(OP)                       \
{                                        \
  torch_mlu::global_dumptool.endLayer();   \
}

#define DUMP_TRY(ARGS) try {ARGS} catch (c10::Error &e) {;} catch (std::exception & e) {;}

#define DUMP_ENABLE(OP) (torch_mlu::global_dumptool.isDumpEnable() && (strcmp(#OP, "copy_") != 0))

#define USE_CPU_COMPARE(OP) \
    (torch_mlu::global_dumptool.isCpuEnable() && (DUMP_ENABLE(OP)))

#define NOT_INPLACE(op_name) (op_name[strlen(op_name) - 1] != '_'          \
                              && (!(op_name[strlen(op_name) - 3] == 'o'    \
                                 && op_name[strlen(op_name) - 2] == 'u'    \
                                 && op_name[strlen(op_name) - 1] == 't'))  \
                             )

namespace torch_mlu {

void _dump_cnnl_gencase(int mode);

class DumpTool{
 public:
  DumpTool();
  ~DumpTool() {}
  inline std::string layerDir() {return dump_dir_ + backend_info_ + "_";}
  inline bool isCpuEnable() {return this->cpu_enable_;}
  inline bool isDumpEnable() {return this->dump_enable_;}
  inline void setBackend(const char* backend) {this->backend_info_ = backend;}
  inline int dumpLevel() {return this-> detail_level_;}
  void initLayer(const char * name, const char * info);
  void endLayer();

  void pause_dump() {dump_enable_ = false;}
  void resume_dump() {dump_enable_ = true;}

  // Python:
  void startDump(const char * dir,
                   bool enable = true,
                   bool dump_cpu = false,
                   int detail_level = 0);
  void endDump();

 private:
  std::string dump_dir_;
  std::vector<std::string> name_stack_;  // the first name is dir name
  std::vector<int> num_stack_;
  bool dump_enable_ = false;
  bool cpu_enable_ = false;
  int detail_level_ = 0;
  std::string backend_info_;  // backend of operator, CNNL, BANG or CPU
};

extern DumpTool global_dumptool;

template<typename T>
inline void basic_item(std::ofstream & dump_os, const T & item) {
  dump_os << item << std::endl;
}

template <>
inline void basic_item(std::ofstream & dump_os, const at::Tensor & item) {
  if (item.defined() == false || item.sizes().vec().size() == 0 || item.numel() == 0) {
    dump_os << "Tensor : Type = Empty" << std::endl;
    return;
  }
  if (item.data_ptr() == nullptr) {
    dump_os << "Tensor : Type = Undefined" << std::endl;
    return;
  }
  try {
    torch_mlu::global_dumptool.pause_dump();
    at::Tensor cpu_tensor = item.cpu().contiguous().view(-1);
    torch_mlu::global_dumptool.resume_dump();
    auto dump_tensor = cpu_tensor;
    size_t max_item = cpu_tensor.sizes().vec()[0];
    if (global_dumptool.dumpLevel() == 0) {  // 10 item level
      max_item = max_item > 10 ? 10 : max_item;
    } else if (global_dumptool.dumpLevel() == 1) {  // sum
      if (cpu_tensor.scalar_type() == at::ScalarType::Bool) {
        dump_tensor = cpu_tensor.sum();  // Bool Tensor can not do abs.
      } else {
        dump_tensor = cpu_tensor.abs().sum();
      }
      max_item = 1;
    }
    dump_os << "Tensor : Type = " << item.scalar_type() << " : Shape = "
            << item.sizes() << " : Stride = " << item.strides()
            << " : Dumped = " << max_item << std::endl;
    for (size_t i = 0; i < max_item; i++) {
      if (dump_tensor.scalar_type() == at::ScalarType::Long) {
          dump_os << ((int*)(dump_tensor.unsafeGetTensorImpl()->data()))[i] << std::endl;
      } else if (dump_tensor.scalar_type() == at::ScalarType::Bool) {
          dump_os << ((bool*)(dump_tensor.unsafeGetTensorImpl()->data()))[i] << std::endl;
      } else {
          dump_tensor = dump_tensor.to(at::kFloat);
          dump_os << ((float*)(dump_tensor.unsafeGetTensorImpl()->data()))[i] << std::endl;
      }
    }
  } catch (std::exception & e) {
    dump_os << "Unknown error" << e.what() << std::endl;
  }
}

template <>
inline void basic_item(std::ofstream & dump_os, const at::Scalar & item) {
  if (item.isBoolean()) {
    dump_os << "Scalar : Type = bool" << std::endl;
    dump_os << item.to<bool>() << std::endl;
  } else if (item.isIntegral()) {
    dump_os << "Scalar : Type = int" << std::endl;
    dump_os << item.to<int>() << std::endl;
  } else {
    dump_os << "Scalar : Type = float" << std::endl;
    dump_os << item.to<float>() << std::endl;
  }
}

template <typename T>
inline void basic_item(std::ofstream & dump_os, const torch::List<T> & item) {
  auto vec = item.vec();
  dump_os << "List : Size = " << vec.size() << std::endl;
  size_t total = 0;
  for (const T & iter : vec) {
    total++;
    basic_item(dump_os, iter);
  }
}

template<typename T>
inline void basic_item(std::ofstream & dump_os, const std::vector<T> & item) {
  size_t total = 0;
  dump_os << "Vector : Size = " << item.size() << std::endl;
  for (const T& i : item) {
    total++;
    basic_item(dump_os, i);
  }
}

template <>
inline void basic_item(std::ofstream & dump_os, const at::TensorList & item) {
  dump_os << "TensorList : Size = " << item.size() << std::endl;
  for (size_t i = 0; i < item.size() ; i++) {
    basic_item(dump_os, item[i]);
  }
}

template <>
inline void basic_item(std::ofstream & dump_os, const at::Generator &item) {
  dump_os << "Generator : " << std::endl;
  dump_os << item.current_seed() << std::endl;
  dump_os << item.device() << std::endl;
}


template<typename T>
inline void basic_item(std::ofstream & dump_os, const c10::optional<T> &item) {
  if (item.has_value()) {
    dump_os << "Optional : " << std::endl;;
    basic_item(dump_os, item.value());
  } else {
    dump_os << "Optional : None" << std::endl;
  }
}

// ---- Dump Tuple
template<int S, typename ... Args,
    typename std::enable_if<!(S < std::tuple_size<std::tuple<Args...>>::value)>::type* = nullptr>
inline void basic_item(std::ofstream & dump_os, const std::tuple<Args ...> &item) {
  return;
}
template<int S, typename ... Args,
    typename std::enable_if<(S < std::tuple_size<std::tuple<Args...>>::value)>::type* = nullptr>
inline void basic_item(std::ofstream & dump_os, const std::tuple<Args ...> &item) {
  basic_item(dump_os, std::get<S>(item));
  basic_item<S+1>(dump_os, item);
}
template<typename ... Args>
inline void basic_item(std::ofstream & dump_os, const std::tuple<Args ...> & item) {
  dump_os << "Tuple : Size = " << std::tuple_size<std::tuple<Args...>>::value << std::endl;
  basic_item<0>(dump_os, item);
}
// --- End of Dump Tuple

template<std::size_t N, typename T>
inline void basic_item(std::ofstream & dump_os, const std::array<T, N> &item) {
  dump_os << "Array : Size = " << N << std::endl;
  for (size_t i = 0; i < item.size(); i++) {
    basic_item(dump_os, item.at(i));
  }
}

template<typename Args>
inline void dump_item(const char* name, const Args & args) {
  if (global_dumptool.isDumpEnable()) {
    std::ofstream dump_file(torch_mlu::global_dumptool.layerDir()+name);
    basic_item(dump_file, args);
    dump_file.close();
  }
}

template<typename A>
inline void recursive_data(const char* name, A data) {
  dump_item(name, data);
}

template<typename A, typename... Args>
inline void recursive_data(const char* name, A data, Args ... args) {
  dump_item(name, data);
  recursive_data(args...);
}

template<typename... Args>
inline void dump_layer_inputs(const char* layer_name, Args ... args) {
  if (global_dumptool.isDumpEnable() && (strcmp(layer_name, "copy_") != 0)) {
    torch_mlu::global_dumptool.initLayer(layer_name, "mlu");
    torch_mlu::global_dumptool.setBackend("mlu");
    recursive_data(args...);
  }
}

}  // namespace torch_mlu
