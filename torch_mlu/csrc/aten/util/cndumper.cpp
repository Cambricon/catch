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


#include <sys/io.h>
#include <stdio.h>
#include <unistd.h>
#include <sys/stat.h>
#include <dirent.h>
#include <ATen/Tensor.h>
#include "aten/util/cndumper.h"

namespace torch_mlu {

// Integrate cnnl gencase API
void _dump_cnnl_gencase(int mode) {
    cnnlSetGenCaseMode(mode);
}

// DumpTool global_dumptool;

DumpTool global_dumptool = DumpTool();

DumpTool::DumpTool() {
  dump_enable_ = false;
  cpu_enable_ = false;
  detail_level_ = 0;
  dump_dir_ = "./dump";
  backend_info_ = "";
}

void DumpTool::startDump(const char * dir, bool enable, bool dump_cpu, int detail_level) {
  if (!enable) return;
  std::string temp_dir(dir);
  if (temp_dir.length() == 0) {
    temp_dir = "./dump";
  }
  if (temp_dir.back() == '/') {
    temp_dir.pop_back();
  }
  std::string full_dir(temp_dir);
  int append = 1;
  // if dir exists, append num like dir1, dir2 ..., till make new dir
  while (access(full_dir.c_str(), 0) != -1) {
    full_dir = temp_dir + std::to_string(append);
    append++;
  }
  mkdir(full_dir.c_str(), 0755);
  full_dir += '/';  // All dirs should has '/' at last
  dump_dir_ = full_dir;
  name_stack_.clear();
  num_stack_.clear();
  name_stack_.push_back(dump_dir_);
  num_stack_.push_back(0);
  this->cpu_enable_ = dump_cpu;
  this->dump_enable_ = enable;
  this->detail_level_ = detail_level;
}

void DumpTool::initLayer(const char * name, const char * info) {
  // Layer dump dir is {parent_dir}/num_{name}
  // while data dump file is {layer_dir}/data
  backend_info_ = info;

  num_stack_.back() += 1;
  int layer_num = num_stack_.back();
  num_stack_.push_back(0);  // Add num for child layer
  dump_dir_ = name_stack_.back() + std::to_string(layer_num) + '_' + name;
  mkdir(dump_dir_.c_str(), 0755);
  dump_dir_ += '/';  // All dirs should has '/' at last
  name_stack_.push_back(dump_dir_);  // name_stack_ always save the right dir-name.
}

void DumpTool::endLayer() {
  num_stack_.pop_back();  // Reset children
  name_stack_.pop_back();

  // After dump finish of a lower layer, we shall go back to the upper layer,
  //   and dump the result(if we want to) of upper layer.
  dump_dir_ = name_stack_.back();
}

void DumpTool::endDump() {
  dump_enable_ = false;
  cpu_enable_ = false;
}

}  // namespace torch_mlu
