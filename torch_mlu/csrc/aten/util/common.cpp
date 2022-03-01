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

#include <string>
#include <map>
#include <cstdlib>

#include "aten/util/common.h"

namespace torch_mlu {

const std::map<std::string, cndevNameEnum_t> device_name_table {{"MLU100", MLU100},
                                                                {"MLU270", MLU270},
                                                                {"MLU220_M2", MLU220_M2},
                                                                {"MLU220_EDGE", MLU220_EDGE},
                                                                {"MLU220_EVB", MLU220_EVB},
                                                                {"MLU220_M2i", MLU220_M2i},
                                                                {"MLU290", MLU290},
                                                                {"MLU370", MLU370}};

Global::Global() {
  is_running_fp32_ = false;
  cndevCardInfo_t card_num_info;
  card_num_info.version = 5;  // CNDEV use API version 5 as default
  card_num_info.number = 0;
  cndevCardName_t card_name;
  card_name.version = 5;  // CNDEV use API version 5 as default

  if (std::getenv("DEFAULT_MLU_DEVICE_NAME") != nullptr) {
    // Use manual device name from env
    std::string default_name{std::getenv("DEFAULT_MLU_DEVICE_NAME")};
    TORCH_MLU_CHECK(device_name_table.find(default_name) != device_name_table.end(),
                    "An unkown device name was given by env::DEFAULT_MLU_DEVICE_NAME as ",
                    default_name);
    CNLOG(WARNING) << "Device name set to "
                   << default_name
                   << " due to env::DEFAULT_MLU_DEVICE_NAME";
    device_name_ = device_name_table.find(default_name)->second;
  } else {
    // Auto get device name from cndev
    TORCH_CNDEV_CHECK(cndevInit(0));
    TORCH_CNDEV_CHECK(cndevGetDeviceCount(&card_num_info));
    TORCH_MLU_CHECK(card_num_info.number != 0,
          "Cannot find any visible MLU devices to specify ",
          "whether floating point can be use in calculations. ",
          "You can set env::DEFAULT_MLU_DEVICE_NAME (e.g. MLU270) as a default device.");
    TORCH_CNDEV_CHECK(cndevGetCardName(&card_name, 0));
    device_name_ = card_name.id;
  }
  if (device_name_ == MLU370) {
    is_running_fp32_ = true;
  }
  return;
}

Global::~Global() {}

}  // namespace torch_mlu
