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

#include "aten/util/matrix_util.h"

namespace torch_mlu {
/*
 * Given a tensor of infos, obtained after a batch operations,
 * this function checks if the computation over all these batches has been
 * successful (info = 0) or not, and report in case of the latter.
 */
void batchCheckErrors(const at::Tensor& infos, const char* name, bool allow_singular) {
  auto batch_size = infos.numel();
  auto infos_cpu = infos.to(at::kCPU);
  auto infos_data = infos_cpu.data_ptr<int>();
  for (int64_t i = 0; i < batch_size; i++) {
    auto info = infos_data[i];
    if (info < 0) {
      CNLOG(ERROR) << name << ": For batch " << i << ": Argument " << -info << " has illegal value";
    } else if (info > 0) {
      if (strstr(name, "svd")) {
        CNLOG(ERROR) << name << ": the "  << i
                     << "-th input matrix SVD computation do not converge, "
                     << "(error: " << info << ") the precision of the results might be low.";
      } else if (strstr(name, "symeig")) {
        CNLOG(ERROR) << name << ": For batch " << i << ": the algorithm failed to converge; "
                     << info << " off-diagonal elements of an intermediate tridiagonal"
                     << " form did not converge to zero.";
      } else if (!allow_singular) {
        CNLOG(ERROR) << name << ": For batch " << i << ": U("
                     << info << "," << info << ") is zero, singular U.";
      }
    }
  }
}

}  // namespace torch_mlu
