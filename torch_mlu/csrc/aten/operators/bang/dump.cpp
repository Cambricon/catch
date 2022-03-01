#include "aten/operators/bang/bang_kernel.h"
#include "aten/operators/bang/internal/bang_internal.h"

namespace torch_mlu {
namespace bang {
namespace ops {

bool bang_dump(const at::Tensor& input) {
    auto input_impl = getMluTensorImpl(input);
    auto input_ptr = input_impl->cnnlMalloc();
    int32_t size = input.numel();
    cnrtDataType_t cnrt_type = fromCnnlType2CnrtType(input_impl->getCnnlType());
    cnrtDim3_t dim = {1, 1, 1};
    cnrtFunctionType_t ktype = CNRT_FUNC_TYPE_BLOCK;
    auto queue = getCurQueue();
    dump(input_ptr, size, dim, ktype, queue, cnrt_type);
    cnrtQueueSync(queue);
    return true;
}

}  // namespace ops
}  // namespace bang
}  // namespace torch_mlu
