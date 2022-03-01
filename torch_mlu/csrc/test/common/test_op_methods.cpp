#include <gtest/gtest.h>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <type_traits>
#include "utils/utils.h"
#include "aten/operators/op_methods.h"
#include "aten/generated/aten_mlu_type_default.h"
#include "aten/util/memory_allocator.h"

namespace torch_mlu {

TEST(ConvolutionTest, convolution_overrideable) {
  OpMethods TestConvolution;
  at::Tensor input = at::randn({1});
  at::Tensor weight = at::randn({1});
  at::Tensor bias = at::randn({1});
  at::IntArrayRef stride = input.sizes();
  at::IntArrayRef padding = input.sizes();
  at::IntArrayRef dilation = input.sizes();
  bool transposed = true;
  at::IntArrayRef output_padding = input.sizes();
  int64_t groups = 1;

  try {
    auto output = TestConvolution.convolution_overrideable(
                    input, weight, bias, stride, padding,
                    dilation, transposed, output_padding,
                    groups);
  } catch (std::exception& e) {}
}

TEST(ConvolutionTest, convolution_backward_overrideable) {
  OpMethods TestConvolutionBackward;
  at::Tensor grad = at::randn({1});
  at::Tensor input = at::randn({1});
  at::Tensor weight = at::randn({1});
  at::IntArrayRef stride = grad.sizes();
  at::IntArrayRef padding = grad.sizes();
  at::IntArrayRef dilation = grad.sizes();
  bool transposed = true;
  at::IntArrayRef output_padding = grad.sizes();
  int64_t groups = 1;
  std::array<bool, 3> output_mask = {true, true, true};

  try {
    auto output = TestConvolutionBackward.convolution_backward_overrideable(
                    grad, input, weight, stride, padding, dilation, transposed,
                    output_padding, groups, output_mask);
  } catch (std::exception& e) {}
}

TEST(NativeBatchNormTest, native_batch_norm) {
  OpMethods TestNativeBatchNorm;
  at::Tensor input = at::randn({1, 10, 10, 1});
  at::Tensor weight = at::randn({10});
  at::Tensor bias = at::randn({10});
  at::Tensor running_mean = at::randn({10});
  at::Tensor running_var = at::randn({10});
  bool training = false;
  double momentum = 0.1;
  double eps = 1e-05;

  auto output_cpu = at::native_batch_norm(input, weight, bias,
                        running_mean, running_var, training, momentum, eps);

  auto output_opmethods = TestNativeBatchNorm.native_batch_norm(
                              input, weight, bias, running_mean,
                              running_var, training, momentum, eps);
  assertTensorsEqual(std::get<0>(output_cpu),
                     std::get<0>(output_opmethods).cpu(), 0.0003, true, false, false);
}

TEST(NativeBatchNormTest, native_batch_norm_backward) {
  OpMethods TestNativeBatchNormBackward;
  at::Tensor grad_out = at::randn({1, 3, 64, 64});
  at::Tensor input = at::randn({1, 3, 64, 64});
  at::Tensor weight = at::randn({3});
  at::Tensor running_mean = at::randn({3});
  at::Tensor running_var = at::randn({3});
  at::Tensor save_mean_mean = at::randn({3});
  at::Tensor save_invstd = at::randn({3});
  bool training = true;
  double eps = 1e-05;
  std::array<bool, 3> output_mask = {true, true, true};
  auto output_cpu = at::native_batch_norm_backward(grad_out, input,
                        weight, running_mean, running_var,
                        save_mean_mean, save_invstd, training, eps, output_mask);
  auto output_opmethods = TestNativeBatchNormBackward.native_batch_norm_backward(
                    grad_out, input, weight, running_mean, running_var,
                    save_mean_mean, save_invstd, training, eps, output_mask);
  assertTensorsEqual(std::get<0>(output_cpu),
                     std::get<0>(output_opmethods).cpu(), 0.0003, true, false, false);
  assertTensorsEqual(std::get<1>(output_cpu),
                     std::get<1>(output_opmethods).cpu(), 0.0003, true, false, false);
  assertTensorsEqual(std::get<2>(output_cpu),
                     std::get<2>(output_opmethods).cpu(), 0.0003, true, false, false);
}

TEST(Resize, resize_) {
  OpMethods TestResize_;
  at::Tensor out = at::randn({1, 2, 3, 4});
  at::Tensor out_mlu = at::clone(out).to(at::Device(at::Device::Type::MLU));
  at::IntArrayRef size = out.sizes();
  auto output_cpu = out.resize_(size);
  auto output_opmethods = TestResize_.resize_(out_mlu, size);
  assertTensorsEqual(output_cpu, output_opmethods.cpu(), 0.0003, true, false, false);
}

TEST(SmoothL1Loss, smooth_l1_loss_forward_out) {
  OpMethods TestSmoothL1Loss;
  at::Tensor output = at::randn({1});
  at::Tensor self = at::randn({1});
  at::Tensor target = at::randn({1});
  auto output_mlu = at::clone(output).to(at::Device(at::Device::Type::MLU));
  auto self_mlu = at::clone(self);
  auto target_mlu = at::clone(target);
  int64_t reduction = 0;
  at::smooth_l1_loss_out(output, self, target, reduction);
  auto output_opmethods = TestSmoothL1Loss.smooth_l1_loss_forward_out(
                output_mlu, self_mlu, target_mlu, reduction);
  assertTensorsEqual(output, output_opmethods.cpu(), 0.0003, true, false, false);
}

TEST(SmoothL1Loss, smooth_l1_loss_backward_out) {
  OpMethods TestSmoothL1LossBackward;
  at::Tensor grad_output = at::randn({1});
  at::Tensor self = at::randn({1});
  at::Tensor target = at::randn({1});
  at::Tensor grad_input = at::empty_like(self);
  int64_t reduction = 0;
  auto grad_output_mlu = at::clone(grad_output);
  auto self_mlu = at::clone(self);
  auto target_mlu = at::clone(target);
  auto grad_input_mlu = at::empty_like(self_mlu).to(at::Device(at::Device::Type::MLU));
  at::smooth_l1_loss_backward_out(grad_input, grad_output, self, target, reduction);
  auto grad_input_opmethods = TestSmoothL1LossBackward.smooth_l1_loss_backward_out(
                                grad_input_mlu, grad_output_mlu, self_mlu,
                                target_mlu, reduction);
  assertTensorsEqual(grad_input, grad_input_opmethods.cpu(), 0.0003, true, false, false);
}

TEST(AdaptiveMaxPool2d, adaptive_max_pool2d_backward_out) {
  OpMethods TestAdaptiveMaxPool2d;
  auto input = at::randn({1, 1, 10, 10});
  auto input_mlu = input.to(at::Device(at::Device::Type::MLU));
  auto output_tuple = at::adaptive_max_pool2d(input, {4, 4});
  auto indices = std::get<1>(output_tuple);
  auto indices_mlu = indices.to(at::Device(at::Device::Type::MLU));

  auto grad_output = at::randn({1, 1, 4, 4});
  auto grad_output_mlu = grad_output.to(at::Device(at::Device::Type::MLU));
  auto grad_input = at::empty_like(input);
  auto grad_input_mlu = at::empty_like(input).to(at::Device(at::Device::Type::MLU));
  at::adaptive_max_pool2d_backward_out(grad_input, grad_output, input, indices);
  TestAdaptiveMaxPool2d.adaptive_max_pool2d_backward_out(grad_input_mlu,
    grad_output_mlu, input_mlu, indices_mlu);
  assertTensorsEqual(grad_input, grad_input_mlu.cpu(), 0.003, true, false, false);
}

TEST(PinMemory, pin_memory) {
  at::TensorOptions options;
  auto cpu_tensor = at::zeros({10}, options);
  auto pin_tensor = at::empty_pinned({10}, options.device(c10::DeviceType::MLU));
  if (!isPinned<void>(pin_tensor.data_ptr())) {
    CNLOG(ERROR) << "pin memory failed!";
  }
  if (pin_tensor.type().backend() != c10::Backend::CPU) {
    CNLOG(ERROR) << "type of pinned tensor is "
                 << pin_tensor.type().toString()
                 << " but not CPU tensors";
  }
  pin_tensor.zero_();
  auto mlu_tensor = pin_tensor.to(at::Device(at::Device::Type::MLU));
  assertTensorsEqual(cpu_tensor, mlu_tensor.cpu(), 0.00, false, false, false);
}
  
TEST(BitwiseOr, bitwise_or) {
  OpMethods TestBitwiseOr;
  at::TensorOptions options;
  at::Tensor output = at::ones({1}, options.dtype(at::ScalarType::Int));
  at::Tensor output_scalar = at::ones({1}, options.dtype(at::ScalarType::Int));
  at::Tensor self = at::ones({1}, options.dtype(at::ScalarType::Int));
  at::Tensor other = at::ones({1}, options.dtype(at::ScalarType::Int));
  at::Scalar other_scalar = 1;
  auto output_mlu = at::clone(output).to(at::Device(at::Device::Type::MLU));
  auto output_scalar_mlu = at::clone(output).to(at::Device(at::Device::Type::MLU));
  auto self_mlu = at::clone(self);
  auto other_mlu = at::clone(other);
  at::bitwise_or_out(output, self, other);
  at::bitwise_or_out(output_scalar, self, other_scalar);
  TestBitwiseOr.bitwise_or_out(output_mlu, self_mlu, other_mlu);
  TestBitwiseOr.bitwise_or_out(output_scalar_mlu, self_mlu, other_scalar);
  assertTensorsEqual(output, output_mlu.cpu(), 0.00, false, false, false);
  assertTensorsEqual(output_scalar, output_scalar_mlu.cpu(), 0.00, false, false, false);
}

TEST(AdvancedIndex, index) {
  OpMethods TestIndex;
  at::TensorOptions options;
  at::Tensor self = at::randn({4,1});
  at::Tensor indice = at::ones({4,1}, options.dtype(at::ScalarType::Bool));
  auto self_mlu = at::clone(self).to(at::Device(at::Device::Type::MLU));
  auto indice_mlu = at::clone(indice).to(at::Device(at::Device::Type::MLU));
  auto output = at::index(self, indice);
  auto output_mlu = TestIndex.index(self_mlu, indice_mlu);
  assertTensorsEqual(output, output_mlu.cpu(), 0.00, false, false, false);
}

}  // namespace torch_mlu
