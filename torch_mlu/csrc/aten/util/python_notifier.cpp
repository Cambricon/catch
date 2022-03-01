#include "aten/util/python_notifier.h"
#include "aten/device/queue.h"
#include "aten/util/cnlog.h"

namespace torch_mlu {

void PyNotifier::place(Queue& queue) {
      notifier_.place(queue);
}

float PyNotifier::elapsed_time(PyNotifier& other) {
  return notifier_.elapsed_time(other.notifier_);
}

float PyNotifier::hardware_time(PyNotifier& other) {
  return notifier_.hardware_time(other.notifier_);
}

void PyNotifier::wait(Queue& queue) {
      notifier_.wait(queue);
}

}  // namespace torch_mlu
