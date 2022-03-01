#pragma once
#include <iostream>
#include <memory>
#include <string>
#include <vector>
#include "aten/device/notifier.h"
#include "aten/device/queue.h"
#include "aten/util/python_interface.h"
namespace torch_mlu {

struct PyNotifier {
  PyNotifier() {}
  ~PyNotifier() {}
  void place() { notifier_.place(); }

  void place(Queue& queue);

  float elapsed_time(PyNotifier& other);

  float hardware_time(PyNotifier& other);

  void synchronize() { notifier_.synchronize(); }

  bool query() { return notifier_.query(); }
  
  void wait(Queue& queue);

 private:
  Notifier notifier_;
};
}
