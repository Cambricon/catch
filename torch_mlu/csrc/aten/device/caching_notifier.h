#pragma once
#include <cstring>
#include <iostream>
#include <sstream>
#include <string>
#include <deque>
#include <memory>
#include "ATen/Utils.h"
#include "aten/device/device.h"
#include "aten/device/mlu_guard.h"
#include "aten/device/queue.h"
#include "aten/device/notifier.h"

namespace torch_mlu {

class CachingNotifier {
  private:
    CachingNotifier() {}
    std::deque<std::shared_ptr<Notifier> > notifier_pool[MLU_DEVICE_NUM_MAX];
    std::mutex notifier_mutex;
    static CachingNotifier instance;
    AT_DISALLOW_COPY_AND_ASSIGN(CachingNotifier);

  public:
    ~CachingNotifier() {
      clean_notifier();
    }

    // alloc a notifier from notifier pool.
    std::shared_ptr<Notifier> alloc_notifier(c10::DeviceIndex device_id);

    // give back notifier to notifier pool.
    void give_back_notifier(std::shared_ptr<Notifier> notifier);

    // Singleton
    static CachingNotifier& get_instance();

    // clear notifier in notifier pool.
    void clean_notifier();
};

#define NotifierPool_Manager CachingNotifier::get_instance()

}  // namespace torch_mlu

