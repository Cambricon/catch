#include "aten/device/caching_notifier.h"

namespace torch_mlu {

CachingNotifier CachingNotifier::instance;

// Get Singleton instance
CachingNotifier& CachingNotifier::get_instance() {
  return instance;
}

std::shared_ptr<Notifier> CachingNotifier::alloc_notifier(c10::DeviceIndex device_id) {
  std::shared_ptr<Notifier> sptr(nullptr);
  std::lock_guard<std::mutex> lock(notifier_mutex);
  if (notifier_pool[device_id].empty()) {
    sptr = std::make_shared<Notifier>();
  } else {
    sptr = notifier_pool[device_id].front();
    notifier_pool[device_id].pop_front();
  }
  return sptr;
}

void CachingNotifier::give_back_notifier(std::shared_ptr<Notifier> sptr) {
  int device_id = static_cast<int>(sptr->device_index());
  std::lock_guard<std::mutex> lock(notifier_mutex);
  notifier_pool[device_id].emplace_back(sptr);
}

void CachingNotifier::clean_notifier() {
  for (int i = 0; i < MLU_DEVICE_NUM_MAX; ++i) {
    for (auto& sptr : notifier_pool[i]) {
      sptr.reset();
    }
    notifier_pool[i].clear();
  }
}

}  // namespace torch_mlu
