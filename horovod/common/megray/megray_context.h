#ifndef HOROVOD_MEGRAY_CONTEXT_H
#define HOROVOD_MEGRAY_CONTEXT_H

#include "../common.h"
#include "../logging.h"

#include "src/megray.h"
#include "src/dist_proxy.h"


namespace horovod {
namespace common {

struct MegrayContext {
  int nrank;
  int rank;
  void Initialize(int cuda_device);

  void Finalize();

  std::shared_ptr<MegRay::Context> GetMegrayContext();

  void Enable() {
    enabled_ = true;
    LOG(DEBUG) << "MegRay context enabled.";
  }

  bool IsEnabled() { return enabled_; }


  std::shared_ptr<MegRay::CudaContext> ctx = nullptr;
  std::shared_ptr<MegRay::Communicator> comm = nullptr;

private:
  // Flag indicating whether megray is enabled.
  bool enabled_ = false;
};

} // namespace common
} // namespace horovod

#endif // HOROVOD_MEGRAY_CONTEXT_H
