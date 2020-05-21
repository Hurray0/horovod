// Copyright 2019 Uber Technologies, Inc. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
// ============================================================================

#include "megray_context.h"

#include <chrono>
#include <memory>

#define HOROVOD_MEGRAY_SERVER "HOROVOD_MEGRAY_SERVER"
#define HOROVOD_MEGRAY_PORT "HOROVOD_MEGRAY_PORT"
//#define HOROVOD_MEGRAY_TIMEOUT_SECONDS "HOROVOD_MEGRAY_TIMEOUT_SECONDS"


#include "../utils/env_parser.h"

namespace horovod {
namespace common {

// Horovod Megray rendezvous knobs.
#define HOROVOD_RANK "HOROVOD_RANK"
#define HOROVOD_LOCAL_RANK "HOROVOD_LOCAL_RANK"

// std::chrono::milliseconds GetTimeoutFromEnv() {
//   auto s = std::chrono::seconds(GetIntEnvOrDefault(HOROVOD_MEGRAY_TIMEOUT_SECONDS, 30));
//   return std::chrono::duration_cast<std::chrono::milliseconds>(s);
// }

void MegrayContext::Initialize(int cuda_device) {
  if (!enabled_) {
    return;
  }

  //Megray::transport::tcp::attr attr;
  //attr.iface = megray_iface;

  //attr.ai_family = AF_UNSPEC;
  //auto dev = megray::transport::tcp::CreateDevice(attr);
  //auto timeout = GetTimeoutFromEnv();

  int nrank = GetIntEnvOrDefault(HOROVOD_RANK, 0);
  //int size = GetIntEnvOrDefault(HOROVOD_SIZE, 1);
  int rank = GetIntEnvOrDefault(HOROVOD_LOCAL_RANK, 0);
  this->nrank = nrank;
  this->rank = rank;
  std::string server_ip = GetStringEnvOrDefault(HOROVOD_MEGRAY_SERVER, "127.0.0.1");
  int server_port = GetIntEnvOrDefault(HOROVOD_MEGRAY_PORT, DEFAULT_PORT); // DEFAULT_PORT @ MegRay

  // megray start
  comm = MegRay::get_communicator(nrank, rank, MegRay::MEGRAY_NCCL);
  std::string uid = comm->get_uid();

  std::vector<std::string> uids;
  if (nrank > 1) {
    MegRay::Proxy::Client client(nrank, rank);
    client.connect(server_ip, server_port);
    uids = client.gather(uid);
  } else {
    uids = std::vector<std::string>{uid};
  }

  CUDA_ASSERT(cudaSetDevice(cuda_device));
  comm->init(uids);
  cudaStream_t stream;
  CUDA_ASSERT(cudaStreamCreate(&stream));
  ctx = MegRay::CudaContext::make(stream);
}

void MegrayContext::Finalize() {
  if (!enabled_) {
    return;
  }

  //ctx.reset();
}

std::shared_ptr<MegRay::Context>
MegrayContext::GetMegrayContext() {
  return ctx;
}

} // namespace common
} // namespace horovod
