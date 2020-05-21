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
// =============================================================================

#include "megray_operations.h"

//#include "megray.h"
//#include "megray/types.h"

#include "../common.h"
#include "../global_state.h"

namespace horovod {
namespace common {

// IMegrayAlgorithms* GetAlgorithmsForType(DataType dtype,
//                                       MegrayContext* megray_context) {
//   switch (dtype) {
//   case HOROVOD_UINT8:
//     return new MegrayAlgorithms<MegRay::DType::MEGRAY_UINT8>(megray_context);
//   case HOROVOD_INT8:
//     return new MegrayAlgorithms<MegRay::DType::MEGRAY_INT8>(megray_context);
//   case HOROVOD_UINT16:
//     return new MegrayAlgorithms<MegRay::DType::MEGRAY_UINT16>(megray_context);
//   case HOROVOD_INT16:
//     return new MegrayAlgorithms<MegRay::DType::MEGRAY_INT16>(megray_context);
//   case HOROVOD_INT32:
//     return new MegrayAlgorithms<MegRay::DType::MEGRAY_INT32>(megray_context);
//   case HOROVOD_INT64:
//     return new MegrayAlgorithms<MegRay::DType::MEGRAY_INT64>(megray_context);
//   case HOROVOD_FLOAT16:
//     return new MegrayAlgorithms<MegRay::DType::MEGRAY_FLOAT16>(megray_context);
//   case HOROVOD_FLOAT32:
//     return new MegrayAlgorithms<MegRay::DType::MEGRAY_FLOAT32>(megray_context);
//   case HOROVOD_FLOAT64:
//     return new MegrayAlgorithms<MegRay::DType::MEGRAY_FLOAT64>(megray_context);
//   // case HOROVOD_BOOL:
//   //   return new MegrayAlgorithms<bool>(megray_context);
//   default:
//     throw std::logic_error("Type " + DataType_Name(dtype) +
//                            " is not supported in Megray mode.");
//   }
// }

MegRay::DType get_megray_type(DataType dtype) {
  switch (dtype) {
  case HOROVOD_UINT8:
    return MegRay::DType::MEGRAY_UINT8;
  case HOROVOD_INT8:
    return MegRay::DType::MEGRAY_INT8;
  // case HOROVOD_UINT16:
  //   return MegRay::DType::MEGRAY_UINT16;
  // case HOROVOD_INT16:
  //   return MegRay::DType::MEGRAY_INT16;
  case HOROVOD_INT32:
    return MegRay::DType::MEGRAY_INT32;
  case HOROVOD_INT64:
    return MegRay::DType::MEGRAY_INT64;
  case HOROVOD_FLOAT16:
    return MegRay::DType::MEGRAY_FLOAT16;
  case HOROVOD_FLOAT32:
    return MegRay::DType::MEGRAY_FLOAT32;
  case HOROVOD_FLOAT64:
    return MegRay::DType::MEGRAY_FLOAT64;
  default:
    throw std::logic_error("Type " + DataType_Name(dtype) +
                           " is not supported in Megray mode.");
  }

}

MegrayAlgorithms::MegrayAlgorithms(MegrayContext* megray_context)
    : megray_context_(megray_context), comm(megray_context->comm) {
    }

void MegrayAlgorithms::Allreduce(void* in_data, void* out_data, int num_elements, DataType dtype) {
  // all_reduce(const void* sendbuff, void* recvbuff, size_t len, DType dtype, 
  // ReduceOp op, std::shared_ptr<Context> ctx)
  MegRay::DType type = get_megray_type(dtype);
	comm->all_reduce(in_data, out_data, num_elements, type, MegRay::MEGRAY_SUM,
      megray_context_->ctx);
}

void MegrayAlgorithms::Allgather(void* in_data, void* out_data,
                                  int in_len, DataType dtype) {
  MegRay::DType type = get_megray_type(dtype);
  comm->all_gather(in_data, out_data, in_len, type, megray_context_->ctx);
}

void MegrayAlgorithms::Broadcast(void* buffer_data, int num_elements,
                                  int root_rank, DataType dtype) {
  MegRay::DType type = get_megray_type(dtype);
  comm->broadcast(buffer_data, buffer_data, num_elements, type, root_rank,
      megray_context_->ctx);
}

MegrayAllreduce::MegrayAllreduce(MegrayContext* megray_context,
                             HorovodGlobalState* global_state)
    : AllreduceOp(global_state), megray_context_(megray_context) {}

Status MegrayAllreduce::Execute(std::vector<TensorTableEntry>& entries,
                              const Response& response) {
  auto& first_entry = entries[0];

  void* buffer_data;
  void* input_data;
  int num_elements = (int)NumElements(entries);

  // Copy memory into the fusion buffer.
  auto& timeline = global_state_->timeline;
  if (entries.size() > 1) {
    timeline.ActivityStartAll(entries, MEMCPY_IN_FUSION_BUFFER);
    const void* fused_input_data;
    size_t buffer_len;
    MemcpyInFusionBuffer(entries, fused_input_data, buffer_data, buffer_len);
    input_data = buffer_data;
    timeline.ActivityEndAll(entries);
  } else {
    buffer_data = (void*)first_entry.output->data();
    input_data = (void*)first_entry.tensor->data();
    // std::memcpy(buffer_data, first_entry.tensor->data(), (size_t)first_entry.tensor->size());
  }

  //// copy to cuda mem
  // void *in_ptr, *out_ptr; // pointer to GPU CUDA memory
  // CUDA_ASSERT(cudaMalloc(&in_ptr, LEN*sizeof(first_entry.tensor->dtype())));
  // CUDA_ASSERT(cudaMalloc(&out_ptr, LEN*sizeof(first_entry.tensor->dype())));
  // CUDA_ASSERT(cudaMemcpy(in_ptr, in_data.data(), LEN*sizeof(float),
  //       cudaMemcpyHostToDevice));
  

  // Do allreduce.
  timeline.ActivityStartAll(entries, MEGRAY_ALLREDUCE);
  // std::unique_ptr<IMegrayAlgorithms> megray_algos(
  //     GetAlgorithmsForType(first_entry.tensor->dtype(), megray_context_));
  std::shared_ptr<MegrayAlgorithms> megray_algos = 
    std::make_shared<MegrayAlgorithms>(megray_context_);
  megray_algos->Allreduce(input_data, buffer_data, num_elements,
      first_entry.tensor->dtype());
  timeline.ActivityEndAll(entries);

  // Copy memory out of the fusion buffer.
  if (entries.size() > 1) {
    timeline.ActivityStartAll(entries, MEMCPY_OUT_FUSION_BUFFER);
    MemcpyOutFusionBuffer(buffer_data, entries);
    timeline.ActivityEndAll(entries);
  }

  return Status::OK();
}

bool MegrayAllreduce::Enabled(const ParameterManager& param_manager,
                            const std::vector<TensorTableEntry>& entries,
                            const Response& response) const {
  return true;
}

MegrayAllgather::MegrayAllgather(MegrayContext* megray_context,
                             HorovodGlobalState* global_state)
    : AllgatherOp(global_state), megray_context_(megray_context) {}

bool MegrayAllgather::Enabled(const ParameterManager& param_manager,
                            const std::vector<TensorTableEntry>& entries,
                            const Response& response) const {
  return true;
}

Status MegrayAllgather::Execute(std::vector<TensorTableEntry>& entries,
                              const Response& response) {
  auto& timeline = global_state_->timeline;

  // Sizes of subcomponents of each entry from all ranks
  auto** entry_component_sizes = new int64_t*[entries.size()];

  // Offset of each subcomponent of every entry in the final buffer after
  // allgatherv
  auto** entry_component_offsets = new int64_t*[entries.size()];

  int global_size = global_state_->controller->GetSize();
  auto* recvcounts = new int[global_size]();
  auto* displcmnts = new int[global_size]();

  for (size_t ec = 0; ec < entries.size(); ++ec) {
    entry_component_sizes[ec] = new int64_t[global_size]();
    entry_component_offsets[ec] = new int64_t[global_size]();
  }

  auto& first_entry = entries[0];

  timeline.ActivityStartAll(entries, ALLOCATE_OUTPUT);
  Status status =
      AllocateOutput(entries, response, entry_component_sizes, recvcounts);
  if (!status.ok()) {
    /* Cleanup */
    for (size_t ec = 0; ec < entries.size(); ++ec) {
      delete[] entry_component_sizes[ec];
      delete[] entry_component_offsets[ec];
    }   
    delete[] entry_component_sizes;
    delete[] entry_component_offsets;
    delete[] recvcounts;
    delete[] displcmnts;
    return status;
  }
  timeline.ActivityEndAll(entries);

  SetDisplacements(recvcounts, displcmnts);
  SetEntryComponentOffsets(entries, entry_component_sizes, recvcounts,
                           entry_component_offsets);

  // std::unique_ptr<IMegrayAlgorithms> megray_algos(
  //     GetAlgorithmsForType(first_entry.tensor->dtype(), megray_context_));
  std::shared_ptr<MegrayAlgorithms> megray_algos = 
    std::make_shared<MegrayAlgorithms>(megray_context_);
  //int element_size = megray_algos->ElementSize();
  int element_size = sizeof(first_entry.tensor->dtype());

  void* sendbuf = nullptr;
  void* buffer_data;

  if (entries.size() > 1) {
    timeline.ActivityStartAll(entries, MEMCPY_IN_FUSION_BUFFER);
    MemcpyInFusionBuffer(entries, displcmnts, element_size, buffer_data);
    sendbuf = buffer_data;
    timeline.ActivityEndAll(entries);
  } else {
    // need to move input data to its corresponding location in the output
    sendbuf = (void*)first_entry.tensor->data();
    buffer_data = (void*)first_entry.output->data();
    int buffer_offset = displcmnts[megray_context_->rank] * element_size;
    std::memcpy((uint8_t*)buffer_data + buffer_offset, sendbuf,
                (size_t)first_entry.tensor->size());
    sendbuf = buffer_data;
  }

  // call megray allgather api
  global_state_->timeline.ActivityStartAll(entries, MEGRAY_ALLGATHER);
  megray_algos->Allgather(sendbuf, buffer_data, recvcounts[0],
      first_entry.tensor->dtype());
  global_state_->timeline.ActivityEndAll(entries);

  // if multiple tensors are gathered, restore the sequence from output
  if (entries.size() > 1) {
    timeline.ActivityStartAll(entries, MEMCPY_OUT_FUSION_BUFFER);
    MemcpyOutFusionBuffer(entry_component_offsets, entry_component_sizes,
                          buffer_data, element_size, entries);
    timeline.ActivityEndAll(entries);
  }

  delete[] recvcounts;
  delete[] displcmnts;

  for (size_t ec = 0; ec < entries.size(); ++ec) {
    delete[] entry_component_sizes[ec];
    delete[] entry_component_offsets[ec];
  }
  delete[] entry_component_sizes;
  delete[] entry_component_offsets;

  return Status::OK();
}

MegrayBroadcast::MegrayBroadcast(MegrayContext* megray_context,
                             HorovodGlobalState* global_state)
    : BroadcastOp(global_state), megray_context_(megray_context) {}

Status MegrayBroadcast::Execute(std::vector<TensorTableEntry>& entries,
                              const Response& response) {
  assert(entries.size() == 1);
  auto e = entries[0];

  // On root rank, MPI_Bcast sends data, on other ranks it receives data.
  // for megray broadcast, only output needs to be set if inplace

  void* data_ptr;
  if (global_state_->controller->GetRank() == e.root_rank) {
    data_ptr = (void*)e.tensor->data();
  } else {
    data_ptr = (void*)e.output->data();
  }

  global_state_->timeline.ActivityStartAll(entries, MEGRAY_BCAST);
  // std::unique_ptr<IMegrayAlgorithms> megray_algos(
  //     GetAlgorithmsForType(e.tensor->dtype(), megray_context_));
  std::shared_ptr<MegrayAlgorithms> megray_algos = 
    std::make_shared<MegrayAlgorithms>(megray_context_);
  megray_algos->Broadcast(data_ptr, (int)e.tensor->shape().num_elements(),
                        e.root_rank, e.tensor->dtype());
  global_state_->timeline.ActivityEndAll(entries);

  return Status::OK();
}

bool MegrayBroadcast::Enabled(const ParameterManager& param_manager,
                            const std::vector<TensorTableEntry>& entries,
                            const Response& response) const {
  return true;
}

} // namespace common
} // namespace horovod
