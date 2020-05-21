// Copyright 2019 MegRay Technologies, Inc. All Rights Reserved.
// =============================================================================

#ifndef HOROVOD_MEGRAY_OPERATIONS_H
#define HOROVOD_MEGRAY_OPERATIONS_H

#include "collective_operations.h"
#include "../megray/megray_context.h"
#include "src/megray.h"

namespace horovod {
namespace common {

class MegrayAlgorithms {
public:
  MegrayAlgorithms(MegrayContext* megray_context);
  void Allreduce(void* in_data, void* out_data, int num_elements, DataType dtype);

  void Allgather(void* in_data, void* out_data, int in_len, DataType dtype);

  void Broadcast(void* buffer_data, int num_elements, int root_rank, DataType dtype);

  //int ElementSize() const;
private:
  MegrayContext* megray_context_;
  std::shared_ptr<MegRay::Communicator> comm = nullptr;
};

// template <typename T> class MegrayAlgorithms : public IMegrayAlgorithms {
// public:
//   MegrayAlgorithms(MegrayContext* megray_context);
// 
//   ~MegrayAlgorithms() = default;
// 
//   void Allreduce(void* in_data, void* out_data, int num_elements) override;
// 
//   void Allgather(void* in_data, void* out_data, int in_len) override;
// 
//   void Broadcast(void* buffer_data, int num_elements, int root_rank) override;
// 
//   int ElementSize() const override;
// 
// private:
//   MegrayContext* megray_context_;
//   std::shared_ptr<MegRay::Communicator> comm = nullptr;
// };

class MegrayAllreduce : public AllreduceOp {
public:
  std::shared_ptr<MegRay::Communicator> comm;

  MegrayAllreduce(MegrayContext* megray_context, HorovodGlobalState* global_state);

  virtual ~MegrayAllreduce() = default;

  Status Execute(std::vector<TensorTableEntry>& entries,
                 const Response& response) override;

  bool Enabled(const ParameterManager& param_manager,
               const std::vector<TensorTableEntry>& entries,
               const Response& response) const override;

protected:
  MegrayContext* megray_context_;
};

class MegrayAllgather : public AllgatherOp {
public:
  MegrayAllgather(MegrayContext* megray_context, HorovodGlobalState* global_state);

  Status Execute(std::vector<TensorTableEntry>& entries,
                 const Response& response) override;

  bool Enabled(const ParameterManager& param_manager,
               const std::vector<TensorTableEntry>& entries,
               const Response& response) const override;

protected:
  MegrayContext* megray_context_;
};

class MegrayBroadcast : public BroadcastOp {
public:
  MegrayBroadcast(MegrayContext* megray_context, HorovodGlobalState* global_state);

  Status Execute(std::vector<TensorTableEntry>& entries,
                 const Response& response) override;

  bool Enabled(const ParameterManager& param_manager,
               const std::vector<TensorTableEntry>& entries,
               const Response& response) const override;

protected:
  MegrayContext* megray_context_;
};

} // namespace common
} // namespace horovod

#endif // HOROVOD_MEGRAY_OPERATIONS_H
