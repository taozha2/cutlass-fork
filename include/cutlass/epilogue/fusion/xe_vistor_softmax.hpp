/***************************************************************************************************
 * Copyright (c) 2024 - 2024 Codeplay Software Ltd. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 * list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its
 * contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 **************************************************************************************************/

/*! \file
  \brief Visitor tree Softmax fusion operation for the Intel PVC epilogue
*/

#pragma once

#include "cutlass/cutlass.h"
#include "cutlass/workspace.h"

#include "cute/tensor.hpp"
#include "sm90_visitor_tma_warpspecialized.hpp"
#ifdef __SYCL_DEVICE_ONLY__
#define SYCL_DEVICE_OCL(x) SYCL_EXTERNAL x
#else
#define SYCL_DEVICE_OCL(x)                                                     \
  inline x { assert(false); }
#endif

SYCL_DEVICE_OCL(float sub_group_reduce_add(float i));
SYCL_DEVICE_OCL(float sub_group_reduce_max(float i));

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass::epilogue::fusion {

/////////////////////////////////////////////////////////////////////////////////////////////////
#undef MAX
#undef EXP
#undef DIV

#define MAX sycl::max
#define EXP sycl::native::exp
#define DIV sycl::native::divide

namespace detail {

CUTLASS_DEVICE
float item_reduce_sum(float val) {
  float res = val;
  return sub_group_reduce_add(res);
}

CUTLASS_DEVICE
float item_reduce_max(float val) {
  float res = val;
  return sub_group_reduce_max(res);
}

template<uint32_t N>
CUTLASS_DEVICE
decltype(auto) sg_reduce_sum(float* vec) {
  CUTLASS_PRAGMA_UNROLL
  for (int i = 0; i < N; i++) {
    vec[i] = item_reduce_sum(vec[i]);    
  }
}

template<uint32_t N>
CUTLASS_DEVICE
decltype(auto) sg_reduce_max(float* vec) {
  CUTLASS_PRAGMA_UNROLL
  for (int i = 0; i < N; i++) {
    vec[i] = item_reduce_max(vec[i]);    
  }
}

template<uint32_t sg_num, uint32_t N, class mem_t>
CUTLASS_DEVICE
decltype(auto) work_group_reduce_sum(mem_t &mem, float* vec) {
  auto item = sycl::ext::oneapi::experimental::this_nd_item<3>();
  auto sg = item.get_sub_group();
  auto group = item.get_group();
  auto sg_group_id_n = sg.get_group_id() % sg_num;
  auto sg_local_id = sg.get_local_id()[0];

  static_assert((sg_num % IntelPVCEpilogue::SubgroupSize) == 0);

  sycl::group_barrier(group);

  static constexpr auto step = sg_num / IntelPVCEpilogue::SubgroupSize;

  if constexpr (sg_num <= N) {
    static constexpr auto n_step = N / sg_num;
    auto base = sg_local_id * N * step + sg_group_id_n * n_step;

    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < N / sg_num; i++) {
      auto sum = 0.f;

      CUTLASS_PRAGMA_UNROLL
      for (int j = 0; j < step; j++) {
        sum += mem[base + i + N * j];
      }

      auto group_sum = item_reduce_sum(sum);

      if (sg_local_id == i) {
        mem[sg_group_id_n * n_step + i] = group_sum;
      }
    }
  }
  else {
    auto sum = 0.f;

    auto base = sg_local_id * N * step + sg_group_id_n;

    if (sg_group_id_n < N) {
    CUTLASS_PRAGMA_UNROLL
      for (int j = 0; j < step; j++) {
        sum += mem[base + N * j];
      }

      auto group_sum = item_reduce_sum(sum);

      if (sg_local_id == 0) {
        mem[sg_group_id_n] = group_sum;
      }
    }
  }
  sycl::group_barrier(group);

  CUTLASS_PRAGMA_UNROLL
  for (int i = 0; i < N; i++) {
    vec[i] = mem[i];
  }
}

template<uint32_t sg_num, uint32_t N, class mem_t>
CUTLASS_DEVICE
void work_group_reduce_max(mem_t &mem, float* vec) {
  auto item = sycl::ext::oneapi::experimental::this_nd_item<3>();
  auto sg = item.get_sub_group();
  auto group = item.get_group();
  auto sg_group_id = sg.get_group_id();
  auto sg_group_id_n = sg_group_id % sg_num;
  auto sg_local_id = sg.get_local_id()[0];

  static_assert((sg_num % IntelPVCEpilogue::SubgroupSize) == 0);

  sycl::group_barrier(group);

  static constexpr auto step = sg_num / IntelPVCEpilogue::SubgroupSize;

  if constexpr (sg_num <= N) {
    static constexpr auto n_step = N / sg_num;
    auto base = sg_local_id * N * step + sg_group_id_n * n_step;

    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < N / sg_num; i++) {
      auto local_max = mem[base + i];

      CUTLASS_PRAGMA_UNROLL
      for (int j = 1; j < step; j++) {
        local_max = MAX(local_max, mem[base + i + N * j]);
      }

      auto group_max = item_reduce_max(local_max);

      if (sg_local_id == i) {
        mem[sg_group_id_n * n_step + i] = group_max;
      }
    }
  } 
  else {
    auto base = sg_local_id * N * step + sg_group_id_n;
    auto local_max = mem[base];

    if (sg_group_id_n < N) {
      CUTLASS_PRAGMA_UNROLL
      for (int j = 1; j < step; j++) {
        local_max = MAX(local_max, mem[base + N * j]);
      }

      auto group_max = item_reduce_max(local_max);

      if (sg_local_id == 0) {
        mem[sg_group_id_n] = group_max;
      }
    }
  }
  sycl::group_barrier(group);

  CUTLASS_PRAGMA_UNROLL
  for (int i = 0; i < N; i++) {
    vec[i] = mem[i];
  }
}

template <class mem_t, uint32_t N, uint32_t sg_per_wg_n>
CUTLASS_DEVICE
void group_reduce_sum(mem_t smem, float *const vec,
                                       float *out) {
  auto item = sycl::ext::oneapi::experimental::this_nd_item<3>();
  auto sg = item.get_sub_group();

  sg_reduce_sum<N>(vec);

  auto sg_group_id = sg.get_group_id();
  auto sg_group_id_n = sg_group_id % sg_per_wg_n;
  auto sg_local_id = sg.get_local_id()[0];

  auto slm_base = smem + (sg_group_id / sg_per_wg_n) * sg_per_wg_n * N;

  if constexpr (N < IntelPVCEpilogue::SubgroupSize) {
    if (sg_local_id < N) {
      slm_base[sg_group_id_n * N + sg_local_id] = vec[sg_local_id];
    }
  }
  else {
    static constexpr auto step = N / IntelPVCEpilogue::SubgroupSize;
    auto base = sg_local_id * step;

    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < step; i++) {
      auto offset = base + i;
      slm_base[sg_group_id_n * N + offset] = vec[offset];
    }
  }

  work_group_reduce_sum<sg_per_wg_n, N, decltype(slm_base)>(slm_base, out);
}

template <class mem_t, uint32_t N, uint32_t sg_per_wg_n>
CUTLASS_DEVICE
void group_reduce_max(mem_t smem, float *const vec,
                                       float *out) {
  auto item = sycl::ext::oneapi::experimental::this_nd_item<3>();
  auto sg = item.get_sub_group();

  sg_reduce_max<N>(vec);

  auto sg_group_id = sg.get_group_id();
  auto sg_group_id_n = sg_group_id % sg_per_wg_n;
  auto sg_local_id = sg.get_local_id()[0];

  auto slm_base = smem + (sg_group_id / sg_per_wg_n) * sg_per_wg_n * N;

  if constexpr (N < IntelPVCEpilogue::SubgroupSize) {
    if (sg_local_id < N) {
      slm_base[sg_group_id_n * N + sg_local_id] = vec[sg_local_id];
    }
  }
  else {
    static constexpr auto step = N / IntelPVCEpilogue::SubgroupSize;
    auto base = sg_local_id * step;

  CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < step; i++) {
      slm_base[sg_group_id_n * N + base + i] = vec[base + i];
    }
  }

    work_group_reduce_max<sg_per_wg_n, N, decltype(slm_base)>(slm_base, out);
}

template <uint32_t sg_num, class mem_t, class RTensor>
CUTLASS_DEVICE
auto group_reduce_sum1(mem_t smem, RTensor const &t, float *out) {
  static constexpr auto row = decltype(size<0>(t))::value;
  static constexpr auto col = decltype(size<1>(t))::value;

  float local_sum[row];

  CUTLASS_PRAGMA_UNROLL
  for (int i = 0; i < row; i++) {
    local_sum[i] = t(i, 0);
  }

  CUTLASS_PRAGMA_UNROLL
  for (int i = 1; i < col; i++) {
    auto tmp = t(_, i);

    CUTLASS_PRAGMA_UNROLL
      for (int j = 0; j < row; j++) {
        local_sum[j] += tmp(j);
      }
  }
  group_reduce_sum<mem_t, row, sg_num>(smem, local_sum, out);
}

template <uint32_t sg_num, class mem_t, class RTensor>
CUTLASS_DEVICE
auto group_reduce_max1(mem_t smem, RTensor const &t, float *out) {
  static constexpr auto row = decltype(size<0>(t))::value;
  static constexpr auto col = decltype(size<1>(t))::value;

  float local_max[row];

  CUTLASS_PRAGMA_UNROLL
  for (int i = 0; i < row; i++) {
    local_max[i] = t(i, 0);
  }

  CUTLASS_PRAGMA_UNROLL
  for (int i = 1; i < col; i++) {
    auto tmp = t(_, i);

    CUTLASS_PRAGMA_UNROLL
    for (int j = 0; j < row; j++) {
      local_max[j] = MAX(local_max[j], tmp(j));
    }
  }
  group_reduce_max<mem_t, row, sg_num>(smem, local_max, out);
}

} // namespace detail

template <
  // int FragmentSize,
  class CtaTileShapeMNK,
  class EpilogueTile,
  class ElementOutput,
  class ElementCompute,
  FloatRoundStyle RoundStyle
>
struct XeSoftmaxRowReduction
{
public:
  static constexpr int FragmentSize = 8;
  static constexpr auto Tile_M = get<0>(CtaTileShapeMNK{});
  static constexpr auto Tile_N = get<1>(CtaTileShapeMNK{});
  static constexpr auto Epi_M = get<0>(EpilogueTile{});
  static constexpr auto Epi_N = get<1>(EpilogueTile{});
  static constexpr auto Sg_M = Tile_M / Epi_M;
  static constexpr auto Sg_N = Tile_N / Epi_N;
  static constexpr auto Sg_Nums = Sg_M * Sg_N;
  struct SharedStorage { };

  struct Arguments { };

  struct Params { };

  template <class ProblemShape>
  static constexpr Params
  to_underlying_arguments(ProblemShape const& problem_shape, Arguments const& args, void* workspace) {
    return {};
  }

  template <class ProblemShape>
  static bool
  can_implement(ProblemShape const& problem_shape, Arguments const& args) {
    auto [M, N, K, L] = problem_shape;
    auto [tile_M, tile_N, tile_K] = CtaTileShapeMNK{};
    // Cross CTA reduction is not possible because there is no guarantee that all CTAs run
    // concurrently.
    // Cross epilogue tile reduction is possible, but re-visiting and applying reduction
    // to accumulators is only possible for the current epilogue tile.
    auto [epi_M, epi_N] = EpilogueTile{};
    return N <= tile_N;
  }

  template <class ProblemShape>
  static size_t
  get_workspace_size(ProblemShape const& problem_shape, Arguments const& args) {
    return 0;
  }

  template <class ProblemShape>
  static cutlass::Status
  initialize_workspace(ProblemShape const& problem_shape, Arguments const& args, void* workspace, cudaStream_t stream,
    CudaHostAdapter* cuda_adapter = nullptr) {
    return Status::kSuccess;
  }

  CUTLASS_DEVICE bool
  is_producer_load_needed() const {
    return false;
  }

  CUTLASS_DEVICE bool
  is_C_load_needed() const {
    return false;
  }

  CUTLASS_HOST_DEVICE
  XeSoftmaxRowReduction() { }
  
  CUTLASS_HOST_DEVICE
  XeSoftmaxRowReduction(Params const& params, SharedStorage const& shared_storage)
      : params(params) { }

  Params params;

  template <class... Args>
  CUTLASS_DEVICE auto
  get_producer_load_callbacks(ProducerLoadArgs<Args...> const& args) {
    return EmptyProducerLoadCallbacks{};
  }

  template<class ArgsTuple>
  struct ConsumerStoreCallbacks : EmptyConsumerStoreCallbacks {
    
    CUTLASS_DEVICE
    ConsumerStoreCallbacks(Params const& params) : params(params) {}

    // ArgsTuple args_tuple;
    Params const& params;
    template <typename ElementInput, typename ElementAccumulator, int FragmentSize>
    CUTLASS_DEVICE auto
    visit(Array<ElementAccumulator, FragmentSize> const& frg_acc, int epi_v, int epi_m, int epi_n,
          Array<ElementInput, FragmentSize> const& frg_input) {
      
      return frg_acc;
    }

    template<class STensor, class SyncFn, class VTensor>
    CUTLASS_DEVICE void
    reduce(STensor&& smem_buffer, SyncFn const& sync_fn, int epi_m, int epi_n, bool is_last_iteration, VTensor visit_results) {
      if(is_last_iteration) {
      constexpr auto vec_size = min(Epi_M, Sg_N);
      constexpr auto loop_cnt = Epi_M / vec_size;

      auto smem = syclcompat::local_mem<float[Sg_Nums * vec_size]>();

      auto t =
          make_tensor(static_cast<decltype(visit_results) &&>(visit_results).data() - epi_m * FragmentSize - epi_n,
                      make_shape(Int<vec_size>{}, Int<loop_cnt>{}, Int<Sg_N / IntelPVCEpilogue::SubgroupSize>{}));

      CUTLASS_PRAGMA_UNROLL
      for (int loop = 0; loop < loop_cnt; loop++) {
        auto loop_t = t(_, loop, _);
        float group_max[vec_size];
        group_reduce_max1<Sg_N>(smem, loop_t, group_max);
        CUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < Sg_N / IntelPVCEpilogue::SubgroupSize; i++) {
          auto tmp = loop_t(_, i);
          CUTLASS_PRAGMA_UNROLL
          for (int j = 0; j < vec_size; j++) {
            tmp(j) -= group_max[j];
          }
        }
      }
      CUTLASS_PRAGMA_UNROLL
      for (int loop = 0; loop < loop_cnt; loop++) {
        auto loop_t = t(_, loop, _);
        CUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < Sg_N / IntelPVCEpilogue::SubgroupSize; i++) {
          auto tmp = loop_t(_, i);
          CUTLASS_PRAGMA_UNROLL
          for (int j = 0; j < vec_size; j++) {
            tmp(j) = EXP(tmp(j));
          }
        }
      }
      CUTLASS_PRAGMA_UNROLL
      for (int loop = 0; loop < loop_cnt; loop++) {
        auto loop_t = t(_, loop, _);

        float group_sum[vec_size];
        group_reduce_sum1<Sg_N>(smem, loop_t, group_sum);
      
        CUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < Sg_N / IntelPVCEpilogue::SubgroupSize; i++) {
          auto tmp = loop_t(_, i);
          CUTLASS_PRAGMA_UNROLL
          for (int j = 0; j < vec_size; j++) {
            tmp(j) = DIV(tmp(j), group_sum[j]);
          }
        }
      }
    }
    }
  };
  
  template <
  bool ReferenceSrc, // do register tensors reference the src or dst layout of the tiled copy
  class... Args
  >
  CUTLASS_DEVICE auto
  get_consumer_store_callbacks(ConsumerStoreArgs<Args...> const& args) {
    auto args_tuple = make_tuple();
    return ConsumerStoreCallbacks<decltype(args_tuple)>(params);
  }

};


/////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace cutlass::epilogue::fusion

/////////////////////////////////////////////////////////////////////////////////////////////////
