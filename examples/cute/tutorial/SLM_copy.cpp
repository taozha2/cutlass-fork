/***************************************************************************************************
 * Copyright (c) 2023 - 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * Copyright (c) 2024 - 2024 Codeplay Software Ltd. All rights reserved.
 * Copyright (C) 2025 Intel Corporation, All rights reserved.
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
#include <sycl/sycl.hpp>
#include <cute/util/compat.hpp>
#include <sycl/ext/intel/experimental/grf_size_properties.hpp>

#include <cute/tensor.hpp>

#include "cutlass/kernel_hardware_info.h"
#include "cutlass/platform/platform.h"
#include "cutlass/tensor_ref.h"
#include "cutlass/util/sycl_event_manager.hpp"
#include "cutlass/util/GPU_Clock.hpp"
#include "cutlass/util/reference/device/gemm_complex.h"
#include "cutlass/util/reference/device/tensor_compare.h"
#include "cutlass/util/reference/host/tensor_fill.h"

#include "../../common/sycl_cute_common.hpp"

#if defined(__clang__)
  #pragma clang diagnostic ignored "-Wpass-failed"
  #pragma clang diagnostic ignored "-Wdeprecated-declarations"
#elif defined(__GNUC__)
  #pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#endif
template<class...> class CopyKernelGlobalName;

using namespace cute;

template<class TensorS, class CtaTiler, class SmemLayout, class ThreadLayout>
void copy_kernel_vector(TensorS S, CtaTiler cta_tiler, SmemLayout smem_layout, ThreadLayout t_layout) {
  static_assert(is_static<SmemLayout>::value);
  using Element = typename TensorS::value_type;
  auto smem = compat::local_mem<Element[cosize_v<SmemLayout>]>();

//   auto cta_coord = make_coord(compat::work_group_id::x(), compat::work_group_id::y(), _);
  
  using COPY_ATOM = XE_LOAD_2D<sizeof_bits_v<Element>, 16, 16>;
  auto copy_global = make_block_2d_copy_X<Element>(COPY_ATOM{}, S.stride(),
                        find_x_mode(S.stride()), find_y_mode(S.stride()),
                        make_tile(_32{}, _256{}),
                        Layout<Shape<_2, _16>>{}).with(S);

  auto thr_copy_global = copy_global.get_slice(compat::local_id::x());
  /* Create proxy coordinate tensors for each global tensor */
  Tensor cC = make_identity_tensor(S.shape());   // (M,K)
  Tensor gS = local_tile(cC, select<0,2>(cta_tiler), make_coord(compat::work_group_id::x(),_));  // (BLK_M,BLK_K,k)
  Tensor sD = make_tensor(make_smem_ptr(smem), smem_layout);          // (BLK_M,BLK_K,stages)
  /* Register fragments for copies */
  auto trS = thr_copy_global.partition_sg_fragment_D(gS(_,_,0));
  /* Partition global tensor (proxies) for copies */
  Tensor tgS = thr_copy_global.partition_S(gS);

#if 0
  #define PRINT(x) print(#x ": "); print(x); print("\n");
  if(cute::thread0()) {
    PRINT(trS);
    PRINT(tgS);
    PRINT(gS);
  }
#endif
  
  int k_tile_count = ceil_div(shape<1>(S), get<2>(cta_tiler));

  #pragma unroll
  for(int k_tile = 0; k_tile < k_tile_count; ++k_tile) {
    copy(copy_global, tgS(_,_,_,k_tile), trS);
  }

}

template<class TensorS, class CtaTiler, class SmemLayout, class ThreadLayout>
void copy_kernel_1d(TensorS S, CtaTiler cta_tiler, SmemLayout smem_layout, ThreadLayout t_layout) {
  static_assert(is_static<SmemLayout>::value);
  using Element = typename TensorS::value_type;
  auto smem = compat::local_mem<Element[cosize_v<SmemLayout>]>();

//   auto cta_coord = make_coord(compat::work_group_id::x(), compat::work_group_id::y(), _);
  
  using COPY_ATOM = XE_LOAD_2D<sizeof_bits_v<Element>, 16, 16>;
  auto copy_global = make_block_2d_copy_X<Element>(COPY_ATOM{}, S.stride(),
                        find_x_mode(S.stride()), find_y_mode(S.stride()),
                        make_tile(_32{}, _256{}),
                        Layout<Shape<_2, _16>>{}).with(S);

  auto thr_copy_global = copy_global.get_slice(compat::local_id::x());
  /* Create proxy coordinate tensors for each global tensor */
  Tensor cC = make_identity_tensor(S.shape());   // (M,K)
  Tensor gS = local_tile(cC, select<0,2>(cta_tiler), make_coord(compat::work_group_id::x(),_));  // (BLK_M,BLK_K,k)
  Tensor sD = make_tensor(make_smem_ptr(smem), smem_layout);          // (BLK_M,BLK_K,stages)
  /* Register fragments for copies */
  auto trS = thr_copy_global.partition_sg_fragment_D(gS(_,_,0));
  /* Partition global tensor (proxies) for copies */
  Tensor tgS = thr_copy_global.partition_S(gS);

#if 0
  #define PRINT(x) print(#x ": "); print(x); print("\n");
  if(cute::thread0()) {
    PRINT(trS);
    PRINT(tgS);
    PRINT(gS);
  }
#endif
  
  int k_tile_count = ceil_div(shape<1>(S), get<2>(cta_tiler));

  #pragma unroll
  for(int k_tile = 0; k_tile < k_tile_count; ++k_tile) {
    copy(copy_global, tgS(_,_,_,k_tile), trS);
  }

}

template<class TensorS, class CtaTiler, class SmemLayout, class ThreadLayout>
void copy_kernel_naive(TensorS S, CtaTiler cta_tiler, SmemLayout smem_layout, ThreadLayout t_layout) {
  static_assert(is_static<SmemLayout>::value);
  using Element = typename TensorS::value_type;
  auto smem = compat::local_mem<Element[cosize_v<SmemLayout>]>();

  auto cta_coord = make_coord(compat::work_group_id::x(), compat::work_group_id::y(), _);
  Tensor gS = local_tile(S, cta_tiler, cta_coord, Step<_1, X,_1>{});  // (BLK_M,BLK_K,k)
  Tensor sD = make_tensor(make_smem_ptr(smem), smem_layout);          // (BLK_M,BLK_K,stages)

  Tensor tgS = local_partition(gS, t_layout, compat::local_id::x());   // (THR_M,THR_N,k)
  Tensor tsD = local_partition(sD, t_layout, compat::local_id::x());   // (THR_M,THR_N,stages)

  auto K_TILE_MAX = size<2>(tgS);
  constexpr auto stages = size<2>(tsD);
  
  #pragma unroll
  for(int k_tile = 0; k_tile < K_TILE_MAX; ++k_tile) {
    copy(tgS(_, _, k_tile), tsD(_, _, k_tile % stages));
  }
  
#if 0
#define PRINT(x) print(#x ": "); print(x); print("\n");
  if(cute::thread0()) {
    PRINT(tgS);
    PRINT(tsD);
  }
#endif

}

int main(int argc, char** argv) {
  int M = 4096;
  int K = 4096 * 4;

  using Element = uint32_t;

  std::vector<Element> host_src(M * K);

  for(size_t i = 0; i < M * K; ++i) {
    host_src[i] = static_cast<Element>(i);
  }
  
  using bM = _32;
  using bK = _256;
  using stages = _2;
  using CtaTiler = Shape<bM, _0, bK>; 
  auto thread_layout = Layout<Shape<_2, _128>, Stride<_128, _1>>{};
  auto smem_layout = Layout<Shape<bM, bK, stages>, Stride<bK, _1, _8192>>{};

  auto device_src = compat::malloc<Element>(M * K);
  compat::memcpy<Element>(device_src, host_src.data(), M * K);
  Tensor S = make_tensor(make_gmem_ptr(device_src),
                         make_layout(make_shape(M, K), make_stride(K, _1{})));

  auto dimBlock = compat::dim3(size(thread_layout));
  auto dimGrid  = compat::dim3(size(ceil_div(M, bM{})));
  //
  // Launch the kernel
  //
  GPU_Clock timer;
  auto iterations = 20;

  timer.start();
  for (int i = 0; i < iterations; ++i) {
    auto event = compat::launch<copy_kernel_naive<decltype(S), CtaTiler, decltype(smem_layout),
                              decltype(thread_layout)>, CopyKernelGlobalName<decltype(S), CtaTiler, decltype(smem_layout),
                              decltype(thread_layout)>>(
        dimGrid, dimBlock, S, CtaTiler{}, smem_layout, thread_layout);
    EventManager::getInstance().addEvent(event);

  }
  compat::wait();
  float cute_time = timer.seconds() / iterations;
  double io = M * K * sizeof(Element) * 1e-12;
  printf("Performance:     [%4.3f]Gb/s  (%6.4f)ms\n", io / cute_time, cute_time*1000);
  return 0;
}
