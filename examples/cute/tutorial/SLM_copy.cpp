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

#define PRINT(x) print(#x ": "); print(x); print("\n");
template<class...> class CopyKernelGlobalName;

using namespace cute;

template<class TensorS, class CtaTiler, class SmemLayout, class ThreadLayout>
void copy_kernel_ocl(TensorS S, CtaTiler cta_tiler, SmemLayout smem_layout, ThreadLayout t_layout) {
  static_assert(is_static<SmemLayout>::value);
  using Element = typename TensorS::value_type;
  auto smem = compat::local_mem<Element[cosize_v<SmemLayout>]>();

//   auto cta_coord = make_coord(compat::work_group_id::x(), compat::work_group_id::y(), _);
  Tensor sD = make_tensor(make_smem_ptr(smem), smem_layout);          // (BLK_M,BLK_N,stages)
  using traits_load = Copy_Traits<XE_2D_U32x16x16_LD_N, decltype(S)>;
  using Atom_load = Copy_Atom<traits_load, Element>;
  auto tiled_copy_load = make_tiled_copy(Atom_load{}.with(S),
                                         Layout<Shape<_1, _16>>{},
                                         make_layout(shape_div(typename traits_load::BlockShape{}, Shape<_1, _16>{})));

  auto S_coord = cute::get_xe_tensor(append(S.shape(),_1{}))(_,_,0);

  Tensor tiled_tensor_S = tiled_divide(
    S_coord, select<0,1>(cta_tiler)); // ((M, N), m', n')
  // Slice work group.
  Tensor tile_wg_S = tiled_tensor_S(make_coord(_, _), BlockIdxX(), BlockIdxY());
  
  auto thr_copy_load =
      tiled_copy_load.get_thread_slice(cutlass::get_sub_group_local_id());
  auto SubgroupShape = make_shape(ceil_div(get<0>(cta_tiler), get<0>(t_layout.shape())), 
                                  ceil_div(get<1>(cta_tiler), get<1>(t_layout.shape()) / _16{}));
  auto sg_id = cutlass::get_sub_group_id();
  Tensor tile_sg_S = local_tile(tile_wg_S, SubgroupShape, sg_id);
  Tensor thr_tile_load_S = thr_copy_load.partition_S(tile_sg_S);
  Tensor thr_tile_load_D = thr_copy_load.partition_D(tile_sg_S);
  Tensor fragment = make_tensor<Element>(thr_tile_load_D.shape());

  
  auto tiled_slm =
      make_tiled_copy(Copy_Atom<UniversalCopy<uint64_t>, Element>{},
                      t_layout,
                      Layout<Shape<_4, _2>, Stride<_2,_1>>{});
  auto thr_copy = tiled_slm.get_slice(ThreadIdxX());
  Tensor thr_local_D = thr_copy.partition_D(sD(_, _, 0));
  auto trD = make_tensor(fragment.data(), thr_local_D.layout());
#if 0
  if(cute::thread0()) {
    PRINT(tiled_tensor_S);
    PRINT(fragment);
    PRINT(tile_wg_S);
    PRINT(thr_tile_load_S);
  }
#endif

  copy(tiled_copy_load, thr_tile_load_S, fragment);
  copy(tiled_slm, trD, thr_local_D);
  compat::wg_barrier();
}


template<class TensorS, class CtaTiler, class SmemLayout, class ThreadLayout>
void copy_kernel_vector(TensorS S, CtaTiler cta_tiler, SmemLayout smem_layout, ThreadLayout t_layout) {
  static_assert(is_static<SmemLayout>::value);
  using Element = typename TensorS::value_type;
  auto smem = compat::local_mem<Element[cosize_v<SmemLayout>]>();

  using COPY_ATOM = XE_LOAD_2D<sizeof_bits_v<Element>, 8, 16>;
  auto SgV = make_layout(make_shape(Shape<_8, _4>{}, Shape<_8, Shape<_8, _2>>{}),
                       make_stride(Stride<_8192,_64>{}, Stride<_1, Stride<_8, _4096>>{}));
  auto copy_global = make_block_2d_copy_X<Element>(COPY_ATOM{}, S.stride(),
                        find_x_mode(S.stride()), find_y_mode(S.stride()),
                        make_tile(get<0>(cta_tiler), get<1>(cta_tiler)),
                        SgV).with(S);

  auto thr_copy_global = copy_global.get_slice(compat::local_id::x());
  /* Create proxy coordinate tensors for each global tensor */
  Tensor cC = make_identity_tensor(S.shape());   // (M,N)
  Tensor gS = local_tile(cC, select<0,1>(cta_tiler), make_coord(compat::work_group_id::x(),compat::work_group_id::y()));  // (BLK_M,BLK_N)
  Tensor sD = make_tensor(make_smem_ptr(smem), smem_layout);          // (BLK_M,BLK_N,stages)
  /* Register fragments for copies */
  auto trS = thr_copy_global.partition_sg_fragment_D(gS);
  /* Partition global tensor (proxies) for copies */
  Tensor tgS = thr_copy_global.partition_S(gS);

  auto tiled_slm =
      make_tiled_copy(Copy_Atom<UniversalCopy<uint64_t>, Element>{},
                      t_layout,
                      Layout<Shape<_4, _2>, Stride<_2,_1>>{});
  auto thr_copy = tiled_slm.get_slice(ThreadIdxX());
  Tensor thr_local_D = thr_copy.partition_D(sD(_, _, 0));
  auto trD = make_tensor(trS.tensor().data(), thr_local_D.layout());

  copy(copy_global, tgS, trS);
  copy(tiled_slm, trD, thr_local_D);
  compat::wg_barrier();
#if 0
  if(cute::thread0()) {
    PRINT(trS);
    PRINT(tgS);
    PRINT(gS);
    PRINT(trD);
  }
#endif
}

template<class TensorS, class CtaTiler, class SmemLayout, class ThreadLayout>
void copy_kernel_1d(TensorS S, CtaTiler cta_tiler, SmemLayout smem_layout, ThreadLayout t_layout) {
  static_assert(is_static<SmemLayout>::value);
  using Element = typename TensorS::value_type;
  auto smem = compat::local_mem<Element[cosize_v<SmemLayout>]>();

  using COPY_ATOM = XE_LOAD_2D<sizeof_bits_v<Element>, 8, 16>;
  auto SgV = make_layout(make_shape(Shape<_8, _4>{}, Shape<_8, Shape<_8, _2>>{}),
                       make_stride(Stride<_8192,_64>{}, Stride<_1, Stride<_8, _4096>>{}));
  auto copy_global = make_block_2d_copy_X<Element>(COPY_ATOM{}, S.stride(),
                        find_x_mode(S.stride()), find_y_mode(S.stride()),
                        make_tile(get<0>(cta_tiler), get<1>(cta_tiler)),
                        SgV).with(S);

  auto thr_copy_global = copy_global.get_slice(compat::local_id::x());
  /* Create proxy coordinate tensors for each global tensor */
  Tensor cC = make_identity_tensor(S.shape());   // (M,N)
  Tensor gS = local_tile(cC, select<0,1>(cta_tiler), make_coord(compat::work_group_id::x(),compat::work_group_id::y()));  // (BLK_M,BLK_N)
  Tensor sD = make_tensor(make_smem_ptr(smem), smem_layout);          // (BLK_M,BLK_N,stages)
  /* Register fragments for copies */
  auto trS = thr_copy_global.partition_sg_fragment_D(gS);
  /* Partition global tensor (proxies) for copies */
  Tensor tgS = thr_copy_global.partition_S(gS);

#if 0
  if(cute::thread0()) {
    PRINT(trS);
    PRINT(tgS);
    PRINT(gS);
  }
#endif
  
  int k_tile_count = ceil_div(shape<1>(S), get<2>(cta_tiler));

  copy(copy_global, tgS, trS);
}

template<class TensorS, class CtaTiler, class SmemLayout, class ThreadLayout>
void copy_kernel_naive(TensorS S, CtaTiler cta_tiler, SmemLayout smem_layout, ThreadLayout t_layout) {
  static_assert(is_static<SmemLayout>::value);
  using Element = typename TensorS::value_type;
  auto smem = compat::local_mem<Element[cosize_v<SmemLayout>]>();

  auto cta_coord = make_coord(compat::work_group_id::x(), compat::work_group_id::y(), _);
  Tensor gS = local_tile(S, cta_tiler, cta_coord, Step<_1, _1, X>{});  // (BLK_M,BLK_N)
  Tensor sD = make_tensor(make_smem_ptr(smem), smem_layout);          // (BLK_M,BLK_N,stages)

  Tensor tgS = local_partition(gS, t_layout, compat::local_id::x());   // (THR_M,THR_N)
  Tensor tsD = local_partition(sD, t_layout, compat::local_id::x());   // (THR_M,THR_N,stages)

  // auto K_TILE_MAX = size<2>(tgS);
  constexpr auto stages = size<2>(tsD);

  copy(tgS, tsD(_, _, 0));
  compat::wg_barrier();

#if 0
  if(cute::thread0()) {
    PRINT(tgS);
  }
#endif

}

int main(int argc, char** argv) {
  constexpr uint M = 256*16;
  constexpr uint N = 256*16;

  using Element = uint32_t;

  std::vector<Element> host_src(M * N);

  for(size_t i = 0; i < M * N; ++i) {
    host_src[i] = static_cast<Element>(i);
  }
  
  using bM = _256;
  using bN = _256;
  using stages = _2;
  using CtaTiler = Shape<bM, bN, _0>; 
  auto thread_layout = Layout<Shape<_4, _128>, Stride<_128, _1>>{};
  // auto smem_layout = Layout<Shape<bM, bN, stages>, Stride<bN, _1, _8192>>{};
  auto smem_layout = composition(
          Swizzle<2,1,3>{},
          Layout<Shape<bM, bN, stages>, Stride<bN, _1, _8192>>{});
  auto device_src = compat::malloc<Element>(M * N);
  compat::memcpy<Element>(device_src, host_src.data(), M * N);
  Tensor S = make_tensor(make_gmem_ptr(device_src),
                         make_layout(make_shape(M, N), make_stride(N, _1{})));

  auto dimBlock = compat::dim3(size(thread_layout));
  auto dimGrid  = compat::dim3(size(ceil_div(M, bM{})), size(ceil_div(N, bN{})));
  //
  // Launch the kernel
  //
  GPU_Clock timer;
  auto iterations = 20;

  timer.start();
  for (int i = 0; i < iterations; ++i) {
    auto event = compat::launch<copy_kernel_vector<decltype(S), CtaTiler, decltype(smem_layout),
                              decltype(thread_layout)>, CopyKernelGlobalName<decltype(S), CtaTiler, decltype(smem_layout),
                              decltype(thread_layout)>>(
        dimGrid, dimBlock, S, CtaTiler{}, smem_layout, thread_layout);
    EventManager::getInstance().addEvent(event);

  }
  compat::wait();
  float cute_time = timer.seconds() / iterations;
  double io = M * N * sizeof(Element) * 1e-12;
  printf("Performance:     [%4.3f]Gb/s  (%6.4f)ms\n", io / cute_time, cute_time*1000);
  return 0;
}
