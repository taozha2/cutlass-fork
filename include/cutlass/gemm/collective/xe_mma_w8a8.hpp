/***************************************************************************************************
 * Copyright (c) 2025 - 2025 Codeplay Software Ltd. All rights reserved.
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
#pragma once

#include "cutlass/cutlass.h"
#include "cutlass/gemm/dispatch_policy.hpp"
#include "cutlass/fp8_to_fp16.h"

#include "cute/algorithm/functional.hpp"
#include "cute/atom/mma_atom.hpp"
#include "cute/algorithm/gemm.hpp"
#include "cute/tensor_predicate.hpp"

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass::gemm::collective {
using namespace cute;
/////////////////////////////////////////////////////////////////////////////////////////////////

template <int Stages, class Schedule, class TileShape_, class ElementA_, class StrideA_, class ElementB_, class StrideB_,
          class TiledMma_, class GmemTiledCopyA_, class SmemLayoutAtomA_, class SmemCopyAtomA_, class TransformA_,
          class GmemTiledCopyB_, class SmemLayoutAtomB_, class SmemCopyAtomB_, class TransformB_>
struct CollectiveMma<MainloopIntelW8A8<Stages, Schedule>, TileShape_, ElementA_, StrideA_, ElementB_, StrideB_, TiledMma_,
                     GmemTiledCopyA_, SmemLayoutAtomA_, SmemCopyAtomA_, TransformA_, GmemTiledCopyB_, SmemLayoutAtomB_,
                     SmemCopyAtomB_, TransformB_> {
  //
  // Type Aliases
  //
  using DispatchPolicy = MainloopIntelW8A8<Stages, Schedule>;
  using WorkgroupTileShape = TileShape_;
  using ElementA = ElementA_;
  using StrideA = StrideA_;
  using ElementB = ElementB_;
  using StrideB = StrideB_;
  using TiledMma = TiledMma_;
  using ElementAccumulator = typename TiledMma::ValTypeC;
  using GmemTiledCopyA = GmemTiledCopyA_;
  using GmemTiledCopyB = GmemTiledCopyB_;
  using SmemLayoutAtomA = SmemLayoutAtomA_;
  using SmemLayoutAtomB = SmemLayoutAtomB_;
  using SmemCopyAtomA = SmemCopyAtomA_;
  using SmemCopyAtomB = SmemCopyAtomB_;
  using TransformA = TransformA_;
  using TransformB = TransformB_;
  using ArchTag = typename DispatchPolicy::ArchTag;

  static_assert(platform::is_same<ElementA, ElementB>::value, "MainloopIntelW8A8 requires that A and B have same type.");
  static_assert(std::is_same_v<ElementA, float_e5m2_t> || std::is_same_v<ElementA, float_e4m3_t>);
  static_assert(std::is_same_v<TransformA, cute::identity>, "Transformation for A is not currently supported on Intel PVC");
  static_assert(std::is_same_v<TransformB, cute::identity>, "Transformation for B is not currently supported on Intel PVC");

  static constexpr int SubgroupSize = DispatchPolicy::SubgroupSize;

  using MmaAtomShape = typename TiledMma::AtomShape_MNK;

  static constexpr auto BLK_M = get<0>(WorkgroupTileShape{});
  static constexpr auto BLK_N = get<1>(WorkgroupTileShape{});
  static constexpr auto BLK_K = get<2>(WorkgroupTileShape{});

  static constexpr auto ATOM_M = get<1>(typename TiledMma::ThrLayoutVMNK{}.shape());
  static constexpr auto ATOM_N = get<2>(typename TiledMma::ThrLayoutVMNK{}.shape());
  static constexpr auto ATOM_K = get<3>(typename TiledMma::ThrLayoutVMNK{}.shape());

  static_assert(BLK_M % TiledMma{}.template tile_size_mnk<0>() == 0, "TiledMma permutation size must match block size.");
  static_assert(BLK_N % TiledMma{}.template tile_size_mnk<1>() == 0, "TiledMma permutation size must match block size.");
  static_assert(BLK_K % TiledMma{}.template tile_size_mnk<2>() == 0, "TiledMma permutation size must match block size.");

  static constexpr auto SG_M = ceil_div(BLK_M, ATOM_M);
  static constexpr auto SG_N = ceil_div(BLK_N, ATOM_N);
  static constexpr auto SG_K = ceil_div(BLK_K, ATOM_K);
  static constexpr auto SG_size_A = sizeof(half_t) * SG_M * SG_K;
  static constexpr auto SG_size_B = sizeof(half_t) * SG_K * SG_N;
  static constexpr auto SLM_size = 128 << 10;
  using SubgroupTileShape = Shape<decltype(SG_M), decltype(SG_N), decltype(SG_K)>;

  // 32
  static constexpr auto Num_SGs = ATOM_N * ATOM_M * ATOM_K;
  static constexpr uint32_t MaxThreadsPerBlock = size(TiledMma{});
  static constexpr auto inner_loop_k = cute::gcd(SLM_size / (SG_size_A * ATOM_M + SG_size_B * ATOM_N), Num_SGs / 2 / ATOM_N);
  static_assert(inner_loop_k == 2, "half of the SGs will be better");
  static constexpr auto allocate_elements = inner_loop_k * (SG_M * SG_K * ATOM_M + SG_N * SG_K * ATOM_N);
  using CopyThreadShape = Shape<_1, Int<SubgroupSize>>;

  using traits_load_A = Copy_Traits<GmemTiledCopyA, StrideA>;
  using atom_load_A = Copy_Atom<traits_load_A, ElementA>;
  using val_layout_load_A = decltype(make_layout(shape_div(typename traits_load_A::BlockShape{}, CopyThreadShape{})));
  using Copy_A = decltype(make_tiled_copy(atom_load_A{}, Layout<CopyThreadShape>{}, val_layout_load_A{}));

  using traits_load_B = Copy_Traits<GmemTiledCopyB, StrideB>;
  using atom_load_B = Copy_Atom<traits_load_B, ElementB>;
  using val_layout_load_B = decltype(make_layout(shape_div(typename traits_load_B::BlockShape{}, CopyThreadShape{})));
  using Copy_B = decltype(make_tiled_copy(atom_load_B{}, Layout<CopyThreadShape>{}, val_layout_load_B{}));

  // Host side kernel arguments
  struct Arguments {
    ElementA const* ptr_A;
    StrideA dA;
    ElementB const* ptr_B;
    StrideB dB;
  };

  struct Params {
    Copy_A tiled_copy_a;
    Copy_B tiled_copy_b;
    ElementA const* ptr_A;
    ElementB const* ptr_B;
  };

  //
  // Methods
  //

  CollectiveMma() = default;

  template <class ProblemShape>
  static constexpr Params
  to_underlying_arguments(ProblemShape const& problem_shape, Arguments const& args, void* workspace) {
    (void) workspace;

    auto [M,N,K,L] = problem_shape;

    auto mA_mkl = make_tensor(make_gmem_ptr(args.ptr_A), make_layout(make_shape(M, K, L), args.dA));
    auto mB_nkl = make_tensor(make_gmem_ptr(args.ptr_B), make_layout(make_shape(N, K, L), args.dB));
    Copy_A tiled_copy_a{Copy_A{}.with(mA_mkl)};
    Copy_B tiled_copy_b{Copy_B{}.with(mB_nkl)};
    
    return Params{tiled_copy_a, tiled_copy_b, args.ptr_A, args.ptr_B};
  }

  template<class ProblemShape>
  static bool
  can_implement(
      ProblemShape problem_shapes,
      Arguments const& args) {
    constexpr int copy_alignment_bits = 128;
    constexpr int batch_alignment_bits = 512;
    auto problem_shape_MNKL = append<4>(problem_shapes, 1);
    auto [M,N,K,L] = problem_shape_MNKL;

    bool implementable = true;

    constexpr int min_aligned_elements_A = copy_alignment_bits / sizeof_bits<ElementA>::value;
    implementable &= cutlass::detail::check_alignment<min_aligned_elements_A>(cute::make_shape(M,K,L), args.dA);
    constexpr int min_aligned_elements_B = copy_alignment_bits / sizeof_bits<ElementB>::value;
    implementable &= cutlass::detail::check_alignment<min_aligned_elements_B>(cute::make_shape(N,K,L), args.dB);

    if (L > 1) {
      constexpr int min_batch_aligned_elements_A = batch_alignment_bits / sizeof_bits<ElementA>::value;
      implementable &= get<2>(args.dA) % min_batch_aligned_elements_A == 0;
      constexpr int min_batch_aligned_elements_B = batch_alignment_bits / sizeof_bits<ElementB>::value;
      implementable &= get<2>(args.dB) % min_batch_aligned_elements_B == 0;
    }

    if (!implementable) {
      CUTLASS_TRACE_HOST("  CAN IMPLEMENT: Problem Size doesn't meet the minimum alignment requirements for XE 2D copy.\n");
    }

    return implementable;
  }

  /// Perform a subgroup-scoped matrix multiply-accumulate
  template <class FrgTensorD, class TensorA, class TensorB, class FrgTensorC, class KTileIterator, class BlkCoord>
  CUTLASS_DEVICE void operator()(FrgTensorD &accum, TensorA gA, TensorB gB, FrgTensorC const &src_accum,
                                 KTileIterator k_tile_iter, int k_tile_count, BlkCoord const &blk_coord, int const &K_start, int thread_idx,
                                 Params const &mainloop) {
    (void)blk_coord;
    static_assert(is_rmem<FrgTensorD>::value, "D tensor must be rmem resident.");
    static_assert(is_rmem<FrgTensorC>::value, "C tensor must be rmem resident.");

    auto smem = syclcompat::local_mem<half_t[allocate_elements]>();
    Tensor sA = make_tensor(make_smem_ptr(smem),
                            make_shape(_16{}, Int<SG_M * SG_K / 16>{}, Int<ATOM_M>{}, Int<inner_loop_k>{}));
    Tensor sB = make_tensor(make_smem_ptr(smem + SG_M * SG_K * ATOM_M * inner_loop_k),
                            make_shape(_16{}, Int<SG_N * SG_K / 16>{}, Int<inner_loop_k>{}, Int<ATOM_N>{}));

    auto thr_copy_A = mainloop.tiled_copy_a.get_slice(thread_idx);
    auto thr_copy_B = mainloop.tiled_copy_b.get_slice(thread_idx);

    // Instantiate the MMA object and get thread slice
    TiledMma tiled_mma;
    // TODO(Codeplay): see if we can make this nicer
    // To make all work items in a subgroup have the same global tensors pass in the index of work item 0 in each subgroup
    Layout sg_layout = Layout<Shape<Int<ATOM_M>, Int<ATOM_N>>, Stride<Int<ATOM_N>,_1>>{};
    Layout sg_layout_A = composition(sg_layout, make_layout(make_shape(Int<ATOM_M>{}, Int<inner_loop_k>{})));
    Layout sg_layout_B = make_layout(make_shape(Int<inner_loop_k>{}, Int<ATOM_N>{}), make_stride(_4{}, _1{}));
    auto sg = syclcompat::get_nd_item<1>().get_sub_group();
    auto wg = syclcompat::get_nd_item<1>().get_group();
    auto sg_id = sg.get_group_linear_id();
    auto wg_id = wg.get_group_linear_id();
    auto logic_sg_id = sg_id >= Num_SGs /2 ? sg_layout_B(sg_id - Num_SGs / 2) : sg_layout_A(sg_id); 
    auto first_thread_in_sg_idx = logic_sg_id * DispatchPolicy::SubgroupSize;
    auto thr_mma = tiled_mma.get_slice(first_thread_in_sg_idx);

    // Partition global counting tensors for MMA
    Tensor tCgA = thr_mma.partition_A(gA);
    Tensor tCgB = thr_mma.partition_B(gB);

    Tensor tCrA_fp16 = make_tensor<half_t>(make_fragment_layout(mainloop.tiled_copy_a, tCgA(_,_,_,0).shape()));
    Tensor tCrB_fp16 = make_tensor<half_t>(make_fragment_layout(mainloop.tiled_copy_b, tCgB(_,_,_,0).shape()));

    Tensor tCrA = make_fragment_like<uint8_t>(tCrA_fp16);
    Tensor tCrB = make_fragment_like<uint8_t>(tCrB_fp16);

    // Retile registers for copies
    Tensor tArA = thr_copy_A.retile_D(tCrA);
    Tensor tBrB = thr_copy_B.retile_D(tCrB);
    
    // Retile global counting tensors for copies
    Tensor tAgA = thr_copy_A.retile_S(tCgA);
    Tensor tBgB = thr_copy_B.retile_S(tCgB);

#if 0
#define PRINT(x) print(#x ": "); print(x); print("\n");
    if (cute::thread(0, 0)) {
      print("======================= A: \n");
      PRINT(sA);
      PRINT(tCgA);
      print(logic_sg_id);print("\n");
      print(idx2crd(4, make_shape(ATOM_M, ATOM_N), make_stride(ATOM_N, 1)));
      PRINT(sg_layout_A);
      PRINT(tAgA);

      PRINT(tCrA);
      PRINT(tArA);
      // PRINT(mainloop.tiled_copy_a);

      print("======================= B: \n");
      PRINT(tCgB);
      PRINT(sg_layout_B);
      PRINT(tBgB);

      PRINT(tCrB);
      PRINT(tBrB);
      // PRINT(mainloop.tiled_copy_b);  
    }
// #undef PRINT
#endif
    
    auto tiled_prefetch_a = cute::prefetch_selector<Shape<Int<BLK_M>,Int<BLK_K>>, Num_SGs>(mainloop.tiled_copy_a);
    auto tiled_prefetch_b = cute::prefetch_selector<Shape<Int<BLK_N>,Int<BLK_K>>, Num_SGs>(mainloop.tiled_copy_b);
    auto thr_prefetch_A = tiled_prefetch_a.get_slice(thread_idx);
    auto thr_prefetch_B = tiled_prefetch_b.get_slice(thread_idx);
    
    // Partition global tile for prefetch
    auto pAgA = thr_prefetch_A.partition_S(gA);
    auto pBgB = thr_prefetch_B.partition_S(gB);

    auto tiled_copy_A = make_tiled_copy(Copy_Atom<UniversalCopy<half_t>, half_t>{}, 
                                      Layout<Shape<_16, _1>, Stride<_1, _0>>{},
                                      Layout<Shape<_1, Int<SG_M * SG_K / 16>>, Stride<_0, _1>>{});
    auto tiled_copy_B = make_tiled_copy(Copy_Atom<UniversalCopy<half_t>, half_t>{}, 
                                        Layout<Shape<_16, _1>, Stride<_1, _0>>{},
                                        Layout<Shape<_1, Int<SG_N * SG_K / 16>>, Stride<_0, _1>>{});
    auto thr_copy_Asmem = tiled_copy_A.get_thread_slice(ThreadIdxX() % 16);
    auto thr_copy_Bsmem = tiled_copy_B.get_thread_slice(ThreadIdxX() % 16);

    //
    // Mainloop
    //
    const auto k_start_idx = crd2idx((*k_tile_iter), make_shape(K_start));
    constexpr int barrier_scope = 2;
    int prefetch_k = k_start_idx;

    // CUTLASS_PRAGMA_UNROLL
    // for (; prefetch_k < DispatchPolicy::Stages; prefetch_k++) {
    //   prefetch(tiled_prefetch_a, pAgA(_, _, _, prefetch_k));
    //   prefetch(tiled_prefetch_b, pBgB(_, _, _, prefetch_k));
    // }

    CUTLASS_PRAGMA_UNROLL
    for (int k_tile = 0; k_tile < ceil_div(k_tile_count, inner_loop_k); k_tile++, prefetch_k +=2) {
      if(sg_id < Num_SGs >> 1){
        auto [m, k] = idx2crd(logic_sg_id, make_shape(ATOM_M, inner_loop_k), make_stride(4, 1));
        copy(mainloop.tiled_copy_a, tAgA(_,_,_,k_tile * inner_loop_k + k), tArA);
        convert_FP8_to_FP16<ElementA>(tCrA, tCrA_fp16);
        Tensor thr_copy_store_A =  thr_copy_Asmem.partition_D(sA(_,_,m,k));
        auto tCrA_fp16_view = make_tensor(static_cast<decltype(tCrA_fp16)&&>(tCrA_fp16).data(),
                                          thr_copy_store_A.shape());
        copy(tiled_copy_A, tCrA_fp16_view, thr_copy_store_A);
      } else {
        auto [k, n] = idx2crd(logic_sg_id, make_shape(inner_loop_k, ATOM_N), make_stride(4,1));
        copy(mainloop.tiled_copy_b, tBgB(_,_,_,k_tile * inner_loop_k + k), tBrB);
        convert_FP8_to_FP16<ElementB>(tCrB, tCrB_fp16);
        Tensor thr_copy_store_B = thr_copy_Bsmem.partition_D(sB(_,_,k,n));
        auto tCrB_fp16_view = make_tensor(static_cast<decltype(tCrB_fp16)&&>(tCrB_fp16).data(),
                                          thr_copy_store_B.shape());
        copy(tiled_copy_B, tCrB_fp16_view, thr_copy_store_B);
      }
      barrier_arrive(barrier_scope,barrier_scope,260);
      barrier_wait(barrier_scope,barrier_scope,258);
      // if (prefetch_k < k_tile_count) {
      //   prefetch(tiled_prefetch_a, pAgA(_, _, _, prefetch_k));
      //   prefetch(tiled_prefetch_b, pBgB(_, _, _, prefetch_k));
      //   prefetch(tiled_prefetch_a, pAgA(_, _, _, prefetch_k+1));
      //   prefetch(tiled_prefetch_b, pBgB(_, _, _, prefetch_k+1));
      // }
      // barrier_wait(barrier_scope);
      // print("");
  
      auto [m, n] = idx2crd(sg_id, make_shape(4, 4), make_stride(4, 1));
      Tensor thr_copy_load_A =  thr_copy_Asmem.partition_S(sA(_,_,m,0));
      auto tCrA_fp16_view = make_tensor(static_cast<decltype(tCrA_fp16)&&>(tCrA_fp16).data(),
                                        thr_copy_load_A.shape());
      Tensor thr_copy_load_B =  thr_copy_Bsmem.partition_S(sB(_,_,0,n));
      auto tCrB_fp16_view = make_tensor(static_cast<decltype(tCrB_fp16)&&>(tCrB_fp16).data(),
                                        thr_copy_load_B.shape());
      
      CUTLASS_PRAGMA_UNROLL
      for(int k = inner_loop_k - 1; k > -1; k--) {
        Tensor thr_copy_load_A = thr_copy_Asmem.partition_S(sA(_,_,m,k));
        Tensor thr_copy_load_B = thr_copy_Bsmem.partition_S(sB(_,_,k,n));
        
        copy(tiled_copy_A, thr_copy_load_A, tCrA_fp16_view);
        copy(tiled_copy_B, thr_copy_load_B, tCrB_fp16_view);
        cute::gemm(tiled_mma, tCrA_fp16, tCrB_fp16, accum);
      }
      barrier_arrive(2,2,260);
      barrier_wait(2,2,258);
    }
  }
};

} // namespace cutlass::gemm::collective

/////////////////////////////////////////////////////////////////////////////////////////////////
