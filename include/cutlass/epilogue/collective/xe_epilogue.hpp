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
  \brief Functor performing elementwise operations used by epilogues.
*/

#pragma once

#include <sycl/sycl.hpp>
#include "cutlass/cutlass.h"
#include "cutlass/epilogue/dispatch_policy.hpp"
#include "cutlass/epilogue/collective/collective_epilogue.hpp"
#include "cutlass/epilogue/collective/detail.hpp"
#include "cutlass/epilogue/fusion/callbacks.hpp"
#include "cutlass/epilogue/fusion/sm90_visitor_tma_warpspecialized.hpp"
#include "cutlass/detail/layout.hpp"

#include "cute/tensor.hpp"

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass {
namespace epilogue {
namespace collective {

/////////////////////////////////////////////////////////////////////////////////////////////////

template <
  class CtaTileMNK_,
  class ElementC_,
  class StrideC_,
  class ElementD_,
  class StrideD_,
  class FusionCallbacks_,
  class CopyOpG2R_,
  class SmemLayoutAtomC_,
  class CopyOpS2R_,
  class CopyOpR2G_,
  class SmemLayoutAtomD_,
  class CopyOpR2S_
>
class CollectiveEpilogue<
    IntelPVCEpilogue,
    CtaTileMNK_,
    ElementC_,
    StrideC_,
    ElementD_,
    StrideD_,
    FusionCallbacks_,
    CopyOpG2R_,
    SmemLayoutAtomC_,
    CopyOpS2R_,
    CopyOpR2G_,
    SmemLayoutAtomD_,
    CopyOpR2S_
> {
public:
  //
  // Type Aliases
  //
  using DispatchPolicy = IntelPVCEpilogue;
  using CtaTileMNK = CtaTileMNK_;
  using FusionCallbacks = FusionCallbacks_;
  using ElementC = ElementC_;
  using ElementAccumulator = ElementC_;
  using StrideC = StrideC_;
  using ElementD = ElementD_;
  using StrideD = StrideD_;
  using CopyOpG2R = CopyOpG2R_;
  using SmemLayoutAtomC = SmemLayoutAtomC_;
  using CopyOpS2R = CopyOpS2R_;
  using CopyOpR2G = CopyOpR2G_;
  using SmemLayoutAtomD = SmemLayoutAtomD_;
  using CopyOpR2S = CopyOpR2S_;

  using ThreadEpilogueOp = typename fusion::FusionCallbacksTraits<FusionCallbacks>::Operation;
  using GmemTiledCopyC = CopyOpG2R;
  using GmemTiledCopyD = CopyOpR2G;
  using ElementOutput = typename FusionCallbacks::ElementOutput;
  using ElementCompute = typename FusionCallbacks::ElementCompute;

  static constexpr int SubgroupSize = DispatchPolicy::SubgroupSize;

  static_assert(cute::rank(CtaTileMNK{}) == 3, "CtaTileMNK must be rank-3: [CTA_M, CTA_N, CTA_K]");
  static_assert(cute::rank(StrideC{}) == 3, "StrideC must be rank-3: [M, N, L]");
  static_assert(cute::rank(StrideD{}) == 3, "StrideD must be rank-3: [M, N, L]");

  static_assert(std::is_same_v<CopyOpS2R, void>, "Copy operation to shared memory is not supported");
  static_assert(std::is_same_v<CopyOpR2S, void>, "Copy operation to shared memory is not supported");
  static_assert(std::is_same_v<SmemLayoutAtomC, void>, "Copy operation to shared memory is not supported");
  static_assert(std::is_same_v<SmemLayoutAtomD, void>, "Copy operation to shared memory is not supported");

  using Trait_C = Copy_Traits<GmemTiledCopyC>;
  using XE_Copy_C = decltype(make_tiled_copy(Copy_Atom<Trait_C, ElementC>{}
                                             .with(static_cast<ElementC const*>(nullptr), int32_t(0), int32_t(0), int32_t(0)),
                                             Layout<Shape<_1, Int<SubgroupSize>>>{},
                                             make_layout(make_shape(get<0>(typename Trait_C::Shape_MN{}),
                                                                    get<1>(typename Trait_C::Shape_MN{}) / Int<SubgroupSize>{}))));
  using Trait_D = Copy_Traits<GmemTiledCopyD>;
  using XE_Copy_D = decltype(make_tiled_copy(Copy_Atom<Trait_D, ElementD>{}
                                             .with(static_cast<ElementD const*>(nullptr),int32_t(0), int32_t(0), int32_t(0)),
                                             Layout<Shape<_1, Int<SubgroupSize>>>{},
                                             make_layout(make_shape(get<0>(typename Trait_D::Shape_MN{}),
                                                                    get<1>(typename Trait_D::Shape_MN{}) / Int<SubgroupSize>{}))));
private:
  constexpr static bool is_source_supported = not cute::is_void_v<ElementC>;
  constexpr static bool is_destination_supported = not cute::is_void_v<ElementD>;

  constexpr static bool is_m_major_C = detail::is_m_major<StrideC>();
  constexpr static bool is_m_major_D = detail::is_m_major<StrideD>();

public:

  using EmptyType = cute::tuple<>;
  using SmemCStorage = EmptyType;
  using SmemDStorage = EmptyType;

  struct TensorStorageImpl: cute::tuple<SmemCStorage, SmemDStorage> {
    using FusionStorage = typename FusionCallbacks::SharedStorage;
    FusionStorage thread;
  };

  struct SharedStorage {
    using TensorStorage = TensorStorageImpl;

    TensorStorage tensors;
  };
  using TensorStorage = typename SharedStorage::TensorStorage;

  // Host side epilogue arguments
  struct Arguments {
    typename FusionCallbacks::Arguments thread{};
    ElementC const* ptr_C;
    StrideC dC;
    ElementD const* ptr_D;
    StrideD dD;
  };

  // Device side epilogue params
  struct Params {
    typename FusionCallbacks::Params thread{};
    XE_Copy_C xe_load_c;
    XE_Copy_D xe_store_d;
  };

  //
  // Methods
  //

  template <class ProblemShape>
  static constexpr Params
  to_underlying_arguments(
      ProblemShape const& problem_shape,
      Arguments const& args,
      [[maybe_unused]] void* workspace) {
    // Optionally append 1s until problem shape is rank-4 in case its is only rank-3 (MNK)
    auto problem_shape_MNKL = append<4>(problem_shape, 1);
    auto [M, N, K, L] = problem_shape_MNKL;

    XE_Copy_C xe_load_c = {};
    if constexpr (is_source_supported) {
      xe_load_c = make_tiled_copy(Copy_Atom<Copy_Traits<CopyOpG2R>, ElementC>{}.with(
                                  args.ptr_C, N, M, N),
                                  Layout<Shape<_1, Int<SubgroupSize>>>{},
                                  make_layout(make_shape(get<0>(typename Trait_C::Shape_MN{}),
                                                         get<1>(typename Trait_C::Shape_MN{}) / Int<SubgroupSize>{})));
    }

    XE_Copy_D xe_store_d = {};
    if constexpr (is_destination_supported) {
      xe_store_d = make_tiled_copy(Copy_Atom<Copy_Traits<CopyOpR2G>, ElementD>{}.with(
                                   args.ptr_D, N, M, N),
                                   Layout<Shape<_1, Int<SubgroupSize>>>{},
                                   make_layout(make_shape(get<0>(typename Trait_D::Shape_MN{}),
                                                          get<1>(typename Trait_D::Shape_MN{}) / Int<SubgroupSize>{})));
    }

    return {
      FusionCallbacks::to_underlying_arguments(problem_shape, args.thread, workspace),
      xe_load_c,
      xe_store_d
    };
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

  template <class ProblemShape>
  CUTLASS_HOST_DEVICE static bool
  can_implement(
      ProblemShape const& problem_shape,
      [[maybe_unused]] Arguments const& args) {
    return true;
  }

  CUTLASS_HOST_DEVICE
  CollectiveEpilogue(Params const& params_, TensorStorage const& shared_storage_)
      : params(params_), fusion_callbacks(params_.thread, shared_storage_.thread) {}

  CUTLASS_DEVICE
  bool
  is_producer_load_needed() const {
    return fusion_callbacks.is_producer_load_needed();
  }

  template<
    class ProblemShapeMNKL,
    class TileShapeMNK,
    class TileCoordMNKL,
    class Accumulator,
    class TiledMma,
    class ResidueMNK
  >
  CUTLASS_DEVICE void
  operator() (
      ProblemShapeMNKL problem_shape_mnkl,
      TileShapeMNK tile_shape_MNK,
      TileCoordMNKL tile_coord_mnkl,
      Accumulator accumulators, 
      TiledMma tiled_mma,
      ResidueMNK residue_mnk,
      int thread_idx,
      char* smem) {
    
    (void) tiled_mma;
    (void) residue_mnk;
    (void) smem;
    using namespace cute;

    static_assert(cute::rank(CtaTileMNK{}) == 3, "CtaTileMNK must be rank-3: [CTA_M, CTA_N, CTA_K]");
    static_assert(cute::rank(StrideC{}) == 3, "StrideC must be rank-3: [M, N, L]");
    static_assert(cute::rank(StrideD{}) == 3, "StrideD must be rank-3: [M, N, L]");

    using MmaAtomShape = typename TiledMma::AtomShape_MNK;
    static constexpr auto BLK_M = get<0>(CtaTileMNK{});
    static constexpr auto BLK_N = get<1>(CtaTileMNK{});
    static constexpr auto BLK_K = get<2>(CtaTileMNK{});
    // static_assert(is_same_v<typename TiledMma::ThrLayoutVMNK, int>, "assertation fail");
    static constexpr auto ATOM_M = get<1>(typename TiledMma::ThrLayoutVMNK{}.shape());
    static constexpr auto ATOM_N = get<2>(typename TiledMma::ThrLayoutVMNK{}.shape());
    static constexpr auto ATOM_K = get<3>(typename TiledMma::ThrLayoutVMNK{}.shape());
    
    static_assert(
      BLK_M % ATOM_M == 0 &&
      BLK_N % ATOM_N == 0 &&
      BLK_K % ATOM_K == 0,
      "expected CTATileMNK to be evenly divided by TiledMma::ThrLayoutVMNK");
    static constexpr auto SG_M = BLK_M / ATOM_M;
    static constexpr auto SG_N = BLK_N / ATOM_N;
    static constexpr auto SG_K = BLK_K / ATOM_K;
    using SubgroupTileShape = Shape<decltype(SG_M), decltype(SG_N), decltype(SG_K)>;

    static constexpr int FragsM = get<0>(SubgroupTileShape{}) / get<0>(MmaAtomShape()); // A frags per sub_group
    static constexpr int FragsN = get<1>(SubgroupTileShape{}) / get<1>(MmaAtomShape()); // B frags per sub_group

    static constexpr int FragmentSize = (get<0>(MmaAtomShape()) * get<1>(MmaAtomShape())) / SubgroupSize;

    // Indexing variables
    auto [M, N, K, L] = problem_shape_mnkl;
    auto [m_coord, n_coord, k_coord, l_coord] = tile_coord_mnkl;
    auto m_offset = m_coord * BLK_M + (get_sub_group_id() / ATOM_N) * SG_M;
    auto n_offset = n_coord * BLK_N + (get_sub_group_id() % ATOM_N) * SG_N;
    auto l_offset = l_coord;

    using EpilogueTile = decltype(get<0>(params.xe_store_d.get_layoutS_MN()).shape());

    auto sg_local_m_coord = get_sub_group_id() / ATOM_N;
    auto sg_local_n_coord = get_sub_group_id() % ATOM_N;

    auto sg_m_coord = m_coord * ATOM_M + sg_local_m_coord;
    auto sg_n_coord = n_coord * ATOM_N + sg_local_n_coord;
    auto sg_coord = make_coord(sg_m_coord, sg_n_coord, k_coord, l_coord);

    bool is_C_load_needed = is_source_supported && fusion_callbacks.is_C_load_needed();

    Tensor trC = make_tensor<typename TiledMma::ValTypeC>(Shape<Int<FragmentSize>>{});
    Tensor trD = make_tensor<typename TiledMma::ValTypeD>(Shape<Int<FragmentSize>>{});
    Tensor tOuti = params.xe_store_d.get_pvc_tensor(
            make_coord(m_offset, n_offset, 0),
            make_shape(_, Int<FragsM>{}, Int<FragsN>{}, L),
            make_stride(Int<get<0>(MmaAtomShape{})>{}, Int<get<1>(MmaAtomShape{})>{}, _1{}));
    // TODO(joe): I replaced tiled_r2s from the sm90 version with tiled_g2r on the assumption
    // that this would give me the equivalent thread layout that I need for XE epilogue (which
    // doesn't use shared memory. Is this the correct/best approach?

    ThrCopy thread_g2r = params.xe_load_c.get_slice(thread_idx);
    // Because Nvidia uses shared memory, they are not tied to using the same accumulator values
    // for MMA and Epilogue. But because we are operating directly in the accumulators, we need to be
    // sure that we are operating on the same values.

    // OOB predication for tile quantization "residue"
    // Absolute coordinate tensors (dynamic)
    Tensor mD_crd = make_identity_tensor(make_shape(M,N));                                                     // (M,N)
    // Is this likely to be wrong, since we have redefined m_coord (?) for subgroup scale logic
    Tensor cD_mn = local_tile(mD_crd, take<0,2>(CtaTileMNK{}), make_coord(m_coord, n_coord));          // (CTA_M,CTA_N)
    Tensor tRS_cD_mn = thread_g2r.partition_S(flat_divide(cD_mn, EpilogueTile{}));     // (G2R,G2R_M,G2R_N,EPI_M,EPI_N)

    Tensor cD = make_counting_tensor(cD_mn.layout());                                                  // (CTA_M,CTA_N)
    Tensor tRS_cD = make_counting_tensor(tRS_cD_mn.layout());                          // (G2R,G2R_M,G2R_N,EPI_M,EPI_N)

    Tensor rw_coord = tOuti(_,_,_,l_coord);

    // Subtract the global "bottom right" corner from the local "top left" corner to get the max relative coordinate
    auto residue_cD = make_coord(M,N) - cD_mn(_0{});                                                           // (m,n)
    auto residue_tRS_cD = make_coord(M,N) - tRS_cD_mn(_0{});                                                   // (m,n)

    // Get the fusion callbacks
    constexpr bool RefSrc = true;
    auto residue_mn = make_coord(M, N);
    auto cst_args = cutlass::epilogue::fusion::detail::ConsumerStoreArgs{
                      problem_shape_mnkl,
                      SubgroupTileShape{},
                      sg_coord,
                      tiled_mma,
                      EpilogueTile{},
                      params.xe_store_d,
                      cD,
                      residue_cD,
                      tRS_cD,
                      residue_tRS_cD,
                      trC,
                      thread_idx,
                    };
    auto cst_callbacks = fusion_callbacks.template get_consumer_store_callbacks<RefSrc>(cst_args);

    cst_callbacks.begin();

    auto acc_frag = recast<Array<ElementOutput, FragmentSize>>(accumulators);
    auto trD_frag = recast<Array<ElementOutput, FragmentSize>>(trD);

    constexpr int ValuesLoaded =
      FragsM * FragsN * FragmentSize * SubgroupSize * ATOM_M * ATOM_N * ATOM_K;
    constexpr int MN = get<0>(CtaTileMNK{}) * get<1>(CtaTileMNK{});
    static_assert(ValuesLoaded == MN, "the total elements loaded by all threads should be the same as MxN" );

    CUTLASS_PRAGMA_UNROLL
    for (int epi_n = 0; epi_n < FragsN; epi_n++) {
      CUTLASS_PRAGMA_UNROLL
      for (int epi_m = 0; epi_m < FragsM; epi_m++) {

        if (is_C_load_needed) {
          copy(params.xe_load_c, rw_coord(_, epi_m, epi_n), trC);
        }

        cst_callbacks.previsit(epi_m, epi_n, 0, is_C_load_needed);

        auto acc_frag_mn = acc_frag(_, epi_m, epi_n);

        CUTLASS_PRAGMA_UNROLL
        for (int epi_v = 0; epi_v < size(trD_frag); ++epi_v) {
          trD_frag(epi_v) = cst_callbacks.visit(acc_frag_mn(epi_v), epi_v, epi_m, epi_n);
        }
        copy(params.xe_store_d, trD, rw_coord(_, epi_m, epi_n));
      }
    }

    cst_callbacks.end();
  }

private:
  Params const& params;
  FusionCallbacks fusion_callbacks;
};


/////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace collective
} // namespace epilogue
} // namespace cutlass

/////////////////////////////////////////////////////////////////////////////////////////////////
