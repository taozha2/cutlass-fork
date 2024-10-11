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

#pragma once

#include "cutlass/epilogue/collective/default_epilogue.hpp"
#include "cutlass/epilogue/collective/intel_pvc_epilogue.hpp"
#include "cutlass/epilogue/fusion/intel_pvc_callbacks.hpp"

using namespace cute;

///////////////////////////////////////////////////////////////////////////////////////////////////

template <
    typename LayoutA,
    typename LayoutB,
    typename LayoutC,
    typename LayoutD
    >
struct PvcGemmBF16BF16FP32 {
  using ElementAccumulator = float;                   // <- data type of accumulator
  using ElementComputeEpilogue = float;               // <- data type of epilogue operations
  using ElementInputA = bfloat16_t;                   // <- data type of elements in input matrix A
  using ElementInputB = bfloat16_t;                   // <- data type of elements in input matrix B
  using ElementOutput = float;                        // <- data type of elements in output matrix D

  // Workgroup-level tile
  using TileShape = Shape<_256, _256, _32>;

  using TiledMma = TiledMMA<
          MMA_Atom<XE_8x16x16_F32BF16BF16F32_TT>,
          Layout<Shape<_1,_1,_1>>,
          Tile<_32,_64,_32>>;                         // Subgroup level-tile

  using GmemTiledCopyA = XE_2D_U16x8x16x4x2_LD_N;
  using GmemTiledCopyB = XE_2D_U16x16x16x2x2_V;

  using PipelineStages = Int<3>;
  using GEMMDispatchPolicy = cutlass::gemm::MainloopIntelPVC<PipelineStages{}>;
  using EpilogueDispatchPolicy = cutlass::epilogue::IntelPVCEpilogue;

  using EpilogueOp = cutlass::epilogue::fusion::LinearCombination<ElementOutput, ElementComputeEpilogue,
          ElementAccumulator, ElementAccumulator, cutlass::FloatRoundStyle::round_to_nearest>;

  using FusionCallBacks = cutlass::epilogue::fusion::FusionCallbacks<EpilogueDispatchPolicy, EpilogueOp, TileShape,
          decltype(tile_shape(TiledMma()))>;
  using CollectiveEpilogue = cutlass::epilogue::collective::CollectiveEpilogue<
          EpilogueDispatchPolicy,
          TileShape,
          ElementAccumulator,
          cutlass::gemm::TagToStrideC_t<LayoutC>,
          ElementOutput,
          cutlass::gemm::TagToStrideC_t<LayoutD>,
          FusionCallBacks,
          XE_2D_U32x8x16x1x1_LD_N,
          void, void,
          XE_2D_U32x8x16x1x1_ST_N,
          void, void>;

  // Mainloop
  using CollectiveMainloop = cutlass::gemm::collective::CollectiveMma<
          GEMMDispatchPolicy,
          TileShape,
          ElementInputA,
          cutlass::gemm::TagToStrideA_t<LayoutA>,
          ElementInputB,
          cutlass::gemm::TagToStrideB_t<LayoutB>,
          TiledMma,
          GmemTiledCopyA, void, void, cute::identity,  // A
          GmemTiledCopyB, void, void, cute::identity   // B
  >;

  using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
  Shape<int, int, int, int>,
  CollectiveMainloop,
  CollectiveEpilogue
  >;

  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
};

using PvcGemmBF16BF16FP32_RRRR = PvcGemmBF16BF16FP32<
    cutlass::layout::RowMajor,
    cutlass::layout::RowMajor,
    cutlass::layout::RowMajor,
    cutlass::layout::RowMajor>;
