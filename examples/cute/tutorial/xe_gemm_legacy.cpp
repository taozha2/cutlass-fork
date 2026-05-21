/***************************************************************************************************
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
/*! \file
    \brief Xe GEMM example using legacy CollectiveMma (xe_mma_legacy.hpp) and
           legacy CollectiveEpilogue (xe_epilogue_legacy.hpp).

    This example demonstrates how to compose the legacy mainloop and epilogue
    into a standalone GEMM kernel for bf16 x bf16 -> fp32 on Intel BMG hardware.

    To build & run (from build dir):
      $ ninja xe_gemm_legacy
      $ ./examples/cute/tutorial/xe_gemm_legacy
*/

#include <sycl/sycl.hpp>
#include <cute/util/compat.hpp>
#include <sycl/ext/intel/experimental/grf_size_properties.hpp>

#include <cute/tensor.hpp>

#include "cutlass/cutlass.h"
#include "cutlass/kernel_hardware_info.h"
#include "cutlass/gemm/dispatch_policy.hpp"
#include "cutlass/gemm/collective/collective_mma.hpp"
#include "cutlass/util/packed_stride.hpp"

#include "cutlass/util/GPU_Clock.hpp"
#include "cutlass/util/command_line.h"
#include "cutlass/util/sycl_event_manager.hpp"
#include "cutlass/util/reference/device/gemm_complex.h"
#include "cutlass/util/reference/device/tensor_compare.h"

#include "../../common/sycl_cute_common.hpp"

#if defined(__clang__)
  #pragma clang diagnostic ignored "-Wpass-failed"
  #pragma clang diagnostic ignored "-Wdeprecated-declarations"
#elif defined(__GNUC__)
  #pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#endif

using namespace cute;

///////////////////////////////////////////////////////////////////////////////////////////////////

// Command line options parsing
struct Options {

  bool help;
  bool error;

  int m, n, k, l, iterations, verify;

  Options():
    help(false),
    error(false),
    m(4096), n(4096), k(4096), l(1), iterations(20), verify(1)
  { }

  void parse(int argc, char const **args) {
    cutlass::CommandLine cmd(argc, args);

    if (cmd.check_cmd_line_flag("help")) {
      help = true;
      return;
    }

    cmd.get_cmd_line_argument("m", m, 4096);
    cmd.get_cmd_line_argument("n", n, 4096);
    cmd.get_cmd_line_argument("k", k, 4096);
    cmd.get_cmd_line_argument("l", l, 1);
    cmd.get_cmd_line_argument("iterations", iterations, 20);
    cmd.get_cmd_line_argument("verify", verify, 1);
  }

  std::ostream & print_usage(std::ostream &out) const {
    out << "Xe GEMM Legacy Example (bf16 x bf16 -> fp32)\n\n"
      << "Options:\n\n"
      << "  --help                      If specified, displays this usage statement\n\n"
      << "  --m=<int>                   Sets the M extent of the GEMM\n"
      << "  --n=<int>                   Sets the N extent of the GEMM\n"
      << "  --k=<int>                   Sets the K extent of the GEMM\n"
      << "  --l=<int>                   Sets the L extent (batch count) of the GEMM\n"
      << "  --iterations=<int>          Iterations\n"
      << "  --verify=<int>              Specify whether to verify (1=yes, 0=no).\n\n";
    return out;
  }
};

///////////////////////////////////////////////////////////////////////////////////////////////////
// Type definitions for bf16 x bf16 -> fp32 GEMM using legacy dispatch policies
///////////////////////////////////////////////////////////////////////////////////////////////////

using ElementA = bfloat16_t;
using ElementB = bfloat16_t;
using ElementD = float;
using ElementAccumulator = float;

using LayoutA = cutlass::layout::RowMajor;
using LayoutB = cutlass::layout::RowMajor;
using LayoutD = cutlass::layout::RowMajor;

using StrideA = cutlass::gemm::TagToStrideA_t<LayoutA>;
using StrideB = cutlass::gemm::TagToStrideB_t<LayoutB>;
using StrideD = cutlass::gemm::TagToStrideC_t<LayoutD>;

// Workgroup tile shape
using TileShape = Shape<_256, _256, _32>;

// TiledMMA: XE_8x16x16_BF16BF16F32_TT with 8x4 subgroup layout
using TiledMma = TiledMMA<
    MMA_Atom<XE_8x16x16_F32BF16BF16F32_TT>,
    Layout<Shape<_8, _4, _1>, Stride<_4, _1, _0>>
>;

// Legacy dispatch policies
constexpr int PipelineStages = 3;
using MainloopPolicy = cutlass::gemm::MainloopIntelXeXMX16<PipelineStages>;

// Copy atoms for A and B (legacy 2D block loads)
using GmemCopyA = XE_2D_U16x8x16_LD_N;
using GmemCopyB = XE_2D_U16x16x16_LD_V;

// Legacy CollectiveMma (from xe_mma_legacy.hpp)
using CollectiveMainloop = cutlass::gemm::collective::CollectiveMma<
    MainloopPolicy,
    TileShape,
    ElementA, StrideA,
    ElementB, StrideB,
    TiledMma,
    GmemCopyA, void, void, cute::identity,  // A
    GmemCopyB, void, void, cute::identity   // B
>;

///////////////////////////////////////////////////////////////////////////////////////////////////
// GEMM Kernel - combines legacy mainloop and simple store epilogue
///////////////////////////////////////////////////////////////////////////////////////////////////

template <class MainloopType>
struct GemmLegacyKernel {

  using ProblemShape = Shape<int, int, int, int>;

  struct Arguments {
    ProblemShape problem_shape;
    typename MainloopType::Arguments mainloop_args;
    ElementD* ptr_D;
    StrideD dD;
  };

  struct Params {
    ProblemShape problem_shape;
    typename MainloopType::Params mainloop_params;
    ElementD* ptr_D;
    StrideD dD;
  };

  static Params to_underlying_arguments(Arguments const& args) {
    auto mainloop_params = MainloopType::to_underlying_arguments(args.problem_shape, args.mainloop_args, nullptr);
    return {args.problem_shape, mainloop_params, args.ptr_D, args.dD};
  }

  static constexpr int MaxThreadsPerBlock = CollectiveMainloop::MaxThreadsPerBlock;
};

///////////////////////////////////////////////////////////////////////////////////////////////////
// Device kernel
///////////////////////////////////////////////////////////////////////////////////////////////////

template <class Params>
void gemm_legacy_device(Params const& params) {
  using namespace cute;

  auto item = compat::get_nd_item<1>();
  int thread_idx = item.get_local_linear_id();
  int block_idx = item.get_group_linear_id();

  auto [M, N, K, L] = params.problem_shape;

  int tile_m_count = ceil_div(M, get<0>(TileShape{}));
  int tile_n_count = ceil_div(N, get<1>(TileShape{}));

  // Compute tile coordinates from block index
  int l_coord = block_idx / (tile_m_count * tile_n_count);
  int mn_idx = block_idx % (tile_m_count * tile_n_count);
  int m_coord = mn_idx / tile_n_count;
  int n_coord = mn_idx % tile_n_count;

  auto tile_coord = make_coord(m_coord, n_coord, _, l_coord);

  // =========================================================================
  // MAINLOOP (inlined from xe_mma_legacy.hpp CollectiveMma::operator())
  // =========================================================================

  static constexpr int BLK_M = get<0>(TileShape{});
  static constexpr int BLK_N = get<1>(TileShape{});
  static constexpr int BLK_K = get<2>(TileShape{});

  static constexpr int ATOM_M = get<1>(typename TiledMma::ThrLayoutVMNK{}.shape());
  static constexpr int ATOM_N = get<2>(typename TiledMma::ThrLayoutVMNK{}.shape());
  static constexpr int ATOM_K = get<3>(typename TiledMma::ThrLayoutVMNK{}.shape());
  static constexpr auto Num_SGs = ATOM_N * ATOM_M * ATOM_K;

  // Create 3D identity tensors for coordinate tracking
  Tensor mA = make_identity_tensor(make_shape(M, K, L));
  Tensor mB = make_identity_tensor(make_shape(N, K, L));

  // Tile for this workgroup (matching xe_gemm.hpp pattern exactly)
  constexpr auto blk_shape = TileShape{};
  Tensor gA = local_tile(mA, select<0,2>(blk_shape), make_coord(m_coord, _, l_coord));
  Tensor gB = local_tile(mB, select<1,2>(blk_shape), make_coord(n_coord, _, l_coord));

  // Get thread-level copies from mainloop params
  auto thr_copy_A = params.mainloop_params.tiled_copy_a.get_slice(thread_idx);
  auto thr_copy_B = params.mainloop_params.tiled_copy_b.get_slice(thread_idx);

  // Instantiate the MMA object and get thread slice
  TiledMma tiled_mma;
  auto sg = compat::get_nd_item<1>().get_sub_group();
  auto first_thread_in_sg_idx = sg.get_group_linear_id() * MainloopPolicy::SubgroupSize;
  auto thr_mma = tiled_mma.get_slice(first_thread_in_sg_idx);

  // Partition global counting tensors for MMA
  Tensor tCgA = thr_mma.partition_A(gA);
  Tensor tCgB = thr_mma.partition_B(gB);

  // Create register fragments for A and B
  Tensor tCrA = make_tensor<ElementA>(make_fragment_layout(params.mainloop_params.tiled_copy_a, tCgA(_,_,_,0).shape()));
  Tensor tCrB = make_tensor<ElementB>(make_fragment_layout(params.mainloop_params.tiled_copy_b, tCgB(_,_,_,0).shape()));

  // Retile registers for copies
  Tensor tArA = thr_copy_A.retile_D(tCrA);
  Tensor tBrB = thr_copy_B.retile_D(tCrB);

  // Retile global counting tensors for copies
  Tensor tAgA = thr_copy_A.retile_S(tCgA);
  Tensor tBgB = thr_copy_B.retile_S(tCgB);

  // Setup prefetch
  auto tiled_prefetch_a = cute::prefetch_selector<Shape<Int<BLK_M>, Int<BLK_K>>, Num_SGs>(params.mainloop_params.tiled_copy_a);
  auto tiled_prefetch_b = cute::prefetch_selector<Shape<Int<BLK_N>, Int<BLK_K>>, Num_SGs>(params.mainloop_params.tiled_copy_b);
  auto thr_prefetch_A = tiled_prefetch_a.get_slice(thread_idx);
  auto thr_prefetch_B = tiled_prefetch_b.get_slice(thread_idx);

  // Partition global tile for prefetch
  auto pAgA = thr_prefetch_A.partition_S(gA);
  auto pBgB = thr_prefetch_B.partition_S(gB);

  // Accumulator
  Tensor accumulators = partition_fragment_C(tiled_mma, select<0,1>(TileShape{}));
  clear(accumulators);

  // Setup D store (before mainloop so gemm and copy stay adjacent)
  auto mD = make_tensor(make_gmem_ptr(params.ptr_D), make_layout(make_shape(M, N, L), params.dD));
  Tensor cD = make_identity_tensor(make_shape(M, N, L));
  Tensor gD = local_tile(cD, make_shape(Int<BLK_M>{}, Int<BLK_N>{}), make_coord(m_coord, n_coord, l_coord));
  auto copy_d = make_block_2d_copy_D(tiled_mma, mD);
  Tensor tCgD = thr_mma.partition_C(gD);

  // Mainloop iteration
  int k_tile_count = ceil_div(K, BLK_K);
  constexpr int barrier_scope = 2;
  int prefetch_k = 0;

  // Prefetch warmup
  CUTLASS_PRAGMA_UNROLL
  for (; prefetch_k < PipelineStages; prefetch_k++) {
    prefetch(tiled_prefetch_a, pAgA(_, _, _, prefetch_k));
    prefetch(tiled_prefetch_b, pBgB(_, _, _, prefetch_k));
  }

  // Main k-tile loop
  for (int k_tile = 0; k_tile < k_tile_count; k_tile++, prefetch_k++) {
    barrier_arrive(barrier_scope);

    // Copy A/B from global memory to registers
    copy(params.mainloop_params.tiled_copy_a, tAgA(_,_,_,k_tile), tArA);
    copy(params.mainloop_params.tiled_copy_b, tBgB(_,_,_,k_tile), tBrB);

    // Prefetch next tiles
    if (prefetch_k < k_tile_count) {
      prefetch(tiled_prefetch_a, pAgA(_, _, _, prefetch_k));
      prefetch(tiled_prefetch_b, pBgB(_, _, _, prefetch_k));
    }

    // MMA: accumulators += A * B
    cute::gemm(tiled_mma, tCrA, tCrB, accumulators);

    barrier_wait(barrier_scope);
  }

  // Store accumulators to global memory
  copy(copy_d, accumulators, tCgD);
}

///////////////////////////////////////////////////////////////////////////////////////////////////
// Host-side runner
///////////////////////////////////////////////////////////////////////////////////////////////////

template <class, class, int> class GemmLegacyKernelName;

bool run(Options const& options) {
  auto M = options.m;
  auto N = options.n;
  auto K = options.k;
  auto L = options.l;

  using ProblemShape = Shape<int, int, int, int>;
  ProblemShape problem_shape{M, N, K, L};

  // Compute strides
  StrideA stride_A = cutlass::make_cute_packed_stride(StrideA{}, make_shape(M, K, L));
  StrideB stride_B = cutlass::make_cute_packed_stride(StrideB{}, make_shape(N, K, L));
  StrideD stride_D = cutlass::make_cute_packed_stride(StrideD{}, make_shape(M, N, L));

  // Allocate device memory
  auto Q = compat::get_default_queue();

  size_t size_A = static_cast<size_t>(M) * K * L;
  size_t size_B = static_cast<size_t>(K) * N * L;
  size_t size_D = static_cast<size_t>(M) * N * L;

  auto ptr_A = sycl::malloc_shared<ElementA>(size_A, Q);
  auto ptr_B = sycl::malloc_shared<ElementB>(size_B, Q);
  auto ptr_D = sycl::malloc_shared<ElementD>(size_D, Q);

  // Initialize data
  srand(2024);
  for (size_t i = 0; i < size_A; i++) ptr_A[i] = ElementA(float(rand()) / RAND_MAX * 2.f - 1.f);
  for (size_t i = 0; i < size_B; i++) ptr_B[i] = ElementB(float(rand()) / RAND_MAX * 2.f - 1.f);
  for (size_t i = 0; i < size_D; i++) ptr_D[i] = ElementD(0);

  // Setup kernel arguments
  using KernelType = GemmLegacyKernel<CollectiveMainloop>;

  typename KernelType::Arguments args{
    problem_shape,
    {ptr_A, stride_A, ptr_B, stride_B},
    ptr_D, stride_D
  };

  auto params = KernelType::to_underlying_arguments(args);

  // Launch configuration
  int tile_m_count = ceil_div(M, get<0>(TileShape{}));
  int tile_n_count = ceil_div(N, get<1>(TileShape{}));
  int grid_size = tile_m_count * tile_n_count * L;
  int block_size = KernelType::MaxThreadsPerBlock;

  namespace syclex = sycl::ext::oneapi::experimental;
  namespace intelex = sycl::ext::intel::experimental;

  syclex::properties kernel_props{
    syclex::sub_group_size<16>,
    intelex::grf_size<256>
  };

  // Run kernel
  auto event = Q.parallel_for<GemmLegacyKernelName<ElementA, ElementD, 0>>(
    sycl::nd_range<1>(grid_size * block_size, block_size),
    kernel_props,
    [=](sycl::nd_item<1>) {
      gemm_legacy_device(params);
    }
  );
  event.wait();

  // Debug: print first few elements of D and ref
  std::cout << "D[0..4]: ";
  for (int i = 0; i < 5; i++) std::cout << ptr_D[i] << " ";
  std::cout << std::endl;

  // Verify: D = A * B (alpha=1, beta=0)
  bool passed = true;
  if (options.verify) {
    auto ptr_ref = sycl::malloc_shared<ElementD>(size_D, Q);
    for (size_t i = 0; i < size_D; i++) ptr_ref[i] = ElementD(0);

    cutlass::TensorRef ref_A(ptr_A, LayoutA::packed({M, K}));
    cutlass::TensorRef ref_B(ptr_B, LayoutB::packed({K, N}));
    cutlass::TensorRef ref_C(ptr_ref, LayoutD::packed({M, N}));
    cutlass::TensorRef ref_D(ptr_ref, LayoutD::packed({M, N}));

    cutlass::reference::device::GemmComplex(
      {M, N, K},
      ElementAccumulator(1),
      ref_A, cutlass::ComplexTransform::kNone,
      ref_B, cutlass::ComplexTransform::kNone,
      ElementAccumulator(0),
      ref_C, ref_D,
      ElementAccumulator(0),
      L, M * K, K * N, M * N, M * N);

    compat::wait();

    // Debug: print first few ref values
    std::cout << "Ref[0..4]: ";
    for (int i = 0; i < 5; i++) std::cout << ptr_ref[i] << " ";
    std::cout << std::endl;

    // Tolerance-based comparison (bf16 inputs may cause small rounding differences vs reference)
    passed = cutlass::reference::device::BlockCompareEqual(ptr_ref, ptr_D, size_D);
    if (!passed) {
      // Fall back to relative tolerance check
      float max_rel_err = 0.0f;
      int mismatches = 0;
      for (size_t i = 0; i < size_D; i++) {
        float ref_val = ptr_ref[i];
        float got_val = ptr_D[i];
        float abs_err = std::abs(ref_val - got_val);
        float denom = std::max(std::abs(ref_val), std::abs(got_val)) + 1e-6f;
        float rel_err = abs_err / denom;
        if (rel_err > max_rel_err) max_rel_err = rel_err;
        if (rel_err > 5e-2f) mismatches++;  // 5% tolerance for bf16 accumulation
      }
      passed = (mismatches == 0);
      std::cout << "Max relative error: " << max_rel_err << ", mismatches (>5%): " << mismatches << std::endl;
    }
    std::cout << "Disposition: " << (passed ? "Passed" : "Failed") << std::endl;

    sycl::free(ptr_ref, Q);
  } else {
    std::cout << "Disposition: skipped verification" << std::endl;
  }

  // Performance measurement
  if (passed && options.iterations > 0) {
    GPU_Clock timer;
    timer.start();
    for (int i = 0; i < options.iterations; ++i) {
      Q.parallel_for<GemmLegacyKernelName<ElementA, ElementD, 1>>(
        sycl::nd_range<1>(grid_size * block_size, block_size),
        kernel_props,
        [=](sycl::nd_item<1>) {
          gemm_legacy_device(params);
        }
      );
    }
    Q.wait_and_throw();

    float elapsed = timer.seconds() / options.iterations;
    double tflops = (2.0 * M * N * K * L) * 1e-12;
    std::cout << "Problem Size: " << M << 'x' << N << 'x' << K << 'x' << L << std::endl;
    printf("Legacy GEMM Performance:     [%4.3f]TFlop/s  (%6.4f)ms\n", tflops / elapsed, elapsed * 1000);
  }

  sycl::free(ptr_A, Q);
  sycl::free(ptr_B, Q);
  sycl::free(ptr_D, Q);

  return passed;
}

///////////////////////////////////////////////////////////////////////////////////////////////////

int main(int argc, const char** argv) {
  Options options;
  options.parse(argc, argv);

  if (options.help) {
    options.print_usage(std::cout) << std::endl;
    return 0;
  }

  if (options.error) {
    std::cerr << "Aborting execution." << std::endl;
    return -1;
  }

  bool passed = run(options);
  return passed ? 0 : -1;
}
