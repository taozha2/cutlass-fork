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

#include "cutlass/epilogue/collective/default_epilogue.hpp"
#include "cutlass/epilogue/collective/xe_epilogue.hpp"
#include "cutlass/epilogue/fusion/xe_callbacks.hpp"
#include "cutlass/gemm/device/gemm_universal.h"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/collective/collective_mma.hpp"
#include "cutlass/util/GPU_Clock.hpp"

#include <cute/tensor.hpp>
#include <random>

#include "cutlass/util/command_line.h"
#include "cutlass/util/device_memory.h"
#include "cutlass/util/packed_stride.hpp"
#include "cutlass/util/reference/device/gemm_complex.h"
#include "cutlass/util/reference/device/tensor_compare.h"
#include "common.hpp"

using namespace cute;

///////////////////////////////////////////////////////////////////////////////////////////////////
#define FLUSH_CACHE 1
// Command line options parsing
struct Options {

  bool help;
  bool error;

  int m, n, k, l, iterations;
  float alpha, beta;

  Options():
    help(false),
    error(false),
    m(5120), n(4096), k(4096), l(1), iterations(20),
    alpha(1.f), beta(0.f)
  { }

  // Parses the command line
  void parse(int argc, char const **args) {
    cutlass::CommandLine cmd(argc, args);

    if (cmd.check_cmd_line_flag("help")) {
      help = true;
      return;
    }

    cmd.get_cmd_line_argument("m", m, 5120);
    cmd.get_cmd_line_argument("n", n, 4096);
    cmd.get_cmd_line_argument("k", k, 4096);
    cmd.get_cmd_line_argument("l", l, 1);
    cmd.get_cmd_line_argument("alpha", alpha, 1.f);
    cmd.get_cmd_line_argument("beta", beta, 0.f);
    cmd.get_cmd_line_argument("iterations", iterations, 100);
  }

  /// Prints the usage statement.
  std::ostream & print_usage(std::ostream &out) const {

    out << "PVC GEMM Example\n\n"
      << "Options:\n\n"
      << "  --help                      If specified, displays this usage statement\n\n"
      << "  --m=<int>                   Sets the M extent of the GEMM\n"
      << "  --n=<int>                   Sets the N extent of the GEMM\n"
      << "  --k=<int>                   Sets the K extent of the GEMM\n"
      << "  --l=<int>                   Sets the L extent (batch count) of the GEMM\n"
      << "  --alpha=<s32>               Epilogue scalar alpha\n"
      << "  --beta=<s32>                Epilogue scalar beta\n\n"
      << "  --iterations=<int>          Iterations\n\n";

    return out;
  }
};

///////////////////////////////////////////////////////////////////////////////////////////////////

template <
  class Gemm
>
struct ExampleRunner {

  using StrideA = typename Gemm::GemmKernel::StrideA;
  using StrideB = typename Gemm::GemmKernel::StrideB;
  using StrideC = typename Gemm::GemmKernel::StrideC;
  using StrideD = typename Gemm::GemmKernel::StrideD;

  using LayoutA = typename Gemm::LayoutA;
  using LayoutB = typename Gemm::LayoutB;
  using LayoutC = typename Gemm::LayoutC;
  using LayoutD = typename Gemm::LayoutD;

  using ElementA = typename Gemm::ElementA;
  using ElementB = typename Gemm::ElementB;
  using ElementAcc = typename Gemm::ElementAccumulator;

  using CollectiveEpilogue = typename Gemm::CollectiveEpilogue;
  using ElementC = typename Gemm::ElementC;
  using ElementOutput = typename CollectiveEpilogue::ElementOutput;
  using ElementCompute = typename CollectiveEpilogue::ElementCompute;
  using ElementAccumulator = typename CollectiveEpilogue::ElementAccumulator;

  using ProblemShapeType = typename Gemm::GemmKernel::ProblemShape;

  //
  // Data members
  //

  /// Initialization
  StrideA stride_A;
  StrideB stride_B;
  StrideC stride_C;
  StrideD stride_D;
  uint64_t seed = 0;

  cutlass::DeviceAllocation<ElementA> block_A;
  cutlass::DeviceAllocation<ElementB> block_B;
  cutlass::DeviceAllocation<ElementC> block_C;

#if FLUSH_CACHE == 1
  cutlass::DeviceAllocation<ElementA> block_A_pingpong;
  cutlass::DeviceAllocation<ElementB> block_B_pingpong;
#endif

  cutlass::DeviceAllocation<ElementOutput> block_D;
  cutlass::DeviceAllocation<ElementOutput> block_ref_D;

  //
  // Methods
  //

  bool verify(const ProblemShapeType& problem_size, ElementCompute alpha, ElementCompute beta) {
    auto [M, N, K, L] = problem_size;

    cutlass::TensorRef ref_A(block_A.get(), LayoutA::packed({M, K}));
    cutlass::TensorRef ref_B(block_B.get(), LayoutB::packed({K, N}));
    cutlass::TensorRef ref_C(block_C.get(), LayoutC::packed({M, N}));
    cutlass::TensorRef ref_D(block_ref_D.get(), LayoutD::packed({M, N}));

    cutlass::reference::device::GemmComplex(
          {M, N, K},
          alpha,
          ref_A,
          cutlass::ComplexTransform::kNone,
          ref_B,
          cutlass::ComplexTransform::kNone,
          beta,
          ref_C,
          ref_D,
          ElementAccumulator(0),
          L,     // batch_count
          M * K, // batch_stride_A
          K * N, // batch_stride_B
          M * N, // batch_stride_C
          M * N  // batch_stride_D
        );

    syclcompat::wait();

    // Check if output from CUTLASS kernel and reference kernel are equal or not
    bool passed = cutlass::reference::device::BlockCompareEqual(
      block_ref_D.get(), block_D.get(), block_D.size());

    return passed;
  }

  /// Initialize operands to be used in the GEMM and reference GEMM
  void initialize(const ProblemShapeType& problem_size) {
    auto problem_shape_MNKL = cute::append<4>(problem_size, 1);
    auto [M, N, K, L] = problem_shape_MNKL;

    stride_A = cutlass::make_cute_packed_stride(StrideA{}, cute::make_shape(M, K, L));
    stride_B = cutlass::make_cute_packed_stride(StrideB{}, cute::make_shape(N, K, L));
    stride_C = cutlass::make_cute_packed_stride(StrideC{}, cute::make_shape(M, N, L));
    stride_D = cutlass::make_cute_packed_stride(StrideD{}, cute::make_shape(M, N, L));

    block_A.reset(M * K * L);
    block_B.reset(K * N * L);
    block_C.reset(M * N * L);
  #if FLUSH_CACHE == 1
    block_A_pingpong.reset(M * K * L);
    block_B_pingpong.reset(K * N * L);
  #endif
    block_D.reset(M * N * L);
    block_ref_D.reset(M * N * L);

    initialize_block(block_A, seed + 2023);
    initialize_block(block_B, seed + 2022);
    initialize_block(block_C, seed + 2021);

  #if FLUSH_CACHE == 1
    initialize_block(block_A_pingpong, seed + 2024);
    initialize_block(block_B_pingpong, seed + 2025);
  #endif
  }

  void run(const Options& options, const cutlass::KernelHardwareInfo& hw_info) {
    ProblemShapeType problem_size = ProblemShapeType{options.m, options.n, options.k, options.l};

    initialize(problem_size);

    typename Gemm::GemmKernel::Arguments arguments{
      cutlass::gemm::GemmUniversalMode::kGemm,
      problem_size,
      {block_A.get(), stride_A, block_B.get(), stride_B},
      {{1, 0}, block_C.get(), stride_C, block_D.get(), stride_D},
      hw_info
    };

    Gemm gemm_op;

    size_t workspace_size = Gemm::get_workspace_size(arguments);
    cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);

    if (gemm_op.can_implement(arguments) != cutlass::Status::kSuccess){
      std::cout << "Invalid Problem Size: " << options.m << 'x' << options.n << 'x' << options.k << 'x' << options.l << std::endl;
      std::exit(1);
    }

    gemm_op.initialize(arguments, workspace.get());

    // Run the GEMM
    gemm_op.run();

    syclcompat::wait();

    // Verify that the result is correct
    bool passed = verify(problem_size, 1, 0);
    std::cout << "Disposition: " << (passed ? "Passed" : "Failed") << std::endl;

    int warm_up = 10;
    std::vector<float> event_times(warm_up + options.iterations);

    if (passed && options.iterations > 0) {
      GPU_Clock timer;

      for (int i = 0; i < warm_up + options.iterations; ++i) {

      #if FLUSH_CACHE == 1
        typename Gemm::GemmKernel::Arguments arguments_ping{
          cutlass::gemm::GemmUniversalMode::kGemm,
          problem_size,
          {block_A.get(), stride_A, block_B.get(), stride_B},
          {{1, 0}, block_C.get(), stride_C, block_D.get(), stride_D},
          hw_info
       };
        typename Gemm::GemmKernel::Arguments arguments_pong{
          cutlass::gemm::GemmUniversalMode::kGemm,
          problem_size,
          {block_A_pingpong.get(), stride_A, block_B_pingpong.get(), stride_B},
          {{1, 0}, block_C.get(), stride_C, block_D.get(), stride_D},
          hw_info
        };
          if (i % 2) {gemm_op.initialize(arguments_ping, workspace.get());}
          else {gemm_op.initialize(arguments_pong, workspace.get());}
        #endif

        #if CACHE_FLUSH == 2
        auto l3_cache_size = 256 * 1024 * 1024;
        auto ref_d_element = max(l3_cache_size / sizeof(float), options.m * options.n * options.l);
        block_ref_D.reset(ref_d_element);
        syclcompat::memset(block_ref_D.get(), 0,
                          ref_d_element * sizeof(float));
        #endif

      timer.start();
      gemm_op.run();
      syclcompat::wait();
//      timer.stop();
      event_times[i] = timer.milliseconds() * float(1e-3);
    }
      //float cute_time = timer.seconds() / options.iterations;
      auto best = 999.f;
      auto worst = 0.f;
      auto average = 0.f;

      auto best_iter = 0;
      auto worst_iter = 0;

      for(uint32_t i = warm_up; i < options.iterations + warm_up; i++) {
        average += event_times[i];
        best = min(best, event_times[i]);
        worst = max(worst, event_times[i]);
        if (best == event_times[i]) { best_iter = i; }
        if (worst == event_times[i]) { worst_iter = i; }
      }
      average = average - best - worst;
      average /= (options.iterations - 2);
      float cute_time =event_times[10];

      double tflops = (2.0 * options.m * options.n * options.k * options.l) * 1e-12;
      double io = options.l *(options.m * options.k * sizeof(bfloat16_t) + options.n * options.k * sizeof(bfloat16_t) + options.m * options.n * sizeof(float)) *1e-9;

      std::cout << "Problem Size: " << options.m << 'x' << options.n << 'x' << options.k << 'x' << options.l << std::endl;
      printf("Cutlass GEMM,\n        Time:    (%6.4f)ms,\n        IO(TFlops):    [%4.3f]TFlop/s,\n        HBM(GBs):    [%f]GB/s\n",
              average, tflops / average, io / average );
    }
    return;
  }
};

template <class GmemTiledCopyA, class GmemTiledCopyB, class MmaTraits,
          class ElementInputA, class ElementInputB, class ElementOutput,
          class LayoutA, class LayoutB>
int run_gemm(int argc, const char** argv)
{
  //
  // Parse options
  //

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

  //
  // Run examples
  //

  // The KernelHardwareInfo struct holds the number of EUs on the GPU with a given device ID. This
  // information is used by the underlying kernel.
  cutlass::KernelHardwareInfo hw_info;

  // Change device_id to another value if you are running on a machine with multiple GPUs and wish
  // to use a GPU other than that with device ID 0.
  hw_info.sm_count = cutlass::KernelHardwareInfo::query_device_multiprocessor_count(hw_info.device_id);

  bool passed;

  using ElementAccumulator = ElementOutput;                   // <- data type of accumulator
  using ElementComputeEpilogue = ElementOutput;               // <- data type of epilogue operations

  using LayoutC = cutlass::layout::RowMajor;
  using LayoutD = cutlass::layout::RowMajor;

  // Workgroup-level tile
  using TileShape = Shape<_256, _256, _32>;

  using TiledMma = TiledMMA<MMA_Atom<MmaTraits>,
          Layout<Shape<_8,_4,_1>>,
          Tile<_64,_64,_32>>; // Subgroup level-tile

  constexpr int PipelineStages = 3;
  using GEMMDispatchPolicy = cutlass::gemm::MainloopIntelPVC<PipelineStages>;
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
          XE_2D_U32x8x16_LD_N,
          void, void,
          XE_2D_U32x8x16_ST_N,
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

  ExampleRunner<Gemm> runner;

  runner.run(options, hw_info);

  return 0;
}

int main(int argc, const char** argv)
{
  // ================  bfloat16  ================
  std::cout << "\n\n==========  bf16, RowMajor, RowMajor  ==========" << std::endl;
  run_gemm<XE_2D_U16x32x32_LD_N, XE_2D_U16x32x32_LD_N, XE_8x16x16_F32BF16BF16F32_TT,
           bfloat16_t, bfloat16_t, float, cutlass::layout::RowMajor, cutlass::layout::RowMajor>(argc, argv);

  std::cout << "\n\n==========  bf16, RowMajor, ColumnMajor  ==========" << std::endl;
  run_gemm<XE_2D_U16x32x32_LD_N, XE_2D_U16x16x16_LD_T, XE_8x16x16_F32BF16BF16F32_TT,
           bfloat16_t, bfloat16_t, float, cutlass::layout::RowMajor, cutlass::layout::ColumnMajor>(argc, argv);

  std::cout << "\n\n==========  bf16, ColumnMajor, RowMajor  ==========" << std::endl;
  run_gemm<XE_2D_U16x16x16_LD_T, XE_2D_U16x32x32_LD_N, XE_8x16x16_F32BF16BF16F32_TT,
           bfloat16_t, bfloat16_t, float, cutlass::layout::ColumnMajor, cutlass::layout::RowMajor>(argc, argv);

  std::cout << "\n\n==========  bf16, ColumnMajor, ColumnMajor  ==========" << std::endl;
  run_gemm<XE_2D_U16x16x16_LD_T, XE_2D_U16x16x16_LD_T, XE_8x16x16_F32BF16BF16F32_TT,
           bfloat16_t, bfloat16_t, float, cutlass::layout::ColumnMajor, cutlass::layout::ColumnMajor>(argc, argv);



  // ================  int8  ================
  std::cout << "\n\n==========  int8, RowMajor, RowMajor  ==========" << std::endl;
  run_gemm<XE_2D_U8x32x32_LD_N, XE_2D_U8x32x32_LD_V, XE_8x16x32_S32S8S8S32_TT,
           int8_t, int8_t, int32_t, cutlass::layout::RowMajor, cutlass::layout::RowMajor>(argc, argv);

  std::cout << "\n\n==========  int8, RowMajor, ColumnMajor  ==========" << std::endl;
  run_gemm<XE_2D_U8x32x32_LD_N, XE_2D_U8x32x16_LD_T, XE_8x16x32_S32S8S8S32_TT,
           int8_t, int8_t, int32_t, cutlass::layout::RowMajor, cutlass::layout::ColumnMajor>(argc, argv);

  // std::cout << "\n\n==========  int8, ColumnMajor, RowMajor  ==========" << std::endl;
  // run_gemm<XE_2D_U8x32x16_LD_T, XE_2D_U8x32x32_LD_V, XE_8x16x32_S32S8S8S32_TT,
  //          int8_t, int8_t, int32_t, cutlass::layout::ColumnMajor, cutlass::layout::RowMajor>(argc, argv);

  // std::cout << "\n\n==========  int8, ColumnMajor, ColumnMajor  ==========" << std::endl;
  // run_gemm<XE_2D_U8x32x16_LD_T, XE_2D_U8x32x16_LD_T, XE_8x16x32_S32S8S8S32_TT,
  //          int8_t, int8_t, int32_t, cutlass::layout::ColumnMajor, cutlass::layout::ColumnMajor>(argc, argv);



  // ================  uint8  ================
  std::cout << "\n\n==========  uint8, RowMajor, RowMajor  ==========" << std::endl;
  run_gemm<XE_2D_U8x32x32_LD_N, XE_2D_U8x32x32_LD_V, XE_8x16x32_S32U8U8S32_TT,
           uint8_t, uint8_t, int32_t, cutlass::layout::RowMajor, cutlass::layout::RowMajor>(argc, argv);

  std::cout << "\n\n==========  uint8, RowMajor, ColumnMajor  ==========" << std::endl;
  run_gemm<XE_2D_U8x32x32_LD_N, XE_2D_U8x32x16_LD_T, XE_8x16x32_S32S8S8S32_TT,
           uint8_t, uint8_t, int32_t, cutlass::layout::RowMajor, cutlass::layout::ColumnMajor>(argc, argv);

  // std::cout << "\n\n==========  uint8, ColumnMajor, RowMajor  ==========" << std::endl;
  // run_gemm<XE_2D_U8x32x16_LD_T, XE_2D_U8x32x32_LD_V, XE_8x16x32_S32S8S8S32_TT,
  //          uint8_t, uint8_t, int32_t, cutlass::layout::ColumnMajor, cutlass::layout::RowMajor>(argc, argv);

  // std::cout << "\n\n==========  uint8, ColumnMajor, ColumnMajor  ==========" << std::endl;
  // run_gemm<XE_2D_U8x32x16_LD_T, XE_2D_U8x32x16_LD_T, XE_8x16x32_S32S8S8S32_TT,
  //          uint8_t, uint8_t, int32_t, cutlass::layout::ColumnMajor, cutlass::layout::ColumnMajor>(argc, argv);



  // ================  fp16  ================
  std::cout << "\n\n==========  fp16, RowMajor, RowMajor  ==========" << std::endl;
  run_gemm<XE_2D_U16x32x32_LD_N, XE_2D_U16x32x32_LD_N, XE_8x16x16_F32F16F16F32_TT,
           half_t, half_t, float, cutlass::layout::RowMajor, cutlass::layout::RowMajor>(argc, argv);

  std::cout << "\n\n==========  fp16, RowMajor, ColumnMajor  ==========" << std::endl;
  run_gemm<XE_2D_U16x32x32_LD_N, XE_2D_U16x16x16_LD_T, XE_8x16x16_F32F16F16F32_TT,
           half_t, half_t, float, cutlass::layout::RowMajor, cutlass::layout::ColumnMajor>(argc, argv);

  std::cout << "\n\n==========  fp16, ColumnMajor, RowMajor  ==========" << std::endl;
  run_gemm<XE_2D_U16x16x16_LD_T, XE_2D_U16x32x32_LD_N, XE_8x16x16_F32F16F16F32_TT,
           half_t, half_t, float, cutlass::layout::ColumnMajor, cutlass::layout::RowMajor>(argc, argv);

  std::cout << "\n\n==========  fp16, ColumnMajor, ColumnMajor  ==========" << std::endl;
  run_gemm<XE_2D_U16x16x16_LD_T, XE_2D_U16x16x16_LD_T, XE_8x16x16_F32F16F16F32_TT,
           half_t, half_t, float, cutlass::layout::ColumnMajor, cutlass::layout::ColumnMajor>(argc, argv);

  // fp32
  // std::cout << "\n\n==========  fp32, RowMajor, RowMajor  ==========" << std::endl;
  // run_gemm<XE_2D_U32x32x16_LD_N, XE_2D_U32x32x16_LD_N, XE_8x16x8_F32TF32TF32F32_TT,
  //          float, float, float, cutlass::layout::RowMajor, cutlass::layout::RowMajor>(argc, argv);

  // std::cout << "\n\n==========  tf32, RowMajor, ColumnMajor  ==========" << std::endl;
  // run_gemm<XE_2D_U32x32x16_LD_N, XE_2D_U32x16x8_LD_T, XE_8x16x8_F32TF32TF32F32_TT,
  //          float, float, float, cutlass::layout::RowMajor, cutlass::layout::ColumnMajor>(argc, argv);

  // std::cout << "\n\n==========  fp32, ColumnMajor, RowMajor  ==========" << std::endl;
  // run_gemm<XE_2D_U32x16x8_LD_T, XE_2D_U32x32x16_LD_N, XE_8x16x8_F32TF32TF32F32_TT,
  //          float, float, float, cutlass::layout::RowMajor, cutlass::layout::RowMajor>(argc, argv);

  // std::cout << "\n\n==========  tf32, ColumnMajor, ColumnMajor  ==========" << std::endl;
  // run_gemm<XE_2D_U32x16x8_LD_T, XE_2D_U32x16x8_LD_T, XE_8x16x8_F32TF32TF32F32_TT,
  //          float, float, float, cutlass::layout::RowMajor, cutlass::layout::ColumnMajor>(argc, argv);

  // tf32
  // std::cout << "\n\n==========  tf32, RowMajor, RowMajor  ==========" << std::endl;
  // run_gemm<XE_2D_TF32x32x16_LD_N, XE_2D_U32x32x16_LD_N, XE_8x16x8_F32TF32TF32F32_TT,
  //          tfloat32_t, tfloat32_t, float, cutlass::layout::RowMajor, cutlass::layout::RowMajor>(argc, argv);

  // std::cout << "\n\n==========  tf32, RowMajor, ColumnMajor  ==========" << std::endl;
  // run_gemm<XE_2D_TF32x32x16_LD_N, XE_2D_U32x16x8_LD_T, XE_8x16x8_F32TF32TF32F32_TT,
  //          tfloat32_t, tfloat32_t, float, cutlass::layout::RowMajor, cutlass::layout::ColumnMajor>(argc, argv);
}