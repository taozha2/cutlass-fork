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

#include "cutlass/util/GPU_Clock.hpp"
#include "cutlass/util/print_error.hpp"

#include <cute/tensor.hpp>
#include <sycl/sycl.hpp>
#include <syclcompat.hpp>

template <class ProblemShape, class CtaTiler, class TA, class AStride, class ASmemLayout,
          class TiledCopyA, class TB, class BStride, class BSmemLayout, class TiledCopyB, class TC,
          class CStride, class CSmemLayout, class TiledMma, class Alpha, class Beta>
void gemm_device(ProblemShape shape_MNK, CtaTiler cta_tiler, TA const* A, AStride dA,
                 ASmemLayout sA_layout, TiledCopyA copy_a, TB const* B, BStride dB,
                 BSmemLayout sB_layout, TiledCopyB copy_b, TC* C, CStride dC, CSmemLayout,
                 TiledMma mma, Alpha alpha, Beta beta) {
  using namespace cute;

  // Preconditions
  CUTE_STATIC_ASSERT_V(rank(shape_MNK) == Int<3>{});  // (M, N, K)
  CUTE_STATIC_ASSERT_V(rank(cta_tiler) == Int<3>{});  // (BLK_M, BLK_N, BLK_K)

  CUTE_STATIC_ASSERT_V(size(copy_a) == size(mma));  // NumThreads
  CUTE_STATIC_ASSERT_V(size(copy_b) == size(mma));  // NumThreads

  static_assert(is_static<ASmemLayout>::value);
  static_assert(is_static<BSmemLayout>::value);
  static_assert(is_static<CSmemLayout>::value);

  CUTE_STATIC_ASSERT_V(size<0>(ASmemLayout{}) == size<0>(cta_tiler));  // BLK_M
  CUTE_STATIC_ASSERT_V(size<1>(CSmemLayout{}) == size<0>(cta_tiler));  // BLK_M
  CUTE_STATIC_ASSERT_V(size<0>(BSmemLayout{}) == size<1>(cta_tiler));  // BLK_N
  CUTE_STATIC_ASSERT_V(size<1>(CSmemLayout{}) == size<1>(cta_tiler));  // BLK_N
  CUTE_STATIC_ASSERT_V(size<1>(ASmemLayout{}) == size<2>(cta_tiler));  // BLK_K
  CUTE_STATIC_ASSERT_V(size<1>(BSmemLayout{}) == size<2>(cta_tiler));  // BLK_K

  CUTE_STATIC_ASSERT_V(congruent(select<0, 2>(shape_MNK), dA));  // dA strides for shape MK
  CUTE_STATIC_ASSERT_V(congruent(select<1, 2>(shape_MNK), dB));  // dB strides for shape NK
  CUTE_STATIC_ASSERT_V(congruent(select<0, 1>(shape_MNK), dC));  // dC strides for shape MN

  //
  // Full and Tiled Tensors
  //

  // Represent the full tensors
  Tensor mA = make_tensor(make_gmem_ptr(A), select<0, 2>(shape_MNK), dA);  // (M,K)
  Tensor mB = make_tensor(make_gmem_ptr(B), select<1, 2>(shape_MNK), dB);  // (N,K)
  Tensor mC = make_tensor(make_gmem_ptr(C), select<0, 1>(shape_MNK), dC);  // (M,N)

  // Get the appropriate blocks for this thread block
  auto cta_coord =
      make_coord(syclcompat::work_group_id::x(), syclcompat::work_group_id::y(), _);  // (m,n,k)
  Tensor gA = local_tile(mA, cta_tiler, cta_coord, Step<_1, X, _1>{});  // (BLK_M,BLK_K,k)
  Tensor gB = local_tile(mB, cta_tiler, cta_coord, Step<X, _1, _1>{});  // (BLK_N,BLK_K,k)
  Tensor gC = local_tile(mC, cta_tiler, cta_coord, Step<_1, _1, X>{});  // (BLK_M,BLK_N)

  // Shared memory buffers
  auto smemA = syclcompat::local_mem<TA[cosize_v<ASmemLayout>]>();
  auto smemB = syclcompat::local_mem<TB[cosize_v<BSmemLayout>]>();
  Tensor sA = make_tensor(make_smem_ptr(smemA), sA_layout);  // (BLK_M,BLK_K)
  Tensor sB = make_tensor(make_smem_ptr(smemB), sB_layout);  // (BLK_N,BLK_K)

  //
  // Partition the copying of A and B tiles across the threads
  //

  // TUTORIAL: Example of partitioning via a TiledCopy

  ThrCopy thr_copy_a = copy_a.get_slice(syclcompat::local_id::x());
  Tensor tAgA = thr_copy_a.partition_S(gA);  // (CPY,CPY_M,CPY_K,k)
  Tensor tAsA = thr_copy_a.partition_D(sA);  // (CPY,CPY_M,CPY_K)
  Tensor tArA = make_fragment_like(tAsA);    // (CPY,CPY_M,CPY_K)

  ThrCopy thr_copy_b = copy_b.get_slice(syclcompat::local_id::x());
  Tensor tBgB = thr_copy_b.partition_S(gB);  // (CPY,CPY_N,CPY_K,k)
  Tensor tBsB = thr_copy_b.partition_D(sB);  // (CPY,CPY_N,CPY_K)
  Tensor tBrB = make_fragment_like(tBsB);    // (CPY,CPY_N,CPY_K)

  CUTE_STATIC_ASSERT_V(size<1>(tAgA) == size<1>(tAsA));  // CPY_M
  CUTE_STATIC_ASSERT_V(size<1>(tAgA) == size<1>(tArA));  // CPY_M
  CUTE_STATIC_ASSERT_V(size<2>(tAgA) == size<2>(tAsA));  // CPY_K
  CUTE_STATIC_ASSERT_V(size<2>(tAgA) == size<2>(tArA));  // CPY_K
  CUTE_STATIC_ASSERT_V(size<1>(tBgB) == size<1>(tBsB));  // CPY_N
  CUTE_STATIC_ASSERT_V(size<1>(tBgB) == size<1>(tBrB));  // CPY_N
  CUTE_STATIC_ASSERT_V(size<2>(tBgB) == size<2>(tBsB));  // CPY_K
  CUTE_STATIC_ASSERT_V(size<2>(tBgB) == size<2>(tBrB));  // CPY_K

  // Copy gmem to rmem for k_tile=0
  copy(copy_a, tAgA(_, _, _, 0), tArA);
  copy(copy_b, tBgB(_, _, _, 0), tBrB);
  //
  // Define A/B partitioning and C accumulators
  //

  // TUTORIAL: Example of partitioning via a TiledMMA

  ThrMMA thr_mma = mma.get_slice(syclcompat::local_id::x());
  Tensor tCsA = thr_mma.partition_A(sA);  // (MMA,MMA_M,MMA_K)
  Tensor tCsB = thr_mma.partition_B(sB);  // (MMA,MMA_N,MMA_K)
  Tensor tCgC = thr_mma.partition_C(gC);  // (MMA,MMA_M,MMA_N)

  // Allocate registers for pipelining
  Tensor tCrA = thr_mma.make_fragment_A(tCsA);  // (MMA,MMA_M,MMA_K)
  Tensor tCrB = thr_mma.make_fragment_B(tCsB);  // (MMA,MMA_N,MMA_K)
  // Allocate the accumulators -- same size as the projected data
  Tensor tCrC = thr_mma.make_fragment_C(tCgC);  // (MMA,MMA_M,MMA_N)

  CUTE_STATIC_ASSERT_V(shape(tCrA) == shape(tCsA));      // (MMA,MMA_M,MMA_K)
  CUTE_STATIC_ASSERT_V(shape(tCrB) == shape(tCsB));      // (MMA,MMA_N,MMA_K)
  CUTE_STATIC_ASSERT_V(shape(tCrC) == shape(tCgC));      // (MMA,MMA_M,MMA_N)
  CUTE_STATIC_ASSERT_V(size<1>(tCgC) == size<1>(tCsA));  // MMA_M
  CUTE_STATIC_ASSERT_V(size<2>(tCgC) == size<1>(tCsB));  // MMA_N
  CUTE_STATIC_ASSERT_V(size<2>(tCsA) == size<2>(tCsB));  // MMA_K

  // Clear the accumulators
  clear(tCrC);

#if CUTLASS_ENABLE_DEBUG_PRINTS
  if(thread0()) {
    print("  mA : "); print(  mA); print("\n");
    print("  gA : "); print(  gA); print("\n");
    print("  sA : "); print(  sA); print("\n");
    print("tAgA : "); print(tAgA); print("\n");
    print("tAsA : "); print(tAsA); print("\n");
    print("tArA : "); print(tArA); print("\n");
  }
#endif

#if CUTLASS_ENABLE_DEBUG_PRINTS
  if(thread0()) {
    print("  mB : "); print(  mB); print("\n");
    print("  gB : "); print(  gB); print("\n");
    print("  sB : "); print(  sB); print("\n");
    print("tBgB : "); print(tBgB); print("\n");
    print("tBsB : "); print(tBsB); print("\n");
    print("tArA : "); print(tArA); print("\n");
  }
#endif

#if CUTLASS_ENABLE_DEBUG_PRINTS
  if(thread0()) {
    print("  mC : "); print(  mC); print("\n");
    print("  gC : "); print(  gC); print("\n");
    print("tCsA : "); print(tCsA); print("\n");
    print("tCsB : "); print(tCsB); print("\n");
    print("tCgC : "); print(tCgC); print("\n");
    print("tCrC : "); print(tCrC); print("\n");
  }
#endif

  // Copy rmem to smem
  copy(tArA, tAsA);
  copy(tBrB, tBsB);
  syclcompat::wg_barrier();

  //
  // PIPELINED MAIN LOOP
  // TUTORIAL: Example of a gemm loop that pipelines shared memory AND register memory
  //   Data is read from global to registers, then to shared via the tA|tB partitions
  //   Data is then copied from shared to registers in multiple waves via the tC partitions
  //     and gemm(.) operates on the current register wave
  //

  // Load A, B shmem->regs for k_block=0
  copy(tCsA(_, _, 0), tCrA(_, _, 0));
  copy(tCsB(_, _, 0), tCrB(_, _, 0));
  auto K_TILE_MAX = size<3>(tAgA);
  auto K_BLOCK_MAX = size<2>(tCrA);

  CUTE_NO_UNROLL
  for (int k_tile = 0; k_tile < K_TILE_MAX; ++k_tile) {
    // Pipeline the k-mode of the block registers
    CUTE_UNROLL
    for (int k_block = 0; k_block < K_BLOCK_MAX; ++k_block) {
      if (k_block == K_BLOCK_MAX - 1) {
        // Copy rmem to smem
        syclcompat::wg_barrier();
        copy(tArA, tAsA);
        copy(tBrB, tBsB);
        syclcompat::wg_barrier();
      }

      // Copy smem to rmem for k_block+1
      int k_block_next = (k_block + 1) % K_BLOCK_MAX;
      copy(tCsA(_, _, k_block_next), tCrA(_, _, k_block_next));
      copy(tCsB(_, _, k_block_next), tCrB(_, _, k_block_next));
      if (k_block == 0) {
        // Copy gmem to rmem for k_tile+1
        int k_tile_next = (k_tile + 1 < K_TILE_MAX) ? k_tile + 1 : k_tile;
        copy(copy_a, tAgA(_, _, _, k_tile_next), tArA);
        copy(copy_b, tBgB(_, _, _, k_tile_next), tBrB);
      }
      // Thread-level register gemm for k_block
      gemm(mma, tCrA(_, _, k_block), tCrB(_, _, k_block), tCrC);
    }  // k_block
  }    // k_tile

  //
  // Epilogue
  //

  axpby(alpha, tCrC, beta, tCgC);
}

template <class TA, class TB, class TC, class Alpha, class Beta>
void gemm_nt(int m, int n, int k, Alpha alpha, TA const* A, int ldA, TB const* B, int ldB,
             Beta beta, TC* C, int ldC) {
  using namespace cute;

  // Define shapes (dynamic)
  auto M = int(m);
  auto N = int(n);
  auto K = int(k);
  auto prob_shape = make_shape(M, N, K);  // (M, N, K)

  // Define NT strides (mixed)
  auto dA = make_stride(Int<1>{}, ldA);  // (dM, dK)
  auto dB = make_stride(Int<1>{}, ldB);  // (dN, dK)
  auto dC = make_stride(Int<1>{}, ldC);  // (dM, dN)

  // Define CTA tile sizes (static)
  auto bM = Int<128>{};
  auto bN = Int<128>{};
  auto bK = Int<8>{};
  auto cta_tiler = make_shape(bM, bN, bK);  // (BLK_M, BLK_N, BLK_K)

  // Define the smem layouts (static)
  auto sA = make_layout(make_shape(bM, bK));  // (m,k) -> smem_idx; m-major
  auto sB = make_layout(make_shape(bN, bK));  // (n,k) -> smem_idx; n-major
  auto sC = make_layout(make_shape(bM, bN));  // (m,n) -> smem_idx; m-major

  // Define the thread layouts (static)
  TiledCopy copyA = make_tiled_copy(Copy_Atom<UniversalCopy<uint128_t>, TA>{},
                                    Layout<Shape<_32, _8>>{},  // Thr layout 32x8 m-major
                                    Layout<Shape<_4, _1>>{});  // Val layout  4x1 m-major
  TiledCopy copyB = make_tiled_copy(Copy_Atom<UniversalCopy<uint128_t>, TB>{},
                                    Layout<Shape<_32, _8>>{},  // Thr layout 32x8 n-major
                                    Layout<Shape<_4, _1>>{});  // Val layout  4x1 n-major

  TiledMMA mmaC = make_tiled_mma(UniversalFMA<TC, TA, TB>{},
                                 Layout<Shape<_16, _16, _1>>{});  // 16x16x1 TiledMMA

#if CUTLASS_ENABLE_DEBUG_PRINTS
  print(copyA);
  print(copyB);
  print(mmaC);
#endif

#if CUTLASS_ENABLE_DEBUG_PRINTS
  print_latex(copyA);
  print_latex(copyB);
  print_latex(mmaC);
#endif

  auto dimBlock = syclcompat::dim3(size(mmaC));
  auto dimGrid = syclcompat::dim3(size(ceil_div(M, bM)), size(ceil_div(N, bN)));
  syclcompat::launch<
      gemm_device<decltype(prob_shape), decltype(cta_tiler), TA, decltype(dA), decltype(sA),
                  decltype(copyA), TB, decltype(dB), decltype(sB), decltype(copyB), TC,
                  decltype(dC), decltype(sC), decltype(mmaC), Alpha, Beta>>(
      dimGrid, dimBlock, prob_shape, cta_tiler, A, dA, sA, copyA, B, dB, sB, copyB, C, dC,
      sC, mmaC, alpha, beta);
}

// Setup params for a TN GEMM
template <class TA, class TB, class TC, class Alpha, class Beta>
void gemm_tn(int m, int n, int k, Alpha alpha, TA const* A, int ldA, TB const* B, int ldB,
             Beta beta, TC* C, int ldC) {
  using namespace cute;

  // Define shapes (dynamic)
  auto M = int(m);
  auto N = int(n);
  auto K = int(k);
  auto prob_shape = make_shape(M, N, K);  // (M, N, K)

  // Define TN strides (mixed)
  auto dA = make_stride(ldA, Int<1>{});  // (dM, dK)
  auto dB = make_stride(ldB, Int<1>{});  // (dN, dK)
  auto dC = make_stride(Int<1>{}, ldC);  // (dM, dN)

  // Define CTA tile sizes (static)
  auto bM = Int<128>{};
  auto bN = Int<128>{};
  auto bK = Int<8>{};
  auto cta_tiler = make_shape(bM, bN, bK);  // (BLK_M, BLK_N, BLK_K)

  // Define the smem layouts (static)
  auto sA = make_layout(make_shape(bM, bK),
                        make_stride(Int<1>{}, bM + Int<1>{}));  // (m,k) -> smem_idx; padded m-major
  auto sB = make_layout(make_shape(bN, bK),
                        make_stride(Int<1>{}, bN + Int<1>{}));  // (n,k) -> smem_idx; padded n-major
  auto sC = make_layout(make_shape(bM, bN));                    // (m,n) -> smem_idx

  // Define the thread layouts (static)

  TiledCopy copyA =
      make_tiled_copy(Copy_Atom<UniversalCopy<TA>, TA>{},
                      Layout<Shape<_32, _8>, Stride<_8, _1>>{},  // Thr layout 32x8 k-major
                      Layout<Shape<_1, _1>>{});                  // Val layout  1x1
  TiledCopy copyB =
      make_tiled_copy(Copy_Atom<UniversalCopy<TB>, TB>{},
                      Layout<Shape<_32, _8>, Stride<_8, _1>>{},  // Thr layout 32x8 k-major
                      Layout<Shape<_1, _1>>{});                  // Val layout  1x1

  TiledMMA mmaC = make_tiled_mma(UniversalFMA<TC, TA, TB>{},
                                 Layout<Shape<_16, _16, _1>>{});  // 16x16x1 TiledMMA

#if CUTLASS_ENABLE_DEBUG_PRINTS
  print(copyA);
  print(copyB);
  print(mmaC);
#endif

#if CUTLASS_ENABLE_DEBUG_PRINTS
  print_latex(copyA);
  print_latex(copyB);
  print_latex(mmaC);
#endif

  auto dimBlock = syclcompat::dim3(size(mmaC));
  auto dimGrid = syclcompat::dim3(size(ceil_div(M, bM)), size(ceil_div(N, bN)));
  syclcompat::launch<
      gemm_device<decltype(prob_shape), decltype(cta_tiler), TA, decltype(dA), decltype(sA),
                  decltype(copyA), TB, decltype(dB), decltype(sB), decltype(copyB), TC,
                  decltype(dC), decltype(sC), decltype(mmaC), Alpha, Beta>>(
      dimGrid, dimBlock, prob_shape, cta_tiler, A, dA, sA, copyA, B, dB, sB, copyB, C, dC,
      sC, mmaC, alpha, beta);
}

template <class TA, class TB, class TC, class Alpha, class Beta>
void gemm(char transA, char transB, int m, int n, int k, Alpha alpha, TA const* A, int ldA,
          TB const* B, int ldB, Beta beta, TC* C, int ldC) {
  if (transA == 'N' && transB == 'T') {
    return gemm_nt(m, n, k, alpha, A, ldA, B, ldB, beta, C, ldC);
  } else if (transA == 'T' && transB == 'N') {
    return gemm_tn(m, n, k, alpha, A, ldA, B, ldB, beta, C, ldC);
  }
  throw std::runtime_error("Not implemented");
}

int main(int argc, char** argv) {
  
  int m = 5120;
  if (argc >= 2) sscanf(argv[1], "%d", &m);

  int n = 5120;
  if (argc >= 3) sscanf(argv[2], "%d", &n);

  int k = 4096;
  if (argc >= 4) sscanf(argv[3], "%d", &k);

  char transA = 'N';
  if (argc >= 5) sscanf(argv[4], "%c", &transA);

  char transB = 'T';
  if (argc >= 6) sscanf(argv[5], "%c", &transB);

  using TA = float;
  using TB = float;
  using TC = float;
  using TI = float;

  TI alpha = 1.0;
  TI beta = 0.0;

  std::cout << "M = " << m << std::endl;
  std::cout << "N = " << n << std::endl;
  std::cout << "K = " << k << std::endl;
  std::cout << "C = A^" << transA << " B^" << transB << std::endl;

  std::vector<TA> h_A(m * k);
  std::vector<TB> h_B(n * k);
  std::vector<TC> h_C(m * n);

  for (int j = 0; j < m * k; ++j) h_A[j] = static_cast<TA>(2 * (rand() / double(RAND_MAX)) - 1);
  for (int j = 0; j < n * k; ++j) h_B[j] = static_cast<TB>(2 * (rand() / double(RAND_MAX)) - 1);
  for (int j = 0; j < m * n; ++j) h_C[j] = static_cast<TC>(-1);

  auto d_A = syclcompat::malloc<TA>(m * k);
  auto d_B = syclcompat::malloc<TB>(k * n);
  auto d_C = syclcompat::malloc<TC>(m * n);

  syclcompat::memcpy<TA>(d_A, h_A.data(), m * k);
  syclcompat::memcpy<TB>(d_B, h_B.data(), k * n);
  syclcompat::memcpy<TC>(d_C, h_C.data(), m * n);

  double gflops = (2.0 * m * n * k) * 1e-9;

  const int timing_iterations = 100;
  GPU_Clock timer;

  int ldA = 0, ldB = 0, ldC = m;

  if (transA == 'N') {
    ldA = m;
  } else if (transA == 'T') {
    ldA = k;
  } else {
    assert(false);
  }

  if (transB == 'N') {
    ldB = k;
  } else if (transB == 'T') {
    ldB = n;
  } else {
    assert(false);
  }
  gemm(transA, transB, m, n, k, alpha, d_A, ldA, d_B, ldB, beta, d_C, ldC);
  syclcompat::wait_and_throw();

  timer.start();
  for (int i = 0; i < timing_iterations; i++) {
    gemm(transA, transB, m, n, k, alpha, d_A, ldA, d_B, ldB, beta, d_C, ldC);
  }
  syclcompat::wait();
  
  double cute_time = timer.seconds() / timing_iterations;
  printf("SYCL_CUTE_GEMM:     [%4.3f]GFlop/s  (%6.4f)ms\n", 
                                                gflops / cute_time, cute_time * 1e3);
  return 0;
}