/***************************************************************************************************
 * Copyright (c) 2024 - 2024 Codeplay Software Ltd. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice,
 *this list of conditions and the following disclaimer.
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
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 *ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 *LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 *CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 *SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 *INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 *CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 *ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 *POSSIBILITY OF SUCH DAMAGE.
 *
 **************************************************************************************************/

#include <cute/tensor.hpp>
#include <sycl/sycl.hpp>
#include <syclcompat.hpp>

#include "cutlass_unit_test.h"

using namespace syclcompat::experimental;

#define SUBGROUP_SIZE (16)

template <class TensorS, class TensorD, uint32_t wg_tile_m, uint32_t wg_tile_n,
          uint32_t sg_tile_m, uint32_t sg_tile_n>
void copy_kernel_vectorized(TensorS S, TensorD D, uint32_t M, uint32_t N,
                            void *) {
  using namespace cute;
  using namespace cutlass;

  using Element = typename TensorS::value_type;

  Tensor tiled_tensor_S = tiled_divide(
      S, Shape<Int<wg_tile_m>, Int<wg_tile_n>>{}); // ((M, N), m', n')
  Tensor tiled_tensor_D = tiled_divide(
      D, Shape<Int<wg_tile_m>, Int<wg_tile_n>>{}); // ((M, N), m', n')

  // Slice work group.
  Tensor tile_wg_S =
      tiled_tensor_S(make_coord(_, _), syclcompat::work_group_id::x(),
                     syclcompat::work_group_id::y());
  Tensor tile_wg_D =
      tiled_tensor_D(make_coord(_, _), syclcompat::work_group_id::x(),
                     syclcompat::work_group_id::y());

  // Slice subgroup.
  auto SubgroupShape = Shape<Int<sg_tile_m>, Int<sg_tile_n>>{};
  auto sg_id = get_sub_group_id();
  Tensor tile_sg_S = local_tile(tile_wg_S, SubgroupShape, sg_id);
  Tensor tile_sg_D = local_tile(tile_wg_D, SubgroupShape, sg_id);

#if 0
  if (thread(1)) {
    print("tile_wg_S:");
    print(tile_wg_S.layout());
    print("\n");

    print("tile_sg_S:");
    print(tile_sg_S.layout());
    print("\n");
  }
#endif

  using traits_load = Copy_Traits<XE_2D_U32x8x16_LD_N, TensorS>;
  using Atom_load = Copy_Atom<traits_load, Element>;
  auto VecLayout = make_layout(
      Shape<Int<traits_load::blk_height>, Int<traits_load::blk_width>>{},
      Stride<Int<traits_load::blk_width>, _0>{});
  auto tiled_copy_load = make_tiled_copy(Atom_load{}.with(&*S.data(), N, M, N),
                                         Layout<Shape<_1, _16>>{}, VecLayout);

  // Construct a Tensor corresponding to each thread's slice.
  auto thr_copy_load =
      tiled_copy_load.get_thread_slice(get_sub_group_local_id());
  Tensor thr_tile_load_S = thr_copy_load.partition_S(tile_sg_S);
  Tensor thr_tile_load_D = thr_copy_load.partition_D(tile_sg_S);

  // Construct a register-backed Tensor with the same shape as each thread's
  // partition Use make_fragment because the first mode is the instruction-local
  // mode
  Tensor fragment = make_fragment_like(thr_tile_load_D);

#if 0
  if (thread(1)) {
    print("thr_tile_load_S: ");
    print(thr_tile_load_S.layout());
    print("\n");

    print("thr_tile_load_D: ");
    print(thr_tile_load_D.layout());
    print("\n");

    print("fragment: ");
    print(fragment.layout());
    print("\n");
  }
#endif

  static constexpr auto sg_per_wg_x = wg_tile_n / sg_tile_n;
  const int m_coord =
      BlockIdxX() * wg_tile_m + (get_sub_group_id() / sg_per_wg_x) * sg_tile_m;
  const int n_coord =
      BlockIdxY() * wg_tile_n + (get_sub_group_id() % sg_per_wg_x) * sg_tile_n;
  const int l_coord = BlockIdxZ();

  // Copy from GMEM to RMEM and from RMEM to GMEM
  auto blk_load_S = tiled_copy_load.get_pvc_tensor(
      make_coord(m_coord, n_coord, l_coord), fragment.shape(),
      make_stride(E<0>{}, E<1>{}));
  copy(tiled_copy_load, blk_load_S, fragment);

  using traits_store = Copy_Traits<XE_2D_U32x8x16_ST_N, TensorD>;
  using Atom_store = Copy_Atom<traits_store, Element>;

  auto tiled_copy_store =
      make_tiled_copy(Atom_store{}.with(&*D.data(), N, M, N),
                      Layout<Shape<_1, _16>, Stride<_0, _1>>{}, VecLayout);
  auto thr_copy_store =
      tiled_copy_store.get_thread_slice(syclcompat::local_id::x());

  Tensor thr_tile_store_D = thr_copy_store.partition_D(tile_sg_D);

#if 0
  if (thread(1)) {
    print("storing to dst from registers ========================\n");
    print("tile_sg_D:");
    print(tile_sg_D.layout());
    print("\n");

    print("thr_tile_store_D: ");
    print(thr_tile_store_D.layout());
    print("\n");
  }
#endif

  auto blk_store_D = tiled_copy_store.get_pvc_tensor(
      make_coord(m_coord, n_coord, l_coord), fragment.shape());

  // onlt run first subgroup
  if (syclcompat::global_id::x() < 16 && !syclcompat::global_id::y() &&
      !syclcompat::global_id::z()) {
    copy(tiled_copy_store, fragment, blk_store_D);
  }
}

template <class dtype, uint32_t wg_tile_m, uint32_t wg_tile_n,
          uint32_t sg_tile_m, uint32_t sg_tile_n>
bool copy(uint32_t M, uint32_t N) {
  using namespace cute;
  //
  // Given a 2D shape, perform an efficient copy
  //

  auto tensor_shape = make_shape(M, N);
  auto block_shape = make_shape(Int<wg_tile_m>{}, Int<wg_tile_n>{});
  auto subgroup_shape = make_shape(Int<sg_tile_m>{}, Int<sg_tile_n>{});

  //
  // Allocate and initialize
  //
  std::vector<dtype> host_src(size(tensor_shape));
  std::vector<dtype> host_output(size(tensor_shape));

  dtype *device_src = syclcompat::malloc<dtype>(size(tensor_shape));
  dtype *device_output = syclcompat::malloc<dtype>(size(tensor_shape));

  for (size_t i = 0; i < host_src.size(); ++i) {
    host_src[i] = static_cast<dtype>(i);
  }

  syclcompat::memcpy<dtype>(device_src, host_src.data(), size(tensor_shape));
  syclcompat::memcpy<dtype>(device_output, host_output.data(),
                            size(tensor_shape));

  //
  // Make tensors
  //

  Tensor tensor_S = make_tensor(make_gmem_ptr(device_src),
                                make_layout(tensor_shape, make_stride(N, 1)));
  Tensor tensor_D = make_tensor(make_gmem_ptr(device_output),
                                make_layout(tensor_shape, make_stride(N, 1)));

  //
  // Tile tensors
  //

  // Tile the tensor (m, n) ==> ((M, N), m', n') where (M, N) is the static tile
  // shape, and modes (m', n') correspond to the number of tiles.
  //
  // These will be used to determine the CUDA kernel grid dimensions.

  // Thread arrangement

  static constexpr auto subgroup_size = 16;

  Layout thr_layout =
      make_layout(Shape<Int<wg_tile_m / sg_tile_m>,
                        Int<wg_tile_n * subgroup_size / sg_tile_n>>{});

  //
  // Determine grid and block dimensions
  //

  auto gridDim = syclcompat::dim3(cute::ceil_div(M, wg_tile_m),
                                  cute::ceil_div(N, wg_tile_n));
  auto blockDim = syclcompat::dim3(size(thr_layout));

  //
  // Launch the kernel
  //
  launch<copy_kernel_vectorized<decltype(tensor_S), decltype(tensor_D),
                                wg_tile_m, wg_tile_n, sg_tile_m, sg_tile_n>>(
      launch_policy{gridDim, blockDim,
                    local_mem_size{static_cast<std::size_t>(0)},
                    kernel_properties{sycl_exp::sub_group_size<SUBGROUP_SIZE>}},
      tensor_S, tensor_D, M, N);

  syclcompat::wait_and_throw();

  //
  // Verify
  //

  syclcompat::memcpy<dtype>(host_output.data(), device_output,
                            size(tensor_shape));

  auto surface_pitch = N;
  for (int i = 0; i < sg_tile_m && i < M; i++) {
    for (int j = 0; j < sg_tile_n && j < N; j++) {
      EXPECT_EQ(host_output[surface_pitch * i + j], surface_pitch * i + j);
    }
  }

  for (int i = sg_tile_m; i < sg_tile_m + 1 && i < M; i++) {
    for (int j = 0; j < sg_tile_n && j < N; j++) {
      EXPECT_NE(host_output[surface_pitch * i + j], surface_pitch * i + j);
    }
  }

  for (int i = 0; i < sg_tile_m && i < M; i++) {
    for (int j = sg_tile_n; j < sg_tile_n + 1 && j < N; j++) {
      EXPECT_NE(host_output[surface_pitch * i + j], surface_pitch * i + j);
    }
  }
  return true;
}

TEST(PVC_CuTe_Xe, block_2d_float_aligned) {
  copy<float, 8, 16, 8, 16>(8, 16);
  copy<float, 16, 128, 8, 16>(32, 128);
  copy<float, 16, 64, 16, 64>(32, 128);
  copy<float, 16, 128, 16, 64>(32, 128);
  copy<float, 64, 128, 16, 64>(1024, 4096);
}

TEST(PVC_CuTe_Xe, block_2d_float_unaligned) {
  copy<float, 64, 128, 16, 64>(1024, 4098);
  copy<float, 64, 128, 16, 64>(1026, 4096);
  copy<float, 64, 128, 16, 64>(1026, 4098);
}
