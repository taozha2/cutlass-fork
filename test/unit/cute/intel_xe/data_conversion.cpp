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

 #include "cutlass/detail/layout.hpp"

 #include <cute/tensor.hpp>
 #include <sycl/sycl.hpp>
 #include <syclcompat.hpp>
 #include <cutlass/gemm/collective/collective_mma.hpp>
 
 #include "cutlass_unit_test.h"
using namespace cute;
using namespace cutlass;
using namespace syclcompat::experimental;

#define SUBGROUP_SIZE (16)
constexpr int row_alignment = 16; // Alignment requirement for Xe 2D Block Copy Instructions

template <class TensorS, class TensorD, class TiledLoad, class TiledStore>
void convert_kernel(TensorS S, TensorD D, TiledLoad load,
                            TiledStore store) {
  const int m_coord = 0;
  const int n_coord = 0;
  const int l_coord = BlockIdxZ();

  // ==========  load   ==========
  auto thr_copy_load = load.get_thread_slice(ThreadIdxX());
  auto coord_tensor_load = cute::get_xe_tensor(append(S.shape(),_1{}));
  auto thr_tile_load_coord = thr_copy_load.partition_S(coord_tensor_load)(_,_,_,0);
  Tensor fragment = make_tensor<uint8_t>(thr_tile_load_coord.shape());
  Tensor fragment_out = make_tensor_like<half_t>(fragment);
  copy(load, thr_tile_load_coord, fragment);

  // ==========  convert ==========
  // vanilla conversion
//   #pragma unroll 16
//   for(size_t i = 0; i < size(fragment); i++) {
//     fragment_out[i] = static_cast<half_t>(float_e4m3_t::bitcast(fragment[i]));
//   }

  /// handwritten conversion
  constexpr int num_elements = decltype(size(fragment))::value;
  constexpr int vec_size = 16;
  Tensor src = make_tensor(static_cast<decltype(fragment)&&>(fragment).data(), make_shape(_16{}, Int<num_elements/vec_size>{}));
  CUTLASS_PRAGMA_UNROLL
  for (int i = 0; i < num_elements / vec_size; ++i) {
      // vectorized load
      cute::intel::uchar16 src_vec;
      CUTLASS_PRAGMA_UNROLL
      for (int j = 0; j < vec_size; ++j) {
          src_vec[j] = src(j, i);
      }
      // // vectorized convert fp8 -> fp16
      cute::intel::ushort16 dst_vec = E4M3_to_FP16_vec16(src_vec);
      // auto src_vec = src(_, i);
      // cute::intel::ushort16 dst_vec = E4M3_to_FP16_vec16(*reinterpret_cast<cute::intel::uchar16*>(&src_vec));
      // vectorized store
      CUTLASS_PRAGMA_UNROLL
      for (int j = 0; j < vec_size; ++j) {
          reinterpret_cast<uint16_t*>(fragment_out.data())[i * vec_size + j] = dst_vec[j];
      }
  }
  // ==========  store   ==========
  auto thr_copy_store = store.get_thread_slice(ThreadIdxX());
  auto coord_tensor_store = cute::get_xe_tensor(append(D.shape(),_1{}));
  auto thr_tile_store_coord = thr_copy_store.partition_D(coord_tensor_store)(_,_,_,0);
  Tensor frag_view =
      make_tensor(static_cast<decltype(fragment_out) &&>(fragment_out).data(),
                  thr_tile_store_coord.shape());
  copy(store, frag_view, thr_tile_store_coord);

#if 0
  if (thread(0)) {
    print("fragment: ");
    print(fragment.layout());
    print("\n");

    print("frag_view: ");
    print(frag_view.layout());
    print("\n");

    print("thr_tile_load_coord: ");
    print(thr_tile_load_coord.layout());
    print("\n");

    print("thr_tile_store_coord: ");
    print(thr_tile_store_coord.layout());
    print("\n");
  }
#endif
}

template <class Src_dtype, class Dst_dtype, class load, class store, uint32_t M, uint32_t N>
struct convert_op {
  void operator()() {
    //
    // Allocate and initialize
    //
    cutlass::host_vector<Src_dtype> host_src(M * N);
    cutlass::host_vector<Dst_dtype> host_output(M * N);
    for (size_t i = 0; i < host_src.size(); ++i) {
      host_src[i] = static_cast<Src_dtype>((i % 5  - 2) * 0.23f);
    }

    cutlass::device_vector<Src_dtype> block_in = host_src;
    cutlass::device_vector<Dst_dtype> block_out(M * N);
    Tensor S =
        make_tensor(make_gmem_ptr(block_in.data()),
                    make_layout(Shape<Int<M>, Int<N>>{}, Stride<Int<N>, _1>{}));
    Tensor D =
        make_tensor(make_gmem_ptr(block_out.data()),
                    make_layout(Shape<Int<M>, Int<N>>{}, Stride<Int<N>, _1>{}));

    auto tiled_load = make_tiled_copy(
        Copy_Atom<Copy_Traits<load, decltype(S)>, Src_dtype>{}.with(block_in.data(), M, N),
        Layout<Shape<_1, Int<SUBGROUP_SIZE>>>{},
        make_layout(shape_div(typename Copy_Traits<load, decltype(S)>::BlockShape{}, Shape<_1, _16>{})));

    auto tiled_store = make_tiled_copy(
        Copy_Atom<Copy_Traits<store, decltype(D)>, Dst_dtype>{}.with(block_out.data(), M, N),
        Layout<Shape<_1, Int<SUBGROUP_SIZE>>>{},
        make_layout(shape_div(typename Copy_Traits<store, decltype(S)>::BlockShape{}, Shape<_1, _16>{})));

    auto blockDim = syclcompat::dim3(SUBGROUP_SIZE);
    
    //
    // Launch the kernel
    //
    launch<
        convert_kernel<decltype(S), decltype(D), decltype(tiled_load),
                               decltype(tiled_store)>>(
        launch_policy{
            syclcompat::dim3(1), blockDim,
            kernel_properties{sycl_exp::sub_group_size<SUBGROUP_SIZE>}},
        S, D, tiled_load, tiled_store);

    syclcompat::wait_and_throw();
    host_output = block_out;
    for (int i = 0; i < M * N; ++i) {
      EXPECT_EQ(host_output[i], static_cast<Dst_dtype>(host_src[i]));
    }
  }
};

TEST(PVC_CuTe_Xe, data_conversion) {
    convert_op<float_e4m3_t, half_t,  XE_2D_U8x32x32_LD_V, XE_2D_U16x8x16_ST_N, 32, 32>{}();
}
