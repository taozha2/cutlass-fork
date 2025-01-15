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
  \brief Fusion callbacks specializations for the Intel PVC epilogue
*/

#pragma once

#include "cutlass/cutlass.h"

#include "cute/tensor.hpp"

#include "cutlass/epilogue/dispatch_policy.hpp"
#include "cutlass/epilogue/fusion/callbacks.hpp"
#include "cutlass/epilogue/fusion/sm90_callbacks_tma_warpspecialized.hpp"
#include "cutlass/epilogue/fusion/sm90_visitor_tma_warpspecialized.hpp"
#include "cutlass/epilogue/fusion/xe_visitor.hpp"
#include "cutlass/epilogue/fusion/sm90_visitor_load_tma_warpspecialized.hpp"
#include "cutlass/epilogue/fusion/sm90_visitor_store_tma_warpspecialized.hpp"
#include "cutlass/epilogue/fusion/sm90_visitor_compute_tma_warpspecialized.hpp"
#include "cutlass/epilogue/fusion/xe_vistor_softmax.hpp"

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass::epilogue::fusion {

/////////////////////////////////////////////////////////////////////////////////////////////////

template <
  class ElementOutput_,
  class ElementCompute_,
  class ElementSource_,
  class ElementScalar_,
  FloatRoundStyle RoundStyle_,
  class CtaTileShapeMNK_,
  class EpilogueTile_
>
struct FusionCallbacks<
    epilogue::IntelPVCEpilogue,
    fusion::LinearCombination<ElementOutput_, ElementCompute_, ElementSource_, ElementScalar_, RoundStyle_>,
    CtaTileShapeMNK_,
    EpilogueTile_
> : Sm90LinearCombination<typename cutlass::detail::get_unpacked_element_type<ElementOutput_>::type, ElementCompute_, ElementSource_, ElementScalar_, RoundStyle_> {

  using Impl = Sm90LinearCombination<typename cutlass::detail::get_unpacked_element_type<ElementOutput_>::type, ElementCompute_, ElementSource_, ElementScalar_, RoundStyle_>;
  using ElementOutput = ElementOutput_;
  using ElementCompute = ElementCompute_;
  using ElementSource = ElementSource_;
  using ElementScalar = ElementScalar_;
  using Operation = fusion::LinearCombination<ElementOutput, ElementCompute, ElementSource_, ElementScalar, RoundStyle_>;

  struct Arguments {
    ElementScalar alpha = ElementScalar(1);
    ElementScalar beta = ElementScalar(0);
    ElementScalar const* alpha_ptr = nullptr;
    ElementScalar const* beta_ptr = nullptr;

    using StrideAlpha = Stride<_0,_0,int64_t>;
    using StrideBeta  = Stride<_0,_0,int64_t>;
    StrideAlpha dAlpha = {_0{}, _0{}, 0};
    StrideBeta  dBeta  = {_0{}, _0{}, 0};

    operator typename Impl::Arguments() const {
      return
        {    // ternary op : beta * C + (alpha * acc)
          {{beta}, {beta_ptr}, {dBeta}}, // leaf args : beta
          {},                   // leaf args : C
          {                     // binary op : alpha * acc
            {{alpha}, {alpha_ptr}, {dAlpha}}, // leaf args : alpha
            {},                     // leaf args : acc
            {}                  // binary args : multiplies
          },                    // end binary op
          {} // ternary args : multiply_add
        };   // end ternary op
    }
  };

  // Ctor inheritance
  using Impl::Impl;
};


template <
  template <class> class ActivationFn_,
  class ElementOutput_,
  class ElementCompute_,
  class ElementSource_,
  class ElementScalar_,
  FloatRoundStyle RoundStyle_,
  class CtaTileShapeMNK_,
  class EpilogueTile_
>
struct FusionCallbacks<
    epilogue::IntelPVCEpilogue,
    fusion::LinCombEltAct<ActivationFn_, ElementOutput_, ElementCompute_, ElementSource_, ElementScalar_, RoundStyle_>,
    CtaTileShapeMNK_,
    EpilogueTile_
> : Sm90LinCombEltAct<ActivationFn_, ElementOutput_, ElementCompute_, ElementSource_, ElementScalar_, RoundStyle_> {

  using Impl = Sm90LinCombEltAct<ActivationFn_, typename cutlass::detail::get_unpacked_element_type<ElementOutput_>::type, ElementCompute_, ElementSource_, ElementScalar_, RoundStyle_>;
  using ElementOutput = ElementOutput_;
  using ElementCompute = ElementCompute_;
  using ElementSource = ElementSource_;
  using ElementScalar = ElementScalar_;
  using Operation = fusion::LinCombEltAct<ActivationFn_, ElementOutput_, ElementCompute_, ElementSource_, ElementScalar_, RoundStyle_>;

  struct Arguments {
    ElementScalar_ alpha = ElementScalar_(1);
    ElementScalar_ beta = ElementScalar_(0);
    ElementScalar_ const* alpha_ptr = nullptr;
    ElementScalar_ const* beta_ptr = nullptr;

    using StrideAlpha = Stride<_0,_0,int64_t>;
    using StrideBeta  = Stride<_0,_0,int64_t>;
    StrideAlpha dAlpha = {_0{}, _0{}, 0};
    StrideBeta  dBeta  = {_0{}, _0{}, 0};

    using ActivationArguments = typename Sm90Compute<ActivationFn_, ElementOutput_, ElementCompute_, RoundStyle_>::Arguments;
    ActivationArguments activation = ActivationArguments();

    operator typename Impl::Arguments() const {
      return
              {    // unary op: activation(beta * C + (alpha * acc))
                        {    // ternary op : beta * C + (alpha * acc)
                          {{beta}, {beta_ptr}, {dBeta}}, // leaf args : beta
                          {},                   // leaf args : C
                          {                     // binary op : alpha * acc
                                        {{alpha}, {alpha_ptr}, {dAlpha}}, // leaf args : alpha
                                        {},                     // leaf args : acc
                                        {}                  // binary args : multiplies
                          },                    // end binary op
                          {} // ternary args : multiply_add
                        },   // end ternary op
                        activation // unary args: activation
                };   // end unary op
    }
  };

  // Ctor inheritance
  using Impl::Impl;
};

// D = softmax(alpha * acc + beta * C)
template<
  // int FragmentSize,
  class CtaTileShapeMNK,
  class EpilogueTile,
  class ElementOutput,
  class ElementCompute,
  class ElementSource = ElementOutput,
  class ElementScalar = ElementCompute,
  FloatRoundStyle RoundStyle = FloatRoundStyle::round_to_nearest
>
using XeLinCombSoftmaxRow =
  Sm90EVT<XeSoftmaxRowReduction<CtaTileShapeMNK, EpilogueTile, ElementOutput, ElementCompute, RoundStyle>, // softmax(beta * C + (alpha * acc))
    Sm90LinearCombination<ElementCompute, ElementCompute, ElementSource, ElementScalar, RoundStyle> // beta * C + (alpha * acc)
  >;

template <
  // int FragmentSize,
  class ElementOutput_,
  class ElementCompute_,
  class ElementSource_,
  class ElementScalar_,
  FloatRoundStyle RoundStyle,
  class CtaTileShapeMNK,
  class EpilogueTile
>
struct FusionCallbacks<
    epilogue::IntelPVCEpilogue,
    fusion::LinCombSoftmaxRow<ElementOutput_, ElementCompute_, ElementSource_, ElementScalar_, RoundStyle>,
    CtaTileShapeMNK,
    EpilogueTile
> : XeLinCombSoftmaxRow<CtaTileShapeMNK, EpilogueTile, ElementOutput_, ElementCompute_, ElementSource_, ElementScalar_, RoundStyle> {

  using ElementOutput = ElementOutput_;
  using ElementCompute = ElementCompute_;
  using ElementSource = ElementSource_;
  using ElementScalar = ElementScalar_;
  using Impl = XeLinCombSoftmaxRow<CtaTileShapeMNK, EpilogueTile, typename cutlass::detail::get_unpacked_element_type<ElementOutput>::type, ElementCompute, ElementSource, ElementScalar, RoundStyle>;
  using Operation = fusion::LinCombSoftmaxRow<ElementOutput_, ElementCompute, ElementSource, ElementScalar, RoundStyle>;

  struct Arguments {
    ElementScalar alpha = ElementScalar(1);
    ElementScalar beta = ElementScalar(0);
    ElementScalar const* alpha_ptr = nullptr;
    ElementScalar const* beta_ptr = nullptr;

    operator typename Impl::Arguments() const {
      return
        {    // unary op: activation(beta * C + (alpha * acc))
          {    // ternary op : beta * C + (alpha * acc)
            {{beta}, {beta_ptr}}, // leaf args : beta
            {},                   // leaf args : C
            {                     // binary op : alpha * acc
              {{alpha}, {alpha_ptr}}, // leaf args : alpha
              {},                     // leaf args : acc
              {}                  // binary args : multiplies
            },                    // end binary op
            {} // ternary args : multiply_add
          },   // end ternary op
          {} // unary args: activation
        };   // end unary op
    }
  };

  // Ctor inheritance
  using Impl::Impl;
};

template<
  class CtaTileShapeMNK,
  class StrideAux,
  class CopyOpG2R,
  template <class> class ActivationFn,
  class ElementOutput,
  class ElementCompute,
  class ElementAux = ElementOutput,
  class ElementSource = ElementOutput,
  class ElementScalar = ElementCompute,
  FloatRoundStyle RoundStyle = FloatRoundStyle::round_to_nearest
>
using XeLinCombDeEltAct =
  Sm90EVT<Sm90Compute<ActivationFn, ElementOutput, ElementCompute, RoundStyle>, // activation(beta * C + (alpha * acc), aux)
    Sm90LinearCombination<ElementCompute, ElementCompute, ElementSource, ElementScalar, RoundStyle>, // beta * C + (alpha * acc)
    XeAuxLoad<CtaTileShapeMNK, ElementAux, StrideAux, CopyOpG2R> // aux
  >;

// Z = Aux
// dY = alpha * acc + beta * C
// D = activation(dY, Z)
//
template <
  class GmemLayoutTagAux,
  template <class> class ActivationFn,
  class ElementOutput_,
  class ElementCompute_,
  class ElementAux,
  class ElementSource,
  class ElementScalar,
  int AlignmentAux,
  FloatRoundStyle RoundStyle,
  class CtaTileShapeMNK,
  class EpilogueTile,
  class CopyOpG2R
>
struct FusionCallbacks<
    epilogue::IntelPVCEpilogue,
    fusion::LinCombDeEltAct<
      GmemLayoutTagAux, ActivationFn, ElementOutput_, ElementCompute_,
      ElementAux, ElementSource, ElementScalar, AlignmentAux, RoundStyle
    >,
    CtaTileShapeMNK,
    EpilogueTile,
    CopyOpG2R
> : XeLinCombDeEltAct<
      CtaTileShapeMNK, cutlass::gemm::TagToStrideC_t<GmemLayoutTagAux>, CopyOpG2R, ActivationFn,
      ElementOutput_, ElementCompute_, ElementAux, ElementSource, ElementScalar, RoundStyle
    > {

  using ElementOutput = ElementOutput_;
  using ElementCompute = ElementCompute_;

  using Impl =
    XeLinCombDeEltAct<
      CtaTileShapeMNK, cutlass::gemm::TagToStrideC_t<GmemLayoutTagAux>, CopyOpG2R, ActivationFn,
      ElementOutput, ElementCompute, ElementAux, ElementSource, ElementScalar, RoundStyle
    >;
  using Operation =
    fusion::LinCombDeEltAct<
      GmemLayoutTagAux, ActivationFn, ElementOutput, ElementCompute,
      ElementAux, ElementSource, ElementScalar, AlignmentAux, RoundStyle
    >;

  struct Arguments {
    ElementScalar alpha = ElementScalar(1);
    ElementScalar beta = ElementScalar(0);
    ElementScalar const* alpha_ptr = nullptr;
    ElementScalar const* beta_ptr = nullptr;

    using StrideAlpha = Stride<_0,_0,int64_t>;
    using StrideBeta  = Stride<_0,_0,int64_t>;
    StrideAlpha dAlpha = {_0{}, _0{}, 0};
    StrideBeta  dBeta  = {_0{}, _0{}, 0};

    using ActivationArguments = typename Sm90Compute<ActivationFn, ElementOutput, ElementCompute, RoundStyle>::Arguments;
    ActivationArguments activation = ActivationArguments();

    using StrideAux = cutlass::gemm::TagToStrideC_t<GmemLayoutTagAux>;
    ElementAux const* aux_ptr = nullptr;
    StrideAux dAux = {};

    operator typename Impl::Arguments() const {
      return
        {    // binary op : activation(beta * C + (alpha * acc), aux)
          {                  // ternary op : beta * C + (alpha * acc)
            {{beta}, {beta_ptr}, {dBeta}}, // leaf args : beta
            {},                   // leaf args : C
            {                     // binary op : alpha * acc
              {{alpha}, {alpha_ptr}, {dAlpha}}, // leaf args : alpha
              {},                     // leaf args : acc
              {}                  // binary args : multiplies
            },                    // end binary op
            {}               // ternary args : multiply_add
          },                 // end ternary op
          {aux_ptr, ElementAux(0), dAux}, // leaf args : aux
          activation // binary args : activation
        };   // end binary op
    }
  };

  // Ctor inheritance
  using Impl::Impl;
};


} // namespace cutlass::epilogue::fusion

/////////////////////////////////////////////////////////////////////////////////////////////////
