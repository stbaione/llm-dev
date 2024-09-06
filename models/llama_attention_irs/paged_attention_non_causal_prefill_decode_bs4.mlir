module @module {
  util.func public @prefill_bs4$async(%arg0: !hal.buffer_view, %arg1: !hal.buffer_view, %arg2: !hal.buffer_view, %arg3: !hal.buffer_view, %arg4: !hal.buffer_view, %arg5: !hal.buffer_view, %arg6: !hal.fence, %arg7: !hal.fence) -> !hal.buffer_view attributes {inlining_policy = #util.inline.never, iree.abi.model = "coarse-fences", iree.abi.stub} {
    %cst = arith.constant 8.837890e-02 : f16
    %c64_i64 = arith.constant 64 : i64
    %c1_i64 = arith.constant 1 : i64
    %0 = hal.tensor.import wait(%arg6) => %arg0 : !hal.buffer_view -> tensor<4x64x32x128xf16>
    %1 = hal.tensor.import wait(%arg6) => %arg1 : !hal.buffer_view -> tensor<4x64x32x128xf16>
    %2 = hal.tensor.import wait(%arg6) => %arg2 : !hal.buffer_view -> tensor<4x64x32x128xf16>
    %3 = hal.tensor.import wait(%arg6) => %arg4 : !hal.buffer_view -> tensor<4x4xi64>
    %4 = hal.tensor.import wait(%arg6) => %arg5 : !hal.buffer_view -> tensor<256x4194304xf16>
    %expanded = tensor.expand_shape %4 [[0], [1, 2, 3, 4, 5]] output_shape [256, 32, 2, 16, 32, 128] : tensor<256x4194304xf16> into tensor<256x32x2x16x32x128xf16>
    %collapsed = tensor.collapse_shape %expanded [[0, 1, 2], [3], [4], [5]] : tensor<256x32x2x16x32x128xf16> into tensor<16384x16x32x128xf16>
    %5 = tensor.empty() : tensor<4x4xi64>
    %6 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%3 : tensor<4x4xi64>) outs(%5 : tensor<4x4xi64>) {
    ^bb0(%in: i64, %out: i64):
      %24 = arith.muli %in, %c64_i64 : i64
      linalg.yield %24 : i64
    } -> tensor<4x4xi64>
    %expanded_0 = tensor.expand_shape %1 [[0], [1, 2], [3], [4]] output_shape [4, 4, 16, 32, 128] : tensor<4x64x32x128xf16> into tensor<4x4x16x32x128xf16>
    %collapsed_1 = tensor.collapse_shape %6 [[0, 1]] : tensor<4x4xi64> into tensor<16xi64>
    %collapsed_2 = tensor.collapse_shape %expanded_0 [[0, 1], [2], [3], [4]] : tensor<4x4x16x32x128xf16> into tensor<16x16x32x128xf16>
    %expanded_3 = tensor.expand_shape %collapsed_1 [[0, 1]] output_shape [16, 1] : tensor<16xi64> into tensor<16x1xi64>
    %expanded_4 = tensor.expand_shape %collapsed_2 [[0], [1, 2], [3], [4]] output_shape [16, 1, 16, 32, 128] : tensor<16x16x32x128xf16> into tensor<16x1x16x32x128xf16>
    %7 = tensor.empty() : tensor<16x1xi32>
    %8 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, 0)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%expanded_3 : tensor<16x1xi64>) outs(%7 : tensor<16x1xi32>) {
    ^bb0(%in: i64, %out: i32):
      %24 = arith.trunci %in : i64 to i32
      linalg.yield %24 : i32
    } -> tensor<16x1xi32>
    %9 = iree_linalg_ext.scatter dimension_map = [0] unique_indices(false) ins(%expanded_4, %8 : tensor<16x1x16x32x128xf16>, tensor<16x1xi32>) outs(%collapsed : tensor<16384x16x32x128xf16>) {
    ^bb0(%arg8: f16, %arg9: f16):
      iree_linalg_ext.yield %arg8 : f16
    } -> tensor<16384x16x32x128xf16>
    %expanded_5 = tensor.expand_shape %2 [[0], [1, 2], [3], [4]] output_shape [4, 4, 16, 32, 128] : tensor<4x64x32x128xf16> into tensor<4x4x16x32x128xf16>
    %10 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%6 : tensor<4x4xi64>) outs(%5 : tensor<4x4xi64>) {
    ^bb0(%in: i64, %out: i64):
      %24 = arith.addi %in, %c1_i64 : i64
      linalg.yield %24 : i64
    } -> tensor<4x4xi64>
    %collapsed_6 = tensor.collapse_shape %10 [[0, 1]] : tensor<4x4xi64> into tensor<16xi64>
    %collapsed_7 = tensor.collapse_shape %expanded_5 [[0, 1], [2], [3], [4]] : tensor<4x4x16x32x128xf16> into tensor<16x16x32x128xf16>
    %expanded_8 = tensor.expand_shape %collapsed_6 [[0, 1]] output_shape [16, 1] : tensor<16xi64> into tensor<16x1xi64>
    %expanded_9 = tensor.expand_shape %collapsed_7 [[0], [1, 2], [3], [4]] output_shape [16, 1, 16, 32, 128] : tensor<16x16x32x128xf16> into tensor<16x1x16x32x128xf16>
    %11 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, 0)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%expanded_8 : tensor<16x1xi64>) outs(%7 : tensor<16x1xi32>) {
    ^bb0(%in: i64, %out: i32):
      %24 = arith.trunci %in : i64 to i32
      linalg.yield %24 : i32
    } -> tensor<16x1xi32>
    %12 = iree_linalg_ext.scatter dimension_map = [0] unique_indices(false) ins(%expanded_9, %11 : tensor<16x1x16x32x128xf16>, tensor<16x1xi32>) outs(%9 : tensor<16384x16x32x128xf16>) {
    ^bb0(%arg8: f16, %arg9: f16):
      iree_linalg_ext.yield %arg8 : f16
    } -> tensor<16384x16x32x128xf16>
    %expanded_10 = tensor.expand_shape %12 [[0, 1, 2], [3], [4], [5]] output_shape [256, 32, 2, 16, 32, 128] : tensor<16384x16x32x128xf16> into tensor<256x32x2x16x32x128xf16>
    %collapsed_11 = tensor.collapse_shape %expanded_10 [[0], [1, 2, 3, 4, 5]] : tensor<256x32x2x16x32x128xf16> into tensor<256x4194304xf16>
    %13 = tensor.empty() : tensor<4x32x64x128xf16>
    %14 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d2, d1, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%0 : tensor<4x64x32x128xf16>) outs(%13 : tensor<4x32x64x128xf16>) {
    ^bb0(%in: f16, %out: f16):
      linalg.yield %in : f16
    } -> tensor<4x32x64x128xf16>
    %15 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d2, d1, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1 : tensor<4x64x32x128xf16>) outs(%13 : tensor<4x32x64x128xf16>) {
    ^bb0(%in: f16, %out: f16):
      linalg.yield %in : f16
    } -> tensor<4x32x64x128xf16>
    %16 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d2, d1, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%2 : tensor<4x64x32x128xf16>) outs(%13 : tensor<4x32x64x128xf16>) {
    ^bb0(%in: f16, %out: f16):
      linalg.yield %in : f16
    } -> tensor<4x32x64x128xf16>
    %collapsed_12 = tensor.collapse_shape %14 [[0, 1], [2], [3]] : tensor<4x32x64x128xf16> into tensor<128x64x128xf16>
    %collapsed_13 = tensor.collapse_shape %15 [[0, 1], [2], [3]] : tensor<4x32x64x128xf16> into tensor<128x64x128xf16>
    %collapsed_14 = tensor.collapse_shape %16 [[0, 1], [2], [3]] : tensor<4x32x64x128xf16> into tensor<128x64x128xf16>
    %17 = tensor.empty() : tensor<128x64x128xf16>
    %18 = iree_linalg_ext.attention {indexing_maps = [affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2)>, affine_map<(d0, d1, d2, d3, d4) -> (d0, d3, d2)>, affine_map<(d0, d1, d2, d3, d4) -> (d0, d3, d4)>, affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d4)>]} ins(%collapsed_12, %collapsed_13, %collapsed_14, %cst : tensor<128x64x128xf16>, tensor<128x64x128xf16>, tensor<128x64x128xf16>, f16) outs(%17 : tensor<128x64x128xf16>) -> tensor<128x64x128xf16>
    %expanded_15 = tensor.expand_shape %18 [[0, 1], [2], [3]] output_shape [4, 32, 64, 128] : tensor<128x64x128xf16> into tensor<4x32x64x128xf16>
    %19 = tensor.empty() : tensor<4x64x32x128xf16>
    %20 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d2, d1, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%expanded_15 : tensor<4x32x64x128xf16>) outs(%19 : tensor<4x64x32x128xf16>) {
    ^bb0(%in: f16, %out: f16):
      linalg.yield %in : f16
    } -> tensor<4x64x32x128xf16>
    %collapsed_16 = tensor.collapse_shape %20 [[0], [1], [2, 3]] : tensor<4x64x32x128xf16> into tensor<4x64x4096xf16>
    %21 = hal.tensor.alias wait(%arg6) => %collapsed_11 : tensor<256x4194304xf16> to %arg5 : !hal.buffer_view
    %22:2 = hal.tensor.barrier join(%21, %collapsed_16 : tensor<256x4194304xf16>, tensor<4x64x4096xf16>) => %arg7 : !hal.fence
    %23 = hal.tensor.export %22#1 : tensor<4x64x4096xf16> -> !hal.buffer_view
    util.return %23 : !hal.buffer_view
  }
  util.func public @prefill_bs4(%arg0: !hal.buffer_view, %arg1: !hal.buffer_view, %arg2: !hal.buffer_view, %arg3: !hal.buffer_view, %arg4: !hal.buffer_view, %arg5: !hal.buffer_view) -> !hal.buffer_view attributes {iree.abi.stub} {
    %0 = util.null : !hal.fence
    %c-1_i32 = arith.constant -1 : i32
    %c0 = arith.constant 0 : index
    %device_0 = hal.devices.get %c0 : !hal.device
    %fence = hal.fence.create device(%device_0 : !hal.device) flags("None") : !hal.fence
    %1 = util.call @prefill_bs4$async(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %0, %fence) : (!hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.fence, !hal.fence) -> !hal.buffer_view
    %status = hal.fence.await until([%fence]) timeout_millis(%c-1_i32) : i32
    util.return %1 : !hal.buffer_view
  }
  util.func public @decode_bs4$async(%arg0: !hal.buffer_view, %arg1: !hal.buffer_view, %arg2: !hal.buffer_view, %arg3: !hal.buffer_view, %arg4: !hal.buffer_view, %arg5: !hal.buffer_view, %arg6: !hal.buffer_view, %arg7: !hal.fence, %arg8: !hal.fence) -> !hal.buffer_view attributes {inlining_policy = #util.inline.never, iree.abi.model = "coarse-fences", iree.abi.stub} {
    %cst = arith.constant 8.837890e-02 : f16
    %c64_i64 = arith.constant 64 : i64
    %c0_i64 = arith.constant 0 : i64
    %c16_i64 = arith.constant 16 : i64
    %c1_i64 = arith.constant 1 : i64
    %c4 = arith.constant 4 : index
    %cst_0 = arith.constant 1.600000e+01 : f32
    %0 = hal.tensor.import wait(%arg7) => %arg0 : !hal.buffer_view -> tensor<4x1x32x128xf16>
    %1 = hal.tensor.import wait(%arg7) => %arg1 : !hal.buffer_view -> tensor<4x1x32x128xf16>
    %2 = hal.tensor.import wait(%arg7) => %arg2 : !hal.buffer_view -> tensor<4x1x32x128xf16>
    %3 = hal.tensor.import wait(%arg7) => %arg4 : !hal.buffer_view -> tensor<4xi64>
    %4 = hal.tensor.import wait(%arg7) => %arg5 : !hal.buffer_view -> tensor<4x4xi64>
    %5 = hal.tensor.import wait(%arg7) => %arg6 : !hal.buffer_view -> tensor<256x4194304xf16>
    %6 = tensor.empty() : tensor<4xi64>
    %7 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%3 : tensor<4xi64>) outs(%6 : tensor<4xi64>) {
    ^bb0(%in: i64, %out: i64):
      %43 = arith.addi %in, %c1_i64 : i64
      linalg.yield %43 : i64
    } -> tensor<4xi64>
    %expanded = tensor.expand_shape %5 [[0], [1, 2, 3, 4, 5]] output_shape [256, 32, 2, 16, 32, 128] : tensor<256x4194304xf16> into tensor<256x32x2x16x32x128xf16>
    %concat = tensor.concat dim(1) %1, %2 : (tensor<4x1x32x128xf16>, tensor<4x1x32x128xf16>) -> tensor<4x2x32x128xf16>
    %8 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%7 : tensor<4xi64>) outs(%6 : tensor<4xi64>) {
    ^bb0(%in: i64, %out: i64):
      %43 = arith.sitofp %in : i64 to f32
      %44 = arith.divf %43, %cst_0 : f32
      %45 = math.floor %44 : f32
      %46 = arith.fptosi %45 : f32 to i64
      linalg.yield %46 : i64
    } -> tensor<4xi64>
    %expanded_1 = tensor.expand_shape %8 [[0, 1]] output_shape [4, 1] : tensor<4xi64> into tensor<4x1xi64>
    %9 = tensor.empty() : tensor<4x1xi64>
    %10 = linalg.fill ins(%c0_i64 : i64) outs(%9 : tensor<4x1xi64>) -> tensor<4x1xi64>
    %11 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%expanded_1 : tensor<4x1xi64>) outs(%10 : tensor<4x1xi64>) {
    ^bb0(%in: i64, %out: i64):
      %43 = linalg.index 0 : index
      %44 = arith.index_cast %in : i64 to index
      %45 = arith.cmpi slt, %44, %c4 : index
      cf.assert %45, "index must be smaller than dim size"
      %46 = arith.cmpi sge, %in, %c0_i64 : i64
      cf.assert %46, "index must be larger or equal to 0"
      %extracted = tensor.extract %4[%43, %44] : tensor<4x4xi64>
      linalg.yield %extracted : i64
    } -> tensor<4x1xi64>
    %12 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%7 : tensor<4xi64>) outs(%6 : tensor<4xi64>) {
    ^bb0(%in: i64, %out: i64):
      %43 = arith.remsi %in, %c16_i64 : i64
      linalg.yield %43 : i64
    } -> tensor<4xi64>
    %13 = tensor.empty() : tensor<2xi64>
    %14 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} outs(%13 : tensor<2xi64>) {
    ^bb0(%out: i64):
      %43 = linalg.index 0 : index
      %44 = arith.index_cast %43 : index to i64
      linalg.yield %44 : i64
    } -> tensor<2xi64>
    %expanded_2 = tensor.expand_shape %11 [[0, 1], [2, 3]] output_shape [1, 4, 1, 1] : tensor<4x1xi64> into tensor<1x4x1x1xi64>
    %15 = tensor.empty() : tensor<1x4x2x1xi64>
    %16 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, 0, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%expanded_2 : tensor<1x4x1x1xi64>) outs(%15 : tensor<1x4x2x1xi64>) {
    ^bb0(%in: i64, %out: i64):
      linalg.yield %in : i64
    } -> tensor<1x4x2x1xi64>
    %17 = tensor.empty() : tensor<4x2xi64>
    %18 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%17 : tensor<4x2xi64>) outs(%17 : tensor<4x2xi64>) {
    ^bb0(%in: i64, %out: i64):
      linalg.yield %c0_i64 : i64
    } -> tensor<4x2xi64>
    %expanded_3 = tensor.expand_shape %12 [[0, 1, 2, 3]] output_shape [1, 4, 1, 1] : tensor<4xi64> into tensor<1x4x1x1xi64>
    %19 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, 0, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%expanded_3 : tensor<1x4x1x1xi64>) outs(%15 : tensor<1x4x2x1xi64>) {
    ^bb0(%in: i64, %out: i64):
      linalg.yield %in : i64
    } -> tensor<1x4x2x1xi64>
    %expanded_4 = tensor.expand_shape %14 [[0, 1, 2, 3]] output_shape [1, 1, 1, 2] : tensor<2xi64> into tensor<1x1x1x2xi64>
    %20 = tensor.empty() : tensor<4x1x1x2xi64>
    %21 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%expanded_4 : tensor<1x1x1x2xi64>) outs(%20 : tensor<4x1x1x2xi64>) {
    ^bb0(%in: i64, %out: i64):
      linalg.yield %in : i64
    } -> tensor<4x1x1x2xi64>
    %collapsed = tensor.collapse_shape %18 [[0, 1]] : tensor<4x2xi64> into tensor<8xi64>
    %collapsed_5 = tensor.collapse_shape %21 [[0, 1, 2, 3]] : tensor<4x1x1x2xi64> into tensor<8xi64>
    %collapsed_6 = tensor.collapse_shape %16 [[0, 1, 2], [3]] : tensor<1x4x2x1xi64> into tensor<8x1xi64>
    %expanded_7 = tensor.expand_shape %collapsed [[0, 1]] output_shape [8, 1] : tensor<8xi64> into tensor<8x1xi64>
    %expanded_8 = tensor.expand_shape %collapsed_5 [[0, 1]] output_shape [8, 1] : tensor<8xi64> into tensor<8x1xi64>
    %collapsed_9 = tensor.collapse_shape %19 [[0, 1, 2], [3]] : tensor<1x4x2x1xi64> into tensor<8x1xi64>
    %concat_10 = tensor.concat dim(1) %collapsed_6, %expanded_7, %expanded_8, %collapsed_9 : (tensor<8x1xi64>, tensor<8x1xi64>, tensor<8x1xi64>, tensor<8x1xi64>) -> tensor<8x4xi64>
    %collapsed_11 = tensor.collapse_shape %concat [[0, 1], [2], [3]] : tensor<4x2x32x128xf16> into tensor<8x32x128xf16>
    %expanded_12 = tensor.expand_shape %collapsed_11 [[0], [1, 2, 3, 4, 5], [6]] output_shape [8, 1, 1, 1, 1, 32, 128] : tensor<8x32x128xf16> into tensor<8x1x1x1x1x32x128xf16>
    %22 = tensor.empty() : tensor<8x4xi32>
    %23 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%concat_10 : tensor<8x4xi64>) outs(%22 : tensor<8x4xi32>) {
    ^bb0(%in: i64, %out: i32):
      %43 = arith.trunci %in : i64 to i32
      linalg.yield %43 : i32
    } -> tensor<8x4xi32>
    %24 = iree_linalg_ext.scatter dimension_map = [0, 1, 2, 3] unique_indices(false) ins(%expanded_12, %23 : tensor<8x1x1x1x1x32x128xf16>, tensor<8x4xi32>) outs(%expanded : tensor<256x32x2x16x32x128xf16>) {
    ^bb0(%arg9: f16, %arg10: f16):
      iree_linalg_ext.yield %arg9 : f16
    } -> tensor<256x32x2x16x32x128xf16>
    %collapsed_13 = tensor.collapse_shape %24 [[0], [1, 2, 3, 4, 5]] : tensor<256x32x2x16x32x128xf16> into tensor<256x4194304xf16>
    %25 = tensor.empty() : tensor<4x4xi64>
    %26 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%4 : tensor<4x4xi64>) outs(%25 : tensor<4x4xi64>) {
    ^bb0(%in: i64, %out: i64):
      %43 = arith.muli %in, %c64_i64 : i64
      linalg.yield %43 : i64
    } -> tensor<4x4xi64>
    %collapsed_14 = tensor.collapse_shape %26 [[0, 1]] : tensor<4x4xi64> into tensor<16xi64>
    %collapsed_15 = tensor.collapse_shape %24 [[0, 1, 2], [3], [4], [5]] : tensor<256x32x2x16x32x128xf16> into tensor<16384x16x32x128xf16>
    %27 = tensor.empty() : tensor<16x16x32x128xf16>
    %28 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%collapsed_14 : tensor<16xi64>) outs(%27 : tensor<16x16x32x128xf16>) {
    ^bb0(%in: i64, %out: f16):
      %43 = arith.index_cast %in : i64 to index
      %44 = linalg.index 1 : index
      %45 = linalg.index 2 : index
      %46 = linalg.index 3 : index
      %extracted = tensor.extract %collapsed_15[%43, %44, %45, %46] : tensor<16384x16x32x128xf16>
      linalg.yield %extracted : f16
    } -> tensor<16x16x32x128xf16>
    %expanded_16 = tensor.expand_shape %28 [[0, 1], [2], [3], [4]] output_shape [4, 4, 16, 32, 128] : tensor<16x16x32x128xf16> into tensor<4x4x16x32x128xf16>
    %collapsed_17 = tensor.collapse_shape %expanded_16 [[0], [1, 2], [3], [4]] : tensor<4x4x16x32x128xf16> into tensor<4x64x32x128xf16>
    %29 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%26 : tensor<4x4xi64>) outs(%25 : tensor<4x4xi64>) {
    ^bb0(%in: i64, %out: i64):
      %43 = arith.addi %in, %c1_i64 : i64
      linalg.yield %43 : i64
    } -> tensor<4x4xi64>
    %collapsed_18 = tensor.collapse_shape %29 [[0, 1]] : tensor<4x4xi64> into tensor<16xi64>
    %30 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%collapsed_18 : tensor<16xi64>) outs(%27 : tensor<16x16x32x128xf16>) {
    ^bb0(%in: i64, %out: f16):
      %43 = arith.index_cast %in : i64 to index
      %44 = linalg.index 1 : index
      %45 = linalg.index 2 : index
      %46 = linalg.index 3 : index
      %extracted = tensor.extract %collapsed_15[%43, %44, %45, %46] : tensor<16384x16x32x128xf16>
      linalg.yield %extracted : f16
    } -> tensor<16x16x32x128xf16>
    %expanded_19 = tensor.expand_shape %30 [[0, 1], [2], [3], [4]] output_shape [4, 4, 16, 32, 128] : tensor<16x16x32x128xf16> into tensor<4x4x16x32x128xf16>
    %collapsed_20 = tensor.collapse_shape %expanded_19 [[0], [1, 2], [3], [4]] : tensor<4x4x16x32x128xf16> into tensor<4x64x32x128xf16>
    %31 = tensor.empty() : tensor<4x32x1x128xf16>
    %32 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d2, d1, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%0 : tensor<4x1x32x128xf16>) outs(%31 : tensor<4x32x1x128xf16>) {
    ^bb0(%in: f16, %out: f16):
      linalg.yield %in : f16
    } -> tensor<4x32x1x128xf16>
    %33 = tensor.empty() : tensor<4x32x64x128xf16>
    %34 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d2, d1, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%collapsed_17 : tensor<4x64x32x128xf16>) outs(%33 : tensor<4x32x64x128xf16>) {
    ^bb0(%in: f16, %out: f16):
      linalg.yield %in : f16
    } -> tensor<4x32x64x128xf16>
    %35 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d2, d1, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%collapsed_20 : tensor<4x64x32x128xf16>) outs(%33 : tensor<4x32x64x128xf16>) {
    ^bb0(%in: f16, %out: f16):
      linalg.yield %in : f16
    } -> tensor<4x32x64x128xf16>
    %collapsed_21 = tensor.collapse_shape %32 [[0, 1], [2], [3]] : tensor<4x32x1x128xf16> into tensor<128x1x128xf16>
    %collapsed_22 = tensor.collapse_shape %34 [[0, 1], [2], [3]] : tensor<4x32x64x128xf16> into tensor<128x64x128xf16>
    %collapsed_23 = tensor.collapse_shape %35 [[0, 1], [2], [3]] : tensor<4x32x64x128xf16> into tensor<128x64x128xf16>
    %36 = tensor.empty() : tensor<128x1x128xf16>
    %37 = iree_linalg_ext.attention {indexing_maps = [affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2)>, affine_map<(d0, d1, d2, d3, d4) -> (d0, d3, d2)>, affine_map<(d0, d1, d2, d3, d4) -> (d0, d3, d4)>, affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d4)>]} ins(%collapsed_21, %collapsed_22, %collapsed_23, %cst : tensor<128x1x128xf16>, tensor<128x64x128xf16>, tensor<128x64x128xf16>, f16) outs(%36 : tensor<128x1x128xf16>) -> tensor<128x1x128xf16>
    %expanded_24 = tensor.expand_shape %37 [[0, 1], [2], [3]] output_shape [4, 32, 1, 128] : tensor<128x1x128xf16> into tensor<4x32x1x128xf16>
    %38 = tensor.empty() : tensor<4x1x32x128xf16>
    %39 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d2, d1, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%expanded_24 : tensor<4x32x1x128xf16>) outs(%38 : tensor<4x1x32x128xf16>) {
    ^bb0(%in: f16, %out: f16):
      linalg.yield %in : f16
    } -> tensor<4x1x32x128xf16>
    %collapsed_25 = tensor.collapse_shape %39 [[0], [1], [2, 3]] : tensor<4x1x32x128xf16> into tensor<4x1x4096xf16>
    %40 = hal.tensor.alias wait(%arg7) => %collapsed_13 : tensor<256x4194304xf16> to %arg6 : !hal.buffer_view
    %41:2 = hal.tensor.barrier join(%40, %collapsed_25 : tensor<256x4194304xf16>, tensor<4x1x4096xf16>) => %arg8 : !hal.fence
    %42 = hal.tensor.export %41#1 : tensor<4x1x4096xf16> -> !hal.buffer_view
    util.return %42 : !hal.buffer_view
  }
  util.func public @decode_bs4(%arg0: !hal.buffer_view, %arg1: !hal.buffer_view, %arg2: !hal.buffer_view, %arg3: !hal.buffer_view, %arg4: !hal.buffer_view, %arg5: !hal.buffer_view, %arg6: !hal.buffer_view) -> !hal.buffer_view attributes {iree.abi.stub} {
    %0 = util.null : !hal.fence
    %c-1_i32 = arith.constant -1 : i32
    %c0 = arith.constant 0 : index
    %device_0 = hal.devices.get %c0 : !hal.device
    %fence = hal.fence.create device(%device_0 : !hal.device) flags("None") : !hal.fence
    %1 = util.call @decode_bs4$async(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %0, %fence) : (!hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.fence, !hal.fence) -> !hal.buffer_view
    %status = hal.fence.await until([%fence]) timeout_millis(%c-1_i32) : i32
    util.return %1 : !hal.buffer_view
  }
}