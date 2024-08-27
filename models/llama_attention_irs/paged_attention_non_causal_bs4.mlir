module @module {
  util.func public @paged_attn_bs4$async(%arg0: !hal.buffer_view, %arg1: !hal.buffer_view, %arg2: !hal.buffer_view, %arg3: !hal.buffer_view, %arg4: !hal.buffer_view, %arg5: !hal.buffer_view, %arg6: !hal.fence, %arg7: !hal.fence) -> !hal.buffer_view attributes {inlining_policy = #util.inline.never, iree.abi.model = "coarse-fences", iree.abi.stub} {
    %cst = arith.constant 8.837890e-02 : f16
    %c1_i64 = arith.constant 1 : i64
    %c64_i64 = arith.constant 64 : i64
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
  util.func public @paged_attn_bs4(%arg0: !hal.buffer_view, %arg1: !hal.buffer_view, %arg2: !hal.buffer_view, %arg3: !hal.buffer_view, %arg4: !hal.buffer_view, %arg5: !hal.buffer_view) -> !hal.buffer_view attributes {iree.abi.stub} {
    %0 = util.null : !hal.fence
    %c-1_i32 = arith.constant -1 : i32
    %c0 = arith.constant 0 : index
    %device_0 = hal.devices.get %c0 : !hal.device
    %fence = hal.fence.create device(%device_0 : !hal.device) flags("None") : !hal.fence
    %1 = util.call @paged_attn_bs4$async(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %0, %fence) : (!hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.fence, !hal.fence) -> !hal.buffer_view
    %status = hal.fence.await until([%fence]) timeout_millis(%c-1_i32) : i32
    util.return %1 : !hal.buffer_view
  }
}