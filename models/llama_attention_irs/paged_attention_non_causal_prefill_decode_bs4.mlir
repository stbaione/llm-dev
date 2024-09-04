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
    %cst = arith.constant dense<1> : tensor<1x1xi64>
    %cst_0 = arith.constant dense<0> : tensor<1x1xi64>
    %cst_1 = arith.constant 8.837890e-02 : f16
    %c64_i64 = arith.constant 64 : i64
    %c16_i64 = arith.constant 16 : i64
    %c1_i64 = arith.constant 1 : i64
    %cst_2 = arith.constant 1.600000e+01 : f32
    %0 = hal.tensor.import wait(%arg7) => %arg0 : !hal.buffer_view -> tensor<4x1x32x128xf16>
    %1 = hal.tensor.import wait(%arg7) => %arg1 : !hal.buffer_view -> tensor<4x1x32x128xf16>
    %2 = hal.tensor.import wait(%arg7) => %arg2 : !hal.buffer_view -> tensor<4x1x32x128xf16>
    %3 = hal.tensor.import wait(%arg7) => %arg4 : !hal.buffer_view -> tensor<4xi64>
    %4 = hal.tensor.import wait(%arg7) => %arg5 : !hal.buffer_view -> tensor<4x4xi64>
    %5 = hal.tensor.import wait(%arg7) => %arg6 : !hal.buffer_view -> tensor<256x4194304xf16>
    %6 = tensor.empty() : tensor<4xi64>
    %7 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%3 : tensor<4xi64>) outs(%6 : tensor<4xi64>) {
    ^bb0(%in: i64, %out: i64):
      %66 = arith.addi %in, %c1_i64 : i64
      linalg.yield %66 : i64
    } -> tensor<4xi64>
    %expanded = tensor.expand_shape %5 [[0], [1, 2, 3, 4, 5]] output_shape [256, 32, 2, 16, 32, 128] : tensor<256x4194304xf16> into tensor<256x32x2x16x32x128xf16>
    %extracted_slice = tensor.extract_slice %7[0] [1] [1] : tensor<4xi64> to tensor<1xi64>
    %collapsed = tensor.collapse_shape %extracted_slice [] : tensor<1xi64> into tensor<i64>
    %extracted_slice_3 = tensor.extract_slice %4[0, 0] [1, 4] [1, 1] : tensor<4x4xi64> to tensor<1x4xi64>
    %collapsed_4 = tensor.collapse_shape %extracted_slice_3 [[0, 1]] : tensor<1x4xi64> into tensor<4xi64>
    %8 = tensor.empty() : tensor<i64>
    %9 = linalg.generic {indexing_maps = [affine_map<() -> ()>, affine_map<() -> ()>], iterator_types = []} ins(%collapsed : tensor<i64>) outs(%8 : tensor<i64>) {
    ^bb0(%in: i64, %out: i64):
      %66 = arith.sitofp %in : i64 to f32
      %67 = arith.divf %66, %cst_2 : f32
      %68 = math.floor %67 : f32
      %69 = arith.fptosi %68 : f32 to i64
      linalg.yield %69 : i64
    } -> tensor<i64>
    %expanded_5 = tensor.expand_shape %9 [] output_shape [1] : tensor<i64> into tensor<1xi64>
    %10 = tensor.empty() : tensor<1xi64>
    %11 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%expanded_5 : tensor<1xi64>) outs(%10 : tensor<1xi64>) {
    ^bb0(%in: i64, %out: i64):
      %66 = arith.index_cast %in : i64 to index
      %extracted = tensor.extract %collapsed_4[%66] : tensor<4xi64>
      linalg.yield %extracted : i64
    } -> tensor<1xi64>
    %12 = linalg.generic {indexing_maps = [affine_map<() -> ()>, affine_map<() -> ()>], iterator_types = []} ins(%collapsed : tensor<i64>) outs(%8 : tensor<i64>) {
    ^bb0(%in: i64, %out: i64):
      %66 = arith.remsi %in, %c16_i64 : i64
      linalg.yield %66 : i64
    } -> tensor<i64>
    %extracted_slice_6 = tensor.extract_slice %1[0, 0, 0, 0] [1, 1, 32, 128] [1, 1, 1, 1] : tensor<4x1x32x128xf16> to tensor<1x1x32x128xf16>
    %collapsed_7 = tensor.collapse_shape %extracted_slice_6 [[0, 1, 2], [3]] : tensor<1x1x32x128xf16> into tensor<32x128xf16>
    %13 = tensor.empty() : tensor<1x32x128xf16>
    %14 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%collapsed_7 : tensor<32x128xf16>) outs(%13 : tensor<1x32x128xf16>) {
    ^bb0(%in: f16, %out: f16):
      linalg.yield %in : f16
    } -> tensor<1x32x128xf16>
    %expanded_8 = tensor.expand_shape %11 [[0, 1]] output_shape [1, 1] : tensor<1xi64> into tensor<1x1xi64>
    %expanded_9 = tensor.expand_shape %12 [] output_shape [1, 1] : tensor<i64> into tensor<1x1xi64>
    %concat = tensor.concat dim(1) %expanded_8, %cst_0, %cst_0, %expanded_9 : (tensor<1x1xi64>, tensor<1x1xi64>, tensor<1x1xi64>, tensor<1x1xi64>) -> tensor<1x4xi64>
    %expanded_10 = tensor.expand_shape %14 [[0], [1, 2, 3, 4, 5], [6]] output_shape [1, 1, 1, 1, 1, 32, 128] : tensor<1x32x128xf16> into tensor<1x1x1x1x1x32x128xf16>
    %15 = tensor.empty() : tensor<1x4xi32>
    %16 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%concat : tensor<1x4xi64>) outs(%15 : tensor<1x4xi32>) {
    ^bb0(%in: i64, %out: i32):
      %66 = arith.trunci %in : i64 to i32
      linalg.yield %66 : i32
    } -> tensor<1x4xi32>
    %17 = iree_linalg_ext.scatter dimension_map = [0, 1, 2, 3] unique_indices(false) ins(%expanded_10, %16 : tensor<1x1x1x1x1x32x128xf16>, tensor<1x4xi32>) outs(%expanded : tensor<256x32x2x16x32x128xf16>) {
    ^bb0(%arg9: f16, %arg10: f16):
      iree_linalg_ext.yield %arg9 : f16
    } -> tensor<256x32x2x16x32x128xf16>
    %extracted_slice_11 = tensor.extract_slice %2[0, 0, 0, 0] [1, 1, 32, 128] [1, 1, 1, 1] : tensor<4x1x32x128xf16> to tensor<1x1x32x128xf16>
    %collapsed_12 = tensor.collapse_shape %extracted_slice_11 [[0, 1, 2], [3]] : tensor<1x1x32x128xf16> into tensor<32x128xf16>
    %18 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%collapsed_12 : tensor<32x128xf16>) outs(%13 : tensor<1x32x128xf16>) {
    ^bb0(%in: f16, %out: f16):
      linalg.yield %in : f16
    } -> tensor<1x32x128xf16>
    %concat_13 = tensor.concat dim(1) %expanded_8, %cst_0, %cst, %expanded_9 : (tensor<1x1xi64>, tensor<1x1xi64>, tensor<1x1xi64>, tensor<1x1xi64>) -> tensor<1x4xi64>
    %expanded_14 = tensor.expand_shape %18 [[0], [1, 2, 3, 4, 5], [6]] output_shape [1, 1, 1, 1, 1, 32, 128] : tensor<1x32x128xf16> into tensor<1x1x1x1x1x32x128xf16>
    %19 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%concat_13 : tensor<1x4xi64>) outs(%15 : tensor<1x4xi32>) {
    ^bb0(%in: i64, %out: i32):
      %66 = arith.trunci %in : i64 to i32
      linalg.yield %66 : i32
    } -> tensor<1x4xi32>
    %20 = iree_linalg_ext.scatter dimension_map = [0, 1, 2, 3] unique_indices(false) ins(%expanded_14, %19 : tensor<1x1x1x1x1x32x128xf16>, tensor<1x4xi32>) outs(%17 : tensor<256x32x2x16x32x128xf16>) {
    ^bb0(%arg9: f16, %arg10: f16):
      iree_linalg_ext.yield %arg9 : f16
    } -> tensor<256x32x2x16x32x128xf16>
    %extracted_slice_15 = tensor.extract_slice %7[1] [1] [1] : tensor<4xi64> to tensor<1xi64>
    %collapsed_16 = tensor.collapse_shape %extracted_slice_15 [] : tensor<1xi64> into tensor<i64>
    %extracted_slice_17 = tensor.extract_slice %4[1, 0] [1, 4] [1, 1] : tensor<4x4xi64> to tensor<1x4xi64>
    %collapsed_18 = tensor.collapse_shape %extracted_slice_17 [[0, 1]] : tensor<1x4xi64> into tensor<4xi64>
    %21 = linalg.generic {indexing_maps = [affine_map<() -> ()>, affine_map<() -> ()>], iterator_types = []} ins(%collapsed_16 : tensor<i64>) outs(%8 : tensor<i64>) {
    ^bb0(%in: i64, %out: i64):
      %66 = arith.sitofp %in : i64 to f32
      %67 = arith.divf %66, %cst_2 : f32
      %68 = math.floor %67 : f32
      %69 = arith.fptosi %68 : f32 to i64
      linalg.yield %69 : i64
    } -> tensor<i64>
    %expanded_19 = tensor.expand_shape %21 [] output_shape [1] : tensor<i64> into tensor<1xi64>
    %22 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%expanded_19 : tensor<1xi64>) outs(%10 : tensor<1xi64>) {
    ^bb0(%in: i64, %out: i64):
      %66 = arith.index_cast %in : i64 to index
      %extracted = tensor.extract %collapsed_18[%66] : tensor<4xi64>
      linalg.yield %extracted : i64
    } -> tensor<1xi64>
    %23 = linalg.generic {indexing_maps = [affine_map<() -> ()>, affine_map<() -> ()>], iterator_types = []} ins(%collapsed_16 : tensor<i64>) outs(%8 : tensor<i64>) {
    ^bb0(%in: i64, %out: i64):
      %66 = arith.remsi %in, %c16_i64 : i64
      linalg.yield %66 : i64
    } -> tensor<i64>
    %extracted_slice_20 = tensor.extract_slice %1[1, 0, 0, 0] [1, 1, 32, 128] [1, 1, 1, 1] : tensor<4x1x32x128xf16> to tensor<1x1x32x128xf16>
    %collapsed_21 = tensor.collapse_shape %extracted_slice_20 [[0, 1, 2], [3]] : tensor<1x1x32x128xf16> into tensor<32x128xf16>
    %24 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%collapsed_21 : tensor<32x128xf16>) outs(%13 : tensor<1x32x128xf16>) {
    ^bb0(%in: f16, %out: f16):
      linalg.yield %in : f16
    } -> tensor<1x32x128xf16>
    %expanded_22 = tensor.expand_shape %22 [[0, 1]] output_shape [1, 1] : tensor<1xi64> into tensor<1x1xi64>
    %expanded_23 = tensor.expand_shape %23 [] output_shape [1, 1] : tensor<i64> into tensor<1x1xi64>
    %concat_24 = tensor.concat dim(1) %expanded_22, %cst_0, %cst_0, %expanded_23 : (tensor<1x1xi64>, tensor<1x1xi64>, tensor<1x1xi64>, tensor<1x1xi64>) -> tensor<1x4xi64>
    %expanded_25 = tensor.expand_shape %24 [[0], [1, 2, 3, 4, 5], [6]] output_shape [1, 1, 1, 1, 1, 32, 128] : tensor<1x32x128xf16> into tensor<1x1x1x1x1x32x128xf16>
    %25 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%concat_24 : tensor<1x4xi64>) outs(%15 : tensor<1x4xi32>) {
    ^bb0(%in: i64, %out: i32):
      %66 = arith.trunci %in : i64 to i32
      linalg.yield %66 : i32
    } -> tensor<1x4xi32>
    %26 = iree_linalg_ext.scatter dimension_map = [0, 1, 2, 3] unique_indices(false) ins(%expanded_25, %25 : tensor<1x1x1x1x1x32x128xf16>, tensor<1x4xi32>) outs(%20 : tensor<256x32x2x16x32x128xf16>) {
    ^bb0(%arg9: f16, %arg10: f16):
      iree_linalg_ext.yield %arg9 : f16
    } -> tensor<256x32x2x16x32x128xf16>
    %extracted_slice_26 = tensor.extract_slice %2[1, 0, 0, 0] [1, 1, 32, 128] [1, 1, 1, 1] : tensor<4x1x32x128xf16> to tensor<1x1x32x128xf16>
    %collapsed_27 = tensor.collapse_shape %extracted_slice_26 [[0, 1, 2], [3]] : tensor<1x1x32x128xf16> into tensor<32x128xf16>
    %27 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%collapsed_27 : tensor<32x128xf16>) outs(%13 : tensor<1x32x128xf16>) {
    ^bb0(%in: f16, %out: f16):
      linalg.yield %in : f16
    } -> tensor<1x32x128xf16>
    %concat_28 = tensor.concat dim(1) %expanded_22, %cst_0, %cst, %expanded_23 : (tensor<1x1xi64>, tensor<1x1xi64>, tensor<1x1xi64>, tensor<1x1xi64>) -> tensor<1x4xi64>
    %expanded_29 = tensor.expand_shape %27 [[0], [1, 2, 3, 4, 5], [6]] output_shape [1, 1, 1, 1, 1, 32, 128] : tensor<1x32x128xf16> into tensor<1x1x1x1x1x32x128xf16>
    %28 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%concat_28 : tensor<1x4xi64>) outs(%15 : tensor<1x4xi32>) {
    ^bb0(%in: i64, %out: i32):
      %66 = arith.trunci %in : i64 to i32
      linalg.yield %66 : i32
    } -> tensor<1x4xi32>
    %29 = iree_linalg_ext.scatter dimension_map = [0, 1, 2, 3] unique_indices(false) ins(%expanded_29, %28 : tensor<1x1x1x1x1x32x128xf16>, tensor<1x4xi32>) outs(%26 : tensor<256x32x2x16x32x128xf16>) {
    ^bb0(%arg9: f16, %arg10: f16):
      iree_linalg_ext.yield %arg9 : f16
    } -> tensor<256x32x2x16x32x128xf16>
    %extracted_slice_30 = tensor.extract_slice %7[2] [1] [1] : tensor<4xi64> to tensor<1xi64>
    %collapsed_31 = tensor.collapse_shape %extracted_slice_30 [] : tensor<1xi64> into tensor<i64>
    %extracted_slice_32 = tensor.extract_slice %4[2, 0] [1, 4] [1, 1] : tensor<4x4xi64> to tensor<1x4xi64>
    %collapsed_33 = tensor.collapse_shape %extracted_slice_32 [[0, 1]] : tensor<1x4xi64> into tensor<4xi64>
    %30 = linalg.generic {indexing_maps = [affine_map<() -> ()>, affine_map<() -> ()>], iterator_types = []} ins(%collapsed_31 : tensor<i64>) outs(%8 : tensor<i64>) {
    ^bb0(%in: i64, %out: i64):
      %66 = arith.sitofp %in : i64 to f32
      %67 = arith.divf %66, %cst_2 : f32
      %68 = math.floor %67 : f32
      %69 = arith.fptosi %68 : f32 to i64
      linalg.yield %69 : i64
    } -> tensor<i64>
    %expanded_34 = tensor.expand_shape %30 [] output_shape [1] : tensor<i64> into tensor<1xi64>
    %31 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%expanded_34 : tensor<1xi64>) outs(%10 : tensor<1xi64>) {
    ^bb0(%in: i64, %out: i64):
      %66 = arith.index_cast %in : i64 to index
      %extracted = tensor.extract %collapsed_33[%66] : tensor<4xi64>
      linalg.yield %extracted : i64
    } -> tensor<1xi64>
    %32 = linalg.generic {indexing_maps = [affine_map<() -> ()>, affine_map<() -> ()>], iterator_types = []} ins(%collapsed_31 : tensor<i64>) outs(%8 : tensor<i64>) {
    ^bb0(%in: i64, %out: i64):
      %66 = arith.remsi %in, %c16_i64 : i64
      linalg.yield %66 : i64
    } -> tensor<i64>
    %extracted_slice_35 = tensor.extract_slice %1[2, 0, 0, 0] [1, 1, 32, 128] [1, 1, 1, 1] : tensor<4x1x32x128xf16> to tensor<1x1x32x128xf16>
    %collapsed_36 = tensor.collapse_shape %extracted_slice_35 [[0, 1, 2], [3]] : tensor<1x1x32x128xf16> into tensor<32x128xf16>
    %33 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%collapsed_36 : tensor<32x128xf16>) outs(%13 : tensor<1x32x128xf16>) {
    ^bb0(%in: f16, %out: f16):
      linalg.yield %in : f16
    } -> tensor<1x32x128xf16>
    %expanded_37 = tensor.expand_shape %31 [[0, 1]] output_shape [1, 1] : tensor<1xi64> into tensor<1x1xi64>
    %expanded_38 = tensor.expand_shape %32 [] output_shape [1, 1] : tensor<i64> into tensor<1x1xi64>
    %concat_39 = tensor.concat dim(1) %expanded_37, %cst_0, %cst_0, %expanded_38 : (tensor<1x1xi64>, tensor<1x1xi64>, tensor<1x1xi64>, tensor<1x1xi64>) -> tensor<1x4xi64>
    %expanded_40 = tensor.expand_shape %33 [[0], [1, 2, 3, 4, 5], [6]] output_shape [1, 1, 1, 1, 1, 32, 128] : tensor<1x32x128xf16> into tensor<1x1x1x1x1x32x128xf16>
    %34 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%concat_39 : tensor<1x4xi64>) outs(%15 : tensor<1x4xi32>) {
    ^bb0(%in: i64, %out: i32):
      %66 = arith.trunci %in : i64 to i32
      linalg.yield %66 : i32
    } -> tensor<1x4xi32>
    %35 = iree_linalg_ext.scatter dimension_map = [0, 1, 2, 3] unique_indices(false) ins(%expanded_40, %34 : tensor<1x1x1x1x1x32x128xf16>, tensor<1x4xi32>) outs(%29 : tensor<256x32x2x16x32x128xf16>) {
    ^bb0(%arg9: f16, %arg10: f16):
      iree_linalg_ext.yield %arg9 : f16
    } -> tensor<256x32x2x16x32x128xf16>
    %extracted_slice_41 = tensor.extract_slice %2[2, 0, 0, 0] [1, 1, 32, 128] [1, 1, 1, 1] : tensor<4x1x32x128xf16> to tensor<1x1x32x128xf16>
    %collapsed_42 = tensor.collapse_shape %extracted_slice_41 [[0, 1, 2], [3]] : tensor<1x1x32x128xf16> into tensor<32x128xf16>
    %36 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%collapsed_42 : tensor<32x128xf16>) outs(%13 : tensor<1x32x128xf16>) {
    ^bb0(%in: f16, %out: f16):
      linalg.yield %in : f16
    } -> tensor<1x32x128xf16>
    %concat_43 = tensor.concat dim(1) %expanded_37, %cst_0, %cst, %expanded_38 : (tensor<1x1xi64>, tensor<1x1xi64>, tensor<1x1xi64>, tensor<1x1xi64>) -> tensor<1x4xi64>
    %expanded_44 = tensor.expand_shape %36 [[0], [1, 2, 3, 4, 5], [6]] output_shape [1, 1, 1, 1, 1, 32, 128] : tensor<1x32x128xf16> into tensor<1x1x1x1x1x32x128xf16>
    %37 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%concat_43 : tensor<1x4xi64>) outs(%15 : tensor<1x4xi32>) {
    ^bb0(%in: i64, %out: i32):
      %66 = arith.trunci %in : i64 to i32
      linalg.yield %66 : i32
    } -> tensor<1x4xi32>
    %38 = iree_linalg_ext.scatter dimension_map = [0, 1, 2, 3] unique_indices(false) ins(%expanded_44, %37 : tensor<1x1x1x1x1x32x128xf16>, tensor<1x4xi32>) outs(%35 : tensor<256x32x2x16x32x128xf16>) {
    ^bb0(%arg9: f16, %arg10: f16):
      iree_linalg_ext.yield %arg9 : f16
    } -> tensor<256x32x2x16x32x128xf16>
    %extracted_slice_45 = tensor.extract_slice %7[3] [1] [1] : tensor<4xi64> to tensor<1xi64>
    %collapsed_46 = tensor.collapse_shape %extracted_slice_45 [] : tensor<1xi64> into tensor<i64>
    %extracted_slice_47 = tensor.extract_slice %4[3, 0] [1, 4] [1, 1] : tensor<4x4xi64> to tensor<1x4xi64>
    %collapsed_48 = tensor.collapse_shape %extracted_slice_47 [[0, 1]] : tensor<1x4xi64> into tensor<4xi64>
    %39 = linalg.generic {indexing_maps = [affine_map<() -> ()>, affine_map<() -> ()>], iterator_types = []} ins(%collapsed_46 : tensor<i64>) outs(%8 : tensor<i64>) {
    ^bb0(%in: i64, %out: i64):
      %66 = arith.sitofp %in : i64 to f32
      %67 = arith.divf %66, %cst_2 : f32
      %68 = math.floor %67 : f32
      %69 = arith.fptosi %68 : f32 to i64
      linalg.yield %69 : i64
    } -> tensor<i64>
    %expanded_49 = tensor.expand_shape %39 [] output_shape [1] : tensor<i64> into tensor<1xi64>
    %40 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%expanded_49 : tensor<1xi64>) outs(%10 : tensor<1xi64>) {
    ^bb0(%in: i64, %out: i64):
      %66 = arith.index_cast %in : i64 to index
      %extracted = tensor.extract %collapsed_48[%66] : tensor<4xi64>
      linalg.yield %extracted : i64
    } -> tensor<1xi64>
    %41 = linalg.generic {indexing_maps = [affine_map<() -> ()>, affine_map<() -> ()>], iterator_types = []} ins(%collapsed_46 : tensor<i64>) outs(%8 : tensor<i64>) {
    ^bb0(%in: i64, %out: i64):
      %66 = arith.remsi %in, %c16_i64 : i64
      linalg.yield %66 : i64
    } -> tensor<i64>
    %extracted_slice_50 = tensor.extract_slice %1[3, 0, 0, 0] [1, 1, 32, 128] [1, 1, 1, 1] : tensor<4x1x32x128xf16> to tensor<1x1x32x128xf16>
    %collapsed_51 = tensor.collapse_shape %extracted_slice_50 [[0, 1, 2], [3]] : tensor<1x1x32x128xf16> into tensor<32x128xf16>
    %42 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%collapsed_51 : tensor<32x128xf16>) outs(%13 : tensor<1x32x128xf16>) {
    ^bb0(%in: f16, %out: f16):
      linalg.yield %in : f16
    } -> tensor<1x32x128xf16>
    %expanded_52 = tensor.expand_shape %40 [[0, 1]] output_shape [1, 1] : tensor<1xi64> into tensor<1x1xi64>
    %expanded_53 = tensor.expand_shape %41 [] output_shape [1, 1] : tensor<i64> into tensor<1x1xi64>
    %concat_54 = tensor.concat dim(1) %expanded_52, %cst_0, %cst_0, %expanded_53 : (tensor<1x1xi64>, tensor<1x1xi64>, tensor<1x1xi64>, tensor<1x1xi64>) -> tensor<1x4xi64>
    %expanded_55 = tensor.expand_shape %42 [[0], [1, 2, 3, 4, 5], [6]] output_shape [1, 1, 1, 1, 1, 32, 128] : tensor<1x32x128xf16> into tensor<1x1x1x1x1x32x128xf16>
    %43 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%concat_54 : tensor<1x4xi64>) outs(%15 : tensor<1x4xi32>) {
    ^bb0(%in: i64, %out: i32):
      %66 = arith.trunci %in : i64 to i32
      linalg.yield %66 : i32
    } -> tensor<1x4xi32>
    %44 = iree_linalg_ext.scatter dimension_map = [0, 1, 2, 3] unique_indices(false) ins(%expanded_55, %43 : tensor<1x1x1x1x1x32x128xf16>, tensor<1x4xi32>) outs(%38 : tensor<256x32x2x16x32x128xf16>) {
    ^bb0(%arg9: f16, %arg10: f16):
      iree_linalg_ext.yield %arg9 : f16
    } -> tensor<256x32x2x16x32x128xf16>
    %extracted_slice_56 = tensor.extract_slice %2[3, 0, 0, 0] [1, 1, 32, 128] [1, 1, 1, 1] : tensor<4x1x32x128xf16> to tensor<1x1x32x128xf16>
    %collapsed_57 = tensor.collapse_shape %extracted_slice_56 [[0, 1, 2], [3]] : tensor<1x1x32x128xf16> into tensor<32x128xf16>
    %45 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%collapsed_57 : tensor<32x128xf16>) outs(%13 : tensor<1x32x128xf16>) {
    ^bb0(%in: f16, %out: f16):
      linalg.yield %in : f16
    } -> tensor<1x32x128xf16>
    %concat_58 = tensor.concat dim(1) %expanded_52, %cst_0, %cst, %expanded_53 : (tensor<1x1xi64>, tensor<1x1xi64>, tensor<1x1xi64>, tensor<1x1xi64>) -> tensor<1x4xi64>
    %expanded_59 = tensor.expand_shape %45 [[0], [1, 2, 3, 4, 5], [6]] output_shape [1, 1, 1, 1, 1, 32, 128] : tensor<1x32x128xf16> into tensor<1x1x1x1x1x32x128xf16>
    %46 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%concat_58 : tensor<1x4xi64>) outs(%15 : tensor<1x4xi32>) {
    ^bb0(%in: i64, %out: i32):
      %66 = arith.trunci %in : i64 to i32
      linalg.yield %66 : i32
    } -> tensor<1x4xi32>
    %47 = iree_linalg_ext.scatter dimension_map = [0, 1, 2, 3] unique_indices(false) ins(%expanded_59, %46 : tensor<1x1x1x1x1x32x128xf16>, tensor<1x4xi32>) outs(%44 : tensor<256x32x2x16x32x128xf16>) {
    ^bb0(%arg9: f16, %arg10: f16):
      iree_linalg_ext.yield %arg9 : f16
    } -> tensor<256x32x2x16x32x128xf16>
    %collapsed_60 = tensor.collapse_shape %47 [[0], [1, 2, 3, 4, 5]] : tensor<256x32x2x16x32x128xf16> into tensor<256x4194304xf16>
    %48 = tensor.empty() : tensor<4x4xi64>
    %49 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%4 : tensor<4x4xi64>) outs(%48 : tensor<4x4xi64>) {
    ^bb0(%in: i64, %out: i64):
      %66 = arith.muli %in, %c64_i64 : i64
      linalg.yield %66 : i64
    } -> tensor<4x4xi64>
    %collapsed_61 = tensor.collapse_shape %49 [[0, 1]] : tensor<4x4xi64> into tensor<16xi64>
    %collapsed_62 = tensor.collapse_shape %47 [[0, 1, 2], [3], [4], [5]] : tensor<256x32x2x16x32x128xf16> into tensor<16384x16x32x128xf16>
    %50 = tensor.empty() : tensor<16x16x32x128xf16>
    %51 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%collapsed_61 : tensor<16xi64>) outs(%50 : tensor<16x16x32x128xf16>) {
    ^bb0(%in: i64, %out: f16):
      %66 = arith.index_cast %in : i64 to index
      %67 = linalg.index 1 : index
      %68 = linalg.index 2 : index
      %69 = linalg.index 3 : index
      %extracted = tensor.extract %collapsed_62[%66, %67, %68, %69] : tensor<16384x16x32x128xf16>
      linalg.yield %extracted : f16
    } -> tensor<16x16x32x128xf16>
    %expanded_63 = tensor.expand_shape %51 [[0, 1], [2], [3], [4]] output_shape [4, 4, 16, 32, 128] : tensor<16x16x32x128xf16> into tensor<4x4x16x32x128xf16>
    %collapsed_64 = tensor.collapse_shape %expanded_63 [[0], [1, 2], [3], [4]] : tensor<4x4x16x32x128xf16> into tensor<4x64x32x128xf16>
    %52 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%49 : tensor<4x4xi64>) outs(%48 : tensor<4x4xi64>) {
    ^bb0(%in: i64, %out: i64):
      %66 = arith.addi %in, %c1_i64 : i64
      linalg.yield %66 : i64
    } -> tensor<4x4xi64>
    %collapsed_65 = tensor.collapse_shape %52 [[0, 1]] : tensor<4x4xi64> into tensor<16xi64>
    %53 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%collapsed_65 : tensor<16xi64>) outs(%50 : tensor<16x16x32x128xf16>) {
    ^bb0(%in: i64, %out: f16):
      %66 = arith.index_cast %in : i64 to index
      %67 = linalg.index 1 : index
      %68 = linalg.index 2 : index
      %69 = linalg.index 3 : index
      %extracted = tensor.extract %collapsed_62[%66, %67, %68, %69] : tensor<16384x16x32x128xf16>
      linalg.yield %extracted : f16
    } -> tensor<16x16x32x128xf16>
    %expanded_66 = tensor.expand_shape %53 [[0, 1], [2], [3], [4]] output_shape [4, 4, 16, 32, 128] : tensor<16x16x32x128xf16> into tensor<4x4x16x32x128xf16>
    %collapsed_67 = tensor.collapse_shape %expanded_66 [[0], [1, 2], [3], [4]] : tensor<4x4x16x32x128xf16> into tensor<4x64x32x128xf16>
    %54 = tensor.empty() : tensor<4x32x1x128xf16>
    %55 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d2, d1, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%0 : tensor<4x1x32x128xf16>) outs(%54 : tensor<4x32x1x128xf16>) {
    ^bb0(%in: f16, %out: f16):
      linalg.yield %in : f16
    } -> tensor<4x32x1x128xf16>
    %56 = tensor.empty() : tensor<4x32x64x128xf16>
    %57 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d2, d1, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%collapsed_64 : tensor<4x64x32x128xf16>) outs(%56 : tensor<4x32x64x128xf16>) {
    ^bb0(%in: f16, %out: f16):
      linalg.yield %in : f16
    } -> tensor<4x32x64x128xf16>
    %58 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d2, d1, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%collapsed_67 : tensor<4x64x32x128xf16>) outs(%56 : tensor<4x32x64x128xf16>) {
    ^bb0(%in: f16, %out: f16):
      linalg.yield %in : f16
    } -> tensor<4x32x64x128xf16>
    %collapsed_68 = tensor.collapse_shape %55 [[0, 1], [2], [3]] : tensor<4x32x1x128xf16> into tensor<128x1x128xf16>
    %collapsed_69 = tensor.collapse_shape %57 [[0, 1], [2], [3]] : tensor<4x32x64x128xf16> into tensor<128x64x128xf16>
    %collapsed_70 = tensor.collapse_shape %58 [[0, 1], [2], [3]] : tensor<4x32x64x128xf16> into tensor<128x64x128xf16>
    %59 = tensor.empty() : tensor<128x1x128xf16>
    %60 = iree_linalg_ext.attention {indexing_maps = [affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2)>, affine_map<(d0, d1, d2, d3, d4) -> (d0, d3, d2)>, affine_map<(d0, d1, d2, d3, d4) -> (d0, d3, d4)>, affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d4)>]} ins(%collapsed_68, %collapsed_69, %collapsed_70, %cst_1 : tensor<128x1x128xf16>, tensor<128x64x128xf16>, tensor<128x64x128xf16>, f16) outs(%59 : tensor<128x1x128xf16>) -> tensor<128x1x128xf16>
    %expanded_71 = tensor.expand_shape %60 [[0, 1], [2], [3]] output_shape [4, 32, 1, 128] : tensor<128x1x128xf16> into tensor<4x32x1x128xf16>
    %61 = tensor.empty() : tensor<4x1x32x128xf16>
    %62 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d2, d1, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%expanded_71 : tensor<4x32x1x128xf16>) outs(%61 : tensor<4x1x32x128xf16>) {
    ^bb0(%in: f16, %out: f16):
      linalg.yield %in : f16
    } -> tensor<4x1x32x128xf16>
    %collapsed_72 = tensor.collapse_shape %62 [[0], [1], [2, 3]] : tensor<4x1x32x128xf16> into tensor<4x1x4096xf16>
    %63 = hal.tensor.alias wait(%arg7) => %collapsed_60 : tensor<256x4194304xf16> to %arg6 : !hal.buffer_view
    %64:2 = hal.tensor.barrier join(%63, %collapsed_72 : tensor<256x4194304xf16>, tensor<4x1x4096xf16>) => %arg8 : !hal.fence
    %65 = hal.tensor.export %64#1 : tensor<4x1x4096xf16> -> !hal.buffer_view
    util.return %65 : !hal.buffer_view
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