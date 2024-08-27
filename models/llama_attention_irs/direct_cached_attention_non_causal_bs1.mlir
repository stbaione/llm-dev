module @module {
  util.func public @sdpa1$async(%arg0: !hal.buffer_view, %arg1: !hal.buffer_view, %arg2: !hal.buffer_view, %arg3: !hal.buffer_view, %arg4: !hal.buffer_view, %arg5: !hal.fence, %arg6: !hal.fence) -> !hal.buffer_view attributes {inlining_policy = #util.inline.never, iree.abi.model = "coarse-fences", iree.abi.stub} {
    %cst = arith.constant 8.837890e-02 : f16
    %0 = hal.tensor.import wait(%arg5) => %arg0 : !hal.buffer_view -> tensor<1x1x32x128xf16>
    %1 = hal.tensor.import wait(%arg5) => %arg1 : !hal.buffer_view -> tensor<1x1x32x128xf16>
    %2 = hal.tensor.import wait(%arg5) => %arg2 : !hal.buffer_view -> tensor<1x1x32x128xf16>
    %3 = hal.tensor.import wait(%arg5) => %arg3 : !hal.buffer_view -> tensor<1x4096x32x128xf16>
    %4 = hal.tensor.import wait(%arg5) => %arg4 : !hal.buffer_view -> tensor<1x4096x32x128xf16>
    %inserted_slice = tensor.insert_slice %1 into %3[0, 0, 0, 0] [1, 1, 32, 128] [1, 1, 1, 1] : tensor<1x1x32x128xf16> into tensor<1x4096x32x128xf16>
    %inserted_slice_0 = tensor.insert_slice %2 into %4[0, 0, 0, 0] [1, 1, 32, 128] [1, 1, 1, 1] : tensor<1x1x32x128xf16> into tensor<1x4096x32x128xf16>
    %5 = tensor.empty() : tensor<1x32x1x128xf16>
    %6 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d2, d1, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%0 : tensor<1x1x32x128xf16>) outs(%5 : tensor<1x32x1x128xf16>) {
    ^bb0(%in: f16, %out: f16):
      linalg.yield %in : f16
    } -> tensor<1x32x1x128xf16>
    %7 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d2, d1, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1 : tensor<1x1x32x128xf16>) outs(%5 : tensor<1x32x1x128xf16>) {
    ^bb0(%in: f16, %out: f16):
      linalg.yield %in : f16
    } -> tensor<1x32x1x128xf16>
    %8 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d2, d1, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%2 : tensor<1x1x32x128xf16>) outs(%5 : tensor<1x32x1x128xf16>) {
    ^bb0(%in: f16, %out: f16):
      linalg.yield %in : f16
    } -> tensor<1x32x1x128xf16>
    %collapsed = tensor.collapse_shape %6 [[0, 1], [2], [3]] : tensor<1x32x1x128xf16> into tensor<32x1x128xf16>
    %collapsed_1 = tensor.collapse_shape %7 [[0, 1], [2], [3]] : tensor<1x32x1x128xf16> into tensor<32x1x128xf16>
    %collapsed_2 = tensor.collapse_shape %8 [[0, 1], [2], [3]] : tensor<1x32x1x128xf16> into tensor<32x1x128xf16>
    %9 = tensor.empty() : tensor<32x1x128xf16>
    %10 = iree_linalg_ext.attention {indexing_maps = [affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2)>, affine_map<(d0, d1, d2, d3, d4) -> (d0, d3, d2)>, affine_map<(d0, d1, d2, d3, d4) -> (d0, d3, d4)>, affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d4)>]} ins(%collapsed, %collapsed_1, %collapsed_2, %cst : tensor<32x1x128xf16>, tensor<32x1x128xf16>, tensor<32x1x128xf16>, f16) outs(%9 : tensor<32x1x128xf16>) -> tensor<32x1x128xf16>
    %expanded = tensor.expand_shape %10 [[0, 1], [2], [3]] output_shape [1, 32, 1, 128] : tensor<32x1x128xf16> into tensor<1x32x1x128xf16>
    %11 = tensor.empty() : tensor<1x1x32x128xf16>
    %12 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d2, d1, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%expanded : tensor<1x32x1x128xf16>) outs(%11 : tensor<1x1x32x128xf16>) {
    ^bb0(%in: f16, %out: f16):
      linalg.yield %in : f16
    } -> tensor<1x1x32x128xf16>
    %collapsed_3 = tensor.collapse_shape %12 [[0], [1], [2, 3]] : tensor<1x1x32x128xf16> into tensor<1x1x4096xf16>
    %13 = hal.tensor.alias wait(%arg5) => %inserted_slice : tensor<1x4096x32x128xf16> to %arg3 : !hal.buffer_view
    %14 = hal.tensor.alias wait(%arg5) => %inserted_slice_0 : tensor<1x4096x32x128xf16> to %arg4 : !hal.buffer_view
    %15:3 = hal.tensor.barrier join(%13, %14, %collapsed_3 : tensor<1x4096x32x128xf16>, tensor<1x4096x32x128xf16>, tensor<1x1x4096xf16>) => %arg6 : !hal.fence
    %16 = hal.tensor.export %15#2 : tensor<1x1x4096xf16> -> !hal.buffer_view
    util.return %16 : !hal.buffer_view
  }
  util.func public @sdpa1(%arg0: !hal.buffer_view, %arg1: !hal.buffer_view, %arg2: !hal.buffer_view, %arg3: !hal.buffer_view, %arg4: !hal.buffer_view) -> !hal.buffer_view attributes {iree.abi.stub} {
    %0 = util.null : !hal.fence
    %c-1_i32 = arith.constant -1 : i32
    %c0 = arith.constant 0 : index
    %device_0 = hal.devices.get %c0 : !hal.device
    %fence = hal.fence.create device(%device_0 : !hal.device) flags("None") : !hal.fence
    %1 = util.call @sdpa1$async(%arg0, %arg1, %arg2, %arg3, %arg4, %0, %fence) : (!hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.fence, !hal.fence) -> !hal.buffer_view
    %status = hal.fence.await until([%fence]) timeout_millis(%c-1_i32) : i32
    util.return %1 : !hal.buffer_view
  }
}
