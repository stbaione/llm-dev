module @module {
  func.func @sdpa1(%arg0: !torch.vtensor<[1,1,32,128],f16>, %arg1: !torch.vtensor<[1,1,32,128],f16>, %arg2: !torch.vtensor<[1,1,32,128],f16>, %arg3: !torch.tensor<[1,4096,32,128],f16>, %arg4: !torch.tensor<[1,4096,32,128],f16>) -> !torch.vtensor<[1,1,4096],f16> attributes {torch.assume_strict_symbolic_shapes} {
    %0 = torch.copy.to_vtensor %arg3 : !torch.vtensor<[1,4096,32,128],f16>
    %1 = torch.copy.to_vtensor %arg4 : !torch.vtensor<[1,4096,32,128],f16>
    %int1 = torch.constant.int 1
    %int0 = torch.constant.int 0
    %int1_0 = torch.constant.int 1
    %int1_1 = torch.constant.int 1
    %2 = torch.aten.slice.Tensor %0, %int1, %int0, %int1_0, %int1_1 : !torch.vtensor<[1,4096,32,128],f16>, !torch.int, !torch.int, !torch.int, !torch.int -> !torch.vtensor<[1,1,32,128],f16>
    %false = torch.constant.bool false
    %3 = torch.aten.copy %2, %arg1, %false : !torch.vtensor<[1,1,32,128],f16>, !torch.vtensor<[1,1,32,128],f16>, !torch.bool -> !torch.vtensor<[1,1,32,128],f16>
    %int1_2 = torch.constant.int 1
    %int0_3 = torch.constant.int 0
    %int1_4 = torch.constant.int 1
    %int1_5 = torch.constant.int 1
    %4 = torch.aten.slice_scatter %0, %3, %int1_2, %int0_3, %int1_4, %int1_5 : !torch.vtensor<[1,4096,32,128],f16>, !torch.vtensor<[1,1,32,128],f16>, !torch.int, !torch.int, !torch.int, !torch.int -> !torch.vtensor<[1,4096,32,128],f16>
    torch.overwrite.tensor.contents %4 overwrites %arg3 : !torch.vtensor<[1,4096,32,128],f16>, !torch.tensor<[1,4096,32,128],f16>
    %int1_6 = torch.constant.int 1
    %int0_7 = torch.constant.int 0
    %int1_8 = torch.constant.int 1
    %int1_9 = torch.constant.int 1
    %5 = torch.aten.slice.Tensor %1, %int1_6, %int0_7, %int1_8, %int1_9 : !torch.vtensor<[1,4096,32,128],f16>, !torch.int, !torch.int, !torch.int, !torch.int -> !torch.vtensor<[1,1,32,128],f16>
    %false_10 = torch.constant.bool false
    %6 = torch.aten.copy %5, %arg2, %false_10 : !torch.vtensor<[1,1,32,128],f16>, !torch.vtensor<[1,1,32,128],f16>, !torch.bool -> !torch.vtensor<[1,1,32,128],f16>
    %int1_11 = torch.constant.int 1
    %int0_12 = torch.constant.int 0
    %int1_13 = torch.constant.int 1
    %int1_14 = torch.constant.int 1
    %7 = torch.aten.slice_scatter %1, %6, %int1_11, %int0_12, %int1_13, %int1_14 : !torch.vtensor<[1,4096,32,128],f16>, !torch.vtensor<[1,1,32,128],f16>, !torch.int, !torch.int, !torch.int, !torch.int -> !torch.vtensor<[1,4096,32,128],f16>
    torch.overwrite.tensor.contents %7 overwrites %arg4 : !torch.vtensor<[1,4096,32,128],f16>, !torch.tensor<[1,4096,32,128],f16>
    %int1_15 = torch.constant.int 1
    %int2 = torch.constant.int 2
    %8 = torch.aten.transpose.int %arg0, %int1_15, %int2 : !torch.vtensor<[1,1,32,128],f16>, !torch.int, !torch.int -> !torch.vtensor<[1,32,1,128],f16>
    %int1_16 = torch.constant.int 1
    %int0_17 = torch.constant.int 0
    %int1_18 = torch.constant.int 1
    %int1_19 = torch.constant.int 1
    %9 = torch.aten.slice.Tensor %4, %int1_16, %int0_17, %int1_18, %int1_19 : !torch.vtensor<[1,4096,32,128],f16>, !torch.int, !torch.int, !torch.int, !torch.int -> !torch.vtensor<[1,1,32,128],f16>
    %int1_20 = torch.constant.int 1
    %int2_21 = torch.constant.int 2
    %10 = torch.aten.transpose.int %9, %int1_20, %int2_21 : !torch.vtensor<[1,1,32,128],f16>, !torch.int, !torch.int -> !torch.vtensor<[1,32,1,128],f16>
    %int1_22 = torch.constant.int 1
    %int0_23 = torch.constant.int 0
    %int1_24 = torch.constant.int 1
    %int1_25 = torch.constant.int 1
    %11 = torch.aten.slice.Tensor %7, %int1_22, %int0_23, %int1_24, %int1_25 : !torch.vtensor<[1,4096,32,128],f16>, !torch.int, !torch.int, !torch.int, !torch.int -> !torch.vtensor<[1,1,32,128],f16>
    %int1_26 = torch.constant.int 1
    %int2_27 = torch.constant.int 2
    %12 = torch.aten.transpose.int %11, %int1_26, %int2_27 : !torch.vtensor<[1,1,32,128],f16>, !torch.int, !torch.int -> !torch.vtensor<[1,32,1,128],f16>
    %float0.000000e00 = torch.constant.float 0.000000e+00
    %true = torch.constant.bool true
    %none = torch.constant.none
    %none_28 = torch.constant.none
    %13:2 = torch.operator "torch.aten._scaled_dot_product_flash_attention_for_cpu"(%8, %10, %12, %float0.000000e00, %true, %none, %none_28) : (!torch.vtensor<[1,32,1,128],f16>, !torch.vtensor<[1,32,1,128],f16>, !torch.vtensor<[1,32,1,128],f16>, !torch.float, !torch.bool, !torch.none, !torch.none) -> (!torch.vtensor<[1,32,1,128],f16>, !torch.vtensor<[1,32,1],f32>) 
    %int1_29 = torch.constant.int 1
    %int2_30 = torch.constant.int 2
    %14 = torch.aten.transpose.int %13#0, %int1_29, %int2_30 : !torch.vtensor<[1,32,1,128],f16>, !torch.int, !torch.int -> !torch.vtensor<[1,1,32,128],f16>
    %int1_31 = torch.constant.int 1
    %int1_32 = torch.constant.int 1
    %int-1 = torch.constant.int -1
    %15 = torch.prim.ListConstruct %int1_31, %int1_32, %int-1 : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %16 = torch.aten.view %14, %15 : !torch.vtensor<[1,1,32,128],f16>, !torch.list<int> -> !torch.vtensor<[1,1,4096],f16>
    return %16 : !torch.vtensor<[1,1,4096],f16>
  }
}
