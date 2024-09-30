
!q_type = tensor<?x?x16x64xf32>
!k_type = tensor<?x?x32x64xf32>
!v_type = tensor<?x?x32x48xf32>
!o_type = tensor<?x?x16x48xf32>
!s_type = tensor<{{scale_type}}>

util.func private @flash_attention(
    %q: !q_type, %k: !k_type, %v: !v_type, %s: !s_type) -> !o_type {

    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %b0 = tensor.dim %q, %c0 : !q_type
    %b1 = tensor.dim %q, %c1 : !q_type

    %scale = tensor.extract %s[] : !s_type

    %empty = tensor.empty(%b0, %b1) : !o_type
    %atten = iree_linalg_ext.attention {indexing_maps = [
                affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d2, d3)>,
                affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d4, d3)>,
                affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d4, d5)>,
                affine_map<(d0, d1, d2, d3, d4, d5) -> ()>,
                affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d2, d5)>]}
                ins(%q, %k, %v, %scale : !q_type, !k_type, !v_type, {{scale_type}}) outs(%empty : !o_type) -> !o_type
    util.return %atten : !o_type
}
