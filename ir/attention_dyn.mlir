
!q_type = tensor<?x?x?x64xf32>
!k_type = tensor<?x?x?x64xf32>
!v_type = tensor<?x?x?x48xf32>
!o_type = tensor<?x?x?x48xf32>

util.func private @flash_attention(%q: !q_type, %k: !k_type, %v: !v_type) -> !o_type {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index

    %b0 = tensor.dim %q, %c0 : !q_type
    %b1 = tensor.dim %q, %c1 : !q_type
    %sl = tensor.dim %q, %c2 : !q_type

    %empty = tensor.empty(%b0, %b1, %sl) : !o_type
    %atten = iree_linalg_ext.attention {indexing_maps = [
                affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d2, d3)>,
                affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d4, d3)>,
                affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d4, d5)>,
                affine_map<(d0, d1, d2, d3, d4, d5) -> ()>,
                affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d2, d5)>]}
                ins(%q, %k, %v, %scale : !q_type, !k_type, !v_type, {{scale_type}}) outs(%empty : !o_type) -> !o_type
    util.return %atten : !o_type
}
