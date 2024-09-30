
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
                ins(%q, %k, %v, %scale : !q_type, !k_type, !v_type, {{scale_type}}) {
        ^bb0(%score : f32, %batch : index, %head : index, %q_idx : index, %kv_idx : index) :
             // default implementation:
            return linalg_ext.yield %score : f32

            // causal masking:
            // %ninf = arith.constant 0xFF800000 : f32
            // %pred = arith.cmpi ult %q_idx, %kv_idx : index
            // %sel = arith.select %pred %score, %ninf : f32
            // linalg_ext.yield %sel : f32

            // soft cap
            // %cap = arith.constant 0.1 : f32
            // %div = arith.divf %score, %cap : f32
            // %tanh = math.tanh %div : f32
            // %mul = arith.mulf %tanh, %cap : f32
            // linalg_ext.yield %mul : f32
    } outs(%empty : !o_type) -> !o_type
    util.return %atten : !o_type
}
