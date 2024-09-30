!q_type = tensor<?x1024x64xf32>
!k_type = tensor<?x2048x64xf32>
!v_type = tensor<?x2048x48xf32>
!o_type = tensor<?x1024x48xf32>
!m_type = tensor<?x1024xf32>
!s_type = tensor<?x1024xf32>

#map = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d0, d1)>
#map2 = affine_map<(d0, d1) -> (d0, d1)>

func.func @custom_op_symbolic_dims(
        %q : !q_type,
        %k : !k_type,
        %v : !v_type,
        %o : !o_type) -> (!o_type) {
  %c0 = arith.constant 0 : index
  %bs = tensor.dim %q, %c0 : !q_type

  %zero = arith.constant 0.000000e+00 : f32
  %ninf = arith.constant 0xFF800000 : f32

  %mx_empty = tensor.empty(%bs) : !m_type
  %sm_empty = tensor.empty(%bs) : !s_type

  %max = linalg.fill ins(%ninf : f32) outs(%mx_empty : tensor<?x1024xf32>) -> tensor<?x1024xf32>
  %sum = linalg.fill ins(%zero : f32) outs(%sm_empty : tensor<?x1024xf32>) -> tensor<?x1024xf32>

  %0 = iree_linalg_ext.custom_op {
        indexing_maps = [affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d4)>,
                         affine_map<(d0, d1, d2, d3, d4) -> (d0, d3, d4)>,
                         affine_map<(d0, d1, d2, d3, d4) -> (d0, d3, d2)>,
                         affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2)>,
                         affine_map<(d0, d1, d2, d3, d4) -> (d0, d1)>,
                         affine_map<(d0, d1, d2, d3, d4) -> (d0, d1)>],
        iterator_types = [#iree_linalg_ext.iterator_type<parallel>,
                          #iree_linalg_ext.iterator_type<parallel>,
                          #iree_linalg_ext.iterator_type<reduction>,
                          #iree_linalg_ext.iterator_type<reduction>,
                          #iree_linalg_ext.iterator_type<reduction>]}
        ins(%q, %k, %v : !q_type, !k_type, !v_type)
        outs(%o, %max, %sum : !o_type, !m_type, !s_type) {
      ^bb0(%qt : tensor<1x1024x64xf32>, %kt : tensor<1x32x64xf32>, %vt : tensor<1x32x64xf32>, %ot : tensor<1x1024x64xf32>, %maxa : tensor<1x1024xf32>, %suma : tensor<1x1024xf32>) :

      // Apply pre-scaling to q
      %q_scale_empty = tensor.empty() : tensor<1x1024x64xf32>

      %q_scale = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%qt : tensor<1x1024x64xf32>) outs(%q_scale_empty : tensor<1x1024x64xf32>) {
      ^bb0(%in: f32, %out: f32):
        %scalef32 = arith.constant 1.44269502 : f32
        %13 = arith.mulf %in, %scalef32 : f32
        linalg.yield %13 : f32
      } -> tensor<1x1024x64xf32>

      // Compute q@k matmul
      %3 = tensor.empty() : tensor<1x1024x32xf32>
      %4 = linalg.fill ins(%zero : f32) outs(%3 : tensor<1x1024x32xf32>) -> tensor<1x1024x32xf32>
      %qk = linalg.matmul_transpose_b ins(%q_scale, %kt : tensor<1x1024x64xf32>, tensor<1x32x64xf32>) outs(%4 : tensor<1x1024x32xf32>) -> tensor<1x1024x32xf32>


      // Compute a new running max
      %new_max = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "parallel", "reduction"]} ins(%qk : tensor<1x1024x32xf32>) outs(%maxa : tensor<1x1024xf32>) {
      ^bb0(%in: f32, %out: f32):
        %13 = arith.maximumf %in, %out : f32
        linalg.yield %13 : f32
      } -> tensor<1x1024xf32>


      // Update the qk using the new max
      %new_qk = linalg.generic {indexing_maps = [#map1, #map], iterator_types = ["parallel", "parallel"]} ins(%new_max : tensor<1x1024xf32>) outs(%qk : tensor<1x1024x32xf32>) {
      ^bb0(%in: f32, %out: f32):
        %13 = arith.subf %out, %in : f32
        %14 = math.exp2 %13 : f32
        linalg.yield %14 : f32
      } -> tensor<1x1024x32xf32>

      // Compute the new max normalization
      %max_norm = linalg.generic {indexing_maps = [#map2, #map2], iterator_types = ["parallel"]} ins(%new_max : tensor<1x1024xf32>) outs(%maxa : tensor<1x1024xf32>) {
      ^bb0(%in: f32, %out: f32):
        %13 = arith.subf %out, %in : f32
        %14 = math.exp2 %13 : f32
        linalg.yield %14 : f32
      } -> tensor<1x1024xf32>


      %sum_norm = linalg.generic {indexing_maps = [#map2, #map2], iterator_types = ["parallel"]} ins(%max_norm : tensor<1x1024xf32>) outs(%suma : tensor<1x1024xf32>) {
      ^bb0(%in: f32, %out: f32):
        %13 = arith.mulf %in, %out : f32
        linalg.yield %13 : f32
      } -> tensor<1x1024xf32>

      %new_sum = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "reduction"]} ins(%new_qk : tensor<1x1024x32xf32>) outs(%sum_norm : tensor<1x1024xf32>) {
      ^bb0(%in: f32, %out: f32):
        %13 = arith.addf %in, %out : f32
        linalg.yield %13 : f32
      } -> tensor<1x1024xf32>

      // Renormalize the output tile
      %norm_ot = linalg.generic {indexing_maps = [#map1, #map], iterator_types = ["parallel", "parallel"]} ins(%max_norm : tensor<1x1024xf32>) outs(%ot : tensor<1x1024x64xf32>) {
      ^bb0(%in: f32, %out: f32):
        %13 = arith.mulf %in, %out : f32
        linalg.yield %13 : f32
      } -> tensor<1x1024x64xf32>

      // Perform the second matmul for hte output tile
      %new_ot = linalg.matmul ins(%new_qk, %vt : tensor<1x1024x32xf32>, tensor<1x32x64xf32>) outs(%norm_ot : tensor<1x1024x64xf32>) -> tensor<1x1024x64xf32>

      iree_linalg_ext.yield %new_ot : tensor<1x1024x64xf32>
    } -> tensor<?x1024x48xf32>
  return %0 : tensor<?x1024x48xf32>
}
