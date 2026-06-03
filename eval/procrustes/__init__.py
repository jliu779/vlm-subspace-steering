"""ProcrustesRotation: low-rank subspace rotation as an extension of CMRM.

Given paired (text-only, text+corrupted-image) anchor hidden states, this
package fits a per-layer orthogonal Procrustes map from the corrupted-multimodal
subspace to the text-only subspace, and applies it to the last input token at
inference time.

Math reference: docs/procrustes_rotation_centes.md.

Geometric caveats from analysis (recorded so the implementation does not
silently hide them):

  1. B_c and B_t are SVD'd independently. They span generally *different*
     subspaces, so `h - p_c + p_t` mixes "subtract from B_c" with "add into
     B_t". When the two subspaces are not the same, the operation does not
     conserve norm. We expose `subspace_mode={split, shared}` to also support a
     joint-SVD shared basis, which we believe is geometrically cleaner.

  2. Q jointly absorbs (a) within-subspace pose alignment, and (b) PC
     ordering / sign mismatch between B_c and B_t. With a shared basis, only
     (a) remains, which is what "rotation" should mean.

  3. The hook fires on prefill_only by default (the LLaVA generation collapses
     into degenerate-token loops if the same correction is applied at every
     decode step; this is the same bug that broke our earlier CMRM runs).
"""
