# Implementation checks

Before any behavioral outcome, smoke-01 stopped at a full-sequence versus
chunked BF16 logit comparison. Different GEMM shapes produced differences
above a tight elementwise tolerance. The gate now compares the custom
unquantized cache to stock DynamicCache with identical chunk shapes, exactly,
and checks causal isolation by changing future tokens with shape held fixed.
This strengthens the intended implementation check without selecting on task
accuracy. The failed smoke manifest is preserved. No held-out result was read.
