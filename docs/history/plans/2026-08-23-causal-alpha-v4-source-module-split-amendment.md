# Causal Alpha V4 Source Module Split Plan Amendment

This is a pre-GREEN responsibility correction for Task 3. It does not alter any scientific, timing, feature, or Gate contract.

The Task 3 RED showed that the next boundary is decision-clock assembly, while `trade_rl/integrations/binance_v4_context.py` already owns immutable archive parsing. To keep files single-purpose, use:

```text
trade_rl/integrations/binance_v4_context.py
  -> official Binance Vision URL contracts and ZIP/CSV parsers

trade_rl/integrations/binance_v4_context_assembly.py
  -> exact 15m decision-clock alignment, funding-event placement,
     mark-price/perpetual/Spot source combination, and V4CrossMarketInputs assembly
```

Tests keep the same behavioral assertions; only assembly symbols are imported from the assembly module. No threshold, source, label, model, or economic parameter changes.
