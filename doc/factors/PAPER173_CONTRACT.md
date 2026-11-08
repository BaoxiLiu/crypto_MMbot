# Paper173 reference contract v1

Status: implemented reference; not yet a trained model or profitable strategy.
Origin skill: academic-research-suite / experiment-agent; mode: implementation.

Source: arXiv2502.18625v2 §7.2 and Appendix B. Names/order are frozen to the
173-row appendix audit. The paper does not specify all numeric conventions;
these choices are explicit engineering interpretations, not author source code.

- All history uses LOCAL integer nanoseconds, complete timestamp batches.
  Windows are (t-(k+1)L,t-kL], k=0,1,2. Epoch gaps reset coverage.
- Buy/sell is normalized to same/opposite aggressor relative to candidate order;
  near/opposite depth follows the order direction. Returns use buy-up/sell-down
  positive; the author's printed label ratio/sign ambiguity is not copied.
- amplitude is max-min trade price; sizes/volumes are base-contract quantities.
  VWAP is all trades weighted by base quantity vs current near-side touch.
- ret_sum uses consecutive trades both inside the window; no artificial zero
  return for a single trade. ret_autocov is mean of consecutive centered return
  products with a common in-window mean, population lag-1 covariance. It needs
  at least 3 trades. trade_intensity is average within-window interval in seconds.
- No-event counts/quantities are 0 only after full epoch coverage. Missing price
  statistics/insufficient history remain NaN. Zero observed price variation is 0.
- stdev_100/500 use 101/501 as-of last trade prices on UTC ten-second endpoints,
  sample std ddof=1 of log returns. No next-trade interpolation.
- L1 liquidity is base quantity. $500K impact walks full observed bid/ask ladders,
  splits the last level to reach fixed quote notional, and measures VWAP distance
  relative to that side's best price. Insufficient depth => NaN.
- age uses the last observed change in either best bid/ask price. Initial
  snapshot age is unknown until a genuine price change. totb_mean uses completed
  price spells ending in (t-600s,t], with known preceding changes. It needs full
  ten-minute observation coverage; current unfinished spell is represented by age.
- Reference query excludes future trade/BBO observations. The replay adapter
  must additionally prove the supplied full depth belongs to the same as-of time.
- Existing202 will be normalized for candidate direction before merging.
  Semantically exact duplicates can be dropped; correlated proxies are retained.
  The present fixed bid-oriented 202-vector is not already order-normalized.

This implementation is a correctness oracle. Monthly extraction must use an
optimized implementation checked against it; no claim that Python meets 100us.
