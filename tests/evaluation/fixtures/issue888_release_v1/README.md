# Issue 888 release readback fixture

These three compressed files are byte-for-byte copies of artificial artifacts from the [Issue 810 development checkpoint release](https://github.com/shuntatsu/trade_rl/releases/tag/research/issue810-development-checkpoint-20261015). The release ZIP SHA-256 is `03bcbd613a516e9e0504d5c074297a896d3e680d87c29adcdbc2fad2d4dc1556`; its source HEAD is `30c5ab54fdb9db7f03b947c1687a09c8b929c461`.

The release `MANIFEST.json` SHA-256 is `9e1c4ff1d0f30e9fcb98fc915edcf10348e8dfdce1a04a14b1b7acabe312935e`.

Fixture file SHA-256 values:

- `native-plan.json.gz`: `ad4b09aade180fbf69d7c53d4f308fe374835879b18a20684a7ae906c181e9e0`
- `ARTIFICIAL_H72_A10_A_BASEL2_nonrl.json.gz`: `7c7a967630816479c3ce351a243ef7da8b451975f063a3da515ff465f3b7cb1e`
- `ARTIFICIAL_H72_A10_A_BASEL2_cash.json.gz`: `d7bd688d6e3190f116fa0d06385e24f6b44ec2541a69c3f5f66034affee68d4d`

The test uses public readers to check artificial payload readback and canonical-byte preservation. This is not current-source execution equivalence, research approval, or live-trading authority.
