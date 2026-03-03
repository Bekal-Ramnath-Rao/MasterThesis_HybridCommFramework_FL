## Communication Model: Analytical Formulas (Diagnostic Pipeline)

This document lists the **exact formulas** used in the diagnostic pipeline to compute predicted round time **T_calc** per protocol. These match the implementation in `Network_Simulation/diagnostic_pipeline.py`.

---

## Table of Contents

1. Notation and Parameters
2. Shared Definitions
3. MQTT
4. AMQP
5. gRPC
6. QUIC
7. HTTP/3
8. DDS
9. Constants Reference
10. Summary Table

---

## Notation and Parameters

| Symbol | Meaning | Unit |
|--------|---------|------|
| **T_calc** | Predicted total round time (client send → server process → client receives) | s |
| **S** | Payload size (model update) | bits |
| **B** | Bandwidth (bottleneck link, measured via iperf3 when available) | bps |
| **RTT_eff** | Effective RTT (delay + jitter from tc/netem or iperf3) | s |
| **p** | Packet loss rate (decimal, e.g. 0.01 = 1%) | — |
| **O_app** | Application + broker overhead (measured in calibration) | s |
| **O_TLS** | TLS 1.3 encryption overhead (QUIC/HTTP3 only) | s |
| **MSS** | Maximum Segment Size (bits) | bits |
| **B_eff** | Effective bandwidth under loss (TCP-like protocols) | bps |
| **T_REPAIR** | Extra time per loss repair (DDS/TCP-like) | s |
| **α_proto** | Protocol-specific scaling factor applied to the network term | — |

---

## Shared Definitions

### Transfer time (wire)

\[
T_{\mathrm{transfer}}(S,\, B) = \frac{S}{B} \quad \text{(seconds)}
\]

(If \(B \leq 0\) or not finite, treat as 0.)

### Effective RTT (from tc)

\[
\mathrm{RTT}_{\mathrm{eff}} = D_{\mathrm{tc}} + J
\]

- \(D_{\mathrm{tc}}\): one-way delay from `tc`
- \(J\): jitter from `tc` (used as additive to RTT here)

### Application overhead

\[
O_{\mathrm{app}} = O_{\mathrm{send}} + O_{\mathrm{recv}} + O_{\mathrm{broker}}
\]

- \(O_{\mathrm{send}}\), \(O_{\mathrm{recv}}\): measured in Phase 1 (calibration)
- \(O_{\mathrm{broker}}\): broker/stack residual from baseline round

---

## MQTT

**Effective bandwidth** (loss-aware, square-root rule):

\[
B_{\mathrm{eff}} =
\begin{cases}
B & \text{if } p = 0 \\
\min\left(B,\; \dfrac{\mathrm{MSS}}{\mathrm{RTT}_{\mathrm{eff}} \sqrt{p}}\right) & \text{if } p > 0,\ \mathrm{RTT}_{\mathrm{eff}} > 0 \\
B & \text{otherwise}
\end{cases}
\]

**Predicted round time:**

\[
\boxed{
T_{\mathrm{calc}}^{\mathrm{MQTT}}
:= 2 \cdot \mathrm{RTT}_{\mathrm{eff}}
+ \frac{S}{B_{\mathrm{eff}}}
+ O_{\mathrm{app}}
}
\]

- \(2 \cdot \mathrm{RTT}_{\mathrm{eff}}\): TCP handshake + CONNECT/CONNACK + PUBLISH/PUBACK (≈2 RTTs in slow networks)
- \(S/B_{\mathrm{eff}}\): transfer time under loss-limited bandwidth
- \(O_{\mathrm{app}}\): protocol + broker overhead

---

## AMQP

Same structure as MQTT: loss-limited effective bandwidth and a TCP-like RTT term, but using \(1.5 \cdot \mathrm{RTT}_{\mathrm{eff}}\) instead of \(2 \cdot \mathrm{RTT}_{\mathrm{eff}}\).

\[
B_{\mathrm{eff}} =
\begin{cases}
B & \text{if } p = 0 \\
\min\left(B,\; \dfrac{\mathrm{MSS}}{\mathrm{RTT}_{\mathrm{eff}} \sqrt{p}}\right) & \text{if } p > 0,\ \mathrm{RTT}_{\mathrm{eff}} > 0 \\
B & \text{otherwise}
\end{cases}
\]

\[
\boxed{
T_{\mathrm{calc}}^{\mathrm{AMQP}}
:= 1.5 \cdot \mathrm{RTT}_{\mathrm{eff}}
+ \frac{S}{B_{\mathrm{eff}}}
+ O_{\mathrm{app}}
}
\]

---

## gRPC

Same as AMQP (TCP/HTTP2, loss-limited bandwidth with \(1.5 \cdot \mathrm{RTT}_{\mathrm{eff}}\) RTT term).

\[
\boxed{
T_{\mathrm{calc}}^{\mathrm{gRPC}}
:= 1.5 \cdot \mathrm{RTT}_{\mathrm{eff}}
+ \frac{S}{B_{\mathrm{eff}}}
+ O_{\mathrm{app}}
}
\]

with \(B_{\mathrm{eff}}\) as above.

---

## QUIC

**TLS 1.3 overhead** (CPU-bound: handshake + per-byte crypto):

\[
O_{\mathrm{TLS}}
:= O_{\mathrm{TLS\_handshake}}
+ K_{\mathrm{TLS}} \cdot \frac{S}{8}
\]

- \(O_{\mathrm{TLS\_handshake}}\): one-time handshake time (s)
- \(K_{\mathrm{TLS}}\): seconds per byte for encrypt+decrypt (s/byte)
- \(S/8\): payload size in bytes

**Predicted round time:**

\[
\boxed{
T_{\mathrm{calc}}^{\mathrm{QUIC}}
:= 1 \cdot \mathrm{RTT}_{\mathrm{eff}}
+ \frac{S}{B}
+ p \cdot \frac{S}{\mathrm{MSS}} \cdot \mathrm{RTT}_{\mathrm{eff}}
+ O_{\mathrm{app}}
+ O_{\mathrm{TLS}}
}
\]

- \(1 \cdot \mathrm{RTT}_{\mathrm{eff}}\): one RTT (e.g. connection/flow)
- \(S/B\): wire transfer time (no \(B_{\mathrm{eff}}\) in code for QUIC)
- \(p \cdot (S/\mathrm{MSS}) \cdot \mathrm{RTT}_{\mathrm{eff}}\): loss-induced retransmit cost
- \(O_{\mathrm{app}}\): application/broker overhead
- \(O_{\mathrm{TLS}}\): TLS 1.3 encryption overhead

---

## HTTP/3

HTTP/3 is QUIC-based; the same analytical model is used:

\[
\boxed{
T_{\mathrm{calc}}^{\mathrm{HTTP/3}}
:= 1 \cdot \mathrm{RTT}_{\mathrm{eff}}
+ \frac{S}{B}
+ p \cdot \frac{S}{\mathrm{MSS}} \cdot \mathrm{RTT}_{\mathrm{eff}}
+ O_{\mathrm{app}}
+ O_{\mathrm{TLS}}
}
\]

with \(O_{\mathrm{TLS}}\) as for QUIC.

---

## DDS

No TLS term; repair uses an extra constant delay per loss round.

\[
\boxed{
T_{\mathrm{calc}}^{\mathrm{DDS}}
:= \frac{S}{B}
+ p \cdot \frac{S}{\mathrm{MSS}} \cdot (\mathrm{RTT}_{\mathrm{eff}} + T_{\mathrm{REPAIR}})
+ O_{\mathrm{app}}
}
\]

- \(S/B\): wire transfer time
- \(p \cdot (S/\mathrm{MSS}) \cdot (\mathrm{RTT}_{\mathrm{eff}} + T_{\mathrm{REPAIR}})\): loss and repair (ACKNACK + retransmit)
- \(O_{\mathrm{app}}\): application/stack overhead

---

## Constants Reference

| Constant | Value (default) | Meaning |
|----------|------------------|--------|
| **MSS** | 11 680 bits | Max segment size used in loss term |
| **T_REPAIR** | 0.005 s | Extra time per repair (DDS) |
| **O_TLS_HANDSHAKE_S** | 0.003 s | TLS 1.3 handshake (QUIC/HTTP3); env `O_TLS_HANDSHAKE_S` |
| **K_TLS_SEC_PER_BYTE** | 2×10⁻⁸ s/byte | Per-byte TLS crypto (QUIC/HTTP3); env `K_TLS_SEC_PER_BYTE` |
| **α_MQTT** | 1.8 | MQTT TCP-like scaling factor (fitted from diagnostics) |
| **α_AMQP** | 2.1 | AMQP TCP-like scaling factor (fitted from diagnostics) |
| **α_gRPC** | 1.7 | gRPC TCP-like scaling factor (fitted from diagnostics) |
| **α_HTTP3** | 4.5 | HTTP/3 (QUIC) scaling factor (fitted from diagnostics) |
| **α_DDS** | 8.2 | DDS chunking/repair scaling factor (fitted from diagnostics) |

---

## Summary Table

| Protocol | T_calc formula |
|----------|----------------|
| **MQTT** | \(\alpha_{\mathrm{MQTT}}\,(2\,\mathrm{RTT}_{\mathrm{eff}} + S/B_{\mathrm{eff}}) + O_{\mathrm{app}}\) |
| **AMQP** | \(\alpha_{\mathrm{AMQP}}\,(1.5\,\mathrm{RTT}_{\mathrm{eff}} + S/B_{\mathrm{eff}}) + O_{\mathrm{app}}\) |
| **gRPC** | \(\alpha_{\mathrm{gRPC}}\,(1.5\,\mathrm{RTT}_{\mathrm{eff}} + S/B_{\mathrm{eff}}) + O_{\mathrm{app}}\) |
| **QUIC** | \(1\,\mathrm{RTT}_{\mathrm{eff}} + S/B + p\,(S/\mathrm{MSS})\,\mathrm{RTT}_{\mathrm{eff}} + O_{\mathrm{app}} + O_{\mathrm{TLS}}\) |
| **HTTP/3** | \(\alpha_{\mathrm{HTTP3}}\,(1\,\mathrm{RTT}_{\mathrm{eff}} + S/B + p\,(S/\mathrm{MSS})\,\mathrm{RTT}_{\mathrm{eff}}) + O_{\mathrm{app}} + O_{\mathrm{TLS}}\) |
| **DDS** | \(\alpha_{\mathrm{DDS}}\,(S/B + p\,(S/\mathrm{MSS})\,(\mathrm{RTT}_{\mathrm{eff}} + T_{\mathrm{REPAIR}})) + O_{\mathrm{app}}\) |

**Effective bandwidth** (MQTT, AMQP, gRPC only):

\[
B_{\mathrm{eff}} = \begin{cases} B & p=0 \\ \min(B,\ \mathrm{MSS}/(\mathrm{RTT}_{\mathrm{eff}}\sqrt{p})) & p>0 \end{cases}
\]

**TLS overhead** (QUIC, HTTP/3 only):

\[
O_{\mathrm{TLS}} = O_{\mathrm{TLS\_handshake}} + K_{\mathrm{TLS}} \cdot (S/8)
\]

---

## Implementation

- Script: `Network_Simulation/diagnostic_pipeline.py`
- **T_calc** is compared to **T_actual** (measured round time); **Error %** = \(100 \cdot (T_{\mathrm{actual}} - T_{\mathrm{calc}})/T_{\mathrm{actual}}\).

### Host network mode

When containers use **network_mode: host**, there is no per-container `eth0`; traffic uses the host’s default interface. The diagnostic pipeline:

1. Applies the scenario on the host via `tc` on the host’s default interface (e.g. `eth0`), so **T_actual** is measured under real delay/loss/bandwidth. This requires `tc` on the host (often `sudo`).
2. Uses scenario parameters for T_calc: \(B\), \(p\), \(D_{\mathrm{tc}}\), \(J\) are taken from the selected scenario (not from `docker exec … tc` inside the container), so the same formulas apply and **T_calc** matches the applied conditions.
3. Resets host tc after the pipeline so the host interface is left clean.

If host `tc` cannot be applied (e.g. no sudo), T_actual remains unshaped (~ calibration time) while T_calc still uses the scenario; the reported error will be large and the run will log a warning.

Do we need a different formula when tc affects all containers (server, client, broker)? No. The same \(T_{\mathrm{calc}}\) formula is used; it already depends on path parameters \(B\), \(p\), \(\mathrm{RTT}_{\mathrm{eff}}\), and \(O_{\mathrm{app}}\). When tc is applied on the host, it shapes the path; we still model that path with the same equation. One caveat: with **network_mode: host**, server and clients typically use **127.0.0.1** (loopback). Tc on the host’s **eth0** does **not** shape loopback traffic, so T_actual may stay unshaped unless the FL traffic actually goes through the shaped interface (e.g. server on another host) or you shape loopback if supported.

For a conceptual overview and worked examples by scenario, see `COMMUNICATION_MODEL.md`.

