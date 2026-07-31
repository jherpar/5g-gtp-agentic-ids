# Experiment Plan

This document is built incrementally as the project progresses. This first
section covers the ground-truth labeling methodology and a finding from its
validation (Phase 4/4C) that materially shapes how the ML baselines (Phase
6) and evaluation (Phase 7) must be designed. The Research Questions,
three-arm comparative design, and full evaluation methodology will be added
in Phase 8 once those phases are implemented.

## Ground-truth labeling methodology

The descriptor paper for 5G-NIDD states that benign traffic keeps flowing
*during* attack windows, so a timestamp window alone cannot safely label an
individual TEID-instance or session — it only bounds when an attack *could*
be present. `preprocessing/labeling.py` therefore combines three
independent evidence levels into an explicit confidence tier instead of a
silent binary label:

- **Level 1 — Attack Schedule**: is this instance inside the attack
  type's published schedule window (`configs/attack_schedule.yaml`,
  mapped onto the file's own observed timestamp span as a relative
  fraction, since the paper's absolute clock windows drift from real
  capture timestamps)?
- **Level 2 — Victim IP**: does this instance's traffic touch the shared
  victim/MEC server (`schedule.victim_ip`)?
- **Level 3 — Traffic pattern validation**: a coarse, attack-type-specific
  structural check, computed with thresholds and code kept entirely
  separate from `agents/rules.py`'s detection logic (an automated test
  enforces this) so label quality is never validated against the same
  logic being evaluated. Two flood sub-families exist because they are
  behaviorally different (see "Connection-oriented floods" below):
  volumetric floods (ICMPflood/UDPflood) check sustained rate + uniform
  payload size + concentrated destination ports; connection-oriented
  floods (SYNflood/Goldeneye) check port-cardinality asymmetry instead.
  Scans check destination-port fan-out; slow-rate attacks check many
  long-lived low-throughput connections.

### Confidence model (Phase 4C)

An earlier version required Level 2 (victim IP) as a mandatory gate for
MEDIUM/HIGH confidence. Evidence-source quantification found Level 2 fires
on only ~3% of instances and is itself an imperfect signal (some
"victim-IP corroborated" instances turned out to be ordinary background
traffic incidentally touching a shared infrastructure IP, not genuine
attack traffic touching the actual target). Requiring it meant a validated
Level-3 pattern match could be silently discarded to LOW whenever
victim-IP corroboration didn't also fire, even when the pattern evidence
was independently strong.

Phase 4C treats Level 2 and Level 3 as interchangeable corroborating
evidence:

| Confidence | Requirement |
|---|---|
| HIGH | Schedule + Victim IP + Pattern |
| MEDIUM | Schedule + (Victim IP OR Pattern), not both |
| LOW | Schedule only |

### Connection-oriented floods

SYNflood and Goldeneye do not fit the volumetric flood assumption (uniform
packet size, concentrated destination ports) that ICMPflood/UDPflood
satisfy. Packet-level inspection of their few victim-IP-corroborated
instances (BS1 and BS2) found a consistent alternate signature instead: one
side of the connection (source ports for an outbound SYN flood, destination
ports for the victim's reply/backscatter traffic) fans out into the
thousands while the other side stays pinned to a handful of well-known
ports — extreme **port-cardinality asymmetry**. Two other candidate
signals (destination-IP concentration, connection churn) were evaluated and
rejected: both showed heavy overlap between corroborated and
non-corroborated instances. Port-cardinality asymmetry showed **zero
overlap** between the two groups across both base stations (SYNflood:
corroborated min 276.5 vs. non-corroborated max 54.0; Goldeneye:
corroborated min 2283.3 vs. non-corroborated max 126.5), so
`connection_flood_pattern.min_port_cardinality_asymmetry = 200.0`
(`configs/label_patterns.yaml`) sits conservatively between those two
bounds rather than being an invented round number.

## Finding: independently corroborated attack labels are inherently sparse

After both the connection-oriented-flood rule and the Phase 4C confidence
redesign, the fraction of attack-labeled instances reaching HIGH+MEDIUM
confidence, aggregated across all 9 real attack types (BS1, weighted by
attack-instance count):

- **TEID level: ~14.5%**
- **Session level: ~12.5%**

Per-type range: 4.4% (UDPflood) to 32.1% (UDPScan) at TEID level.

This is **not an artifact of an overly strict confidence formula or
mistuned pattern thresholds**. Investigation (in order) ruled out each
alternative explanation:

1. Widening the flood pattern check to use entropy/port-concentration
   instead of a bare rate floor did not change the aggregate ceiling,
   because MEDIUM/HIGH were still gated on Level 2, and Level 3 can only
   redistribute *within* the Level-2-corroborated set, never add to it.
2. Building a correctly-signatured connection-oriented-flood rule for
   SYNflood/Goldeneye (validated with zero-overlap separation) still did
   not move the aggregate ceiling, for the same gating reason.
3. Removing the Level-2 gate entirely (Phase 4C: MEDIUM = schedule +
   *either* victim IP or pattern) moved the true aggregate by less than
   half a percentage point (~14.1% → ~14.5%). Cross-tabulating the three
   evidence sources (`outputs/reports/evidence_quantification/`) showed
   why: the case Phase 4C was designed to rescue — Level 3 firing
   *without* Level 2 — is empirically ~0% for 8 of the 9 attack types.
   Level 2 and Level 3 are not complementary signals in this dataset; they
   are almost perfectly redundant, agreeing on the same small sliver of
   traffic (UDPScan, at 3.6%, was the sole exception and the only type
   that improved).

**Conclusion**: the vast majority of traffic inside a published attack
window is genuinely indistinguishable from background traffic by either
victim-IP contact or structural pattern — consistent with the descriptor
paper's own statement that benign traffic continues throughout attack
windows. This is a property of the dataset and the attack-schedule-based
labeling approach itself, not a defect in the labeling pipeline's
thresholds or gating logic.

### Consequence for Phase 6/7: three preserved label views

Because independently-corroborated labels are too sparse to serve as the
sole training signal, every `TEIDFeatureRecord`/`PDUSessionRecord` retains
all labeling evidence rather than collapsing it to one binary field. Phase
6 (ML baselines) and Phase 7 (evaluation) code MUST expose, and must not
silently collapse to a single view of, all three of:

- **(A) Schedule-based labels** — `is_attack` (Level 1 only, the full
  attack-window population; noisy but complete).
- **(B) Corroborated labels** — `is_attack` instances filtered to
  `label_confidence in {HIGH, MEDIUM}` (~14.5%/~12.5% of attack instances;
  higher-trust but small).
- **(C) Confidence-tier metadata** — `label_confidence` +
  `label_evidence` themselves, kept as data rather than discarded, so
  training/evaluation can use them as sample weights or an uncertainty
  axis instead of a hard filter.

This enables comparative experiments Phase 6/7 can run without re-deriving
labels: train on (A) vs. (B) and measure the effect of label noise;
evaluate all three arms (official baseline / GTP-ML / agentic) against (B)
as the higher-trust test set while reporting (A)-based metrics separately;
or use (C) directly as per-sample confidence weights in training.

Full per-type distribution tables and the underlying evidence trail are in
`outputs/reports/labeling_validation_all/`,
`outputs/reports/confidence_diagnosis/`,
`outputs/reports/evidence_quantification/`,
`outputs/reports/flood_evidence_inspection/`, and
`outputs/reports/connection_flood_hypothesis/` (gitignored analysis
artifacts, not part of the committed pipeline).
