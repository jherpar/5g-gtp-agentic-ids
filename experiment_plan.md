# Experiment Plan

This is the thesis-facing methodology and results document. It covers, in
order: the ground-truth labeling methodology and its Phase 4/4C findings,
Phase 5B's agent threshold calibration, the four-arm comparative ML
baseline design (Phase 6), and the Research Questions findings, limitations,
threats to validity, and reproducibility instructions (Phase 8). All
results below are **frozen** as of the Phase 7 commit — no further model
tuning or threshold recalibration was performed after Phase 7 concluded.
See [`architecture.md`](architecture.md) for code structure.

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

## Phase 5B: agent threshold calibration and the inverted temporal-entropy finding

Phase 5's agents (`TEIDAgent`/`PDUSessionAgent`/`SupervisorAgent`) were unit-
tested only against synthetic data; `scripts/validate_agents.py` ran them
against the real labeled BS1 dataset for the first time and found high
false-positive rates on HIGH-confidence (trustworthy) benign traffic for
the DoS-type attacks (44-73% FPR), driven mainly by `PDUSessionAgent`'s
`high_state_transition` and `low_temporal_entropy` rules.
`scripts/calibrate_agent_thresholds.py` computed real distribution
statistics (p50/p75/p90/p95/p99), pooled across all 9 attack types, split
by HIGH-confidence-benign / HIGH-confidence-attack / MEDIUM-confidence-
attack, to recalibrate them (`outputs/reports/agent_threshold_calibration/`).

`high_state_transition_rate` separated cleanly in its assumed direction
(benign p99 = 11.47, attack median = 106.8) and was recalibrated from 0.5
to 15.0.

`low_temporal_entropy` did not: the real data shows attack sessions have
**higher** temporal entropy than benign ones (attack median = 2.69 vs.
benign median = 0.50) — the opposite of the rule's "low entropy = bursty,
flood-like" assumption. This is not a threshold-value problem; no cutoff
can fix a reversed relationship without flipping the rule's `<=` comparison
to `>=`, which is a logic change outside Phase 5B's scope (calibrate
existing thresholds, not redesign rules). The rule was neutralized
(threshold set below the valid range of Shannon entropy, so it
mathematically cannot trigger) rather than left at a value that would
still generate false positives. **The inverted relationship itself is
carried forward as an open research question, not treated as solved** —
plausibly floods sustained across a full window arrive uniformly (spreading
packets evenly across the window's time sub-bins, i.e. high entropy) while
ordinary app traffic is bursty/request-response shaped (concentrated in a
few sub-bins, i.e. low entropy) for this dataset, but this has not been
independently verified and the rule should not be re-enabled without doing
so.

Full per-type distribution tables and the underlying evidence trail are in
`outputs/reports/labeling_validation_all/`,
`outputs/reports/confidence_diagnosis/`,
`outputs/reports/evidence_quantification/`,
`outputs/reports/flood_evidence_inspection/`, and
`outputs/reports/connection_flood_hypothesis/` (gitignored analysis
artifacts, not part of the committed pipeline).

## Phase 6: four-arm comparative ML baseline design

Extended from the original three-arm plan to four, after inspecting the
dataset authors' two processed CSVs (see `architecture.md`):

- **A1_combined** (primary official baseline): `data/processed/Combined/Combined.csv`,
  our own documented preprocessing (readable categorical columns —
  `Proto`, `sDSb`, `dDSb`, `Cause`, `State` — one-hot encoded before
  splitting; NaN imputed to a `-1` sentinel).
- **A2_encoded** (secondary reproducibility check only): `data/processed/Encoded/Encoded.csv`,
  the authors' own pre-encoded columns used near-verbatim. Demoted from
  primary-baseline status because many of its one-hot column names are
  uninterpretable artifacts of the authors' own encoding step (e.g.
  `" *    V   "`), not citable as specific features.
- **B_gtp_ml**: RandomForest/XGBoost on the real labeled GTP/TEID/session
  features (the 9 BS1 attack-type files), TEID-safe split (a
  `(ue_ip, teid)` group never crosses train/test, computed independently
  per attack type so no type is entirely held out).
- **C_agentic**: the existing, *untrained*, Phase-5B-calibrated
  `TEIDAgent`/`PDUSessionAgent`/`SupervisorAgent` pipeline, evaluated on
  the identical test sessions arm B was scored on.

**A split-strategy bug found and fixed during Phase 6**: `Combined.csv`/
`Encoded.csv`'s `Seq` and `RunTime` columns both reset per source capture
file rather than forming a global timeline (verified empirically — e.g.
`ICMPFlood`'s `Seq` spans 885–3074 while `Benign`'s spans 1–137210). A
naive global chronological split by either column interleaves unrelated
captures and silently drops most attack types from the test set entirely
(verified: 7 of 9 types vanished from test, leaving it almost pure
Benign+UDPFlood) — this produced degenerate "predict everything as
attack" results (recall=1.0 across all of RF/XGBoost/Combined/Encoded)
before being caught. Fixed with `per_group_chronological_split`: the
chronological split is computed independently within each `Attack Type`
group, then concatenated — the same principle already used for arm B/C's
TEID-safe split.

Arms B and C are evaluated under the two label views defined above:
**view A** (full schedule-labeled population) and **view B**
(HIGH/MEDIUM-confidence corroborated only). Arms A1/A2 have no such
confidence-tier concept (the dataset authors' `Label` column is a flat
binary) and are reported once.

## Research Questions — findings (Phase 6/7, frozen)

### Primary results table

All numbers below are from `outputs/reports/phase6_training/results.csv`
(Phase 6) and independently reproduced bit-for-bit by
`scripts/run_phase7_analysis.py` (10/10 confusion-matrix checks passed).

| Arm | Model | Accuracy | Precision | Recall | F1 | ROC-AUC | FPR |
|---|---|---|---|---|---|---|---|
| A1 (Combined, ours) | RandomForest | 0.940 | 0.937 | 0.967 | 0.952 | 0.959 | 0.101 |
| A1 (Combined, ours) | XGBoost | 0.955 | 0.965 | 0.961 | 0.963 | 0.994 | 0.054 |
| A2 (Encoded, authors') | RandomForest | 0.940 | 0.936 | 0.967 | 0.951 | 0.959 | 0.102 |
| A2 (Encoded, authors') | XGBoost | 0.955 | 0.965 | 0.961 | 0.963 | 0.994 | 0.053 |
| B (GTP-ML), view A | RandomForest | 0.666 | 0.268 | 0.958 | 0.418 | 0.945 | 0.376 |
| B (GTP-ML), view A | XGBoost | 0.622 | 0.243 | 0.952 | 0.387 | 0.954 | 0.426 |
| B (GTP-ML), view B | RandomForest | 0.646 | 0.143 | 0.987 | 0.250 | 0.947 | 0.375 |
| B (GTP-ML), view B | XGBoost | 0.599 | 0.129 | 0.987 | 0.227 | 0.972 | 0.425 |
| C (Agentic), view A | — | 0.874 | 0.488 | 0.120 | 0.192 | 0.710 | 0.018 |
| C (Agentic), view B | — | 0.932 | 0.344 | 0.149 | 0.208 | 0.947 | 0.018 |

A1 and A2 agree almost exactly with each other — a sanity check that the
Combined.csv preprocessing is faithful to the authors' own encoding.

### RQ1 — Can GTP-U/TEID features improve detection over the official flow dataset?

**No, on raw classification metrics, for this dataset as constructed.**
Arm A reaches F1 ≈ 0.95–0.96; arm B reaches F1 ≈ 0.23–0.42 at the default
0.5 cutoff. But the ROC-AUC gap is far smaller (0.94–0.99 vs. 0.94–0.97),
meaning arm B's models discriminate about as well as arm A's once
ranking is separated from the classification threshold. The gap is best
read as a data-scale and label-quality confound rather than evidence that
GTP-U/TEID information itself is uninformative: arm A trains on ~1.2M
flow records with dozens of mature Argus features; arm B trains on
~19,500 sessions with 8 simple aggregate features, built from data where
only ~14.5% of attack labels are independently corroborated (this
document's own earlier finding).

### RQ2 — Can the agentic architecture match/exceed ML given identical features?

**A genuine precision/recall trade-off, not a clean win or loss.** Arm C
has dramatically better precision (0.34–0.49 vs. 0.13–0.27) and FPR
(0.018 vs. 0.38–0.43) than arm B — exactly what Phase 5B's threshold
calibration was built to achieve — but far worse recall (0.12–0.15 vs.
0.95–0.99). By ROC-AUC alone, arm C's view-B score (0.947) is competitive
with arm B's trained models (0.94–0.97): the agent's fused risk score
ranks attacks about as well, it is simply evaluated at a far more
conservative fixed decision threshold than the ML models' default 0.5
cutoff. The threshold-sensitivity plots (`outputs/figures/phase7/`,
discussion only — no threshold was changed) show how each arm's
precision/recall would trade off across the full range.

### RQ3 — Does TEID/session-level reasoning improve explainability?

**Qualitatively yes.** The agentic system produces a concrete,
human-readable, per-decision explanation built from named triggered
rules and their measured values, e.g.:

> `TEIDAgent: syn_flood (syn_count=2001 (>= 20), ack/syn=0.000 (<= 0.1)); scan (unique_dst_ports=1001 (>= 15), packets_per_port=2.01 (<= 3.0)) | PDUSessionAgent: state=WATCH; high_diversity (port_diversity=1000, destination_diversity=3 (peak 1000 >= 15))`

Classical ML (arms A/B) offers only a model-level, non-per-instance
explanation (global feature importances) unless a post-hoc method (e.g.
SHAP) were added, which this project does not implement. This is the
intended comparison from the original plan — a genuine capability
difference between the two paradigms, not a gap to "fix" on the ML side.
Per-attack-type worked examples (one TP/FN/FP each, where present in the
test split) are in `outputs/reports/phase7_analysis/report.md`.

### RQ4 — Can attacks be detected earlier via TEID/session reasoning?

**Partial answer from available evidence, now including an empirical
time-to-first-flag measurement (below).** Both arm B and arm C have
sub-millisecond per-sample inference latency once trained/configured, so
raw wall-clock inference speed is not the differentiator. The qualitative
distinction is architectural: `PDUSessionAgent`'s state machine
(NORMAL→WATCH→SUSPICIOUS→ATTACK) produces staged early-warning signal
across a session's own timeline that a single-shot flow classifier does
not have by construction — a session can surface as WATCH/SUSPICIOUS
before ever reaching a final ATTACK verdict. This staged-signal
*capability* is real (the state machine architecturally can flag before a
final verdict), but the follow-up measurement below tests whether it
actually happens *earlier in wall-clock time* than a traditional
classifier's first positive — and finds, on the primary comparison, that
it typically does not.

#### RQ4 follow-up: detection-latency measurement study

A pure measurement addendum (`scripts/analyze_rq4_detection_latency.py`),
run after Phase 7 with the experimental configuration completely frozen —
no model retrained, no threshold/rule/label/split/config changed. Two
things were of necessity *re-derived* rather than reused as saved
artifacts, both exactly reproducing what Phase 6 already did (not new
training in any sense that could change a result): the arm-B XGBoost fit
(no model was persisted to disk after Phase 6/7; same hyperparameters,
same seed, same `build_gtp_session_dataset` train split) and "attack
start," taken directly from the existing Level 1 ground truth
(`preprocessing.labeling._approximate_attack_subwindow`, the same
function `_classify()` already uses internally) rather than any new
definition.

**Unit of measurement**: per `(ue_ip, teid)` conversation group with at
least one attack-labeled session, not per file — one file has exactly one
`attack_start` but typically many independent groups, which is what makes
a per-attack-type mean/median/min/max distribution meaningful without
requiring new (BS2 or otherwise) data. `PDUSessionAgent.annotate_series`
ran over each group's full chronological session history for that file
(no train/test split applies to a rule-based, never-trained agent — no
leakage concern). The ML model was scored the same way, over the same
full chronological sequence, which for most attack types means **some
scored sessions were part of the model's own training set** — an
explicit, load-bearing limitation, not a footnote (see below). A
stricter, leakage-free "test-split-only" measurement is also reported
but is only defined for 14 of 122 groups (most attack types have zero
attack-labeled groups in the held-out test split at all, the same finding
already reported in the Error Analysis section above for Goldeneye/
ICMPflood).

**Overall summary (pooled across all 9 BS1 attack types, 122
attack-labeled conversation groups):**

| Metric | n | mean | median | min | max |
|---|---|---|---|---|---|
| Time to first WATCH (s, relative to attack_start) | 49 | 232.4 | 77.3 | -293.0 | 1462.3 |
| Time to first SUSPICIOUS (s) | 14 | 117.0 | 94.7 | -71.6 | 365.0 |
| Time to first ATTACK (s) | 12 | 181.5 | 189.4 | 7.3 | 375.0 |
| Time to first ML detection, full timeline (s) | 116 | -66.7 | 2.3 | -315.1 | 512.4 |
| Time to first ML detection, test-split only (s) | 14 | 157.2 | 68.1 | 17.4 | 478.9 |
| Detection lead time: ML − WATCH (s) | 49 | -307.9 | -60.0 | -1675.0 | 280.0 |
| Detection lead time: ML − SUSPICIOUS (s) | 14 | -108.2 | -22.5 | -525.0 | 0.0 |
| Detection lead time: ML − ATTACK (s) | 12 | -132.1 | -102.5 | -365.0 | -5.0 |

(Positive lead time = agent event happened before ML's first detection,
i.e. agent earlier; negative = ML detected first. Full per-attack-type
tables — all 9 types individually — are in
`outputs/reports/rq4_detection_latency/report.md`.)

**The result is negative on the primary comparison, and is reported as
such.** Median lead time is negative for all three agent milestones
(WATCH: -60.0s, SUSPICIOUS: -22.5s, ATTACK: -102.5s) — under this
measurement, the ML classifier's first positive prediction typically
precedes even the agent's *earliest* escalation state (WATCH), let alone
SUSPICIOUS or ATTACK. This directly contradicts the capability argument's
implicit assumption that staged escalation would also mean *earlier*
detection in wall-clock time. A plausible architectural explanation:
`PDUSessionAgent`'s state machine is deliberately rate-limited to escalate
at most one level per observed window (`pdu_session_agent.py`, to avoid a
single noisy window jumping straight to ATTACK) — that rate limit is a
real, intentional design choice for stability, but it mechanically costs
wall-clock time relative to a classifier that can flag "positive" on the
very first window whose features cross its threshold. This is offered as
an interpretation of *why*, not an excuse for the negative result.

The effect is not universal: individual conversation groups do show the
agent flagging before ML (e.g. UDPflood's WATCH-vs-ML lead time reaches
+280.0s at the max, Goldeneye +205.0s, ICMPflood +15.0s), so a minority of
cases and attack types benefit from earlier agent escalation — but this is
not the typical/median pattern across BS1's 9 attack types, and is not
represented as one.

**A specific, load-bearing limitation on this finding**: the "full
timeline" ML measurement is optimistic for ML (potential training-set
leakage for most attack types), which means the true gap could be
narrower — but it cannot make the reported negative result an artifact of
unfairness *against* the agent, since the leakage only advantages ML.
The leakage-free "test-split-only" comparison exists for too few groups
(14/122) to support an independent lead-time conclusion on its own; it is
reported (median 68.1s after attack_start) for transparency, not
interpreted further.

Three representative timeline figures (one flood — ICMPflood, one scan —
SYNScan, one slow-rate — Slowloris; each picks the attack-labeled group
with the most complete state trajectory, or the first available group if
none reached ATTACK) are in
`outputs/figures/rq4_detection_latency/timeline_{flood,scan,slow-rate}_*.png`,
plotting the agent's state over time against attack_start and ML's first
detection.

**Conclusion**: the data does not support the hypothesis that the
agentic architecture provides earlier warning in wall-clock time via its
WATCH/SUSPICIOUS states, in the typical/median case, across BS1's 9
attack types — the opposite pattern is what was measured, and is reported
without softening. The architectural capability for staged early warning
before a final verdict is real (RQ4's qualitative point stands), but
capability and demonstrated earliness are not the same claim, and only
the former is supported by this project's evidence.

### Error analysis (arm C, per attack type, view A)

| Attack type | n | TP | FP | FN | TN | Recall | FPR |
|---|---|---|---|---|---|---|---|
| Goldeneye | 621 | 0 | 0 | 0 | 621 | n/a | 0.000 |
| ICMPflood | 104 | 0 | 0 | 0 | 104 | n/a | 0.000 |
| SYNScan | 43 | 7 | 0 | 36 | 0 | 0.163 | n/a |
| SYNflood | 265 | 0 | 5 | 58 | 202 | 0.000 | 0.024 |
| Slowloris | 207 | 0 | 2 | 0 | 205 | n/a | 0.010 |
| TCPConnect | 49 | 9 | 0 | 40 | 0 | 0.184 | n/a |
| Torshammer | 6 | 0 | 5 | 0 | 1 | n/a | 0.833 |
| UDPScan | 7 | 1 | 0 | 6 | 0 | 0.143 | n/a |
| UDPflood | 30 | 3 | 9 | 7 | 11 | 0.300 | 0.450 |

The aggregate view-A metrics (accuracy 0.874, FPR 0.018) hide real
per-type variance: **Torshammer's FPR is 0.833** (5 false alarms out of 6
test sessions) — the aggregate FPR is low mainly because most attack
types' test sessions are overwhelmingly benign, not because Torshammer is
well-handled. **Goldeneye and ICMPflood have zero attack sessions in the
test split at all** — the TEID-safe grouping happened to place all of
their few corroborated attack instances entirely in train for this
particular split.

## Limitations

- **BS1 only.** All Phase 4–7 results (labeling validation, agent
  calibration, ML baselines, RQ1–4 findings) use `data/raw/BS1/*.pcapng`
  exclusively. `data/raw/BS2/*.pcapng` was used only for the Phase 4C
  connection-oriented-flood hypothesis validation (to confirm
  port-cardinality asymmetry generalizes across base stations), never for
  Phase 6/7 training or evaluation. A full BS1+BS2 run was explicitly
  deferred, not attempted and hidden.
- **Small, noisy training data for arms B/C.** ~19,500 GTP sessions built
  from ~14.5%-corroborated labels, vs. arm A's ~1.2M mature flow records —
  the arm A vs. B comparison (RQ1) is confounded by this scale/quality
  gap, as discussed above.
- **Arm A1/A2 vs. B/C are not the same unit of analysis.** A1/A2 predict
  per-flow (Argus biflow record); B/C predict per-session (fixed
  5s/30s window) or per-TEID-instance. Metrics are reported per arm, not
  forced into a single shared confusion matrix.
- **Fixed decision thresholds, not threshold-matched.** RQ2's B-vs-C
  comparison uses each arm's own default operating point (ML: 0.5
  probability cutoff; agentic: Phase-5B-calibrated
  `attack_decision_threshold`), not a threshold-matched comparison (e.g.
  both arms compared at equal FPR). The threshold-sensitivity plots
  partially address this but a formal iso-FPR comparison was not built.
- **The `low_temporal_entropy` inverted-relationship finding is
  unexplained**, not just unfixed — the hypothesis offered (uniform
  floods vs. bursty background traffic) was not independently verified
  against packet-level evidence the way the connection-oriented-flood
  finding was.
- **RQ4's detection-latency measurement uses a leakage-prone ML timeline**
  for most attack types (the re-derived arm-B model scored some sessions
  it was fit on, since a leakage-free measurement is only defined for
  14/122 conversation groups) — see the RQ4 follow-up subsection for the
  full caveat. Time-to-first-flag is measured per conversation group, not
  independently replicated across base stations (BS1 only, same scope as
  the rest of this document).
- **The scan-type attack labeling limitation is structural, not fixed**:
  per `preprocessing/labeling.py`'s own documented KNOWN LIMITATION, Table
  III of the descriptor paper gives SYNScan/TCPConnect/UDPScan no
  separate collection-period window distinct from the attack window, so
  Level 1 fires for effectively the entire file and confidence tier (not
  the binary label) carries the real signal for these three types.
- **`Settings`/`configs/base.yaml` exists but was not the actual driver**
  of the scripts that produced these results — see `architecture.md`'s
  "Scripts actually used" section. A reader trying to reproduce results
  via `Settings`/`base.yaml`-driven CLI commands (as the original plan
  described) would not find one; the actual reproduction path is the
  specific `scripts/*.py` files documented below.

## Threats to validity

- **Internal validity**: ground-truth labels are schedule/victim-IP/
  pattern-derived, not independently hand-verified per instance beyond
  the case studies and validation reports produced during Phase 4. The
  ~14.5% HIGH+MEDIUM corroboration rate means most of the training signal
  for arms B/C (in view A) is LOW-confidence, i.e. plausibly mislabeled
  concurrent benign traffic per the descriptor paper's own caveat.
- **Construct validity**: "detection" is operationalized differently per
  arm (per-flow vs. per-session vs. per-TEID-instance), and the agentic
  system's PDU-session state machine only sees the test-period portion of
  each session sequence during evaluation (`evaluate_arm_c`'s docstring),
  not the full pre-split history — a session scored in isolation from its
  training-period predecessors may reach a different `final_state` than
  it would in a live, continuously-running deployment.
- **External validity**: findings are specific to the 5G-NIDD dataset's
  attack tools/traffic mix and to BS1. Generalization to other 5G
  deployments, other attack tools, or BS2 is not established by this
  work.
- **Statistical power for the connection-oriented-flood threshold**: the
  `min_port_cardinality_asymmetry=200.0` calibration is based on only 9
  victim-IP-corroborated instances total (5 SYNflood + 4 Goldeneye,
  BS1+BS2 combined) — the zero-overlap separation is compelling given the
  order-of-magnitude gap, but the sample is small.
- **Non-adversarial evaluation**: all attack traffic is the original
  5G-NIDD capture; no adaptive or evasion-aware adversary was simulated
  against either the agentic rules or the trained ML models.

## Reproducibility

- **Environment**: Python 3.11, Poetry (`poetry install`). All seeds
  fixed to 42 (`ml/dataset.py::SEED`, propagated to `RandomForestModel`/
  `XGBoostModel`/`per_group_chronological_split`).
- **Data**: place `data/raw/BS1/*.pcapng` and
  `data/processed/{Combined,Encoded}/*.csv` as described in `README.md`
  (not committed to git). BS2 is only needed to reproduce the Phase 4C
  connection-flood hypothesis validation, not Phase 6/7.
- **Config**: `configs/thresholds.yaml`, `configs/label_patterns.yaml`,
  `configs/attack_schedule.yaml` are committed and are the exact frozen
  values behind every number in this document.
- **Reproduction path** (each step's outputs are cached under
  `outputs/cache/packets/`, gitignored, so re-runs after the first are
  fast):
  1. `poetry run python scripts/validate_labeling_all.py` — labeling
     validation report for all 9 BS1 types.
  2. `poetry run python scripts/validate_agents.py` — Phase 5 agent
     validation against real data.
  3. `poetry run python scripts/run_phase6_training.py` — trains/
     evaluates all four arms, writes
     `outputs/reports/phase6_training/results.csv`.
  4. `poetry run python scripts/run_phase7_analysis.py` — regenerates
     ROC/PR curves, case studies, and the RQ writeup; independently
     re-derives every Phase 6 confusion matrix and asserts it matches
     bit-for-bit (10/10 passed as of this writing).
  5. `poetry run python scripts/analyze_rq4_detection_latency.py` — the
     RQ4 detection-latency measurement study (post-Phase-7 addendum, pure
     measurement, no configuration changed).
- **Tests**: `poetry run pytest` (198 tests, ~93% coverage, no real pcap
  data required — all synthetic/hand-built fixtures).
- **What is NOT reproducible from git alone**: `outputs/**` (reports,
  figures, cache) is gitignored by design (regenerable, and some report
  directories are large). This document inlines every number load-bearing
  for the RQ1–4 findings so the thesis narrative does not depend on those
  artifacts being present.
