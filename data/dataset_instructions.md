# Dataset preparation

This project uses the **5G-NIDD** dataset (5G Network Intrusion Detection
Dataset), published on IEEE DataPort:
<https://ieee-dataport.org/documents/5g-nidd-comprehensive-network-intrusion-detection-dataset-generated-over-5g-wireless>.

**The dataset is NOT redistributed in this repository** and is not
covered by this repository's [MIT license](../LICENSE) — obtain it
directly from IEEE DataPort under the original authors' own terms, and
cite it separately (see [`CITATION.cff`](../CITATION.cff)'s dataset
reference).

## What to download

From the IEEE DataPort listing, you need:

1. **Raw pcap captures**, one per attack type per base station:
   `Goldeneye`, `ICMPflood`, `Slowloris`, `SSH` (benign-only baseline,
   not an attack), `SYNflood`, `SYNScan`, `TCPConnect`, `Torshammer`,
   `UDPflood`, `UDPScan` — for base stations `BS1` and `BS2` (20 files
   total, ~3GB). **This project's reported results use BS1 only** (see
   `docs/experiment_plan.md`'s Limitations) — `BS2` is only needed to
   reproduce the Phase 4C connection-flood cross-base-station check.
2. **Processed CSV exports** (the dataset authors' own Argus-derived
   flow features): `Combined.csv` and `Encoded.csv`.
3. *(Optional, appendix only)* the per-attack-type flow CSV export with
   real IP addresses (referred to in this project as
   `BS1_each_attack_csv`) — only needed to reproduce
   `docs/appendix_ground_truth_validation.md`. If the dataset listing
   organizes this differently than described here, place the per-type
   CSVs (one file per attack type, e.g. `ICMPFlood1.csv`, `SYNScan1.csv`,
   ...) under `data/raw/BS1_each_attack_csv/BS1_each_attack_csv/`.

## Expected layout

```
data/raw/BS1/Goldeneye_BS1.pcapng
data/raw/BS1/ICMPflood_BS1.pcapng
data/raw/BS1/SSH_BS1.pcapng
data/raw/BS1/SYNflood_BS1.pcapng
data/raw/BS1/SYNScan_BS1.pcapng
data/raw/BS1/Slowloris_BS1.pcapng
data/raw/BS1/TCPConnect_BS1.pcapng
data/raw/BS1/Torshammer_BS1.pcapng
data/raw/BS1/UDPflood_BS1.pcapng
data/raw/BS1/UDPScan_BS1.pcapng

data/raw/BS2/<same 10 files, _BS2 suffix>          (optional, see above)

data/raw/BS1_each_attack_csv/BS1_each_attack_csv/
    Goldeneye1.csv  ICMPFlood1.csv  SSH1.csv  SYNFlood1.csv
    SYNScan1.csv  Slowloris1.csv  TCPConnect1.csv  Torshammer1.csv
    UDPFlood1.csv  UDPScan1.csv                     (optional, appendix only)

data/processed/Combined/Combined.csv
data/processed/Encoded/Encoded.csv
```

File naming must match exactly (`<AttackType>_BS{1,2}.pcapng`) —
`configs/attack_schedule.yaml` and every script in `scripts/` key off
these filenames via `FILENAME_TOKEN_TO_ATTACK_TYPE`
(`src/agente_5g/models/labels.py`).

## Verifying the download

There is no checksum manifest published in this repository (dataset
integrity should be verified against whatever the IEEE DataPort listing
itself provides). As a sanity check once files are in place:

```bash
poetry run python -c "
from pathlib import Path
for bs in ['BS1', 'BS2']:
    d = Path(f'data/raw/{bs}')
    files = sorted(d.glob('*.pcapng')) if d.exists() else []
    print(f'{bs}: {len(files)} pcapng files')
"
```

Expect 10 files per base station if both are downloaded (10 for BS1
alone if you only need the primary reported results).

## Next step

Once the data is in place, see
[`reproducibility/run_order.md`](../reproducibility/run_order.md) for the
exact script sequence used to produce this project's reported results.
