# 📐 Geometry-Based RF Wireless Power Transfer

Python tooling to coordinate a geometry-aware wireless power transfer experiment over distributed tiles (Raspberry Pi + USRP B210) using a ZMQ control plane and tile-management Ansible playbooks.

---

## ✨ At a glance
- **Control plane:** ZMQ sync + Ansible orchestration
- **Clients:** Raspberry Pi tiles running USRP B210 scripts
- **Outputs:** Heatmaps, per-iteration energy-ball logs, TX phase YAMLs
- **Typical loop:** generate phases → deploy → run → record → plot → iterate

---

## 🧭 Repository layout
- `experiment-settings.yaml`: central experiment config (tile groups, RF params, server host, client script + args)
- `server/`: control-node tooling for provisioning tiles, updating experiment artifacts, starting/stopping clients, and the ZMQ coordinator
- `client/`: tile scripts + calibration data (`cal-settings.yml`) + best-phase snapshots (`tx-phases-*.yml`)
- `processing/`: TX phase generation, post-processing, and plotting (matplotlib + Plotly/Dash)
- `lib/`: shared helpers (energy profiler, YAML utilities)
- `pictures/`: diagrams/results

---

<details>
<summary>✅ Prerequisites</summary>

- Control machine with Python 3, Git/SSH access to the tiles, and Ansible available
- `tile-management` repo checked out at `~/tile-management` (or let `server/setup-server.sh` clone/update it)
- Tiles listed in `~/tile-management/inventory/hosts.yaml` and reachable via SSH; hostnames follow `rpi-<id>`
- UHD/B210 stack on the tiles (validated by `server/setup-clients.py`)

</details>

---

<details>
<summary>🛠️ Setup (control node)</summary>

1) **Prep TX phase files**

- Sionna ray-tracing helper:
```bash
cd processing
python compute-tx-weights-sionna.py
cd ..
```

- Energy-ball workflow (post-process latest run):
```bash
python processing/process-energy-ball.py --plot
```

2) **Bootstrap the virtualenv and pull tile-management**
```bash
cd server
./setup-server.sh
source bin/activate
```

3) **Configure** `experiment-settings.yaml` (server host/IP, tile group(s), RF params, `client_script_name`/`client_script_args`, extra apt packages)

4) **Prepare tiles** (apt, repos, UHD)
```bash
python server/setup-clients.py --ansible-output
```
Flags: `--skip-apt`, `--repos-only`, `--install-only`, `--check-uhd-only`

5) **Push code/settings to tiles**
```bash
python server/update-experiment.py --ansible-output
```

6) **Start/stop the experiment service**
```bash
python server/run-clients.py --start   # or --stop
```

</details>

---

<details>
<summary>📡 Running an experiment</summary>

- Start the ZMQ server (with venv active):
```bash
python server/run_server.py
```

- Tiles run the client defined in `experiment-settings.yaml` (e.g. `client/run_energy_ball.py`).  
Manual example on a tile:
```bash
python client/run_quasi_multi_tone.py \
  --config-file /home/pi/geometry-based-wireless-power-transfer/experiment-settings.yaml
```

Clients wait for `tx-start`, transmit for the requested duration, then reply with `tx-done`.

</details>

---

<details>
<summary>🧪 Recording measurements</summary>

- Energy-profiler recorder (update `FOLDER` in `server/record/record-meas-energy-profiler.py`):
```bash
python server/record/record-meas-energy-profiler.py
```

- Autosaves every `SAVE_EVERY` seconds to:
`data/<FOLDER>/<timestamp>_{positions,values}.npy`

</details>

---

<details>
<summary>🧠 Experiment workflows</summary>

**Fixed-phase transmit (`run_gbwpt_phases.py`)**
- Uses phases defined in `settings.yml`
- Requires: `sync-server.py`, XY plotter (TTRPI5), Qualisys positioner, `server/record/record-meas-energy-profiler.py`
- Plotting: `processing/plot_all_folders_heatmap.py` or `processing/plot-values-positions-2d.py`

**Energy-ball (simulated annealing) transmit**
- Iteratively updates phases via simulated annealing
- Server: `server/record/server-energy-ball.py` or `server-energy-ball-max.py`
- Clients: `client/run_energy_ball.py` or `client/run_energy_ball_max.py`
- Post-process: `processing/process-energy-ball.py --plot`
- Plotting: matplotlib heatmaps or `processing/plot_all_folders_heatmap_live.py`

```bash
python run-clients.py --start -a
python run-clients.py --stop -a
```

```bash
python server/record/record-meas-scope.py
```

```bash
python server/record/sync-BF-server.py --num-subscribers 42 --num-pilots 1
```

```bash
python usrp_pilot.py --pilot 1
```


</details>

---

<details>
<summary>📈 Plotting & post-processing</summary>

- Per-folder heatmap:
```bash
cd processing
python plot-values-positions-2d.py
```

- Aggregate heatmaps (newest folder by default):
```bash
python processing/plot_all_folders_heatmap.py
```
Options: `--plot-all`, `--plot-movement`

- Live Plotly/Dash heatmap:
```bash
python processing/plot_all_folders_heatmap_live.py \
  --host 0.0.0.0 --port 8050 --target 3.181 1.774 0.266
```

- Energy-ball YAML summarizer:
```bash
python processing/process-energy-ball.py [path/to/exp-YYYYMMDDHHMMSS.yml]
```

</details>

---

<details>
<summary>⚙️ TX phase generation</summary>

- Friis model:
```bash
python processing/compute-tx-phases.py
```

- Energy-ball best-phase update:
```bash
python processing/process-energy-ball.py
```

</details>

---

<details>
<summary>🧾 Reference waveform for reciprocity tests</summary>

```bash
python3 client/run-ref.py --args "type=b200" \
  --freq 920e6 --rate 250e3 --duration 1E6 --channels 0 \
  --wave-ampl 0.8 --gain 73 -w sine --wave-freq 0
```

</details>

---

<details>
<summary>🧰 Maintenance utilities</summary>

- `server/cleanup-clients.py`, `server/reboot-clients.py`
- `client/usrp-cal-bf.py`
- `ref-RF-cable.yml` and `tx-phases-*.yml` hold calibration data

</details>

---

## Random benchmarks
- `RANDOM`: 42 ceiling antennas with fixed phases (reference)
- `RANDOM-1`: 41 ceiling antennas with a new random phase every baseband sample
- `RANDOM-2`: 42 ceiling antennas with phases re-randomized every 100 ms (held constant within each 100 ms window)

---

## 📜 License
MIT (see `LICENSE`).
