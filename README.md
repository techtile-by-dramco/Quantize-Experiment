# Geometry-Based RF Wireless Power Transfer

Python tooling to coordinate a geometry-aware wireless power transfer experiment over distributed tiles (Raspberry Pi + USRP B210) using a ZMQ control plane and tile-management Ansible playbooks.

## Repository layout
- `experiment-settings.yaml`: central experiment config (tile groups, RF params, server host, client script + args).
- `server/`: control-node tooling for provisioning tiles, updating experiment artifacts, starting/stopping clients, and the ZMQ coordinator.
- `client/`: scripts that run on the tiles plus calibration data for the USRPs.
- `processing/`: helper scripts used to prepare experiment inputs (e.g., TX phase generation).
- `lib/`: shared helpers (energy profiler, YAML utilities); `pictures/`: diagrams/results.

## Prerequisites
- Control machine with Python 3, Git/SSH access to the tiles, and Ansible available.
- `tile-management` repo checked out at `~/tile-management` (or let `server/setup-server.sh` clone/update it).
- Tiles listed in `~/tile-management/inventory/hosts.yaml` and reachable via SSH; hostnames follow `rpi-<id>` so the client IDs resolve correctly.
- UHD/B210 stack on the tiles (validated by `server/setup-clients.py`).

## Setup (control node)
1) Bootstrap the virtualenv and pull tile-management:
```
cd server
./setup-server.sh         # clones/updates tile-management and installs deps
source bin/activate
```
2) Configure `experiment-settings.yaml` with the server host/IP, target tile group(s), RF params (`frequency`, `gain`, `rate`, `duration`), `client_script_name`/`client_script_args`, and any extra apt packages.
3) Prepare the tiles (apt update/upgrade, install extras, pull repos, check UHD):
```
python server/setup-clients.py --ansible-output
```
   Flags: `--skip-apt`, `--repos-only`, `--install-only`, `--check-uhd-only` to narrow the actions.
4) Push updated experiment code/settings to the tiles:
```
python server/update-experiment.py --ansible-output
```
5) Start or stop the experiment service on the tiles:
```
python server/run-clients.py --start   # or --stop
```

## Running an experiment
- Launch the ZMQ control server on the control node (with the venv active):
```
python server/run_server.py
```
- Tiles run the client defined in `experiment-settings.yaml`. The default quasi multi-tone client can be started manually on a tile if needed:
```
python client/run_quasi_multi_tone.py --config-file /home/pi/geometry-based-wireless-power-transfer/experiment-settings.yaml
```
  Clients wait for `tx-start`, transmit for the requested duration, then reply with `tx-done`.

## Preparing TX phases
- `processing/compute-tx-phases.py` fetches the tile geometry from `techtile-description` and generates `client/tx-phases-friis.yml` (phase-aligned) and `client/tx-phases-benchmark.yml` (all zeros). Run it from the repo root:
```
python processing/compute-tx-phases.py
```

## Maintenance utilities
- `server/cleanup-clients.py`, `server/reboot-clients.py`: quick management helpers for the tiles.
- `client/usrp-cal-bf.py`: USRP calibration/beamforming helper; `ref-RF-cable.yml` and `tx-phases-*.yml` hold calibration data.

## License
MIT (see `LICENSE`).
