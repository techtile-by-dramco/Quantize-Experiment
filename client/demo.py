import numpy as np

def load_tx_seq(path: str, L: int) -> np.ndarray:
    s = np.fromfile(path, dtype=np.complex64)
    if s.size < L:
        raise ValueError(f"{path} length {s.size} < L={L}")
    s = s[:L].astype(np.complex64, copy=False)
    # unit RMS
    s = s / (np.sqrt(np.mean(np.abs(s)**2)) + 1e-12)
    return s

def find_best_cyclic_shift(y: np.ndarray, s: np.ndarray, max_search: int = 512) -> int:
    """
    Find shift k in [0, max_search) maximizing |<y, roll(s,k)>|.
    Assumes y,s length L.
    """
    best_k, best_v = 0, -1.0
    for k in range(max_search):
        v = np.abs(np.vdot(np.roll(s, k), y))  # <roll(s,k), y>
        if v > best_v:
            best_v, best_k = v, k
    return best_k

def corr_leakage_sinr(
    y: np.ndarray,
    s1: np.ndarray,
    s2: np.ndarray,
    user_id: int = 1,          # UE1 desired=s1; UE2 desired=s2
    max_shift_search: int = 512,
    eps: float = 1e-12,
):
    """
    Return dict with:
      a_des, a_int, P_des, P_int, leak_ratio, leak_db, P_noise, sinr, sinr_db, shift
    """
    L = len(s1)
    assert len(y) == L and len(s2) == L

    s_des = s1 if user_id == 1 else s2
    s_int = s2 if user_id == 1 else s1

    # 1) coarse alignment by cyclic shift
    shift = find_best_cyclic_shift(y, s_des, max_search=max_shift_search)
    s_des_s = np.roll(s_des, shift)
    s_int_s = np.roll(s_int, shift)

    # 2) projections (matched filter / correlation)
    a_des = np.vdot(s_des_s, y) / float(L)
    a_int = np.vdot(s_int_s, y) / float(L)

    P_des = float(np.abs(a_des) ** 2)
    P_int = float(np.abs(a_int) ** 2)

    leak_ratio = P_int / max(P_des, eps)
    leak_db = 10.0 * np.log10(max(leak_ratio, eps))

    # 3) residual-based noise+model error estimate
    e = y - a_des * s_des_s - a_int * s_int_s
    P_noise = float(np.mean(np.abs(e) ** 2))

    # 4) SINR estimate
    sinr = P_des / max(P_int + P_noise, eps)
    sinr_db = 10.0 * np.log10(max(sinr, eps))

    return {
        "shift": int(shift),
        "a_des_re": float(np.real(a_des)),
        "a_des_im": float(np.imag(a_des)),
        "a_int_re": float(np.real(a_int)),
        "a_int_im": float(np.imag(a_int)),
        "P_des": P_des,
        "P_int": P_int,
        "leak_ratio": float(leak_ratio),
        "leak_db": float(leak_db),
        "P_noise": P_noise,
        "sinr": float(sinr),
        "sinr_db": float(sinr_db),
    }
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, sys, time, json
import numpy as np
import zmq

# ---------- import the functions above ----------
# (You can paste load_tx_seq / corr_leakage_sinr here)

TOPIC = b"CH0"   # 你 AP 端若按 TOPIC 发，就订阅对应 topic；不带 topic 就订阅 b""
L = 4096
MAX_SHIFT_SEARCH = 512

def main():
    # ---- config (modify as needed) ----
    AP_IP = os.environ.get("AP_IP", "192.168.1.10")   # 改成你的 AP IP
    AP_PORT = int(os.environ.get("AP_PORT", "50001"))

    USER_ID = int(os.environ.get("USER_ID", "2"))     # UE1=1, UE2=2

    TX1 = os.environ.get("TX1_PATH", "./tx1.bin")
    TX2 = os.environ.get("TX2_PATH", "./tx2.bin")

    OUT_PUB_PORT = int(os.environ.get("OUT_PUB_PORT", "52001"))

    # ---- load sequences ----
    s1 = load_tx_seq(TX1, L)
    s2 = load_tx_seq(TX2, L)

    # ---- ZMQ setup ----
    ctx = zmq.Context.instance()

    sub = ctx.socket(zmq.SUB)
    sub.connect(f"tcp://{AP_IP}:{AP_PORT}")
    # 如果 AP 端是 send_multipart([topic, payload])，就用 topic 订阅
    sub.setsockopt(zmq.SUBSCRIBE, TOPIC)  # 若不确定，用 b"" 订阅全部

    pub = ctx.socket(zmq.PUB)
    pub.bind(f"tcp://*:{OUT_PUB_PORT}")

    # ---- receive buffer ----
    buf = np.zeros(L, dtype=np.complex64)
    fill = 0

    print(f"[UE] SUB from tcp://{AP_IP}:{AP_PORT} topic={TOPIC!r}")
    print(f"[UE] PUB metrics on tcp://*:{OUT_PUB_PORT} (USER_ID={USER_ID})")

    time.sleep(0.2)  # PUB/SUB warmup

    while True:
        try:
            # ----- receive one ZMQ message -----
            # 情况1：AP 发 multipart [topic, raw_bytes]
            parts = sub.recv_multipart()
            if len(parts) == 2:
                topic, payload = parts
            else:
                # 情况2：AP 直接发 raw bytes（无 topic）
                topic, payload = b"", parts[0]

            # payload 解析为 complex64
            x = np.frombuffer(payload, dtype=np.complex64)

            # ----- append into window buffer -----
            i = 0
            while i < x.size:
                take = min(L - fill, x.size - i)
                buf[fill:fill+take] = x[i:i+take]
                fill += take
                i += take

                if fill == L:
                    y = buf.copy()
                    fill = 0

                    metrics = corr_leakage_sinr(
                        y=y,
                        s1=s1,
                        s2=s2,
                        user_id=USER_ID,
                        max_shift_search=MAX_SHIFT_SEARCH,
                    )

                    msg = {
                        "t": time.time(),
                        "user_id": USER_ID,
                        "topic": topic.decode(errors="ignore") if topic else "",
                        "L": L,
                        "metrics": metrics,
                    }

                    pub.send_string(json.dumps(msg))

        except KeyboardInterrupt:
            break
        except Exception as e:
            # 不要让脚本死掉
            err = {"t": time.time(), "user_id": USER_ID, "error": str(e)}
            try:
                pub.send_string(json.dumps(err))
            except Exception:
                pass
            time.sleep(0.05)

    sub.close(0)
    pub.close(0)
    ctx.term()

if __name__ == "__main__":
    main()
