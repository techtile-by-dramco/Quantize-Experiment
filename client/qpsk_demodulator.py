#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#
# SPDX-License-Identifier: GPL-3.0
#
# GNU Radio Python Flow Graph
# Title: qpsk_demodulator
# Author: Dramco_Tianzheng
# GNU Radio version: 3.10.10.0
#
# Modified:
# - Replace file sink with probe (already done in your graph)
# - Expose get_evm_pct() for external scripts
# - Make moving average scale consistent with length

from gnuradio import blocks
from gnuradio import digital
from gnuradio import gr
from gnuradio.filter import firdes
from gnuradio import uhd

import sys
import signal
from argparse import ArgumentParser


class qpsk_demodulator(gr.top_block):
    """
    QPSK demodulator with EVM measurement.

    EVM path:
        Costas Loop (complex) -> EVM Measurement (float, percent)
            -> Moving Average (float) -> Probe Signal (float)

    Use get_evm_pct() to read the (smoothed) EVM in percent.
    """

    def __init__(
        self,
        eq_gain=0.0001,
        phase_bw=6.28 / 200,
        timing_loop_bw=6.28 / 200,
        evm_avg_len=200,
    ):
        gr.top_block.__init__(self, "qpsk_demodulator", catch_exceptions=True)

        ##################################################
        # Parameters
        ##################################################
        self.eq_gain = eq_gain
        self.phase_bw = phase_bw
        self.timing_loop_bw = timing_loop_bw
        self.evm_avg_len = int(evm_avg_len)

        ##################################################
        # Variables
        ##################################################
        self.d = d = 1 / (2 ** (1 / 2))
        self.sps = sps = 4
        self.qpsk_mg = qpsk_mg = digital.constellation_rect(
            [d + d * 1j, -d + d * 1j, -d - d * 1j, d - d * 1j],
            [0, 1, 3, 2],
            4,
            2,
            2,
            1,
            1,
        ).base()
        self.nfilts = nfilts = 32
        self.excess_bw = excess_bw = 0.35
        self.variable_adaptive_algorithm_0 = variable_adaptive_algorithm_0 = (
            digital.adaptive_algorithm_cma(qpsk_mg, eq_gain, 1).base()
        )
        self.samp_rate = samp_rate = 250000
        self.rrc_taps = rrc_taps = firdes.root_raised_cosine(
            nfilts, nfilts, 1.0 / float(sps), excess_bw, 11 * sps * nfilts
        )
        self.freq = freq = 920e6
        self.arity = arity = 4

        ##################################################
        # Blocks
        ##################################################
        self.uhd_usrp_source_0 = uhd.usrp_source(
            ",".join(("serial=31DB5AB", "")),
            uhd.stream_args(
                cpu_format="fc32",
                args="",
                channels=list(range(0, 1)),
            ),
        )
        self.uhd_usrp_source_0.set_samp_rate(samp_rate)
        # No synchronization enforced.
        self.uhd_usrp_source_0.set_center_freq(freq, 0)
        self.uhd_usrp_source_0.set_antenna("TX/RX", 0)
        self.uhd_usrp_source_0.set_gain(40, 0)
        self.uhd_usrp_source_0.set_auto_dc_offset(False, 0)
        self.uhd_usrp_source_0.set_auto_iq_balance(False, 0)

        self.digital_symbol_sync_xx_0 = digital.symbol_sync_cc(
            digital.TED_SIGNAL_TIMES_SLOPE_ML,
            sps,
            timing_loop_bw,
            1.0,
            1.0,
            1.5,
            2,
            digital.constellation_bpsk().base(),
            digital.IR_PFB_MF,
            nfilts,
            rrc_taps,
        )
        self.digital_meas_evm_cc_0 = digital.meas_evm_cc(
            qpsk_mg.base(), digital.evm_measurement_t.EVM_PERCENT
        )
        self.digital_linear_equalizer_0 = digital.linear_equalizer(
            15, 2, variable_adaptive_algorithm_0, True, [], "corr_est"
        )
        self.digital_diff_decoder_bb_0_0 = digital.diff_decoder_bb(
            4, digital.DIFF_DIFFERENTIAL
        )
        self.digital_costas_loop_cc_0 = digital.costas_loop_cc(phase_bw, arity, False)
        self.digital_constellation_decoder_cb_0 = digital.constellation_decoder_cb(
            qpsk_mg.base()
        )
        self.blocks_unpack_k_bits_bb_0 = blocks.unpack_k_bits_bb(2)
        self.blocks_char_to_float_0_0 = blocks.char_to_float(1, 1)

        # EVM smoothing + probe
        if self.evm_avg_len < 1:
            raise ValueError("evm_avg_len must be >= 1")
        self.blocks_moving_average_xx_0 = blocks.moving_average_ff(
            self.evm_avg_len,
            1.0 / float(self.evm_avg_len),
            4000,
            1,
        )
        self.blocks_probe_signal_x_0 = blocks.probe_signal_f()

        # Keep your original null sink (for decoded bits path)
        self.blocks_null_sink_0 = blocks.null_sink(gr.sizeof_float * 1)

        ##################################################
        # Connections
        ##################################################
        self.connect((self.digital_symbol_sync_xx_0, 0), (self.digital_linear_equalizer_0, 0))
        self.connect((self.digital_linear_equalizer_0, 0), (self.digital_costas_loop_cc_0, 0))

        # Decoder chain (unchanged)
        self.connect((self.digital_costas_loop_cc_0, 0), (self.digital_constellation_decoder_cb_0, 0))
        self.connect((self.digital_constellation_decoder_cb_0, 0), (self.digital_diff_decoder_bb_0_0, 0))
        self.connect((self.digital_diff_decoder_bb_0_0, 0), (self.blocks_unpack_k_bits_bb_0, 0))
        self.connect((self.blocks_unpack_k_bits_bb_0, 0), (self.blocks_char_to_float_0_0, 0))
        self.connect((self.blocks_char_to_float_0_0, 0), (self.blocks_null_sink_0, 0))

        # EVM path
        self.connect((self.digital_costas_loop_cc_0, 0), (self.digital_meas_evm_cc_0, 0))
        self.connect((self.digital_meas_evm_cc_0, 0), (self.blocks_moving_average_xx_0, 0))
        self.connect((self.blocks_moving_average_xx_0, 0), (self.blocks_probe_signal_x_0, 0))

        # Source
        self.connect((self.uhd_usrp_source_0, 0), (self.digital_symbol_sync_xx_0, 0))

    # -------------------------
    # Public helper for EVM
    # -------------------------
    def get_evm_pct(self) -> float:
        """Return current (smoothed) EVM in percent."""
        return float(self.blocks_probe_signal_x_0.level())

    # -------------------------
    # Auto-generated getters/setters (kept)
    # -------------------------
    def get_eq_gain(self):
        return self.eq_gain

    def set_eq_gain(self, eq_gain):
        self.eq_gain = eq_gain

    def get_phase_bw(self):
        return self.phase_bw

    def set_phase_bw(self, phase_bw):
        self.phase_bw = phase_bw
        self.digital_costas_loop_cc_0.set_loop_bandwidth(self.phase_bw)

    def get_timing_loop_bw(self):
        return self.timing_loop_bw

    def set_timing_loop_bw(self, timing_loop_bw):
        self.timing_loop_bw = timing_loop_bw
        self.digital_symbol_sync_xx_0.set_loop_bandwidth(self.timing_loop_bw)

    def get_d(self):
        return self.d

    def set_d(self, d):
        self.d = d

    def get_sps(self):
        return self.sps

    def set_sps(self, sps):
        self.sps = sps
        self.set_rrc_taps(
            firdes.root_raised_cosine(
                self.nfilts,
                self.nfilts,
                1.0 / float(self.sps),
                self.excess_bw,
                11 * self.sps * self.nfilts,
            )
        )
        self.digital_symbol_sync_xx_0.set_sps(self.sps)

    def get_qpsk_mg(self):
        return self.qpsk_mg

    def set_qpsk_mg(self, qpsk_mg):
        self.qpsk_mg = qpsk_mg

    def get_nfilts(self):
        return self.nfilts

    def set_nfilts(self, nfilts):
        self.nfilts = nfilts
        self.set_rrc_taps(
            firdes.root_raised_cosine(
                self.nfilts,
                self.nfilts,
                1.0 / float(self.sps),
                self.excess_bw,
                11 * self.sps * self.nfilts,
            )
        )

    def get_excess_bw(self):
        return self.excess_bw

    def set_excess_bw(self, excess_bw):
        self.excess_bw = excess_bw
        self.set_rrc_taps(
            firdes.root_raised_cosine(
                self.nfilts,
                self.nfilts,
                1.0 / float(self.sps),
                self.excess_bw,
                11 * self.sps * self.nfilts,
            )
        )

    def get_variable_adaptive_algorithm_0(self):
        return self.variable_adaptive_algorithm_0

    def set_variable_adaptive_algorithm_0(self, variable_adaptive_algorithm_0):
        self.variable_adaptive_algorithm_0 = variable_adaptive_algorithm_0

    def get_samp_rate(self):
        return self.samp_rate

    def set_samp_rate(self, samp_rate):
        self.samp_rate = samp_rate
        self.uhd_usrp_source_0.set_samp_rate(self.samp_rate)

    def get_rrc_taps(self):
        return self.rrc_taps

    def set_rrc_taps(self, rrc_taps):
        self.rrc_taps = rrc_taps

    def get_freq(self):
        return self.freq

    def set_freq(self, freq):
        self.freq = freq
        self.uhd_usrp_source_0.set_center_freq(self.freq, 0)

    def get_arity(self):
        return self.arity

    def set_arity(self, arity):
        self.arity = arity


def argument_parser():
    parser = ArgumentParser()
    # 如果你后面想 CLI 里调 evm_avg_len，也可以加参数
    # parser.add_argument("--evm-avg-len", type=int, default=200)
    return parser


def main(top_block_cls=qpsk_demodulator, options=None):
    if options is None:
        options = argument_parser().parse_args()

    tb = top_block_cls()

    # -----------------------------
    # ZMQ PUB: publish EVM to server
    # -----------------------------
    import threading
    import json
    import zmq
    import time as _time

    ctx = zmq.Context.instance()
    pub = ctx.socket(zmq.PUB)
    pub.bind("tcp://*:52001")  # user/client side opens port 50001

    stop_flag = {"stop": False}

    def evm_publisher():
        # PUB/SUB may drop messages until subscribers connect
        _time.sleep(0.2)
        while not stop_flag["stop"]:
            try:
                evm_pct = float(tb.get_evm_pct())  # EVM in percent from probe
                msg = {"t": _time.time(), "evm_pct": evm_pct}
                pub.send_string(json.dumps(msg))
            except Exception:
                # keep the publisher alive even if tb temporarily fails
                pass
            _time.sleep(0.05)  # 20 Hz

    th = threading.Thread(target=evm_publisher, daemon=True)

    def _cleanup_and_exit(code=0):
        stop_flag["stop"] = True
        try:
            tb.stop()
            tb.wait()
        except Exception:
            pass
        try:
            pub.close(0)
        except Exception:
            pass
        sys.exit(code)

    def sig_handler(sig=None, frame=None):
        _cleanup_and_exit(0)

    signal.signal(signal.SIGINT, sig_handler)
    signal.signal(signal.SIGTERM, sig_handler)

    tb.start()
    th.start()

    try:
        input("Press Enter to quit: ")
    except EOFError:
        pass
    except KeyboardInterrupt:
        pass

    _cleanup_and_exit(0)


if __name__ == "__main__":
    main()
