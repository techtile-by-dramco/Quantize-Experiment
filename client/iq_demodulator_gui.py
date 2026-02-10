#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#
# SPDX-License-Identifier: GPL-3.0
#
# GNU Radio Python Flow Graph
# Title: iq_demodulator_gui
# Author: Tianzheng
# GNU Radio version: 3.10.10.0

from PyQt5 import Qt
from gnuradio import qtgui
from PyQt5 import QtCore
from gnuradio import analog
from gnuradio import blocks
from gnuradio import digital
from gnuradio import filter
from gnuradio import gr
from gnuradio.filter import firdes
from gnuradio.fft import window
import sys
import signal
from PyQt5 import Qt
from argparse import ArgumentParser
from gnuradio.eng_arg import eng_float, intx
from gnuradio import eng_notation
from gnuradio import uhd
import time
import sip
import threading
import json
import zmq
import time as _time



class iq_demodulator_gui(gr.top_block, Qt.QWidget):

    def __init__(self):
        gr.top_block.__init__(self, "iq_demodulator_gui", catch_exceptions=True)
        Qt.QWidget.__init__(self)
        self.setWindowTitle("iq_demodulator_gui")
        qtgui.util.check_set_qss()
        try:
            self.setWindowIcon(Qt.QIcon.fromTheme('gnuradio-grc'))
        except BaseException as exc:
            print(f"Qt GUI: Could not set Icon: {str(exc)}", file=sys.stderr)
        self.top_scroll_layout = Qt.QVBoxLayout()
        self.setLayout(self.top_scroll_layout)
        self.top_scroll = Qt.QScrollArea()
        self.top_scroll.setFrameStyle(Qt.QFrame.NoFrame)
        self.top_scroll_layout.addWidget(self.top_scroll)
        self.top_scroll.setWidgetResizable(True)
        self.top_widget = Qt.QWidget()
        self.top_scroll.setWidget(self.top_widget)
        self.top_layout = Qt.QVBoxLayout(self.top_widget)
        self.top_grid_layout = Qt.QGridLayout()
        self.top_layout.addLayout(self.top_grid_layout)

        self.settings = Qt.QSettings("GNU Radio", "iq_demodulator_gui")

        try:
            geometry = self.settings.value("geometry")
            if geometry:
                self.restoreGeometry(geometry)
        except BaseException as exc:
            print(f"Qt GUI: Could not restore geometry: {str(exc)}", file=sys.stderr)

        ##################################################
        # Variables
        ##################################################
        self.d = d = 1/(2**(1/2))
        self.sps = sps = 4
        self.qpsk_mg = qpsk_mg = digital.constellation_rect([d+d*1j, -d+d*1j,-d-d*1j, d-d*1j], [0, 1, 3, 2],
        4, 2, 2, 1, 1).base()
        self.nfilts = nfilts = 32
        self.excess_bw = excess_bw = 0.35
        self.eq_gain = eq_gain = 0.0001
        self.variable_adaptive_algorithm_0 = variable_adaptive_algorithm_0 = digital.adaptive_algorithm_cma( qpsk_mg, eq_gain, 1).base()
        self.timing_loop_bw = timing_loop_bw = 6.28/200
        self.taps_per_filt = taps_per_filt = int((11*sps*nfilts)/nfilts)
        self.taps_1 = taps_1 = [1,0,0,0.5]
        self.taps = taps = [0.825,0,0,0,0.526]
        self.samp_rate = samp_rate = 250000
        self.rrc_taps_tx = rrc_taps_tx = firdes.root_raised_cosine(nfilts, nfilts, 1.0, excess_bw, 11*sps*nfilts)
        self.rrc_taps = rrc_taps = firdes.root_raised_cosine(nfilts, nfilts, 1.0/float(sps), excess_bw, 11*sps*nfilts)
        self.phase_bw = phase_bw = 6.28/200
        self.noise_volt = noise_volt = 0
        self.gain_tx = gain_tx = 50
        self.gain_rx = gain_rx = 40
        self.freq_offset = freq_offset = 0
        self.freq = freq = 920e6
        self.delay = delay = 32
        self.arity = arity = 4

        ##################################################
        # Blocks
        ##################################################

        self.controls = Qt.QTabWidget()
        self.controls_widget_0 = Qt.QWidget()
        self.controls_layout_0 = Qt.QBoxLayout(Qt.QBoxLayout.TopToBottom, self.controls_widget_0)
        self.controls_grid_layout_0 = Qt.QGridLayout()
        self.controls_layout_0.addLayout(self.controls_grid_layout_0)
        self.controls.addTab(self.controls_widget_0, 'TX')
        self.controls_widget_1 = Qt.QWidget()
        self.controls_layout_1 = Qt.QBoxLayout(Qt.QBoxLayout.TopToBottom, self.controls_widget_1)
        self.controls_grid_layout_1 = Qt.QGridLayout()
        self.controls_layout_1.addLayout(self.controls_grid_layout_1)
        self.controls.addTab(self.controls_widget_1, 'RX')
        self.top_grid_layout.addWidget(self.controls, 0, 0, 1, 2)
        for r in range(0, 1):
            self.top_grid_layout.setRowStretch(r, 1)
        for c in range(0, 2):
            self.top_grid_layout.setColumnStretch(c, 1)
        self._timing_loop_bw_range = qtgui.Range(0.0, 0.1, 0.001, 6.28/200, 200)
        self._timing_loop_bw_win = qtgui.RangeWidget(self._timing_loop_bw_range, self.set_timing_loop_bw, "timing_loop_bw", "counter_slider", float, QtCore.Qt.Horizontal)
        self.controls_grid_layout_1.addWidget(self._timing_loop_bw_win, 0, 1, 1, 1)
        for r in range(0, 1):
            self.controls_grid_layout_1.setRowStretch(r, 1)
        for c in range(1, 2):
            self.controls_grid_layout_1.setColumnStretch(c, 1)
        self._phase_bw_range = qtgui.Range(0.0, 0.1, 0.01, 6.28/200, 200)
        self._phase_bw_win = qtgui.RangeWidget(self._phase_bw_range, self.set_phase_bw, "Phase: Bandwidth", "counter_slider", float, QtCore.Qt.Horizontal)
        self.controls_grid_layout_1.addWidget(self._phase_bw_win, 1, 1, 1, 1)
        for r in range(1, 2):
            self.controls_grid_layout_1.setRowStretch(r, 1)
        for c in range(1, 2):
            self.controls_grid_layout_1.setColumnStretch(c, 1)
        self._gain_rx_range = qtgui.Range(0, 74, 1, 40, 200)
        self._gain_rx_win = qtgui.RangeWidget(self._gain_rx_range, self.set_gain_rx, "gain_rx", "counter_slider", float, QtCore.Qt.Horizontal)
        self.controls_grid_layout_1.addWidget(self._gain_rx_win, 0, 0, 1, 1)
        for r in range(0, 1):
            self.controls_grid_layout_1.setRowStretch(r, 1)
        for c in range(0, 1):
            self.controls_grid_layout_1.setColumnStretch(c, 1)
        self.uhd_usrp_source_0 = uhd.usrp_source(
            ",".join(("serial=31DEA81", "")),
            uhd.stream_args(
                cpu_format="fc32",
                args='',
                channels=list(range(0,1)),
            ), # T01
        # self.uhd_usrp_source_0 = uhd.usrp_source(
        #     ",".join(("serial=31DEAB8", "")),
        #     uhd.stream_args(
        #         cpu_format="fc32",
        #         args='',
        #         channels=list(range(0,1)),
        #     ), # T03
        )
        self.uhd_usrp_source_0.set_samp_rate(samp_rate)
        # No synchronization enforced.

        self.uhd_usrp_source_0.set_center_freq(freq, 0)
        self.uhd_usrp_source_0.set_antenna('TX/RX', 0)
        self.uhd_usrp_source_0.set_gain(gain_rx, 0)
        self.uhd_usrp_source_0.set_auto_dc_offset(False, 0)
        self.uhd_usrp_source_0.set_auto_iq_balance(False, 0)
        self.qtgui_sink_x_1_0 = qtgui.sink_c(
            1024, #fftsize
            window.WIN_BLACKMAN_hARRIS, #wintype
            freq, #fc
            samp_rate, #bw
            "Rx_Signal", #name
            True, #plotfreq
            True, #plotwaterfall
            True, #plottime
            True, #plotconst
            None # parent
        )
        self.qtgui_sink_x_1_0.set_update_time(1.0/10)
        self._qtgui_sink_x_1_0_win = sip.wrapinstance(self.qtgui_sink_x_1_0.qwidget(), Qt.QWidget)

        self.qtgui_sink_x_1_0.enable_rf_freq(False)

        self.top_grid_layout.addWidget(self._qtgui_sink_x_1_0_win, 1, 1, 1, 1)
        for r in range(1, 2):
            self.top_grid_layout.setRowStretch(r, 1)
        for c in range(1, 2):
            self.top_grid_layout.setColumnStretch(c, 1)
        self.qtgui_number_sink_1 = qtgui.number_sink(
            gr.sizeof_float,
            0,
            qtgui.NUM_GRAPH_HORIZ,
            1,
            None # parent
        )
        self.qtgui_number_sink_1.set_update_time(0.05)
        self.qtgui_number_sink_1.set_title("EVM")

        labels = ['', '', '', '', '',
            '', '', '', '', '']
        units = ['', '', '', '', '',
            '', '', '', '', '']
        colors = [("black", "red"), ("black", "black"), ("black", "black"), ("black", "black"), ("black", "black"),
            ("black", "black"), ("black", "black"), ("black", "black"), ("black", "black"), ("black", "black")]
        factor = [1, 1, 1, 1, 1,
            1, 1, 1, 1, 1]

        for i in range(1):
            self.qtgui_number_sink_1.set_min(i, -1)
            self.qtgui_number_sink_1.set_max(i, 1)
            self.qtgui_number_sink_1.set_color(i, colors[i][0], colors[i][1])
            if len(labels[i]) == 0:
                self.qtgui_number_sink_1.set_label(i, "Data {0}".format(i))
            else:
                self.qtgui_number_sink_1.set_label(i, labels[i])
            self.qtgui_number_sink_1.set_unit(i, units[i])
            self.qtgui_number_sink_1.set_factor(i, factor[i])

        self.qtgui_number_sink_1.enable_autoscale(True)
        self._qtgui_number_sink_1_win = sip.wrapinstance(self.qtgui_number_sink_1.qwidget(), Qt.QWidget)
        self.top_layout.addWidget(self._qtgui_number_sink_1_win)
        self.qtgui_const_sink_x_0 = qtgui.const_sink_c(
            1024, #size
            "RX_Constellation", #name
            1, #number of inputs
            None # parent
        )
        self.qtgui_const_sink_x_0.set_update_time(0.10)
        self.qtgui_const_sink_x_0.set_y_axis((-2), 2)
        self.qtgui_const_sink_x_0.set_x_axis((-2), 2)
        self.qtgui_const_sink_x_0.set_trigger_mode(qtgui.TRIG_MODE_FREE, qtgui.TRIG_SLOPE_POS, 0.0, 0, "")
        self.qtgui_const_sink_x_0.enable_autoscale(False)
        self.qtgui_const_sink_x_0.enable_grid(True)
        self.qtgui_const_sink_x_0.enable_axis_labels(True)


        labels = ['', '', '', '', '',
            '', '', '', '', '']
        widths = [1, 1, 1, 1, 1,
            1, 1, 1, 1, 1]
        colors = ["blue", "red", "red", "red", "red",
            "red", "red", "red", "red", "red"]
        styles = [0, 0, 0, 0, 0,
            0, 0, 0, 0, 0]
        markers = [0, 0, 0, 0, 0,
            0, 0, 0, 0, 0]
        alphas = [1.0, 1.0, 1.0, 1.0, 1.0,
            1.0, 1.0, 1.0, 1.0, 1.0]

        for i in range(1):
            if len(labels[i]) == 0:
                self.qtgui_const_sink_x_0.set_line_label(i, "Data {0}".format(i))
            else:
                self.qtgui_const_sink_x_0.set_line_label(i, labels[i])
            self.qtgui_const_sink_x_0.set_line_width(i, widths[i])
            self.qtgui_const_sink_x_0.set_line_color(i, colors[i])
            self.qtgui_const_sink_x_0.set_line_style(i, styles[i])
            self.qtgui_const_sink_x_0.set_line_marker(i, markers[i])
            self.qtgui_const_sink_x_0.set_line_alpha(i, alphas[i])

        self._qtgui_const_sink_x_0_win = sip.wrapinstance(self.qtgui_const_sink_x_0.qwidget(), Qt.QWidget)
        self.top_grid_layout.addWidget(self._qtgui_const_sink_x_0_win, 1, 0, 1, 1)
        for r in range(1, 2):
            self.top_grid_layout.setRowStretch(r, 1)
        for c in range(0, 1):
            self.top_grid_layout.setColumnStretch(c, 1)
        self._noise_volt_range = qtgui.Range(0, 2, 0.01, 0, 200)
        self._noise_volt_win = qtgui.RangeWidget(self._noise_volt_range, self.set_noise_volt, "Noise Voltage", "counter_slider", float, QtCore.Qt.Horizontal)
        self.controls_grid_layout_0.addWidget(self._noise_volt_win, 1, 0, 1, 1)
        for r in range(1, 2):
            self.controls_grid_layout_0.setRowStretch(r, 1)
        for c in range(0, 1):
            self.controls_grid_layout_0.setColumnStretch(c, 1)
        self._gain_tx_range = qtgui.Range(0, 74, 1, 50, 200)
        self._gain_tx_win = qtgui.RangeWidget(self._gain_tx_range, self.set_gain_tx, "gain_tx", "counter_slider", int, QtCore.Qt.Horizontal)
        self.controls_grid_layout_0.addWidget(self._gain_tx_win, 0, 0, 1, 1)
        for r in range(0, 1):
            self.controls_grid_layout_0.setRowStretch(r, 1)
        for c in range(0, 1):
            self.controls_grid_layout_0.setColumnStretch(c, 1)
        self._freq_offset_range = qtgui.Range(-0.1, 0.1, 0.001, 0, 200)
        self._freq_offset_win = qtgui.RangeWidget(self._freq_offset_range, self.set_freq_offset, "Frequency Offset", "counter_slider", float, QtCore.Qt.Horizontal)
        self.controls_grid_layout_0.addWidget(self._freq_offset_win, 1, 1, 1, 1)
        for r in range(1, 2):
            self.controls_grid_layout_0.setRowStretch(r, 1)
        for c in range(1, 2):
            self.controls_grid_layout_0.setColumnStretch(c, 1)
        self._eq_gain_range = qtgui.Range(0.0, 0.001, 0.0001, 0.0001, 200)
        self._eq_gain_win = qtgui.RangeWidget(self._eq_gain_range, self.set_eq_gain, "Equalizer: rate", "counter_slider", float, QtCore.Qt.Horizontal)
        self.controls_grid_layout_1.addWidget(self._eq_gain_win, 1, 0, 1, 1)
        for r in range(1, 2):
            self.controls_grid_layout_1.setRowStretch(r, 1)
        for c in range(0, 1):
            self.controls_grid_layout_1.setColumnStretch(c, 1)
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
            rrc_taps)
        self.digital_mpsk_snr_est_cc_0 = digital.mpsk_snr_est_cc(0, 10000, 0.001)
        self.digital_meas_evm_cc_0 = digital.meas_evm_cc(qpsk_mg.base(),digital.evm_measurement_t.EVM_PERCENT)
        self.digital_linear_equalizer_0 = digital.linear_equalizer(15, 2, variable_adaptive_algorithm_0, True, [ ], 'corr_est')
        self.digital_costas_loop_cc_0 = digital.costas_loop_cc(phase_bw, arity, False)
        self._delay_range = qtgui.Range(0, 200, 1, 32, 200)
        self._delay_win = qtgui.RangeWidget(self._delay_range, self.set_delay, "Delay", "counter_slider", int, QtCore.Qt.Horizontal)
        self.controls_grid_layout_0.addWidget(self._delay_win, 0, 1, 1, 1)
        for r in range(0, 1):
            self.controls_grid_layout_0.setRowStretch(r, 1)
        for c in range(1, 2):
            self.controls_grid_layout_0.setColumnStretch(c, 1)
        self.blocks_tag_debug_0 = blocks.tag_debug(gr.sizeof_gr_complex*1, 'SNR', "")
        self.blocks_tag_debug_0.set_display(True)
        self.blocks_probe_signal_x_0 = blocks.probe_signal_f()
        self.blocks_moving_average_xx_0_0 = blocks.moving_average_ff(400, 0.002, 4000, 1)
        self.analog_agc_xx_0 = analog.agc_cc((1e-4), 1.0, 1.0)


        ##################################################
        # Connections
        ##################################################
        self.connect((self.analog_agc_xx_0, 0), (self.digital_symbol_sync_xx_0, 0))
        self.connect((self.analog_agc_xx_0, 0), (self.qtgui_sink_x_1_0, 0))
        self.connect((self.blocks_moving_average_xx_0_0, 0), (self.blocks_probe_signal_x_0, 0))
        self.connect((self.blocks_moving_average_xx_0_0, 0), (self.qtgui_number_sink_1, 0))
        self.connect((self.digital_costas_loop_cc_0, 0), (self.digital_meas_evm_cc_0, 0))
        self.connect((self.digital_costas_loop_cc_0, 0), (self.digital_mpsk_snr_est_cc_0, 0))
        self.connect((self.digital_costas_loop_cc_0, 0), (self.qtgui_const_sink_x_0, 0))
        self.connect((self.digital_linear_equalizer_0, 0), (self.digital_costas_loop_cc_0, 0))
        self.connect((self.digital_meas_evm_cc_0, 0), (self.blocks_moving_average_xx_0_0, 0))
        self.connect((self.digital_mpsk_snr_est_cc_0, 0), (self.blocks_tag_debug_0, 0))
        self.connect((self.digital_symbol_sync_xx_0, 0), (self.digital_linear_equalizer_0, 0))
        self.connect((self.uhd_usrp_source_0, 0), (self.analog_agc_xx_0, 0))
    # -----------------------------
    # EVM getter (from probe)
    # -----------------------------
    def get_evm_pct(self):
        # blocks_probe_signal_x_0 is fed by moving average of meas_evm_cc_0 output
        return float(self.blocks_probe_signal_x_0.level())

    # -----------------------------
    # ZMQ PUB: publish EVM
    # -----------------------------
    def start_evm_publisher(self, bind_addr="tcp://*:52001", period_s=0.05):
        """
        Start a background thread that publishes EVM (%).
        Message format: {"t": <unix_time>, "evm_pct": <float>}
        """
        # avoid double-start
        if getattr(self, "_evm_pub_running", False):
            return

        self._evm_pub_running = True
        self._evm_pub_period_s = period_s

        self._zmq_ctx = zmq.Context.instance()
        self._evm_pub = self._zmq_ctx.socket(zmq.PUB)
        self._evm_pub.bind(bind_addr)

        self._evm_stop_flag = {"stop": False}

        def _publisher():
            # PUB/SUB may drop messages until subscribers connect
            _time.sleep(0.2)
            while not self._evm_stop_flag["stop"]:
                try:
                    evm_pct = self.get_evm_pct()
                    msg = {"t": _time.time(), "evm_pct": evm_pct}
                    self._evm_pub.send_string(json.dumps(msg))
                except Exception:
                    # keep alive even if temporarily not ready
                    pass
                _time.sleep(self._evm_pub_period_s)

        self._evm_thread = threading.Thread(target=_publisher, daemon=True)
        self._evm_thread.start()

    def stop_evm_publisher(self):
        if not getattr(self, "_evm_pub_running", False):
            return
        self._evm_pub_running = False

        try:
            self._evm_stop_flag["stop"] = True
        except Exception:
            pass

        try:
            # Do not hang on close
            self._evm_pub.close(0)
        except Exception:
            pass


    def closeEvent(self, event):
        self.settings = Qt.QSettings("GNU Radio", "iq_demodulator_gui")
        self.settings.setValue("geometry", self.saveGeometry())

        # stop publisher first
        self.stop_evm_publisher()

        self.stop()
        self.wait()
        event.accept()


    def get_d(self):
        return self.d

    def set_d(self, d):
        self.d = d

    def get_sps(self):
        return self.sps

    def set_sps(self, sps):
        self.sps = sps
        self.set_rrc_taps(firdes.root_raised_cosine(self.nfilts, self.nfilts, 1.0/float(self.sps), self.excess_bw, 11*self.sps*self.nfilts))
        self.set_rrc_taps_tx(firdes.root_raised_cosine(self.nfilts, self.nfilts, 1.0, self.excess_bw, 11*self.sps*self.nfilts))
        self.set_taps_per_filt(int((11*self.sps*self.nfilts)/self.nfilts))
        self.digital_symbol_sync_xx_0.set_sps(self.sps)

    def get_qpsk_mg(self):
        return self.qpsk_mg

    def set_qpsk_mg(self, qpsk_mg):
        self.qpsk_mg = qpsk_mg

    def get_nfilts(self):
        return self.nfilts

    def set_nfilts(self, nfilts):
        self.nfilts = nfilts
        self.set_rrc_taps(firdes.root_raised_cosine(self.nfilts, self.nfilts, 1.0/float(self.sps), self.excess_bw, 11*self.sps*self.nfilts))
        self.set_rrc_taps_tx(firdes.root_raised_cosine(self.nfilts, self.nfilts, 1.0, self.excess_bw, 11*self.sps*self.nfilts))
        self.set_taps_per_filt(int((11*self.sps*self.nfilts)/self.nfilts))

    def get_excess_bw(self):
        return self.excess_bw

    def set_excess_bw(self, excess_bw):
        self.excess_bw = excess_bw
        self.set_rrc_taps(firdes.root_raised_cosine(self.nfilts, self.nfilts, 1.0/float(self.sps), self.excess_bw, 11*self.sps*self.nfilts))
        self.set_rrc_taps_tx(firdes.root_raised_cosine(self.nfilts, self.nfilts, 1.0, self.excess_bw, 11*self.sps*self.nfilts))

    def get_eq_gain(self):
        return self.eq_gain

    def set_eq_gain(self, eq_gain):
        self.eq_gain = eq_gain

    def get_variable_adaptive_algorithm_0(self):
        return self.variable_adaptive_algorithm_0

    def set_variable_adaptive_algorithm_0(self, variable_adaptive_algorithm_0):
        self.variable_adaptive_algorithm_0 = variable_adaptive_algorithm_0

    def get_timing_loop_bw(self):
        return self.timing_loop_bw

    def set_timing_loop_bw(self, timing_loop_bw):
        self.timing_loop_bw = timing_loop_bw
        self.digital_symbol_sync_xx_0.set_loop_bandwidth(self.timing_loop_bw)

    def get_taps_per_filt(self):
        return self.taps_per_filt

    def set_taps_per_filt(self, taps_per_filt):
        self.taps_per_filt = taps_per_filt

    def get_taps_1(self):
        return self.taps_1

    def set_taps_1(self, taps_1):
        self.taps_1 = taps_1

    def get_taps(self):
        return self.taps

    def set_taps(self, taps):
        self.taps = taps

    def get_samp_rate(self):
        return self.samp_rate

    def set_samp_rate(self, samp_rate):
        self.samp_rate = samp_rate
        self.qtgui_sink_x_1_0.set_frequency_range(self.freq, self.samp_rate)
        self.uhd_usrp_source_0.set_samp_rate(self.samp_rate)

    def get_rrc_taps_tx(self):
        return self.rrc_taps_tx

    def set_rrc_taps_tx(self, rrc_taps_tx):
        self.rrc_taps_tx = rrc_taps_tx

    def get_rrc_taps(self):
        return self.rrc_taps

    def set_rrc_taps(self, rrc_taps):
        self.rrc_taps = rrc_taps

    def get_phase_bw(self):
        return self.phase_bw

    def set_phase_bw(self, phase_bw):
        self.phase_bw = phase_bw
        self.digital_costas_loop_cc_0.set_loop_bandwidth(self.phase_bw)

    def get_noise_volt(self):
        return self.noise_volt

    def set_noise_volt(self, noise_volt):
        self.noise_volt = noise_volt

    def get_gain_tx(self):
        return self.gain_tx

    def set_gain_tx(self, gain_tx):
        self.gain_tx = gain_tx

    def get_gain_rx(self):
        return self.gain_rx

    def set_gain_rx(self, gain_rx):
        self.gain_rx = gain_rx
        self.uhd_usrp_source_0.set_gain(self.gain_rx, 0)

    def get_freq_offset(self):
        return self.freq_offset

    def set_freq_offset(self, freq_offset):
        self.freq_offset = freq_offset

    def get_freq(self):
        return self.freq

    def set_freq(self, freq):
        self.freq = freq
        self.qtgui_sink_x_1_0.set_frequency_range(self.freq, self.samp_rate)
        self.uhd_usrp_source_0.set_center_freq(self.freq, 0)

    def get_delay(self):
        return self.delay

    def set_delay(self, delay):
        self.delay = delay

    def get_arity(self):
        return self.arity

    def set_arity(self, arity):
        self.arity = arity




def main(top_block_cls=iq_demodulator_gui, options=None):

    qapp = Qt.QApplication(sys.argv)

    tb = top_block_cls()

    tb.start()

    # Start ZMQ publisher (same as your non-GUI version)
    tb.start_evm_publisher(bind_addr="tcp://*:52001", period_s=0.05)

    tb.show()

    def sig_handler(sig=None, frame=None):
        try:
            tb.stop_evm_publisher()
        except Exception:
            pass

        tb.stop()
        tb.wait()
        Qt.QApplication.quit()

    signal.signal(signal.SIGINT, sig_handler)
    signal.signal(signal.SIGTERM, sig_handler)

    timer = Qt.QTimer()
    timer.start(500)
    timer.timeout.connect(lambda: None)

    qapp.exec_()


if __name__ == '__main__':
    main()
