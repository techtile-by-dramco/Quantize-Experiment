"""
Interface skeleton for the Server–USRP command set.

Each method defines the expected arguments from the protocol; implement the
transport and payload formatting later. All commands are intended to be
non-blocking and time-referenced to USRP time (PPS aligned).
"""

import os

from pyvisa import logger
import uhd
from typing import Iterable, Optional, Sequence


class USRPClient:
    """Define the command interface; methods must be implemented by the user."""

    def load_config(self, config_path: str) -> None:
        """Load configuration from a YAML file.

        Args:
            config_path: Path to the configuration YAML file.
        """
        self.config = {}
        raise NotImplementedError

    def __init__(self) -> None:
        self.load_config("cal-settings.yml")
        fpga_path = "usrp_b210_fpga_loopback.bin"

        # Initialize USRP device with custom FPGA image and integer mode
        self.usrp = uhd.usrp.MultiUSRP(
            "enable_user_regs, " f"fpga={fpga_path}, " "mode_n=integer"
        )

    def setup_usrp(self):
        mcr = 20e6
        assert (
            mcr / rate
        ).is_integer(), f"The masterclock rate {mcr} should be an integer multiple of the sampling rate {rate}"
        # Manual selection of master clock rate may also be required to synchronize multiple B200 units in time.
        usrp.set_master_clock_rate(mcr)
        channels = [0, 1]
        setup_usrp_clock(usrp, "external", usrp.get_num_mboards())
        setup_usrp_pps(usrp, "external")
        # smallest as possible (https://files.ettus.com/manual/page_usrp_b200.html#b200_fe_bw)
        rx_bw = 200e3
        for chan in channels:
            usrp.set_rx_rate(rate, chan)
            usrp.set_tx_rate(rate, chan)
            # NOTE DC offset is enabled
            usrp.set_rx_dc_offset(True, chan)
            usrp.set_rx_bandwidth(rx_bw, chan)
            usrp.set_rx_agc(False, chan)
        # specific settings from loopback/REF PLL
        usrp.set_tx_gain(LOOPBACK_TX_GAIN, LOOPBACK_TX_CH)
        usrp.set_tx_gain(LOOPBACK_TX_GAIN, FREE_TX_CH)

        usrp.set_rx_gain(LOOPBACK_RX_GAIN, LOOPBACK_RX_CH)
        usrp.set_rx_gain(REF_RX_GAIN, REF_RX_CH)
        # streaming arguments
        st_args = uhd.usrp.StreamArgs("fc32", "sc16")
        st_args.channels = channels
        # streamers
        tx_streamer = usrp.get_tx_stream(st_args)
        rx_streamer = usrp.get_rx_stream(st_args)

        self.tx_streamer, self.rx_streamer = tx_streamer, rx_streamer


    def tune_usrp(at_time):
        """Synchronously set the device's frequency.
        If a channel is using an internal LO it will be tuned first
        and every other channel will be manually tuned based on the response.
        This is to account for the internal LO channel having an offset in the actual DSP frequency.
        Then all channels are synchronously tuned."""
        treq = uhd.types.TuneRequest(freq)
        usrp.set_command_time(uhd.types.TimeSpec(at_time))
        treq.dsp_freq = 0.0
        treq.target_freq = freq
        treq.rf_freq = freq
        treq.rf_freq_policy = uhd.types.TuneRequestPolicy(ord("M"))
        treq.dsp_freq_policy = uhd.types.TuneRequestPolicy(ord("M"))
        args = uhd.types.DeviceAddr("mode_n=integer")
        treq.args = args
        rx_freq = freq - 1e3
        rreq = uhd.types.TuneRequest(rx_freq)
        rreq.rf_freq = rx_freq
        rreq.target_freq = rx_freq
        rreq.dsp_freq = 0.0
        rreq.rf_freq_policy = uhd.types.TuneRequestPolicy(ord("M"))
        rreq.dsp_freq_policy = uhd.types.TuneRequestPolicy(ord("M"))
        rreq.args = uhd.types.DeviceAddr("mode_n=fractional")
        for chan in channels:
            print_tune_result(usrp.set_rx_freq(rreq, chan))
            print_tune_result(usrp.set_tx_freq(treq, chan))
        while not usrp.get_rx_sensor("lo_locked").to_bool():
            print(".")
            time.sleep(0.01)
        logger.info("RX LO is locked")
        while not usrp.get_tx_sensor("lo_locked").to_bool():
            print(".")
            time.sleep(0.01)
        logger.info("TX LO is locked")

    def sync(self, mode: str) -> None:
        """Align all tiles to a common time reference.

        Args:
            mode: "ON_NEXT_PPS" (align at next PPS edge) or "IMMEDIATE" (debug).
        """
        # Step1: wait for the last pps time to transition to catch the edge
        # Step2: set the time at the next pps (synchronous for all boards)
        # this is better than set_time_next_pps as we wait till the next PPS to transition and after that we set the time.
        # this ensures that the FPGA has enough time to clock in the new timespec (otherwise it could be too close to a PPS edge)
        logger.info("Setting device timestamp to 0...")
        # Synchronize the times across all motherboards in this configuration.
        usrp.set_time_unknown_pps(uhd.types.TimeSpec(0.0))
        logger.debug("[SYNC] Resetting time.")
        logger.info("RX GAIN PROFILE CH0: %s", usrp.get_rx_gain_names(0))
        logger.info("RX GAIN PROFILE CH1: %s", usrp.get_rx_gain_names(1))
        # we wait 3 seconds to ensure a PPS rising edge occurs and latches the 0.000s value to both USRPs.
        time.sleep(3)
        tune_usrp(at_time=BEGIN_TIME)
        logger.info(
            "USRP has been tuned and setup. (%s)", usrp.get_time_now().get_real_secs()
        )

    def cal(
        self,
        at_ms: Optional[int] = None,
        delay_ms: Optional[int] = None,
        mode: str = "LB",
    ) -> None:
        """Schedule calibration.

        Args:
            at_ms: Absolute USRP time in milliseconds.
            delay_ms: Relative delay in milliseconds from command receipt.
            mode: Calibration mode; currently "LB" (loopback).
        """
        raise NotImplementedError

    def pilot(
        self,
        at_ms: Optional[int] = None,
        delay_ms: Optional[int] = None,
        tx_tiles: Optional[Sequence[str]] = None,
        rx_tiles: Optional[Sequence[str]] = None,
        waveform: Optional[str] = None,
    ) -> None:
        """Schedule a pilot transmission/reception.

        Args:
            at_ms: Absolute USRP time in milliseconds.
            delay_ms: Relative delay in milliseconds from command receipt.
            tx_tiles: Tiles to transmit the pilot (None for all).
            rx_tiles: Tiles to receive the pilot (None for all).
            waveform: Pilot waveform file name.
        """
        raise NotImplementedError

    def setup(
        self,
        waveform: str,
        weights: str,
        direction: str,
        tiles: Optional[Iterable[str]] = None,
    ) -> None:
        """Load static experiment configuration (no RF activity).

        Args:
            waveform: IQ waveform file name.
            weights: Beamforming weights file name.
            direction: "tx" or "rx".
            tiles: Target tiles (None for all).
        """
        raise NotImplementedError

    def start(
        self,
        at_ms: Optional[int] = None,
        delay_ms: Optional[int] = None,
        mode: str = "CONTINUOUS",
        direction: str = "tx",
        duration_ms: Optional[int] = None,
        tiles: Optional[Iterable[str]] = None,
        waveform: Optional[str] = None,
        weights: Optional[str] = None,
    ) -> None:
        """Schedule RF activity.

        Args:
            at_ms: Absolute USRP time in milliseconds.
            delay_ms: Relative delay in milliseconds from command receipt.
            mode: "CONTINUOUS" or "BURST".
            direction: "tx" or "rx".
            duration_ms: Duration for BURST mode; ignored for CONTINUOUS.
            tiles: Target tiles (None for all).
            waveform: Override waveform file (optional).
            weights: Override weights file (optional).
        """
        raise NotImplementedError

    def stop(
        self,
        at_ms: Optional[int] = None,
        delay_ms: Optional[int] = None,
        direction: str = "both",
        tiles: Optional[Iterable[str]] = None,
    ) -> None:
        """Stop RF activity and cancel pending starts.

        Args:
            at_ms: Absolute USRP time in milliseconds.
            delay_ms: Relative delay in milliseconds from command receipt.
            direction: "tx", "rx", or "both".
            tiles: Target tiles (None for all).
        """
        raise NotImplementedError

    def status(self, query: str, tiles: Optional[Iterable[str]] = None) -> None:
        """Query system state.

        Args:
            query: "TIME", "STATE", or "SETUP".
            tiles: Target tiles (None for all).
        """
        raise NotImplementedError

    def abort(self, tiles: Optional[Iterable[str]] = None) -> None:
        """Immediate safety stop; clears pending schedules but keeps last setup.

        Args:
            tiles: Target tiles (None for all).
        """
        raise NotImplementedError
