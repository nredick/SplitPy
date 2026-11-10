#!/usr/bin/env python

# Copyright 2019 Pascal Audet & Andrew Schaeffer
#
# This file is part of SplitPy.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# -*- coding: utf-8 -*-
# -*- coding: utf-8 -*-
# Optimized driver for SplitPy automatic splitting (RC + SC methods).
# The splitting calculation itself (Split.analyze / is_null / get_quality and
# everything upstream of it) is called exactly as in the original script.
import os

# One BLAS/OpenMP thread per worker: parallelism comes from the process pool.
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import io
import copy
import json
import time
import pickle
import contextlib
import traceback
import multiprocessing as mp

import numpy as np
import matplotlib
matplotlib.use('Agg')          # headless-safe; required for worker processes
import matplotlib.pyplot as plt
import stdb

from obspy.clients.fdsn import Client as FDSN_Client
from obspy.clients.filesystem.sds import Client as SDS_Client
from obspy import UTCDateTime

from splitpy import utils
from splitpy import Split, DiagPlot

from argparse import ArgumentParser
from os.path import exists as exist
from pathlib import Path


def get_arguments_calc_auto(argv=None):

    parser = ArgumentParser(
        usage="%(prog)s [arguments] <station database>",
        description="Script wrapping "
        "together the python-based implementation of SplitLab by " +
        "Wustefeld and others. This version " +
        "requests data on the fly for a given date range. Data is " +
        "requested from the internet using " +
        "the client services framework or from data provided on a " +
        "local disk. The stations are processed " +
        "one by one with the SKS Splitting parameters measured " +
        "individually using both the " +
        "Rotation-Correlation (RC) and Silver & Chan (SC) methods.")
    parser.add_argument(
        "indb",
        help="Station Database to process from.",
        type=str)
    parser.add_argument(
        "--keys",
        action="store",
        type=str,
        dest="stkeys",
        default="",
        help="Specify a comma separated list of station keys for " +
        "which to perform the analysis. These must be " +
        "contained within the station database. Partial keys " +
        "will be used to match against those in the " +
        "dictionary. For instance, providing IU will match with " +
        "all stations in the IU network [Default processes " +
        "all stations in the database]")
    parser.add_argument(
        "-V", "--verbose",
        action="store_true",
        dest="verb",
        default=False,
        help="Specify to increase verbosity.")
    parser.add_argument(
        "-O", "--overwrite",
        action="store_true",
        dest="ovr",
        default=False,
        help="Force the overwriting of pre-existing Split results. " +
        "Default behaviour prompts for those that " +
        "already exist. Selecting overwrite and skip (ie, both flags) " +
        "negate each other, and both are set to " +
        "false (every repeat is prompted). [Default False]")
    parser.add_argument(
        "--zcomp", 
        dest="zcomp",
        type=str,
        default="Z",
        help="Specify the Vertical Component Channel Identifier. "+
        "[Default Z].")
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        dest="skip",
        default=False,
        help="Skip any event for which existing splitting results are " +
        "saved to disk. Default behaviour prompts for " +
        "each event. Selecting skip and overwrite (ie, both flags) " +
        "negate each other, and both are set to " +
        "False (every repeat is prompted). [Default False]")
    parser.add_argument(
        "--calc",
        action="store_true",
        dest="calc",
        default=False,
        help="Analyze data for shear-wave splitting. [Default saves data "+
        "to folders for subsequent analysis]")
    parser.add_argument(
        "--plot-diagnostic",
        action="store_true",
        dest="diagplot",
        default=False,
        help="Plot diagnostic window at end of process. [Default False]")
    parser.add_argument(
        "--recalc",
        action="store_true",
        dest="recalc",
        default=False,
        help="Re-calculate estimates and overwrite existing splitting "+
        "results without re-downloading data. [Default False]")

    # Server Settings
    ServerGroup = parser.add_argument_group(
        title="Server Settings",
        description="Settings associated with which "
        "datacenter to log into.")
    ServerGroup.add_argument(
        "--server",
        action="store",
        type=str,
        dest="server",
        default="IRIS",
        help="Base URL of FDSN web service compatible "
        "server (e.g. “http://service.iris.edu”) or key string for recognized "
        "server (one of 'AUSPASS', 'BGR', 'EARTHSCOPE', 'EIDA', 'EMSC', 'ETH', "
        "'GEOFON', 'GEONET', 'GFZ', 'ICGC', 'IESDMC', 'INGV', 'IPGP', 'IRIS', "
        "'IRISPH5', 'ISC', 'KNMI', 'KOERI', 'LMU', 'NCEDC', 'NIEP', 'NOA', "
        "'NRCAN', 'ODC', 'ORFEUS', 'RASPISHAKE', 'RESIF', 'RESIFPH5', 'SCEDC', "
        "'TEXNET', 'UIB-NORSAR', 'USGS', 'USP'). [Default 'IRIS']")
    ServerGroup.add_argument(
        "--user-auth",
        action="store",
        type=str,
        dest="userauth",
        default=None,
        help="Authentification Username and Password for the " +
        "waveform server (--user-auth='username:authpassword') to access " +
        "and download restricted data. [Default no user and password]")
    ServerGroup.add_argument(
        "--eida-token", 
        action="store", 
        type=str,
        dest="tokenfile", 
        default=None, 
        help="Token for EIDA authentication mechanism, see " +
        "http://geofon.gfz-potsdam.de/waveform/archive/auth/index.php. "
        "If a token is provided, argument --user-auth will be ignored. "
        "This mechanism is only available on select EIDA nodes. The token can "
        "be provided in form of the PGP message as a string, or the filename of "
        "a local file with the PGP message in it. [Default None]")

    # Database Settings
    DataGroup = parser.add_argument_group(
        title="Local Data Settings",
        description="Settings associated with defining " +
        "and using a local data base of pre-downloaded " +
        "day-long SAC or MSEED files.")
    DataGroup.add_argument(
        "--local-data",
        action="store",
        type=str,
        dest="localdata",
        default=None,
        help="Specify absolute path to a SeisComP Data Structure (SDS) " +
        "archive containing day-long SAC or MSEED files" +
        "(e.g., --local-data=/Home/username/Data/SDS). " +
        "See https://www.seiscomp.de/seiscomp3/doc/applications/slarchive/SDS.html " +
        "for details on the SDS format. If this option is used, it takes " +
        "precedence over the --server settings.")
    DataGroup.add_argument(
        "--dtype",
        action="store",
        type=str,
        dest="dtype",
        default='MSEED',
        help="Specify the data archive file type, either SAC " +
        " or MSEED. Note the default behaviour is to search for " +
        "SAC files. Local archive files must have extensions of " +
        "'.SAC'  or '.MSEED'. These are case dependent, so specify " +
        "the correct case here.")
    # DataGroup.add_argument(
    #     "--no-data-zero",
    #     action="store_true",
    #     dest="ndval",
    #     default=False,
    #     help="Specify to force missing data to be set as zero, rather " +
    #     "than default behaviour which sets to nan.")
    # DataGroup.add_argument(
    #     "--no-local-net",
    #     action="store_false",
    #     dest="useNet",
    #     default=True,
    #     help="Specify to prevent using the Network code in the " +
    #     "search for local data (sometimes for CN stations " +
    #     "the dictionary name for a station may disagree with that " +
    #     "in the filename. [Default Network used]")

    # Constants Settings
    ConstGroup = parser.add_argument_group(
        title='Parameter Settings',
        description="Miscellaneous default values and settings")
    ConstGroup.add_argument(
        "--sampling-rate",
        action="store",
        type=float,
        dest="new_sampling_rate",
        default=10.,
        help="Specify new sampling rate in Hz. [Default 10.]")
    ConstGroup.add_argument(
        "--min-snr",
        action="store",
        type=float,
        dest="msnr",
        default=5.,
        help="Minimum SNR value calculated on the radial (Q) component "+
        "to proceed with analysis (dB). [Default 5.]")
    ConstGroup.add_argument(
        "--window",
        action="store",
        type=float,
        dest="dts",
        default=120.,
        help="Specify time window length before and after the SKS "
        "arrival. The total window length is 2*dst (sec). [Default 120]")
    ConstGroup.add_argument(
        "--max-delay",
        action="store",
        type=float,
        dest="maxdt",
        default=4.,
        help="Specify the maximum delay time in search (sec). "+
        "[Default 4]")
    ConstGroup.add_argument(
        "--dt-delay",
        action="store",
        type=float,
        dest="ddt",
        default=0.1,
        help="Specify the time delay increment in search (sec). "+
        "[Default 0.1]")
    ConstGroup.add_argument(
        "--dphi",
        action="store",
        type=float,
        dest="dphi",
        default=1.,
        help="Specify the fast angle increment in search (degree). "+
        "[Default 1.]")
    ConstGroup.add_argument(
        "--snrT",
        action="store",
        type=float,
        dest="snrTlim",
        default=1.,
        help="Specify the minimum SNR Threshold for the Transverse " +
        "component to be considered Non-Null. [Default 1.]")
    ConstGroup.add_argument(
        "--fmin",
        action="store",
        type=float,
        dest="fmin",
        default=0.02,
        help="Specify the minimum frequency corner for bandpass " +
        "filter (Hz). [Default 0.02]")
    ConstGroup.add_argument(
        "--fmax",
        action="store",
        type=float,
        dest="fmax",
        default=0.5,
        help="Specify the maximum frequency corner for bandpass " +
        "filter (Hz). [Default 0.5]")

    # Event Selection Criteria
    EventGroup = parser.add_argument_group(
        title="Event Settings",
        description="Settings associated with refining "
        "the events to include in matching station pairs")
    EventGroup.add_argument(
        "--start",
        action="store",
        type=str,
        dest="startT",
        default="",
        help="Specify a UTCDateTime compatible string representing " +
        "the start time for the event search. This will override any " +
        "station start times. [Default start date of each station]")
    EventGroup.add_argument(
        "--end",
        action="store",
        type=str,
        dest="endT",
        default="",
        help="Specify a UTCDateTime compatible string representing " +
        "the end time for the event search. This will override any " +
        "station end times [Default end date of each station]")
    EventGroup.add_argument(
        "--reverse",
        action="store_true",
        dest="reverse",
        default=False,
        help="Reverse order of events. Default behaviour starts at " +
        "oldest event and works towards most recent. " +
        "Specify reverse order and instead the program will start " +
        "with the most recent events and work towards older")
    EventGroup.add_argument(
        "--min-mag",
        action="store",
        type=float,
        dest="minmag",
        default=6.0,
        help="Specify the minimum magnitude of event for which to " +
        "search. [Default 6.0]")
    EventGroup.add_argument(
        "--max-mag",
        action="store",
        type=float,
        dest="maxmag",
        default=None,
        help="Specify the maximum magnitude of event for which to " +
        "search. [Default None, i.e. no limit]")

    # Geometry Settings
    GeomGroup = parser.add_argument_group(
        title="Geometry Settings",
        description="Settings associatd with the "
        "event-station geometries")
    GeomGroup.add_argument(
        "--min-dist",
        action="store",
        type=float,
        dest="mindist",
        default=85.,
        help="Specify the minimum great circle distance (degrees) " +
        "between the station and event. [Default 85]")
    GeomGroup.add_argument(
        "--max-dist",
        action="store",
        type=float,
        dest="maxdist",
        default=120.,
        help="Specify the maximum great circle distance (degrees) " +
        "between the station and event. [Default 120]")
    GeomGroup.add_argument(
        "--phase",
        action="store",
        type=str,
        dest="phase",
        default='SKS',
        help="Specify the phase name to use. Be careful with the distance. " +
        "setting. Options are 'SKS' or 'SKKS'. [Default 'SKS']")

    # Performance / caching
    PerfGroup = parser.add_argument_group(
        title="Performance Settings",
        description="Parallelism and download-avoidance settings")
    PerfGroup.add_argument(
        "--nproc",
        action="store",
        type=int,
        dest="nproc",
        default=max(1, (os.cpu_count() or 2) - 1),
        help="Number of worker processes (one event per task). "
        "[Default: CPU count - 1]")
    PerfGroup.add_argument(
        "--ndl",
        action="store",
        type=int,
        dest="ndl",
        default=4,
        help="Maximum number of simultaneous waveform requests across "
        "all workers (be polite to the data center). [Default 4]")
    PerfGroup.add_argument(
        "--redownload",
        action="store_true",
        dest="redownload",
        default=False,
        help="Ignore locally saved ZNE data and the rejected-event cache "
        "and download again. [Default False]")
    PerfGroup.add_argument(
        "--summary-file",
        action="store",
        type=str,
        dest="summary_file",
        default="run_summary.json",
        help="JSON file written at the end of the run with event counts, "
        "download volume and timing. [Default run_summary.json]")
    PerfGroup.add_argument(
        "--reject-dir",
        action="store",
        type=str,
        dest="rejdir",
        default="REJECTED",
        help="Folder holding the cache of events rejected on SNR, so they "
        "are never downloaded again. [Default REJECTED]")

    args = parser.parse_args(argv)

    # Check inputs
    if not exist(args.indb):
        parser.error("Input file " + args.indb + " does not exist")

    # create station key list
    if len(args.stkeys) > 0:
        args.stkeys = args.stkeys.split(',')

    # construct start time
    if len(args.startT) > 0:
        try:
            args.startT = UTCDateTime(args.startT)
        except:
            parser.error(
                "Cannot construct UTCDateTime from start time: " +
                args.startT)
    else:
        args.startT = None

    # construct end time
    if len(args.endT) > 0:
        try:
            args.endT = UTCDateTime(args.endT)
        except:
            parser.error(
                "Cannot construct UTCDateTime from end time: " +
                args.endT)
    else:
        args.endT = None

    # Parse restricted data settings
    if args.tokenfile is not None:
        args.userauth = [None, None]
    else:
        if args.userauth is not None:
            tt = args.userauth.split(':')
            if not len(tt) == 2:
                msg = (
                    "Error: Incorrect Username and Password Strings " +
                    "for User Authentification")
                parser.error(msg)
            else:
                args.userauth = tt
        else:
            args.userauth = [None, None]

    if args.nproc < 1 or args.ndl < 1:
        parser.error("--nproc and --ndl must be >= 1")

    # Check existing file behaviour
    if args.skip and args.ovr:
        args.skip = False
        args.ovr = False

    # Check Datatype specification
    if args.dtype.upper() not in ['MSEED', 'SAC']:
        parser.error(
            "Error: Local Data Archive must be of types 'SAC'" +
            "or MSEED. These must match the file extensions for " +
            " the archived data.")

    # Check selected phase
    if args.phase not in ['SKS', 'SKKS', 'PKS']:
        parser.error(
            "Error: choose between 'SKS', 'SKKS and 'PKS'.")

    # Check distances for all phases
    if not args.mindist:
        if args.phase == 'SKS':
            args.mindist = 85.
        elif args.phase == 'SKKS':
            args.mindist = 90.
        elif args.phase == 'PKS':
            args.mindist = 130.
    if not args.maxdist:
        if args.phase == 'SKS':
            args.maxdist = 120.
        elif args.phase == 'SKKS':
            args.maxdist = 130.
        elif args.phase == 'PKS':
            args.maxdist = 150.
    if args.mindist < 85. or args.maxdist > 180.:
        parser.error(
            "Distances should be between 85 and 180 deg. for " +
            "teleseismic 'SKS', 'SKKS' and 'PKS' waves.")

    return args


# ---------------------------------------------------------------------------
# Worker side
# ---------------------------------------------------------------------------
_G = {}


_STATS = {'requests': 0, 'mseed_bytes': 0, 'sample_bytes': 0, 'traces': 0}


def _install_counters():
    """Count waveform requests and bytes (best effort, per process).

    requests      : number of get_waveforms calls
    sample_bytes  : uncompressed bytes of the samples returned
    mseed_bytes   : size of the raw responses, when obspy exposes it
                    (stays 0 if it cannot be measured)
    """
    try:
        from obspy.clients.fdsn.client import Client as _C
        if not getattr(_C.get_waveforms, '_counted', False):
            _orig = _C.get_waveforms

            def get_waveforms(self, *a, **k):
                _STATS['requests'] += 1
                st = _orig(self, *a, **k)
                try:
                    _STATS['sample_bytes'] += int(
                        sum(tr.data.nbytes for tr in st))
                    _STATS['traces'] += len(st)
                except Exception:
                    pass
                return st
            get_waveforms._counted = True
            _C.get_waveforms = get_waveforms
    except Exception:
        pass
    try:
        import obspy.clients.fdsn.client as _cm
        if hasattr(_cm, 'download_url') and not getattr(
                _cm.download_url, '_counted', False):
            _odl = _cm.download_url

            def download_url(*a, **k):
                res = _odl(*a, **k)
                try:
                    data = res[1]
                    if isinstance(data, (bytes, bytearray)):
                        n = len(data)
                    elif hasattr(data, 'getbuffer'):
                        n = data.getbuffer().nbytes
                    else:
                        n = 0
                    _STATS['mseed_bytes'] += n
                except Exception:
                    pass
                return res
            download_url._counted = True
            _cm.download_url = download_url
    except Exception:
        pass


def _init_worker(args, stas, events, sem):
    _install_counters()
    # Meta.__init__ builds TauPyModel('iasp91') for every event. Reuse one
    # instance per process: identical numbers, no repeated model loading.
    try:
        import functools
        import obspy.taup as _taup
        if not hasattr(_taup.TauPyModel, 'cache_info'):
            _taup.TauPyModel = functools.lru_cache(
                maxsize=None)(_taup.TauPyModel)
    except Exception:
        pass    # purely an optimisation; never let it kill a worker
    _G['args'] = args
    _G['stas'] = stas
    _G['events'] = events
    _G['sem'] = sem
    _G['client'] = None


def _get_client():
    if _G['client'] is None:
        args = _G['args']
        if args.localdata is None:
            _G['client'] = FDSN_Client(
                base_url=args.server,
                user=args.userauth[0],
                password=args.userauth[1],
                eida_token=args.tokenfile)
        else:
            _G['client'] = SDS_Client(args.localdata, format=args.dtype)
    return _G['client']


def _dump_atomic(objs, path):
    """Pickle one or more objects sequentially into path, atomically."""
    tmp = Path(str(path) + '.tmp.{}'.format(os.getpid()))
    try:
        with open(tmp, 'wb') as fh:
            for o in objs:
                pickle.dump(o, fh)
        os.replace(tmp, path)
    except BaseException:
        if tmp.exists():
            tmp.unlink()
        raise


def _load(path):
    with open(path, 'rb') as fh:
        return pickle.load(fh)


def _param_key(args):
    # Everything that determines the SNR value of a downloaded event
    return "|".join(str(x) for x in (
        args.new_sampling_rate, args.dts, args.fmin, args.fmax,
        args.zcomp, args.phase))


def _paths(args, stkey, timekey):
    datadir = Path('DATA') / stkey / timekey
    return dict(
        datadir=datadir,
        ZNE=datadir / 'ZNE_data.pkl',
        LQT=datadir / 'LQT_data.pkl',
        meta=datadir / 'Meta_data.pkl',
        sta=datadir / 'Station_data.pkl',
        split=datadir / 'Split_results_auto.pkl',
        rej=Path(args.rejdir) / stkey / (timekey + '.json'))


def _process_event(stkey, ev):
    args = _G['args']
    sta = _G['stas'][stkey]

    # Initialize Split object with station info
    split = Split(sta, zcomp=args.zcomp)

    # Add event to split object (exact geometry test)
    accept = split.add_event(
        ev,
        gacmin=args.mindist,
        gacmax=args.maxdist,
        phase=args.phase,
        returned=True)
    if not accept:
        return None, 'geometry'

    timekey = split.meta.time.strftime("%Y%m%d_%H%M%S")
    P = _paths(args, stkey, timekey)
    datadir = P['datadir']
    do_calc = args.calc or args.recalc

    if args.verb:
        print("*" * 50)
        print("* {0:13s} {1}".format(timekey, stkey))
        print("*   Phase: {}".format(args.phase))
        print("*   Origin Time: " +
              split.meta.time.strftime("%Y-%m-%d %H:%M:%S"))
        print("*   Lat: {0:6.2f};        Lon: {1:7.2f}".format(
            split.meta.lat, split.meta.lon))
        print("*   Dep: {0:6.2f} km;     Mag: {1:3.1f}".format(
            split.meta.dep, split.meta.mag))
        print("*   Dist: {0:7.2f} km;".format(split.meta.epi_dist) +
              "   Epi dist: {0:6.2f} deg\n".format(split.meta.gac) +
              "*   Baz:  {0:6.2f} deg;".format(split.meta.baz) +
              "   Az: {0:6.2f} deg".format(split.meta.az))

    # Existing results: skip unless overwrite (recalc implies overwrite)
    if P['split'].exists() and not (args.ovr or args.recalc):
        return timekey, 'exists'

    have_local = all(P[k].exists() for k in ('ZNE', 'meta', 'sta'))

    # Download-only run and everything is already on disk
    if (not do_calc) and have_local and not args.redownload and not args.ovr:
        return timekey, 'exists'

    use_local = (args.recalc or have_local) and not args.redownload

    if args.recalc and not have_local:
        return timekey, 'no-local-data'

    if use_local:
        # --- Reuse data already on disk: no download ---------------------
        sta_l = _load(P['sta'])
        split = Split(sta_l, zcomp=args.zcomp)
        split.meta = _load(P['meta'])
        split.dataZNE = _load(P['ZNE'])

        split.rotate(align='LQT')
        split.dataLQT.filter(
            'bandpass', freqmin=args.fmin, freqmax=args.fmax)
        split.calc_snr()

        if not args.recalc:
            # Same acceptance logic as a fresh download
            if split.meta.snrq < args.msnr or np.isnan(split.meta.snrq):
                return timekey, 'snr'

        _dump_atomic([split.dataLQT], P['LQT'])

    else:
        # --- Rejected-on-SNR cache: avoid re-downloading ------------------
        if P['rej'].exists() and not args.redownload:
            try:
                with open(P['rej']) as fh:
                    r = json.load(fh)
                if r.get('key') == _param_key(args) and (
                        r.get('snrq') is None or r['snrq'] < args.msnr):
                    return timekey, 'cached-reject'
            except Exception:
                pass

        # --- Download ----------------------------------------------------
        with _G['sem']:
            has_data = split.download_data(
                client=_get_client(),
                new_sr=args.new_sampling_rate,
                dts=args.dts,
                returned=True,
                verbose=args.verb)
        if not has_data:
            return timekey, 'nodata'

        split.rotate(align='LQT')
        split.dataLQT.filter(
            'bandpass', freqmin=args.fmin, freqmax=args.fmax)
        split.calc_snr()

        snrq = split.meta.snrq
        rejected = None
        if snrq < args.msnr:
            rejected = 'snr'
            if args.verb:
                print("* SNRQ < {0:.1f}, continuing".format(args.msnr))
        elif np.isnan(snrq):
            rejected = 'nan'
            if args.verb:
                print("* SNR NaN, continuing")
        if rejected is not None:
            try:
                P['rej'].parent.mkdir(parents=True, exist_ok=True)
                with open(P['rej'], 'w') as fh:
                    json.dump(dict(
                        key=_param_key(args),
                        snrq=None if np.isnan(snrq) else float(snrq)), fh)
            except Exception:
                pass
            return timekey, rejected

        datadir.mkdir(parents=True, exist_ok=True)
        _dump_atomic([split.dataZNE], P['ZNE'])
        _dump_atomic([split.dataLQT], P['LQT'])
        # Meta/Station are saved right away so that a later --calc run can
        # reuse this download even if the analysis below fails.
        _dump_atomic([split.meta], P['meta'])
        _dump_atomic([split.sta], P['sta'])

    if args.verb:
        print("* SNRQ: {}".format(split.meta.snrq))
        print("* SNRT: {}".format(split.meta.snrt))

    if do_calc:
        # ---- Splitting calculation: unchanged --------------------------
        split.analyze(verbose=args.verb)

        # Continue if problem with analysis
        if split.RC_res.edtt is None or split.SC_res.edtt is None:
            if args.verb:
                print("* !!! DOF Error. --> Skipping...")
            _dump_atomic([split.meta], P['meta'])
            _dump_atomic([split.sta], P['sta'])
            return timekey, 'dof'

        split.is_null(args.snrTlim, verbose=args.verb)
        split.get_quality(verbose=args.verb)

    # Display results
    if args.verb:
        split.display_meta()
        if do_calc:
            split.display_results()
            split.display_null_quality()

    # Save event meta data and station data
    _dump_atomic([split.meta], P['meta'])
    _dump_atomic([split.sta], P['sta'])

    if do_calc:
        _dump_atomic([split.SC_res, split.RC_res, split.null,
                      split.quality], P['split'])

        # Diagnostic figure (saved to disk instead of shown)
        if args.diagplot:
            dplot = DiagPlot(split)
            dplot.plot_diagnostic()
            plt.figure(dplot.axes[0].number)
            plt.gcf().savefig(str(datadir / 'Diagnostic_auto.png'), dpi=150)
            plt.close('all')

    return timekey, 'done'


def process_event(task):
    stkey, iev = task
    buf = io.StringIO()
    before = dict(_STATS)
    try:
        with contextlib.redirect_stdout(buf):
            timekey, status = _process_event(stkey, _G['events'][iev])
    except Exception as e:
        timekey, status = None, 'error: {!r}'.format(e)
        buf.write(traceback.format_exc())
        plt.close('all')
    delta = {k: _STATS[k] - before[k] for k in _STATS}
    return stkey, timekey, status, buf.getvalue(), delta


# ---------------------------------------------------------------------------
# Main-process side: catalogue (one query), pre-filtering, dispatch
# ---------------------------------------------------------------------------
def _fetch_catalog(client, tmin, tmax, minmag, maxmag, chunk_years=10):
    """One catalogue for the union of all station time ranges."""
    tmax = min(tmax, UTCDateTime())
    events, seen = [], set()
    t0 = tmin
    while t0 < tmax:
        t1 = min(t0 + chunk_years * 365.25 * 86400., tmax)
        cat = client.get_events(starttime=t0, endtime=t1,
                                minmagnitude=minmag, maxmagnitude=maxmag)
        for ev in cat:
            rid = str(ev.resource_id)
            if rid not in seen:
                seen.add(rid)
                events.append(ev)
        t0 = t1
    return events


def _gcd_deg(lat1, lon1, lat2, lon2):
    """Spherical great-circle distance (deg); used only as a loose
    pre-filter. The exact test is Split.add_event in the worker."""
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dl = np.radians(lon2 - lon1)
    c = np.sin(p1) * np.sin(p2) + np.cos(p1) * np.cos(p2) * np.cos(dl)
    return np.degrees(np.arccos(np.clip(c, -1., 1.)))


def _write_summary(args, t0, ntot, counts, dl, note=''):
    out = dict(
        note=note,
        tasks=ntot,
        counts=counts,
        download=dl,
        elapsed_s=round(time.time() - t0, 1),
        finished=UTCDateTime().strftime("%Y-%m-%dT%H:%M:%S"))
    tmp = str(args.summary_file) + '.tmp'
    with open(tmp, 'w') as fh:
        json.dump(out, fh, indent=2)
    os.replace(tmp, args.summary_file)


def main(args=None):

    if args is None:
        # Run Input Parser
        args = get_arguments_calc_auto()

    # Load Database
    # stdb=0.1.4
    try:
        db, stkeys = stdb.io.load_db(fname=args.indb, keys=args.stkeys)

    # stdb=0.1.3
    except:
        db = stdb.io.load_db(fname=args.indb)

        # Construct station key loop
        allkeys = db.keys()
        sorted(allkeys)

        # Extract key subset
        if len(args.stkeys) > 0:
            stkeys = []
            for skey in args.stkeys:
                stkeys.extend([s for s in allkeys if skey in s])
        else:
            stkeys = db.keys()
            sorted(stkeys)

    t0 = time.time()
    dl = {'requests': 0, 'mseed_bytes': 0, 'sample_bytes': 0, 'traces': 0}

    # Station time ranges
    stas, ranges = {}, {}
    for stkey in list(stkeys):
        sta = db[stkey]
        tstart = sta.startdate if args.startT is None else args.startT
        tend = sta.enddate if args.endT is None else args.endT
        if tstart > sta.enddate or tend < sta.startdate:
            continue
        stas[stkey] = sta
        ranges[stkey] = (tstart, tend)
        (Path('DATA') / stkey).mkdir(parents=True, exist_ok=True)

    if not stas:
        print("No stations to process.")
        _write_summary(args, t0, 0, {}, dl, note='no stations in date range')
        return

    # One catalogue query for all stations
    print("Querying event catalogue (once) for {} stations ...".format(
        len(stas)))
    tmin = min(r[0] for r in ranges.values())
    tmax = max(r[1] for r in ranges.values())
    events = _fetch_catalog(
        FDSN_Client(), tmin, tmax, args.minmag, args.maxmag)

    # Origin arrays for vectorized per-station pre-filtering
    keep, elat, elon, etime = [], [], [], []
    for ev in events:
        try:
            o = ev.origins[0]   # same origin Split/Meta uses
            elat.append(o.latitude)
            elon.append(o.longitude)
            etime.append(o.time.timestamp)
            keep.append(ev)
        except Exception:
            continue
    events = keep
    elat, elon, etime = map(np.array, (elat, elon, etime))
    order = np.argsort(etime)
    if args.reverse:
        order = order[::-1]
    print("Catalogue: {} events".format(len(events)))

    # Build task list
    margin = 1.0   # deg; loose, so the exact add_event test decides
    tasks = []
    for stkey, sta in stas.items():
        tstart, tend = ranges[stkey]
        d = _gcd_deg(sta.latitude, sta.longitude, elat, elon)
        m = ((etime >= tstart.timestamp) & (etime <= tend.timestamp) &
             (d >= args.mindist - margin) & (d <= args.maxdist + margin))
        n0 = len(tasks)
        for i in order[m[order]]:
            # Cheap skip of events that are already finished
            tk = UTCDateTime(etime[i]).strftime("%Y%m%d_%H%M%S")
            sp = Path('DATA') / stkey / tk / 'Split_results_auto.pkl'
            if sp.exists() and not (args.ovr or args.recalc):
                continue
            tasks.append((stkey, int(i)))
        print("  {0:12s} {1:5d} candidate events".format(
            stkey, len(tasks) - n0))

    print("Total tasks: {}   workers: {}   max simultaneous downloads: {}"
          .format(len(tasks), args.nproc, args.ndl))
    if not tasks:
        _write_summary(args, t0, 0, {}, dl, note='no tasks')
        return

    sem = mp.BoundedSemaphore(max(1, args.ndl))
    initargs = (args, stas, events, sem)
    counts = {}
    ntot = len(tasks)

    def _report(k, res):
        stkey, timekey, status, text, delta = res
        for kk in dl:
            dl[kk] += delta.get(kk, 0)
        counts[status] = counts.get(status, 0) + 1
        if status == 'geometry':
            return
        if args.verb and text:
            print(text, end='')
        if status.startswith('error'):
            print(text, end='')
        print("[{0}/{1}] {2} {3}: {4}".format(
            k, ntot, stkey, timekey, status), flush=True)

    if args.nproc == 1:
        _init_worker(*initargs)
        for k, t in enumerate(tasks, 1):
            _report(k, process_event(t))
    else:
        pool = mp.Pool(args.nproc, initializer=_init_worker,
                       initargs=initargs)
        try:
            for k, res in enumerate(
                    pool.imap_unordered(process_event, tasks, chunksize=1), 1):
                _report(k, res)
            pool.close()
        except KeyboardInterrupt:
            pool.terminate()
            raise
        finally:
            pool.join()

    print("\nSummary:")
    for k in sorted(counts):
        print("  {0:15s} {1}".format(k, counts[k]))
    print("Download: {0} requests, {1:.1f} MB mseed, {2:.1f} MB samples"
          .format(dl['requests'], dl['mseed_bytes'] / 1e6,
                  dl['sample_bytes'] / 1e6))
    _write_summary(args, t0, ntot, counts, dl, note='complete')


if __name__ == "__main__":

    # Run main program
    main()
