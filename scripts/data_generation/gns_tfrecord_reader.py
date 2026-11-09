# -*- coding: utf-8 -*-
"""Pure-Python reader for DeepMind GNS TFRecord datasets (no TensorFlow).

A TFRecord file is a sequence of records:
    uint64 length | uint32 crc(length) | length bytes payload | uint32 crc(payload)

Each payload is a serialized `tf.train.SequenceExample` protobuf:
    SequenceExample { Features context = 1; FeatureLists feature_lists = 2; }
    Features      { map<string, Feature> feature = 1; }
    FeatureLists  { map<string, FeatureList> feature_list = 1; }
    FeatureList   { repeated Feature feature = 1; }
    Feature       { BytesList bytes_list = 1; FloatList = 2; Int64List = 3; }
    BytesList     { repeated bytes value = 1; }

Protobuf map entries are encoded as repeated messages with key=field1, value=field2,
so they are parsed like any other nested message. CRCs are not verified (the files
ship intact and verification would cost a full extra pass).
"""
from __future__ import annotations

import struct
from pathlib import Path

import numpy as np

_WIRE_VARINT, _WIRE_64BIT, _WIRE_LEN, _WIRE_32BIT = 0, 1, 2, 5


def _varint(buf: bytes, i: int):
    """Decode a base-128 varint; returns (value, next_index)."""
    val = shift = 0
    while True:
        b = buf[i]
        i += 1
        val |= (b & 0x7F) << shift
        if not b & 0x80:
            return val, i
        shift += 7


def _fields(buf: bytes, start: int = 0, end: int | None = None):
    """Yield (field_number, payload) for every field in a protobuf message.

    payload is bytes for length-delimited fields, int for varint/fixed fields.
    """
    if end is None:
        end = len(buf)
    i = start
    while i < end:
        key, i = _varint(buf, i)
        fnum, wire = key >> 3, key & 7
        if wire == _WIRE_LEN:
            ln, i = _varint(buf, i)
            yield fnum, buf[i:i + ln]
            i += ln
        elif wire == _WIRE_VARINT:
            val, i = _varint(buf, i)
            yield fnum, val
        elif wire == _WIRE_64BIT:
            yield fnum, buf[i:i + 8]
            i += 8
        elif wire == _WIRE_32BIT:
            yield fnum, buf[i:i + 4]
            i += 4
        else:
            raise ValueError(f"unsupported protobuf wire type {wire}")


def _bytes_list(feature: bytes) -> list[bytes]:
    """Extract BytesList values from a Feature message."""
    out = []
    for fnum, payload in _fields(feature):
        if fnum == 1:                       # bytes_list
            for vnum, val in _fields(payload):
                if vnum == 1:
                    out.append(val)
    return out


def _map_entries(container: bytes):
    """Yield (key, value_bytes) for a protobuf map field container."""
    for fnum, payload in _fields(container):
        if fnum != 1:                       # map field is always field 1 here
            continue
        key = value = None
        for enum, epayload in _fields(payload):
            if enum == 1:
                key = epayload.decode("utf-8")
            elif enum == 2:
                value = epayload
        if key is not None:
            yield key, value


def iter_records(path: str | Path, max_records: int | None = None):
    """Stream raw payloads out of a TFRecord file."""
    n = 0
    with open(path, "rb") as f:
        while True:
            head = f.read(8)
            if len(head) < 8:
                return
            (length,) = struct.unpack("<Q", head)
            f.read(4)                       # crc(length) — not verified
            payload = f.read(length)
            if len(payload) < length:
                return
            f.read(4)                       # crc(payload) — not verified
            yield payload
            n += 1
            if max_records is not None and n >= max_records:
                return


def parse_trajectory(payload: bytes, dim: int):
    """Decode one SequenceExample into (positions, particle_types).

    positions       float32 [T, N, dim]
    particle_types  int64   [N]
    """
    ptype = None
    positions = []
    for fnum, sub in _fields(payload):
        if fnum == 1:                       # context Features
            for key, feat in _map_entries(sub):
                if key == "particle_type":
                    vals = _bytes_list(feat)
                    if vals:
                        ptype = np.frombuffer(vals[0], dtype=np.int64)
        elif fnum == 2:                     # feature_lists FeatureLists
            for key, flist in _map_entries(sub):
                if key != "position":
                    continue
                for inum, feat in _fields(flist):
                    if inum != 1:           # repeated Feature
                        continue
                    for raw in _bytes_list(feat):
                        positions.append(np.frombuffer(raw, dtype=np.float32))
    if not positions:
        raise ValueError("no position data found in record")
    pos = np.stack(positions).reshape(len(positions), -1, dim)
    return pos, ptype


def read_trajectories(path, dim, max_trajectories=None):
    """Yield (index, positions, particle_types) for each trajectory."""
    for idx, payload in enumerate(iter_records(path, max_trajectories)):
        pos, ptype = parse_trajectory(payload, dim)
        yield idx, pos, ptype


if __name__ == "__main__":
    import json
    import sys

    root = Path(__file__).resolve().parents[2]
    ds = root / "datasets" / "gns_datasets" / (sys.argv[1] if len(sys.argv) > 1 else "Water-3D")
    meta = json.loads((ds / "metadata.json").read_text())
    dim = meta["dim"]
    print(f"dataset={ds.name} dim={dim} seq_len={meta['sequence_length']} "
          f"radius={meta['default_connectivity_radius']}")
    for idx, pos, ptype in read_trajectories(ds / "valid.tfrecord", dim, max_trajectories=2):
        types, counts = np.unique(ptype, return_counts=True)
        print(f"  traj{idx}: frames={pos.shape[0]} particles={pos.shape[1]} "
              f"types={dict(zip(types.tolist(), counts.tolist()))} "
              f"bounds=[{pos.min():.3f}, {pos.max():.3f}]")
