"""Downloads and extracts authentic adverse-weather driving video sequences for Mission 9.

Leverages contiguous HTTP range requests on the remote BDD100K MOT val archive
to extract selected 200-frame sequences with zero wasted bandwidth.
"""
from __future__ import annotations

import io
import json
import time
import zipfile
import zlib
from pathlib import Path
from PIL import Image
import requests

SELECTED_SEQUENCES = {
    "b1e9ee0e-67e26f2e": {"weather": "rainy", "timeofday": "daytime", "scene": "highway", "desc": "Rainy Daytime Highway/Urban"},
    "b2064e61-2beadd45": {"weather": "rainy", "timeofday": "night", "scene": "city street", "desc": "Rainy Night Wet Asphalt & Headlights"},
    "b1d0a191-65deaeef": {"weather": "snowy", "timeofday": "daytime", "scene": "city street", "desc": "Snowy Daytime Dense Urban Traffic"},
    "b1d0a191-03dcecc2": {"weather": "snowy", "timeofday": "daytime", "scene": "residential", "desc": "Snowy Daytime Roadside Snowbanks"},
    "b24071b8-b3ee1196": {"weather": "overcast", "timeofday": "daytime", "scene": "city street", "desc": "Overcast Daytime High-Density Pedestrian/Cyclist"},
    "b1d968b9-ce42734f": {"weather": "overcast", "timeofday": "dawn/dusk", "scene": "highway", "desc": "Overcast Dusk Low-Light Highway"},
    "b2208b0f-2796a692": {"weather": "clear", "timeofday": "daytime", "scene": "city street", "desc": "Clear Daytime Dense Baseline"},
    "b1ca2e5d-84cf9134": {"weather": "clear", "timeofday": "night", "scene": "city street", "desc": "Clear Night Urban Street Lighting Baseline"},
}

REMOTE_ZIP_URL = "https://huggingface.co/datasets/mrhwl123/BDD100kMOT/resolve/main/images20-track-val-1.zip"
BASE_DIR = Path(__file__).resolve().parent.parent / "data" / "video_dataset"
SEQ_DIR = BASE_DIR / "sequences"
LABELS_DIR = BASE_DIR / "labels"


class RemoteZipStream(io.RawIOBase):
    def __init__(self, url: str):
        r = requests.head(url, allow_redirects=True, timeout=10)
        self.url = r.url
        self.size = int(r.headers.get("Content-Length", 0))
        self.pos = 0

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def seek(self, offset: int, whence: int = io.SEEK_SET) -> int:
        if whence == io.SEEK_SET:
            self.pos = offset
        elif whence == io.SEEK_CUR:
            self.pos += offset
        elif whence == io.SEEK_END:
            self.pos = self.size + offset
        return self.pos

    def tell(self) -> int:
        return self.pos

    def readinto(self, b) -> int:
        if self.pos >= self.size:
            return 0
        l = len(b)
        end = min(self.pos + l - 1, self.size - 1)
        headers = {"Range": f"bytes={self.pos}-{end}"}
        r = requests.get(self.url, headers=headers, timeout=30)
        data = r.content
        n = len(data)
        b[:n] = data
        self.pos += n
        return n


def download_all_sequences() -> None:
    SEQ_DIR.mkdir(parents=True, exist_ok=True)
    LABELS_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Extract labels from local box_track_labels_trainval.zip
    labels_zip_path = Path(__file__).resolve().parent.parent / "data" / "bdd100k" / "box_track_labels_trainval.zip"
    if labels_zip_path.exists():
        with zipfile.ZipFile(labels_zip_path) as lz:
            for sid in SELECTED_SEQUENCES:
                dest = LABELS_DIR / f"{sid}.json"
                if not dest.exists():
                    zf_path = f"bdd100k/labels/box_track_20/val/{sid}.json"
                    with open(dest, "wb") as f:
                        f.write(lz.read(zf_path))
                    print(f"Extracted label: {dest.name}")

    # 2. Check pending image downloads
    pending = []
    for sid in SELECTED_SEQUENCES:
        s_path = SEQ_DIR / sid
        existing = list(s_path.glob("*.jpg")) if s_path.exists() else []
        if len(existing) < 150:
            pending.append(sid)

    if not pending:
        print("All 8 sequences already downloaded and intact!")
        return

    print(f"Connecting to remote archive to fetch {len(pending)} sequences: {pending}...")
    stream = RemoteZipStream(REMOTE_ZIP_URL)
    zf = zipfile.ZipFile(stream)

    for sid in pending:
        t0 = time.time()
        prefix = f"bdd100k/images/track/val/{sid}/"
        seq_infos = [info for info in zf.infolist() if info.filename.startswith(prefix) and info.filename.endswith(".jpg")]
        if not seq_infos:
            print(f"Warning: No frames found for {sid} in remote zip!")
            continue

        min_off = min(i.header_offset for i in seq_infos)
        last_info = max(seq_infos, key=lambda i: i.header_offset)
        max_off = last_info.header_offset + last_info.compress_size + 4096

        span_mb = (max_off - min_off) / (1024 * 1024)
        print(f"Downloading {sid} ({len(seq_infos)} frames, {span_mb:.2f} MB)...")
        headers = {"Range": f"bytes={min_off}-{max_off}"}
        resp = requests.get(stream.url, headers=headers, timeout=60)
        chunk_bytes = resp.content

        out_dir = SEQ_DIR / sid
        out_dir.mkdir(parents=True, exist_ok=True)

        extracted = 0
        for info in seq_infos:
            rel_off = info.header_offset - min_off
            sig = chunk_bytes[rel_off : rel_off + 4]
            if sig != b"PK\x03\x04":
                continue
            fn_len = int.from_bytes(chunk_bytes[rel_off + 26 : rel_off + 28], "little")
            extra_len = int.from_bytes(chunk_bytes[rel_off + 28 : rel_off + 30], "little")
            c_size = info.compress_size
            data_start = rel_off + 30 + fn_len + extra_len
            data_bytes = chunk_bytes[data_start : data_start + c_size]

            if info.compress_type == 0:
                raw_img = data_bytes
            elif info.compress_type == 8:
                raw_img = zlib.decompress(data_bytes, -15)
            else:
                continue

            fn = Path(info.filename).name
            with open(out_dir / fn, "wb") as f:
                f.write(raw_img)
            extracted += 1

        print(f"Successfully extracted {extracted}/{len(seq_infos)} frames for {sid} in {time.time()-t0:.2f}s!")

    # Verify all downloaded sequences
    print("\nVerifying all sequence integrity...")
    total_frames = 0
    for sid, meta in SELECTED_SEQUENCES.items():
        s_path = SEQ_DIR / sid
        frames = sorted(list(s_path.glob("*.jpg")))
        total_frames += len(frames)
        assert len(frames) >= 190, f"Sequence {sid} incomplete: {len(frames)} frames"
        with Image.open(frames[0]) as im:
            assert im.size == (1280, 720), f"Invalid resolution {im.size} for {sid}"
        print(f"Sequence {sid} [{meta['weather']}/{meta['timeofday']}]: {len(frames)} frames OK (1280x720 RGB)")

    print(f"\nALL 8 SEQUENCES VERIFIED: Total {total_frames} continuous video frames ready for Mission 9!")


if __name__ == "__main__":
    download_all_sequences()
