"""Geometry and reference-data seam for the 2-D periodic-hill case.

This module is deliberately independent of the neural closure code.  It fixes
the benchmark geometry, downloads/parses the *independent* LES reference from
the ERCOFTAC Classic Collection (case 081), and exposes the wall-normal
coordinate used by the 2-D observation and evaluation splits.

The reference is Temmerman & Leschziner's wall-resolving dynamic-SGS LES, not a
synthetic solution.  The archive is kept outside the MIT-licensed source tree:
:func:`download_reference` fetches it into a caller-supplied directory and
verifies the SHA-256 before extraction.
"""

from __future__ import annotations

import hashlib
import math
import re
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Mapping, Sequence, Tuple

import numpy as np

__all__ = [
    "HILL_HEIGHT",
    "CHANNEL_HEIGHT",
    "PERIOD",
    "RE_H",
    "BULK_VELOCITY",
    "KINEMATIC_VISCOSITY",
    "REFERENCE_STATIONS",
    "REFERENCE_ARCHIVE_URL",
    "REFERENCE_ARCHIVE_SHA256",
    "PeriodicHillProfile",
    "hill_height",
    "wall_normal_coordinate",
    "parse_profile",
    "load_reference",
    "reference_wall_shear",
    "download_reference",
]

#: Non-dimensional benchmark constants.  Lengths are in hill heights.
HILL_HEIGHT = 1.0
CHANNEL_HEIGHT = 3.035
PERIOD = 9.0
BULK_VELOCITY = 1.0
RE_H = 10595.0
KINEMATIC_VISCOSITY = 1.0 / RE_H

#: x/h locations distributed with the ERCOFTAC case-081 LES archive.
REFERENCE_STATIONS: Tuple[float, ...] = (0.05, 0.5, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0)

#: Source and integrity check for ``perh-allfiles.zip``.  The archive contains
#: ``geom.dat`` and ten ``data-0NN.dat`` profiles.
REFERENCE_ARCHIVE_URL = (
    "http://cfd.mace.manchester.ac.uk/ercoftac/lib/exe/fetch.php"
    "?media=cdata:case081:perh-allfiles.zip"
)
REFERENCE_ARCHIVE_SHA256 = (
    "61ca8053a187c67d914f1ab951a003136bd92c4b61e7de7b359d18002978030b"
)

#: Geometry from ``geom.dat``.  The original x coordinates are in mm and the
#: hill height is 28 mm; this implementation converts them to h = 1 units.
_HILL_MM = 28.0
_GEOM_X_MAX_MM = 54.0


def _hill_height_mm(x_mm: np.ndarray) -> np.ndarray:
    """Evaluate the published piecewise-cubic hill shape in millimetres."""
    x = np.asarray(x_mm, dtype=float)
    y = np.zeros_like(x)

    mask = (x >= 0.0) & (x < 9.0)
    xx = x[mask]
    y[mask] = np.minimum(
        28.0,
        28.0 + 6.775070969851e-03 * xx**2 - 2.124527775800e-03 * xx**3,
    )

    mask = (x >= 9.0) & (x < 14.0)
    xx = x[mask]
    y[mask] = (
        25.07355893131
        + 9.754803562315e-01 * xx
        - 1.016116352781e-01 * xx**2
        + 1.889794677828e-03 * xx**3
    )

    mask = (x >= 14.0) & (x < 20.0)
    xx = x[mask]
    y[mask] = (
        25.79601052357
        + 8.206693007457e-01 * xx
        - 9.055370274339e-02 * xx**2
        + 1.626510569859e-03 * xx**3
    )

    mask = (x >= 20.0) & (x < 30.0)
    xx = x[mask]
    y[mask] = (
        40.46435022819
        - 1.379581654948e00 * xx
        + 1.945884504128e-02 * xx**2
        - 2.070318932190e-04 * xx**3
    )

    mask = (x >= 30.0) & (x < 40.0)
    xx = x[mask]
    y[mask] = (
        17.92461334664
        + 8.743920332081e-01 * xx
        - 5.567361123058e-02 * xx**2
        + 6.277731764683e-04 * xx**3
    )

    mask = (x >= 40.0) & (x <= _GEOM_X_MAX_MM)
    xx = x[mask]
    y[mask] = np.maximum(
        0.0,
        56.39011190988
        - 2.010520359035e00 * xx
        + 1.644919857549e-02 * xx**2
        + 2.674976141766e-05 * xx**3,
    )
    return y


def hill_height(x_over_h: np.ndarray) -> np.ndarray:
    """Return the lower-wall height ``b(x)/h`` for ``x/h`` values.

    ``x/h`` is periodic with period ``PERIOD``.  The published spline is only
    defined on the hill itself (``0 <= x/h <= 54/28``); the remainder of the
    period is the flat plate at ``b = 0``.
    """
    x = np.asarray(x_over_h, dtype=float)
    # Fold into the geometric period without jumping at the right edge.
    folded = np.mod(x, PERIOD)
    mm = folded * _HILL_MM
    out = _hill_height_mm(mm) / _HILL_MM
    # ``_hill_height_mm`` already returns zero outside the spline support, but
    # folding can leave values like 54.000000000000014 mm at the seam.
    out = np.where(folded > _GEOM_X_MAX_MM / _HILL_MM, 0.0, out)
    return out


def wall_normal_coordinate(y_over_h: np.ndarray, x_over_h: np.ndarray) -> np.ndarray:
    """Return the distance above the lower wall, ``y - b(x)``.

    The reference profiles are tabulated in absolute ``Y/h``.  Observations and
    wall functions need the wall-normal coordinate, so this is the one place
    where the shift is applied.
    """
    y = np.asarray(y_over_h, dtype=float)
    x = np.asarray(x_over_h, dtype=float)
    return y - hill_height(x)


@dataclass(frozen=True)
class PeriodicHillProfile:
    """One station from the ERCOFTAC case-081 LES archive."""

    x_over_h: float
    y_over_h: np.ndarray
    u_over_ub: np.ndarray
    v_over_ub: np.ndarray
    uu_over_ub2: np.ndarray
    vv_over_ub2: np.ndarray
    uv_over_ub2: np.ndarray
    k_over_ub2: np.ndarray

    @property
    def wall_normal(self) -> np.ndarray:
        return wall_normal_coordinate(self.y_over_h, self.x_over_h)


_PROFILE_HEADER = re.compile(r"X/h\s*=\s*([0-9.EeEe+-]+)")


def parse_profile(path: Path) -> PeriodicHillProfile:
    """Parse a single ``data-0NN.dat`` file from the case-081 archive."""
    path = Path(path)
    text = path.read_text(encoding="utf-8", errors="replace")
    match = _PROFILE_HEADER.search(text)
    if match is None:
        raise ValueError(f"{path}: missing 'X/h = ...' header")
    x_over_h = float(match.group(1))
    data = np.loadtxt(path, comments="#")
    if data.ndim != 2 or data.shape[1] != 7:
        raise ValueError(f"{path}: expected seven columns, got shape {data.shape}")
    return PeriodicHillProfile(
        x_over_h=x_over_h,
        y_over_h=data[:, 0],
        u_over_ub=data[:, 1],
        v_over_ub=data[:, 2],
        uu_over_ub2=data[:, 3],
        vv_over_ub2=data[:, 4],
        uv_over_ub2=data[:, 5],
        k_over_ub2=data[:, 6],
    )


def load_reference(directory: Path) -> Dict[float, PeriodicHillProfile]:
    """Load and key all profiles in an extracted case-081 directory by ``x/h``."""
    directory = Path(directory)
    profiles: Dict[float, PeriodicHillProfile] = {}
    for path in sorted(directory.glob("data-*.dat")):
        profile = parse_profile(path)
        nearest = min(REFERENCE_STATIONS, key=lambda s: abs(s - profile.x_over_h))
        if abs(nearest - profile.x_over_h) > 1e-6:
            raise ValueError(
                f"{path}: x/h={profile.x_over_h} does not match a published station"
            )
        profiles[nearest] = profile
    missing = [station for station in REFERENCE_STATIONS if station not in profiles]
    if missing:
        raise FileNotFoundError(
            f"{directory}: missing reference stations {missing}; found {sorted(profiles)}"
        )
    return profiles
def reference_wall_shear(profile: PeriodicHillProfile) -> float:
    """Estimate ``tau_w`` from the first two LES samples.

    The LES archive does not tabulate the wall derivative.  A one-sided linear
    estimate ``mu * (U_1 - U_0) / (y_1 - y_0)`` is used, with the molecular
    viscosity; this is an *independent reference estimate*, not a solver input.
    """
    if profile.y_over_h.size < 2:
        raise ValueError("need at least two reference samples for a wall-shear fit")
    dy = profile.y_over_h[1] - profile.y_over_h[0]
    if dy <= 0.0:
        raise ValueError("reference y coordinates must increase")
    grad = (profile.u_over_ub[1] - profile.u_over_ub[0]) / dy
    return KINEMATIC_VISCOSITY * grad


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_reference(
    destination: Path,
    *,
    url: str = REFERENCE_ARCHIVE_URL,
    sha256: str = REFERENCE_ARCHIVE_SHA256,
    timeout: float = 60.0,
    force: bool = False,
) -> Path:
    """Download and extract the independent LES archive.

    Returns the directory containing ``data-0NN.dat`` and ``geom.dat``.  The
    archive is verified before extraction, and a previously extracted tree is
    reused unless ``force`` is true.
    """
    destination = Path(destination)
    extracted = destination / "extracted"
    marker = extracted / "data-001.dat"
    if marker.is_file() and not force:
        return extracted

    destination.mkdir(parents=True, exist_ok=True)
    archive = destination / "perh-allfiles.zip"
    request = urllib.request.Request(url, headers={"User-Agent": "PINN-periodic-hill/1.0"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        archive.write_bytes(response.read())

    actual = _sha256(archive)
    expected = sha256.lower()
    if actual != expected:
        archive.unlink(missing_ok=True)
        raise ValueError(
            f"reference archive checksum mismatch: got {actual}, expected {expected}"
        )

    extracted.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as zf:
        zf.extractall(extracted)
    if not marker.is_file():
        raise FileNotFoundError(f"{archive}: archive did not contain data-001.dat")
    return extracted
