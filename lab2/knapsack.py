import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

DEFAULT_INSTANCE = Path(__file__).parent / "data" / "instance.json"


@dataclass(frozen=True)
class Instance:
    mass: np.ndarray
    volume: np.ndarray
    value: np.ndarray
    cap_mass: int
    cap_volume: int
    meta: dict

    @property
    def n(self) -> int:
        return len(self.value)

    def loads(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        X = np.atleast_2d(X).astype(np.int64)
        return X @ self.mass, X @ self.volume, X @ self.value

    def excess(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        m, v, _ = self.loads(X)
        return np.maximum(0, m - self.cap_mass), np.maximum(0, v - self.cap_volume)

    def feasible(self, X: np.ndarray) -> np.ndarray:
        em, ev = self.excess(X)
        return (em == 0) & (ev == 0)

    def efficiency(self) -> np.ndarray:
        return self.value / (self.mass / self.cap_mass + self.volume / self.cap_volume)

    def to_json(self, path: str | Path) -> None:
        data = {"meta": self.meta, "cap_mass": int(self.cap_mass),
                "cap_volume": int(self.cap_volume),
                "items": [{"id": i, "mass": int(m), "volume": int(v), "value": int(p)}
                          for i, (m, v, p) in enumerate(zip(self.mass, self.volume, self.value))]}
        Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

    @classmethod
    def from_json(cls, path: str | Path = DEFAULT_INSTANCE) -> "Instance":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        items = data["items"]
        return cls(mass=np.array([it["mass"] for it in items], dtype=np.int64),
                   volume=np.array([it["volume"] for it in items], dtype=np.int64),
                   value=np.array([it["value"] for it in items], dtype=np.int64),
                   cap_mass=int(data["cap_mass"]), cap_volume=int(data["cap_volume"]),
                   meta=data.get("meta", {}))
