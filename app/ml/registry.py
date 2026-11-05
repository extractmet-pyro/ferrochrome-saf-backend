"""Versioned model registry with explicit promotion workflow (shadow -> active, rollback).

Online models are NEVER retrained or promoted automatically: activation requires an
authorised user (role engineer/administrator) and is audit-logged.
"""
from __future__ import annotations

import os
import pickle
import threading
import time
import uuid


class ModelRegistry:
    def __init__(self, root: str):
        self.root = root
        os.makedirs(root, exist_ok=True)
        self.lock = threading.Lock()
        self.meta: dict[str, dict] = {}
        self._load()

    def _load(self):
        for fn in os.listdir(self.root):
            if fn.endswith(".pkl"):
                try:
                    with open(os.path.join(self.root, fn), "rb") as fh:
                        obj = pickle.load(fh)
                    self.meta[obj["id"]] = obj["meta"]
                except Exception:  # noqa: BLE001
                    continue

    def register(self, trained: dict, manifest: dict) -> str:
        mid = f"{trained['target']}-{trained['kind']}-{uuid.uuid4().hex[:8]}"
        meta = {k: v for k, v in trained.items() if k not in ("model",)}
        meta.update({"id": mid, "status": "shadow", "registered_at": time.time(), "manifest": manifest})
        with self.lock, open(os.path.join(self.root, f"{mid}.pkl"), "wb") as fh:
            pickle.dump({"id": mid, "meta": meta, "model": trained["model"]}, fh)
            self.meta[mid] = meta
        return mid

    def load(self, mid: str):
        with open(os.path.join(self.root, f"{mid}.pkl"), "rb") as fh:
            return pickle.load(fh)["model"]

    def list(self) -> list[dict]:
        return [{k: v for k, v in m.items() if k not in ("test_series", "ood", "x_ref")} for m in self.meta.values()]

    def activate(self, mid: str, user: str) -> dict:
        if mid not in self.meta:
            raise KeyError(mid)
        tgt = self.meta[mid]["target"]
        with self.lock:
            for m in self.meta.values():
                if m["target"] == tgt and m["status"] == "active":
                    m["status"] = "retired"
            self.meta[mid]["status"] = "active"
            self.meta[mid]["activated_by"] = user
            self.meta[mid]["activated_at"] = time.time()
            self._persist(mid)
        return self.meta[mid]

    def _persist(self, mid):
        path = os.path.join(self.root, f"{mid}.pkl")
        with open(path, "rb") as fh:
            obj = pickle.load(fh)
        obj["meta"] = self.meta[mid]
        with open(path, "wb") as fh:
            pickle.dump(obj, fh)

    def active(self, target: str) -> tuple[str, dict] | None:
        for mid, m in self.meta.items():
            if m["target"] == target and m["status"] == "active":
                return mid, m
        return None
