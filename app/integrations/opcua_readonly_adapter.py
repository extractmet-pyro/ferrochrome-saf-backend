"""READ-ONLY OPC UA adapter boundary for an on-prem gateway.

Design rules (IEC 62443 zones & conduits): the gateway lives in the plant DMZ, subscribes to an
allow-list of node IDs with a read-only certificate/user, and replicates OUTBOUND ONLY to the
cloud twin. This class deliberately exposes no write method; any attempt raises.
Requires the `asyncua` package on the gateway host (not installed in the cloud service).
"""
from __future__ import annotations


class WriteForbidden(PermissionError):
    pass


class OPCUAReadOnlyAdapter:
    def __init__(self, endpoint: str, node_map: dict[str, str], cert_path: str | None = None):
        self.endpoint, self.node_map, self.cert_path = endpoint, node_map, cert_path

    async def read_all(self) -> dict:
        try:
            from asyncua import Client  # type: ignore
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("asyncua not installed: run this adapter on the on-prem gateway host") from exc
        out = {}
        async with Client(self.endpoint) as c:  # pragma: no cover - requires plant server
            for tag, nid in self.node_map.items():
                out[tag] = await c.get_node(nid).read_value()
        return out

    def write(self, *_a, **_k):
        raise WriteForbidden("write access is not implemented by design (advisory-only twin)")
