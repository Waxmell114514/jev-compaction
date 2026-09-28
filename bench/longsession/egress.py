"""An allowlisting CONNECT proxy: the benchmark container's only way out.

Agent containers sit on an ``--internal`` Docker network, so the only thing they
can reach is the host. This proxy, bound to that network's gateway, tunnels
CONNECT requests to allowlisted hosts (the model endpoint) and refuses the rest.
Without it the agent can ``pip download`` a later release or ``git fetch`` the
upstream fix, which makes a SWE-bench score meaningless.

If ``HTTPS_PROXY`` is set on the host, tunnels are chained through it.

    python bench/swebench/egress.py --bind 172.30.0.1 --port 3128 --allow opencode.ai
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
from urllib.parse import urlparse


async def pipe(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
    try:
        while data := await reader.read(65536):
            writer.write(data)
            await writer.drain()
    except (ConnectionError, asyncio.CancelledError):
        pass
    finally:
        writer.close()


async def open_upstream(host: str, port: int, upstream: str | None):
    if not upstream:
        return await asyncio.open_connection(host, port)
    proxy = urlparse(upstream)
    reader, writer = await asyncio.open_connection(proxy.hostname, proxy.port or 80)
    writer.write(f"CONNECT {host}:{port} HTTP/1.1\r\nHost: {host}:{port}\r\n\r\n".encode())
    await writer.drain()
    status = await reader.readline()
    while (await reader.readline()) not in (b"\r\n", b""):
        pass
    if b" 200" not in status:
        writer.close()
        raise ConnectionError(status.decode(errors="replace").strip())
    return reader, writer


def handler(allow: set[str], upstream: str | None, log):
    async def handle(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        try:
            request = (await reader.readline()).decode(errors="replace").split()
            while (await reader.readline()) not in (b"\r\n", b""):
                pass
            if len(request) < 2 or request[0] != "CONNECT":
                writer.write(b"HTTP/1.1 405 Method Not Allowed\r\n\r\n")
                return
            host, _, port = request[1].rpartition(":")
            if host not in allow:
                log(f"denied {host}")
                writer.write(b"HTTP/1.1 403 Forbidden\r\n\r\n")
                return
            up_reader, up_writer = await open_upstream(host, int(port), upstream)
            writer.write(b"HTTP/1.1 200 Connection Established\r\n\r\n")
            await writer.drain()
            await asyncio.gather(pipe(reader, up_writer), pipe(up_reader, writer))
        except (ConnectionError, OSError, ValueError) as exc:
            log(f"error {type(exc).__name__}: {exc}")
            writer.write(b"HTTP/1.1 502 Bad Gateway\r\n\r\n")
        finally:
            writer.close()

    return handle


async def serve(bind: str, port: int, allow: set[str], upstream: str | None) -> None:
    def log(message: str) -> None:
        print(message, file=sys.stderr, flush=True)

    server = await asyncio.start_server(handler(allow, upstream, log), bind, port)
    print(f"egress proxy on {bind}:{port}, allowing {sorted(allow)}", flush=True)
    async with server:
        await server.serve_forever()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--bind", required=True)
    parser.add_argument("--port", type=int, default=3128)
    parser.add_argument("--allow", nargs="+", default=["opencode.ai"])
    args = parser.parse_args(argv)
    asyncio.run(serve(args.bind, args.port, set(args.allow), os.environ.get("HTTPS_PROXY")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
