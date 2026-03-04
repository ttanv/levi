#!/usr/bin/env python3
"""Simple round-robin load balancer for vLLM endpoints."""

import itertools
from aiohttp import web, ClientSession, ClientTimeout

BACKENDS = [
    "http://localhost:8000",
    "http://10.130.0.12:8000",
]

backend_cycle = itertools.cycle(BACKENDS)

async def proxy_handler(request: web.Request) -> web.Response:
    backend = next(backend_cycle)
    target_url = f"{backend}{request.path_qs}"

    timeout = ClientTimeout(total=300)
    async with ClientSession(timeout=timeout) as session:
        body = await request.read()
        async with session.request(
            method=request.method,
            url=target_url,
            headers={k: v for k, v in request.headers.items() if k.lower() != 'host'},
            data=body,
        ) as resp:
            response_body = await resp.read()
            return web.Response(
                body=response_body,
                status=resp.status,
                headers={k: v for k, v in resp.headers.items()
                        if k.lower() not in ('transfer-encoding', 'content-encoding')},
            )

app = web.Application()
app.router.add_route('*', '/{path:.*}', proxy_handler)

if __name__ == '__main__':
    print("Load balancer running on http://localhost:8001")
    print(f"Backends: {BACKENDS}")
    web.run_app(app, host='0.0.0.0', port=8001, print=None)
