"""Async client for vLLM OpenAI-compatible servers: resumable, incremental, multi-server.

Input JSONL rows: {"id", "messages": [...], "thinking": bool, "max_tokens": int, "temperature": float, ...extra kept}
Output JSONL rows: {"id", "text", "reasoning", "completion_tokens", "finish_reason"} written as each request finishes.
  python -m jevlab.tclient --inp REQ.jsonl --out RESP.jsonl --servers http://127.0.0.1:8100,http://127.0.0.1:8101 --conc 96
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
import time

import httpx


async def worker(name, q, client, url, model, out, stats, lock):
    while True:
        r = await q.get()
        if r is None:
            q.task_done()
            return
        body = {"model": model, "messages": r["messages"], "max_tokens": r.get("max_tokens", 8192),
                "temperature": r.get("temperature", 0.8), "top_p": r.get("top_p", 0.95), "top_k": r.get("top_k", 20),
                "chat_template_kwargs": {"enable_thinking": bool(r.get("thinking", True))}}
        for attempt in range(4):
            try:
                resp = await client.post(url + "/v1/chat/completions", json=body, timeout=7200)
                if resp.status_code != 200:
                    raise RuntimeError(f"HTTP {resp.status_code}: {resp.text[:200]}")
                d = resp.json()
                ch = d["choices"][0]
                msg = ch["message"]
                rec = {"id": r["id"], "text": msg.get("content") or "", "reasoning": msg.get("reasoning_content") or msg.get("reasoning"),
                       "completion_tokens": d["usage"]["completion_tokens"], "finish_reason": ch.get("finish_reason")}
                async with lock:
                    out.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    out.flush()
                    stats["done"] += 1
                    stats["tok"] += rec["completion_tokens"]
                break
            except Exception as e:  # noqa: BLE001
                if attempt == 3:
                    stats["fail"] += 1
                    print(f"{name}: giving up on {r['id']}: {e}", flush=True)
                else:
                    await asyncio.sleep(5 * (attempt + 1))
        q.task_done()


async def run(a):
    done = set()
    if os.path.exists(a.out):
        for l in open(a.out):
            try:
                done.add(json.loads(l)["id"])
            except Exception:  # noqa: BLE001
                pass
    reqs = [json.loads(l) for l in open(a.inp) if l.strip()]
    reqs = [r for r in reqs if r["id"] not in done]
    random.Random(0).shuffle(reqs)
    servers = a.servers.split(",")
    async with httpx.AsyncClient(timeout=7200, limits=httpx.Limits(max_connections=a.conc * len(servers) + 8)) as client:
        models = {}
        for s in servers:
            for _ in range(120):
                try:
                    models[s] = (await client.get(s + "/v1/models")).json()["data"][0]["id"]
                    break
                except Exception:  # noqa: BLE001
                    await asyncio.sleep(10)
        print(f"{len(reqs)} requests to do ({len(done)} done) on {len(models)} servers", flush=True)
        stats = {"done": 0, "tok": 0, "fail": 0}
        lock = asyncio.Lock()
        with open(a.out, "a") as out:
            queues, tasks = [], []
            for s in models:
                q = asyncio.Queue()
                queues.append(q)
                for i in range(a.conc):
                    tasks.append(asyncio.create_task(worker(f"{s}#{i}", q, client, s, models[s], out, stats, lock)))
            for i, r in enumerate(reqs):
                queues[i % len(queues)].put_nowait(r)
            for q in queues:
                for _ in range(a.conc):
                    q.put_nowait(None)
            t0 = time.time()
            while any(not t.done() for t in tasks):
                await asyncio.sleep(60)
                el = time.time() - t0
                print(f"[{el / 60:.1f} min] done {stats['done']}/{len(reqs)} fail {stats['fail']} "
                      f"{stats['tok'] / max(el, 1):.0f} out-tok/s", flush=True)
            await asyncio.gather(*tasks)
    print("complete", stats, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inp", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--servers", required=True)
    ap.add_argument("--conc", type=int, default=64)
    asyncio.run(run(ap.parse_args()))


if __name__ == "__main__":
    main()
