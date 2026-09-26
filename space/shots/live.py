"""What a browser check needs to run against the live Space as well as a local mock server.

    BLINK_URL=https://thegovind-blink.hf.space/ BLINK_WAIT_MS=150000 uv run --with playwright python shots/check_integrity.py

- BLINK_WAIT_MS (default unset: the local waits) lets a check wait longer for a model run. On the Space, the first
  call a fresh GPU worker takes can run for about 30 s.
- On Spaces, Gradio's page asks the Hub for the owner's avatar as an organization; for a user account that request
  gets a 404. That console error comes from Gradio's page, not from the app, so checks don't count it.
- BLINK_HF_TOKEN (default unset) signs the checks that press Decide in with a Hub token, so their runs count against
  that account's quota and not the anonymous one, which a full check can use up. Only an https://*.hf.space
  BLINK_URL gets the token, on requests to that Space alone; a local server and every other host never see it.
"""
import os
import re
from urllib.parse import urlsplit

WAIT_MS = int(os.environ.get("BLINK_WAIT_MS") or 0)
HF_TOKEN = os.environ.get("BLINK_HF_TOKEN") or ""
GRADIO_AVATAR = re.compile(r"^https://huggingface\.co/api/organizations/[^/]+/avatar$")


def app_error(msg) -> bool:
    """A console message a check should count: an error that isn't the Spaces avatar lookup."""
    if msg.type != "error":
        return False
    return not GRADIO_AVATAR.match((msg.location or {}).get("url") or "")


def answer_wait(default_ms: int) -> int:
    """How long to wait for a model run: the local default, or longer when BLINK_WAIT_MS asks for it."""
    return max(default_ms, WAIT_MS)


def space_origin(url: str) -> str:
    """The origin a token may go to: an https://*.hf.space URL's own, or '' for anything else."""
    parts = urlsplit(url)
    host = parts.hostname or ""
    return f"https://{host}/" if parts.scheme == "https" and host.endswith(".hf.space") else ""


async def sign_in(page) -> None:
    """With BLINK_HF_TOKEN set and a Space URL, send the token on requests to that Space and nowhere else."""
    origin = space_origin(os.environ.get("BLINK_URL", ""))
    if not HF_TOKEN or not origin:
        return

    async def add(route):
        await route.continue_(headers={**route.request.headers, "authorization": f"Bearer {HF_TOKEN}"})

    await page.route(lambda url: url.startswith(origin), add)
