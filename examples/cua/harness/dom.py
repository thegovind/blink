"""Actionable, on-screen hit regions; never included as text in a blink request."""

from __future__ import annotations

from dataclasses import dataclass

from playwright.async_api import Page

MAX_CANDIDATES = 16

_EXTRACT = r"""() => {
  const query = 'a[href], button, input:not([type="hidden"]), select, textarea, ' +
    '[role="button"], [role="link"], [role="tab"], [role="checkbox"], [role="radio"], ' +
    '[role="switch"], [role="menuitem"], [role="option"]';
  const viewport = {width: innerWidth, height: innerHeight};
  const rectOf = el => {
    const r = el.getBoundingClientRect();
    const x = Math.max(0, r.left), y = Math.max(0, r.top);
    return {x, y, w: Math.max(0, Math.min(viewport.width, r.right) - x),
      h: Math.max(0, Math.min(viewport.height, r.bottom) - y)};
  };
  const usable = el => {
    if (!el || !el.isConnected || el.matches(':disabled') ||
        el.closest('[inert], [aria-hidden="true"]') ||
        el.getAttribute('aria-disabled') === 'true') return false;
    for (let node = el; node && node.nodeType === 1; node = node.parentElement) {
      const s = getComputedStyle(node);
      if (s.display === 'none' || s.visibility === 'hidden' ||
          s.visibility === 'collapse' || Number(s.opacity) === 0) return false;
    }
    const r = rectOf(el);
    if (r.w < 2 || r.h < 2) return false;
    const hit = document.elementFromPoint(r.x + r.w / 2, r.y + r.h / 2);
    return !!hit && (el === hit || el.contains(hit) || hit.contains(el));
  };
  const text = el => (el?.innerText || el?.textContent || '').replace(/\s+/g, ' ').trim();
  const nameOf = el => {
    const labelled = (el.getAttribute('aria-labelledby') || '').split(/\s+/)
      .map(id => text(document.getElementById(id))).filter(Boolean).join(' ');
    const label = el.labels && [...el.labels].map(text).filter(Boolean).join(' ');
    const value = el.matches('input[type="button"], input[type="submit"]') ? el.value : '';
    return (labelled || el.getAttribute('aria-label') || label || value ||
      (el.tagName === 'SELECT' ? '' : text(el)) ||
      el.getAttribute('placeholder') || el.getAttribute('title') || el.getAttribute('alt') || '').trim();
  };
  const roleOf = el => {
    if (el.getAttribute('role')) return el.getAttribute('role');
    if (el.tagName === 'A') return 'link';
    if (el.tagName === 'SELECT') return 'select';
    if (el.tagName === 'TEXTAREA') return 'textbox';
    if (el.tagName === 'INPUT') {
      if (['checkbox', 'radio'].includes(el.type)) return el.type;
      return ['button', 'submit', 'reset', 'image'].includes(el.type) ? 'button' : 'textbox';
    }
    return 'button';
  };
  const selectorOf = el => {
    if (el.id && document.querySelectorAll('#' + CSS.escape(el.id)).length === 1)
      return '#' + CSS.escape(el.id);
    const parts = [];
    for (let node = el; node && node.nodeType === 1; node = node.parentElement) {
      const tag = node.tagName.toLowerCase();
      let index = 1;
      for (let prev = node.previousElementSibling; prev; prev = prev.previousElementSibling)
        if (prev.tagName === node.tagName) index++;
      parts.unshift(tag + ':nth-of-type(' + index + ')');
    }
    return parts.join(' > ');
  };
  const readingOrder = items => {
    const rows = [];
    for (const item of items.sort((a,b) => a.rect.y-b.rect.y || a.rect.x-b.rect.x)) {
      const center = item.rect.y + item.rect.h/2;
      let row = rows.find(r => Math.abs(center-r.center) <= Math.min(24, Math.min(item.rect.h,r.h)*.65));
      if (!row) {
        row = {y:item.rect.y,center,h:item.rect.h,items:[]};
        rows.push(row);
      }
      row.items.push(item);
      row.center = row.items.reduce((sum, member) =>
        sum + member.rect.y + member.rect.h/2, 0) / row.items.length;
      row.y = Math.min(row.y,item.rect.y);
    }
    return rows.sort((a,b) => a.y-b.y)
      .flatMap(row => row.items.sort((a,b) => a.rect.x-b.rect.x || a.rect.y-b.rect.y));
  };
  if (typeof window.blinkScenario?.targets === 'function') {
    return {canvas: true, candidates: readingOrder(window.blinkScenario.targets()
      .filter(t => t && typeof t.id === 'string' && t.rect &&
        Number.isFinite(t.rect.x) && Number.isFinite(t.rect.y) &&
        Number.isFinite(t.rect.w) && Number.isFinite(t.rect.h))
      .map(t => {
        const r = t.rect;
        const x = Math.max(0, r.x), y = Math.max(0, r.y);
        return {target: t.id, name: String(t.name || ''), role: String(t.role || 'button'),
          tag: 'canvas', input_type: '',
          rect: {x, y, w: Math.max(0, Math.min(viewport.width, r.x + r.w) - x),
            h: Math.max(0, Math.min(viewport.height, r.y + r.h) - y)}};
      }).filter(t => t.rect.w >= 2 && t.rect.h >= 2))
      .slice(0, 16)};
  }
  return {canvas: false, candidates: readingOrder([...document.querySelectorAll(query)]
    .filter(el => usable(el) && !(() => {
      for (let parent = el.parentElement; parent; parent = parent.parentElement)
        if (parent.matches(query) && usable(parent)) return true;
      return false;
    })())
    .map(el => {
      // The associated label is clickable and shows a checkbox/radio's readable name.
      const box = ['checkbox', 'radio'].includes(el.type) && el.labels?.length &&
        usable(el.labels[0]) ? el.labels[0] : el;
      return {selector: selectorOf(el), role: roleOf(el), name: nameOf(el),
        tag: el.tagName.toLowerCase(), input_type: el.tagName === 'INPUT' ? el.type : '',
        rect: rectOf(box)};
    }))
    .slice(0, 16)};
}"""

_MATCH = r"""({expected, candidate}) => {
  const a = document.querySelector(expected);
  const b = document.querySelector(candidate);
  if (!a || !b) return false;
  return a === b || a.contains(b) || b.contains(a);
}"""


@dataclass(frozen=True)
class Candidate:
    n: int
    rect: dict[str, float]
    role: str
    name: str
    tag: str
    input_type: str
    selector: str | None = None
    target: str | None = None

    @property
    def center(self) -> tuple[float, float]:
        return (self.rect["x"] + self.rect["w"] / 2, self.rect["y"] + self.rect["h"] / 2)


async def candidates(page: Page) -> list[Candidate]:
    found = await page.evaluate(_EXTRACT)
    return [Candidate(n=i, **item) for i, item in enumerate(found["candidates"], 1)]


async def matches_expected(page: Page, candidate: Candidate, expected: dict) -> bool:
    if candidate.target is not None:
        return candidate.target == expected.get("target")
    selector = expected.get("selector")
    if not selector:
        return False
    return bool(await page.evaluate(_MATCH, {"expected": selector, "candidate": candidate.selector}))
