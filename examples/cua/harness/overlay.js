// Video-only HUD. The caller inserts it after the clean screenshot and removes it before acting.
(() => {
  "use strict";
  if (window.blinkCUAOverlay) return;

  const style = document.createElement("style");
  style.id = "blink-cua-hud-style";
  style.textContent = `
    #blink-cua-hud { position:fixed; inset:0; z-index:2147483647; pointer-events:none;
      font-family:Inter,ui-sans-serif,system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;
      color:#fff; }
    #blink-cua-hud *, #blink-cua-hud *::before { box-sizing:border-box; }
    .blink-cua-box { position:absolute; border:var(--cuastroke) solid #2447f5;
      box-shadow:0 0 0 var(--cuahalo) #fff; border-radius:1px;
      transition:box-shadow .24s ease,border-color .24s ease,background .24s ease; }
    .blink-cua-box.chosen { background:rgba(36,71,245,.10);
      box-shadow:0 0 0 var(--cuahalo) #fff,0 0 0 5px rgba(36,71,245,.32),
        0 0 24px rgba(36,71,245,.85); }
    .blink-cua-tag { position:absolute; background:#2447f5; color:#fff; font-weight:750;
      display:flex; align-items:center; justify-content:center; letter-spacing:-.02em; line-height:1; }
    .blink-cua-status { position:absolute; right:16px; bottom:16px; width:min(252px,calc(100vw - 32px));
      padding:13px 15px 14px; border-radius:12px; background:rgba(13,20,42,.94);
      box-shadow:0 12px 36px rgba(13,20,42,.22); backdrop-filter:blur(12px);
      border:1px solid rgba(255,255,255,.12); animation:blink-cua-in .24s ease both; }
    #blink-cua-hud.blink-cua-phone .blink-cua-status { right:9px; bottom:92px;
      width:205px; padding:10px 12px; }
    #blink-cua-hud[data-kind="game"] .blink-cua-status { right:8px; bottom:12px;
      width:186px; padding:9px 10px; }
    #blink-cua-hud[data-kind="game"] .blink-cua-action { font-size:14px; }
    #blink-cua-hud[data-kind="game"] .blink-cua-row { gap:4px; font-size:9px; }
    .blink-cua-eyebrow { display:flex; justify-content:space-between; align-items:center;
      color:#abb9d6; font-size:10px; font-weight:650; letter-spacing:.10em; text-transform:uppercase; }
    .blink-cua-brand { overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }
    .blink-cua-dot { display:inline-block; width:7px; height:7px; margin-right:5px;
      border-radius:50%; background:#6682ff; box-shadow:0 0 8px #6682ff;
      animation:blink-cua-pulse 1.2s ease-in-out infinite; }
    .blink-cua-action { margin:7px 0 10px; font-size:17px; line-height:1.22;
      font-weight:720; letter-spacing:-.025em; }
    .blink-cua-row { display:flex; align-items:center; gap:7px; height:17px;
      font-size:10px; color:#dce5ff; font-variant-numeric:tabular-nums; }
    .blink-cua-label { width:43px; flex:none; white-space:nowrap; }
    .blink-cua-track { flex:1; height:4px; border-radius:3px; background:#34405b; overflow:hidden; }
    .blink-cua-fill { display:block; height:100%; border-radius:3px; background:#7190ff;
      transform-origin:left; animation:blink-cua-grow .38s ease both; }
    .blink-cua-value { width:27px; text-align:right; }
    .blink-cua-footer { margin-top:9px; padding-top:8px; border-top:1px solid #39445f;
      display:flex; justify-content:space-between; font-size:10px; color:#b8c4de;
      font-variant-numeric:tabular-nums; }
    .blink-cua-confirm { color:#a9bbff; font-size:10px; margin-top:8px; }
    @keyframes blink-cua-in { from { opacity:0; transform:translateY(7px) }
      to { opacity:1; transform:translateY(0) } }
    @keyframes blink-cua-pulse { 50% { opacity:.42; transform:scale(.78) } }
    @keyframes blink-cua-grow { from { transform:scaleX(0) } to { transform:scaleX(1) } }
  `;
  document.head.appendChild(style);

  let root, status, shownModel;
  const part = (cls, content) => {
    const el = document.createElement("div");
    el.className = cls;
    if (content !== undefined) el.textContent = String(content);
    return el;
  };
  const pct = p => `${Math.round(p * 100)}%`;
  const position = (el, r) => {
    Object.assign(el.style, {
      left:`${r[0]}px`, top:`${r[1]}px`,
      width:`${r[2] - r[0]}px`, height:`${r[3] - r[1]}px`,
    });
  };

  window.blinkCUAOverlay = {
    show(marks, scale, width, kind, model) {
      root?.remove();
      root = part("");
      root.id = "blink-cua-hud";
      root.dataset.kind = kind;
      shownModel = String(model);
      root.dataset.model = shownModel;
      if (innerWidth < 500) root.classList.add("blink-cua-phone");
      const stroke = Math.max(2, Math.round(width / 360)) / scale;
      const halo = Math.max(1, Math.round(width / 360) >> 1) / scale;
      root.style.setProperty("--cuastroke", `${stroke}px`);
      root.style.setProperty("--cuahalo", `${halo}px`);
      for (const mark of marks) {
        const box = part("blink-cua-box");
        box.dataset.number = mark.n;
        position(box, [mark.box[0] - stroke, mark.box[1] - stroke,
          mark.box[2] + stroke, mark.box[3] + stroke]);
        root.appendChild(box);
        const tag = part("blink-cua-tag", mark.n);
        position(tag, mark.tag);
        tag.style.fontSize = `${Math.max(11, Math.round(Math.max(16, Math.round(width / 42)) * .72)) / scale}px`;
        root.appendChild(tag);
      }
      status = part("blink-cua-status");
      const brow = part("blink-cua-eyebrow");
      const name = part("blink-cua-brand", `●  ${shownModel}`);
      brow.append(name, part("", "DECIDING"));
      status.append(brow, part("blink-cua-action", "Looking at the screen"));
      root.appendChild(status);
      document.body.appendChild(root);
    },
    decide({number, action, probabilities, done, risky, ms, gated}) {
      root?.querySelector(`.blink-cua-box[data-number="${number}"]`)?.classList.add("chosen");
      status.replaceChildren();
      const brow = part("blink-cua-eyebrow");
      brow.append(part("blink-cua-brand", `●  ${shownModel}`), part("", gated ? "CONFIRM" : "DECISION"));
      status.append(brow, part("blink-cua-action", gated ? `Ask before clicking ${number}` : action));
      for (const [label, p] of Object.entries(probabilities).sort((a,b) => b[1] - a[1]).slice(0,3)) {
        const row = part("blink-cua-row"), track = part("blink-cua-track");
        const fill = part("blink-cua-fill");
        fill.style.width = `${Math.round(100 * p)}%`;
        track.appendChild(fill);
        row.append(part("blink-cua-label", `box ${label}`), track, part("blink-cua-value", pct(p)));
        status.appendChild(row);
      }
      const foot = part("blink-cua-footer");
      foot.append(part("", `done ${pct(done)} · risk ${pct(risky)}`),
        part("", `${Math.round(ms)} ms`));
      status.appendChild(foot);
      if (gated) status.appendChild(part("blink-cua-confirm", "Confirmation requested · auto-approving"));
    },
    approve(number) {
      const title = status?.querySelector(".blink-cua-action");
      if (title) title.textContent = `Approved · Click ${number}`;
      const confirm = status?.querySelector(".blink-cua-confirm");
      if (confirm) confirm.textContent = "Proceeding after the visible confirmation beat";
    },
    remove() {
      root?.remove();
      root = null;
      status = null;
    },
  };
})();
