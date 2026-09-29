// Shared runtime for the shop, mail, settings, calendar and files scenario apps.
// Each app is a model plus pure view(): every click re-renders from the model, so expected(),
// check() and state() read one source of truth. While a dialog or interrupt is open, controls
// behind it are neutralised (disabled / no href / no role) so they are never candidates.
(function () {
  "use strict";
  const K = window.BlinkScenarioKit;
  const P = K.params;

  const esc = (s) => String(s == null ? "" : s).replace(/[&<>"']/g, (c) =>
    ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));

  const ICONS = {
    search: '<circle cx="11" cy="11" r="6.5"/><path d="M20 20l-4.2-4.2"/>',
    cart: '<path d="M3 4h2.2l2.2 10.4a1.8 1.8 0 0 0 1.8 1.4h7.9a1.8 1.8 0 0 0 1.7-1.3L20.6 8H6.3"/><circle cx="10" cy="20" r="1.2"/><circle cx="17.2" cy="20" r="1.2"/>',
    close: '<path d="M6.5 6.5l11 11M17.5 6.5l-11 11"/>',
    check: '<path d="M5 12.5l4.5 4.5L19 7.5"/>',
    left: '<path d="M14.5 5.5L8 12l6.5 6.5"/>',
    right: '<path d="M9.5 5.5L16 12l-6.5 6.5"/>',
    down: '<path d="M6 9.5l6 6 6-6"/>',
    trash: '<path d="M4.5 7h15M10 11v6M14 11v6M6.5 7l.9 11.2A2 2 0 0 0 9.4 20h5.2a2 2 0 0 0 2-1.8L17.5 7M9.5 7V4.5h5V7"/>',
    archive: '<rect x="3.5" y="4.5" width="17" height="4" rx="1"/><path d="M5.5 8.5V18a1.5 1.5 0 0 0 1.5 1.5h10a1.5 1.5 0 0 0 1.5-1.5V8.5M10 12.5h4"/>',
    reply: '<path d="M10 7.5L4.5 12.5l5.5 5"/><path d="M4.5 12.5H14a5.5 5.5 0 0 1 5.5 5.5v1"/>',
    forward: '<path d="M14 7.5l5.5 5-5.5 5"/><path d="M19.5 12.5H10A5.5 5.5 0 0 0 4.5 18v1"/>',
    star: '<path d="M12 3.8l2.5 5.2 5.6.8-4 4 1 5.6-5.1-2.7-5 2.7 1-5.6-4.1-4 5.7-.8z"/>',
    inbox: '<path d="M3.5 13l2.7-7.3a1.5 1.5 0 0 1 1.4-1h8.8a1.5 1.5 0 0 1 1.4 1l2.7 7.3v5.5a1.5 1.5 0 0 1-1.5 1.5h-14A1.5 1.5 0 0 1 3.5 18.5z"/><path d="M3.5 13h4.8l1.4 2.6h4.6l1.4-2.6h4.8"/>',
    send: '<path d="M20.5 3.5L10 14"/><path d="M20.5 3.5l-6.5 17-4-6.5-6.5-4z"/>',
    file: '<path d="M13.5 3.5H7A1.5 1.5 0 0 0 5.5 5v14A1.5 1.5 0 0 0 7 20.5h10a1.5 1.5 0 0 0 1.5-1.5V8.5z"/><path d="M13.5 3.5v5h5"/>',
    folder: '<path d="M3.5 7A1.5 1.5 0 0 1 5 5.5h4.2l2 2H19A1.5 1.5 0 0 1 20.5 9v8.5A1.5 1.5 0 0 1 19 19H5a1.5 1.5 0 0 1-1.5-1.5z"/>',
    share: '<circle cx="17.5" cy="5.5" r="2.3"/><circle cx="6.5" cy="12" r="2.3"/><circle cx="17.5" cy="18.5" r="2.3"/><path d="M8.5 10.8l7-4.1M8.5 13.2l7 4.1"/>',
    more: '<circle cx="5.5" cy="12" r="1.3" fill="currentColor"/><circle cx="12" cy="12" r="1.3" fill="currentColor"/><circle cx="18.5" cy="12" r="1.3" fill="currentColor"/>',
    gear: '<circle cx="12" cy="12" r="3"/><path d="M12 3.5v2.2M12 18.3v2.2M3.5 12h2.2M18.3 12h2.2M6 6l1.6 1.6M16.4 16.4L18 18M6 18l1.6-1.6M16.4 7.6L18 6"/>',
    bell: '<path d="M6.5 16v-5a5.5 5.5 0 0 1 11 0v5l1.5 2h-14z"/><path d="M10 20.5a2 2 0 0 0 4 0"/>',
    lock: '<rect x="5" y="10.5" width="14" height="10" rx="2"/><path d="M8.5 10.5V8a3.5 3.5 0 0 1 7 0v2.5"/>',
    key: '<circle cx="8" cy="15" r="3.8"/><path d="M10.8 12.2l8.7-8.7M16.5 6.5l2.5 2.5M14.3 8.7l2 2"/>',
    shield: '<path d="M12 3.5l7.5 2.8v5.4c0 4.6-3.2 7.7-7.5 8.8-4.3-1.1-7.5-4.2-7.5-8.8V6.3z"/><path d="M9 12l2.2 2.2L15.5 10"/>',
    plus: '<path d="M12 5.5v13M5.5 12h13"/>',
    user: '<circle cx="12" cy="8.5" r="3.8"/><path d="M4.5 20a7.5 7.5 0 0 1 15 0"/>',
    users: '<circle cx="9" cy="9" r="3.3"/><path d="M3 19a6 6 0 0 1 12 0"/><path d="M15.5 5.9a3.3 3.3 0 0 1 0 6.3M17.5 13.4A6 6 0 0 1 21 19"/>',
    calendar: '<rect x="3.5" y="5" width="17" height="15.5" rx="2"/><path d="M3.5 10h17M8 3v4M16 3v4"/>',
    clock: '<circle cx="12" cy="12" r="8.5"/><path d="M12 7.5V12l3 2"/>',
    pin: '<path d="M12 20.5s6.5-5.8 6.5-11a6.5 6.5 0 0 0-13 0c0 5.2 6.5 11 6.5 11z"/><circle cx="12" cy="9.5" r="2.3"/>',
    door: '<path d="M5.5 20.5h13M7.5 20.5V4.5h9v16"/><circle cx="13.8" cy="12.5" r=".9" fill="currentColor"/>',
    truck: '<path d="M3.5 6.5h10v9h-10zM13.5 9.5h4l3 3v3h-7"/><circle cx="7" cy="17.5" r="1.8"/><circle cx="17" cy="17.5" r="1.8"/>',
    card: '<rect x="3" y="5.5" width="18" height="13" rx="2"/><path d="M3 9.5h18M6.5 15h4"/>',
    tag: '<path d="M3.5 12.2V4.5h7.7l9 9-7.7 7.7z"/><circle cx="7.8" cy="8.8" r="1.3"/>',
    heart: '<path d="M12 19.5s-7.5-4.5-7.5-10a4.2 4.2 0 0 1 7.5-2.6 4.2 4.2 0 0 1 7.5 2.6c0 5.5-7.5 10-7.5 10z"/>',
    return: '<path d="M8.5 5L4 9.5 8.5 14"/><path d="M4 9.5h10.5a5.5 5.5 0 0 1 0 11H10"/>',
    grid: '<rect x="4" y="4" width="6.5" height="6.5" rx="1.2"/><rect x="13.5" y="4" width="6.5" height="6.5" rx="1.2"/><rect x="4" y="13.5" width="6.5" height="6.5" rx="1.2"/><rect x="13.5" y="13.5" width="6.5" height="6.5" rx="1.2"/>',
    list: '<path d="M8.5 6.5h12M8.5 12h12M8.5 17.5h12"/><circle cx="4.5" cy="6.5" r=".9" fill="currentColor"/><circle cx="4.5" cy="12" r=".9" fill="currentColor"/><circle cx="4.5" cy="17.5" r=".9" fill="currentColor"/>',
    upload: '<path d="M12 15.5V4.5M7.5 9L12 4.5 16.5 9"/><path d="M4.5 15v3.5A1.5 1.5 0 0 0 6 20h12a1.5 1.5 0 0 0 1.5-1.5V15"/>',
    download: '<path d="M12 4.5v11M7.5 11l4.5 4.5 4.5-4.5"/><path d="M4.5 15v3.5A1.5 1.5 0 0 0 6 20h12a1.5 1.5 0 0 0 1.5-1.5V15"/>',
    move: '<path d="M3.5 7A1.5 1.5 0 0 1 5 5.5h4.2l2 2H19A1.5 1.5 0 0 1 20.5 9v8.5A1.5 1.5 0 0 1 19 19H5a1.5 1.5 0 0 1-1.5-1.5z"/><path d="M9 13.2h6M13 11l2.2 2.2L13 15.4"/>',
    link: '<path d="M10 14a4 4 0 0 0 5.7 0l3-3a4 4 0 0 0-5.7-5.7l-1.2 1.2"/><path d="M14 10a4 4 0 0 0-5.7 0l-3 3a4 4 0 0 0 5.7 5.7l1.2-1.2"/>',
    info: '<circle cx="12" cy="12" r="8.5"/><path d="M12 11v5.5"/><circle cx="12" cy="7.8" r=".9" fill="currentColor"/>',
    alert: '<path d="M12 4l9 15.5H3z"/><path d="M12 10v4.5"/><circle cx="12" cy="17" r=".9" fill="currentColor"/>',
    sparkle: '<path d="M12 3.5l1.8 5.2 5.2 1.8-5.2 1.8L12 17.5l-1.8-5.2L5 10.5l5.2-1.8z"/>',
    paperclip: '<path d="M19 11.5l-7.3 7.3a4.6 4.6 0 0 1-6.5-6.5l7.8-7.8a3 3 0 0 1 4.3 4.3l-7.7 7.7a1.5 1.5 0 0 1-2.2-2.1l7-7"/>',
    edit: '<path d="M4.5 19.5l1-4L16 5a2 2 0 0 1 2.8 2.8L8.5 18.4z"/>',
    mail: '<rect x="3.5" y="5.5" width="17" height="13" rx="1.8"/><path d="M4 7l8 6 8-6"/>',
    sun: '<circle cx="12" cy="12" r="3.8"/><path d="M12 3v2M12 19v2M3 12h2M19 12h2M5.6 5.6L7 7M17 17l1.4 1.4M5.6 18.4L7 17M17 7l1.4-1.4"/>',
    image: '<rect x="3.5" y="4.5" width="17" height="15" rx="2"/><circle cx="9" cy="10" r="1.8"/><path d="M20.5 16l-5-5-8.5 8.5"/>',
    video: '<rect x="3.5" y="6" width="12.5" height="12" rx="2"/><path d="M16 10.5l4.5-2.5v8l-4.5-2.5z"/>',
    cloud: '<path d="M7 18.5a4.5 4.5 0 0 1-.4-9 6 6 0 0 1 11.5 1.6A3.8 3.8 0 0 1 17.5 18.5z"/>',
    home: '<path d="M4 10.5L12 4l8 6.5V19a1.5 1.5 0 0 1-1.5 1.5H15v-6h-6v6H5.5A1.5 1.5 0 0 1 4 19z"/>',
    history: '<path d="M4.5 12a7.5 7.5 0 1 0 2.2-5.3L4.5 9"/><path d="M4.5 4.5V9H9M12 8v4.3l2.8 1.7"/>',
    globe: '<circle cx="12" cy="12" r="8.5"/><path d="M3.5 12h17M12 3.5c2.4 2.4 3.5 5.3 3.5 8.5S14.4 18.1 12 20.5M12 3.5C9.6 5.9 8.5 8.8 8.5 12s1.1 6.1 3.5 8.5"/>',
    chart: '<path d="M4.5 19.5h15M7.5 16v-5M12 16V7M16.5 16v-7.5"/>',
    logout: '<path d="M14 4.5H6A1.5 1.5 0 0 0 4.5 6v12A1.5 1.5 0 0 0 6 19.5h8M10 12h10M16.5 8.5L20 12l-3.5 3.5"/>',
    filter: '<path d="M4 5.5h16l-6.2 7.3v5.2l-3.6 1.8v-7z"/>',
    menu: '<path d="M4.5 7h15M4.5 12h15M4.5 17h15"/>',
    copy: '<rect x="8.5" y="8.5" width="11" height="11" rx="1.8"/><path d="M15.5 8.5V6a1.5 1.5 0 0 0-1.5-1.5H6A1.5 1.5 0 0 0 4.5 6v8A1.5 1.5 0 0 0 6 15.5h2.5"/>',
    video2: '<circle cx="12" cy="12" r="8.5"/><path d="M10 8.8l5 3.2-5 3.2z"/>',
  };

  function icon(name, size, cls) {
    const s = size || 18;
    return `<svg class="ic${cls ? " " + cls : ""}" width="${s}" height="${s}" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">${ICONS[name] || ""}</svg>`;
  }

  const money = (n, cur) => (cur || "$") + Number(n).toFixed(2);
  const norm = (s) => String(s == null ? "" : s).trim().replace(/\s+/g, " ").toLowerCase();
  const initials = (name) => String(name).split(/\s+/).map((w) => w[0] || "").join("").slice(0, 2).toUpperCase();

  const ACTIONABLE = 'a[href], button, input:not([type="hidden"]), select, textarea, [role="button"], [role="link"], [role="tab"], [role="checkbox"], [role="radio"], [role="switch"], [role="menuitem"], [role="option"]';

  // Everything outside the topmost [data-overlay] stops being a candidate, without changing its look.
  function neutralise(root) {
    const overlays = root.querySelectorAll("[data-overlay]");
    if (!overlays.length) return;
    const top = overlays[overlays.length - 1];
    root.querySelectorAll(ACTIONABLE).forEach((el) => {
      if (top.contains(el)) return;
      el.classList.add("is-inert");
      el.setAttribute("tabindex", "-1");
      if (el.matches("a[href]")) { el.dataset.href = el.getAttribute("href"); el.removeAttribute("href"); }
      if (el.hasAttribute("role")) { el.dataset.role = el.getAttribute("role"); el.removeAttribute("role"); }
      if (/^(BUTTON|INPUT|SELECT|TEXTAREA)$/.test(el.tagName)) el.disabled = true;
    });
    Array.from(root.children).forEach((child) => { if (!child.contains(top)) child.setAttribute("inert", ""); });
  }

  // def: {id, optimal:[per task], setup(ctx)->{task, values, S}, view(S), act(S, name, arg, el),
  //       bind(S, key, value)?, expected(S), check(S), state(S), modalOpen(S)?, interrupt:{kinds, view(kind, S)}}
  function run(def) {
    const ctx = { P, K, esc, icon, money, norm, initials };
    const made = def.setup(ctx);
    const S = made.S;
    const optimal = (def.optimal && def.optimal[P.task]) || 4;
    const ir = K.stream("a-interrupt");
    const intr = {
      enabled: P.interrupts,
      at: K.between(ir, 1, Math.max(1, Math.min(3, optimal - 2))),
      kind: K.pick(ir, (def.interrupt && def.interrupt.kinds) || ["notice"]),
      shown: false, dismissed: false, clicks: 0,
    };
    S.intr = intr;
    const root = document.getElementById("app");

    function render() {
      const a = document.activeElement;
      const keep = a && a.id && /^(INPUT|TEXTAREA)$/.test(a.tagName)
        ? { id: a.id, s: a.selectionStart, e: a.selectionEnd } : null;
      let html = def.view(S);
      if (intr.shown && def.interrupt) html += def.interrupt.view(intr.kind, S);
      root.innerHTML = html;
      neutralise(root);
      if (keep) {
        const el = document.getElementById(keep.id);
        if (el && !el.disabled) { el.focus(); try { el.setSelectionRange(keep.s, keep.e); } catch (e) { /* not a text field */ } }
      }
    }

    function after() {
      intr.clicks += 1;
      if (intr.enabled && !intr.shown && !intr.dismissed && intr.clicks >= intr.at
          && !(def.modalOpen && def.modalOpen(S)) && !safeCheck().done) {
        intr.shown = true;
      }
      render();
    }

    function safeCheck() { try { return def.check(S); } catch (e) { return { done: false, success: false, detail: String(e) }; } }

    document.addEventListener("click", (e) => {
      const el = e.target.closest("[data-act]");
      if (!el || el.closest("[inert]") || el.disabled || el.classList.contains("is-inert")) {
        if (e.target.closest("a")) e.preventDefault();
        return;
      }
      if (el.tagName === "SELECT") return;
      e.preventDefault();
      if (el.dataset.act === "a-dismiss") { intr.shown = false; intr.dismissed = true; intr.clicks += 1; render(); return; }
      def.act(S, el.dataset.act, el.dataset.arg, el);
      after();
    });
    document.addEventListener("change", (e) => {
      const el = e.target;
      if (el.tagName === "SELECT" && el.dataset.act && !el.disabled) { def.act(S, el.dataset.act, el.value, el); after(); }
    });
    document.addEventListener("input", (e) => {
      const el = e.target;
      if (el.dataset && el.dataset.bind && !el.disabled) {
        if (def.bind) def.bind(S, el.dataset.bind, el.value, el); else S.fields[el.dataset.bind] = el.value;
      }
    });
    document.addEventListener("submit", (e) => {
      e.preventDefault();
      const f = e.target;
      if (f.dataset && f.dataset.submit && !f.closest("[inert]")) { def.act(S, f.dataset.submit, f.dataset.arg, f); after(); }
    });

    K.register({
      id: def.id,
      task: made.task,
      values: made.values,
      check: () => def.check(S),
      expected: () => {
        if (intr.shown) return { selector: "#a-dismiss", action: "click", value: null, done: false, risky: false };
        return def.expected(S);
      },
      state: () => def.state(S),
    });
    window.__aApp = { S, render, intr, T: made.T || null };
    render();
  }

  // shared oracle shapes
  const click = (selector, risky) => ({ selector, action: "click", value: null, done: false, risky: !!risky });
  const type = (selector, value) => ({ selector, action: "type", value, done: false, risky: false });
  const choose = (selector, value) => ({ selector, action: "select", value, done: false, risky: false });
  const finished = () => ({ selector: null, action: "none", value: null, done: true, risky: false });

  window.AKit = { P, K, esc, icon, money, norm, initials, run, click, type, choose, finished, ACTIONABLE };
})();
