/* blink docs: copy buttons, code tabs and the on-page contents. Nothing here is needed to read a page. */
(() => {
  document.documentElement.classList.add("js");
  const body = document.body;

  // a copy pill on every code block
  const copyLabel = body.dataset.copy || "Copy";
  const copiedLabel = body.dataset.copied || "Copied";
  const copyText = async (text) => {
    try {
      await navigator.clipboard.writeText(text);
      return true;
    } catch (err) {
      const area = document.createElement("textarea");
      area.value = text;
      area.setAttribute("readonly", "");
      area.style.position = "fixed";
      area.style.opacity = "0";
      document.body.append(area);
      area.select();
      let ok = false;
      try {
        ok = document.execCommand("copy");
      } catch (e) {
        ok = false;
      }
      area.remove();
      return ok;
    }
  };
  document.querySelectorAll(".code").forEach((block) => {
    const head = block.querySelector(".code-head");
    const shown = () => block.querySelector(".code-panel.on pre") || block.querySelector("pre");
    if (!head || !shown()) return;
    const button = document.createElement("button");
    button.type = "button";
    button.className = "copy";
    button.textContent = copyLabel;
    button.addEventListener("click", async () => {
      if (!(await copyText(shown().innerText.replace(/\n$/, "")))) return;
      button.textContent = copiedLabel;
      button.classList.add("done");
      clearTimeout(button.blinkTimer);
      button.blinkTimer = setTimeout(() => {
        button.textContent = copyLabel;
        button.classList.remove("done");
      }, 1600);
    });
    head.append(button);
  });

  // code tabs: one panel at a time; arrow keys, Home and End move between tabs
  document.querySelectorAll(".code-tabs").forEach((group) => {
    const tabs = [...group.querySelectorAll("[role=tab]")];
    const select = (tab, focus) => {
      for (const t of tabs) {
        const on = t === tab;
        t.setAttribute("aria-selected", String(on));
        t.tabIndex = on ? 0 : -1;
        document.getElementById(t.getAttribute("aria-controls")).classList.toggle("on", on);
      }
      if (focus) tab.focus();
    };
    tabs.forEach((tab, i) => {
      tab.addEventListener("click", () => select(tab, false));
      tab.addEventListener("keydown", (e) => {
        const to = { ArrowRight: i + 1, ArrowLeft: i - 1, Home: 0, End: tabs.length - 1 }[e.key];
        if (to === undefined) return;
        e.preventDefault();
        select(tabs[(to + tabs.length) % tabs.length], true);
      });
    });
  });

  // the on-page contents mark the section being read
  const links = [...document.querySelectorAll(".toc a[href^='#']")];
  if (!links.length) return;
  const byId = new Map(links.map((a) => [decodeURIComponent(a.hash.slice(1)), a]));
  const heads = [...document.querySelectorAll(".prose h2[id], .prose h3[id]")].filter((h) => byId.has(h.id));
  let current = null;
  let queued = false;
  const mark = () => {
    queued = false;
    const line = 120;
    let found = heads[0];
    for (const h of heads) {
      if (h.getBoundingClientRect().top <= line) found = h;
      else break;
    }
    const link = found ? byId.get(found.id) : null;
    if (link === current) return;
    if (current) current.removeAttribute("aria-current");
    current = link;
    if (current) current.setAttribute("aria-current", "location");
  };
  window.addEventListener("scroll", () => {
    if (!queued) {
      queued = true;
      requestAnimationFrame(mark);
    }
  }, { passive: true });
  mark();
})();
