/* blink docs: copy buttons, math, diagrams without a saved drawing, and the on-page contents.
   Nothing here is needed to read a page. */
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
    const pre = block.querySelector("pre");
    if (!head || !pre) return;
    const button = document.createElement("button");
    button.type = "button";
    button.className = "copy";
    button.textContent = copyLabel;
    button.addEventListener("click", async () => {
      if (!(await copyText(pre.innerText.replace(/\n$/, "")))) return;
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

  // math, typeset by KaTeX where the page has any
  if (window.katex) {
    document.querySelectorAll(".math").forEach((el) => {
      try {
        window.katex.render(el.textContent, el, {
          displayMode: el.classList.contains("display"),
          throwOnError: false,
          output: "htmlAndMathml",
        });
      } catch (err) {
        /* the source stays readable */
      }
    });
  }

  // a diagram with no saved drawing is drawn here, with the same settings
  const pending = document.querySelectorAll("pre.mermaid");
  const config = document.getElementById("mermaid-config");
  if (pending.length && window.mermaid && config) {
    window.mermaid.initialize(JSON.parse(config.textContent));
    window.mermaid.run({ nodes: pending });
  }

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
