// Shared helper for blink CUA scenario apps (see ../../SPEC.md).
// Seeded randomness, URL params and the window.blinkScenario contract. No network, no globals
// beyond window.blinkScenario and window.BlinkScenarioKit.
(function () {
  "use strict";

  const q = new URLSearchParams(location.search);
  const intParam = (name, dflt) => {
    const v = parseInt(q.get(name) || "", 10);
    return Number.isFinite(v) ? v : dflt;
  };

  const params = {
    seed: intParam("seed", 1),
    task: intParam("task", 0),
    interrupts: q.get("interrupts") === "1",
    theme: q.get("theme") === "dark" ? "dark" : "light",
    state: q.get("state") || "",
  };

  // mulberry32: small, fast, deterministic
  function rng(seed) {
    let a = (seed >>> 0) || 1;
    return function () {
      a = (a + 0x6d2b79f5) | 0;
      let t = Math.imul(a ^ (a >>> 15), 1 | a);
      t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  }

  // one stream per purpose, so adding a random draw in one place does not shift another
  function stream(label) {
    let h = 2166136261;
    const s = String(label);
    for (let i = 0; i < s.length; i++) h = Math.imul(h ^ s.charCodeAt(i), 16777619);
    return rng((params.seed * 1000003) ^ (h >>> 0));
  }

  const pick = (r, arr) => arr[Math.floor(r() * arr.length)];
  function shuffle(r, arr) {
    const a = arr.slice();
    for (let i = a.length - 1; i > 0; i--) {
      const j = Math.floor(r() * (i + 1));
      [a[i], a[j]] = [a[j], a[i]];
    }
    return a;
  }
  const between = (r, lo, hi) => lo + Math.floor(r() * (hi - lo + 1));

  // Register the contract object. Missing methods get safe defaults; check() never throws.
  function register(spec) {
    const safe = (fn, fallback) => function () {
      try { return fn ? fn.apply(this, arguments) : fallback; } catch (e) { return fallback; }
    };
    const obj = {
      id: spec.id,
      seed: params.seed,
      taskIndex: params.task,
      task: spec.task || "",
      values: Array.isArray(spec.values) ? spec.values.slice() : [],
      check: safe(spec.check, { done: false, success: false, detail: "check() failed" }),
      expected: safe(spec.expected, { selector: null, action: "none", value: null, done: false, risky: false }),
      state: safe(spec.state, ""),
      targets: spec.targets || null,
    };
    window.blinkScenario = obj;
    document.documentElement.dataset.theme = params.theme;
    return obj;
  }

  window.BlinkScenarioKit = { params, rng, stream, pick, shuffle, between, register };
})();
