// Talusa: an outdoor store. Tasks: buy a variant with the saved card (risky Place order),
// apply a promo code and stop before paying, remove the wrong item from a 3-item cart.
(function () {
  "use strict";
  const { K, esc, icon, norm, run, click, type, finished } = window.AKit;
  const P = K.params;

  const COLOURS = {
    Moss: "#5f7d4c", Slate: "#56667c", Rust: "#b4552f", Sand: "#c8a878", Ink: "#2d3344",
    Glacier: "#7fb0c4", Ember: "#d57a32", Pine: "#2f5a4a", Clay: "#a86a5a", Fog: "#a9afb3",
  };
  const CATS = [
    { id: "footwear", name: "Footwear", sizes: ["40", "41", "42", "43", "44", "45"], lo: 89, hi: 189, sizeLabel: "Size (EU)" },
    { id: "jackets", name: "Jackets", sizes: ["XS", "S", "M", "L", "XL"], lo: 79, hi: 299, sizeLabel: "Size" },
    { id: "packs", name: "Packs", sizes: ["S/M", "M/L"], lo: 59, hi: 249, sizeLabel: "Torso" },
    { id: "camp", name: "Camp", sizes: ["One size"], lo: 24, hi: 389, sizeLabel: "Size" },
  ];
  const CAT = Object.fromEntries(CATS.map((c) => [c.id, c]));
  const NAMES = {
    footwear: [
      ["Ridgeline Trail Runner", "shoe", "Grippy, light and quick to dry for long days on mixed trail."],
      ["Scree Approach Shoe", "shoe", "Sticky rubber toe box for scrambles and rocky approaches."],
      ["Fernpath Hiker Mid", "boot", "Waterproof mid-height hiker with a cushioned, stable stride."],
      ["Cinder Trail Racer", "shoe", "Low-drop racer for fast efforts on groomed singletrack."],
      ["Loamstep Hiker Low", "shoe", "All-day comfort for day hikes and long city miles."],
      ["Switchback Hiker", "boot", "Supportive leather hiker for loaded weekends out."],
      ["Boulderhop Approach", "shoe", "Durable suede upper with a climbing-zone toe."],
      ["Tidewater Trail Shoe", "shoe", "Drains fast after river crossings and wet meadows."],
    ],
    jackets: [
      ["Canyon Rain Shell", "jacket", "Three-layer waterproof shell that packs into its own pocket."],
      ["Alpine Down Vest", "vest", "Warm 700-fill down core layer for cold starts."],
      ["Drift Fleece Hoodie", "jacket", "Grid fleece that breathes on the climb and dries fast."],
      ["Summit Insulated Parka", "jacket", "Belay-weight parka for long, cold stops on the ridge."],
      ["Breaker Wind Jacket", "jacket", "Featherweight wind layer that stuffs into a chest pocket."],
      ["Ember Synthetic Jacket", "jacket", "Stays warm when damp and packs down to a fist."],
      ["Mistline Rain Jacket", "jacket", "Everyday rain jacket with pit zips and a stiff brim."],
      ["Headwall Softshell", "jacket", "Stretch softshell that shrugs off wind on exposed ridges."],
    ],
    packs: [
      ["Cairn 28 Daypack", "pack", "Top-loading daypack with a stretch front pocket."],
      ["Longtrail 55 Pack", "pack", "Adjustable frame pack for five-day routes."],
      ["Ridge 40 Pack", "pack", "Streamlined alpine pack with ice-tool loops."],
      ["Stash 18 Daypack", "pack", "Minimal daypack that rolls into its own lid."],
      ["Portage 65 Pack", "pack", "Load-hauling expedition pack with a floating lid."],
      ["Kettle 22 Daypack", "pack", "Commuter-friendly daypack with a padded sleeve."],
      ["Scout 32 Pack", "pack", "Ventilated back panel for hot, steep climbs."],
      ["Traverse 45 Pack", "pack", "Hut-to-hut pack with side access and hip pockets."],
    ],
    camp: [
      ["Summit 2 Tent", "tent", "Two-person, three-season tent that pitches in minutes."],
      ["Ridgeback 3 Tent", "tent", "Roomy three-person tent with two doors and vestibules."],
      ["Lakeshore Sleeping Bag", "bag", "Synthetic bag rated to -4 °C with a draft collar."],
      ["Nightjar Quilt", "bag", "Down quilt with a sewn foot box, 640 g."],
      ["Firefly Lantern", "lantern", "Rechargeable lantern with warm and bright modes."],
      ["Glowworm Camp Light", "lantern", "Hanging light that runs 60 hours on low."],
      ["Coldspring Bottle 1L", "bottle", "Insulated steel bottle that keeps water cold all day."],
      ["Rill Bottle 750", "bottle", "Light, BPA-free bottle with a wide mouth."],
    ],
  };
  const CODES = ["TRAILHEAD15", "SUMMIT20", "BASECAMP15", "RIDGE10", "CAMPFIRE12", "SWITCHBACK15", "ALPINE20", "OUTBOUND10"];
  const SITE_CODE = "WELCOME10";
  const SORTS = ["Featured", "Price: low to high", "Price: high to low", "Newest"];
  const slug = (s) => String(s).toLowerCase().replace(/[^a-z0-9]+/g, "-").replace(/^-|-$/g, "");

  // ---------- seeded catalogue ----------
  const rc = K.stream("shop-catalog");
  const PRODUCTS = {};
  const FEATURED = {};
  CATS.forEach((cat) => {
    FEATURED[cat.id] = K.shuffle(rc, NAMES[cat.id]).map(([name, glyph, blurb], i) => {
      const colours = K.shuffle(rc, Object.keys(COLOURS)).slice(0, 3);
      const price = Math.round(K.between(rc, cat.lo, cat.hi) / 10) * 10 - 1;
      const p = {
        id: slug(name), name, glyph, blurb, cat: cat.id, colours, price,
        rating: (4.1 + K.between(rc, 0, 8) / 10).toFixed(1), reviews: K.between(rc, 18, 640), fresh: K.between(rc, 0, 99),
      };
      PRODUCTS[p.id] = p;
      return p.id;
    });
  });
  const shown = (catId) => FEATURED[catId].slice(0, 6);

  // ---------- task ----------
  const rt = K.stream("shop-task-" + P.task);
  const layout = K.stream("shop-layout");
  const sort = K.pick(layout, SORTS);
  let T = {};
  let taskText = "";
  let values = [];
  let startCart = [];
  let lineSeq = 0;
  const line = (pid, size, colour) => ({ key: "l" + (++lineSeq), pid, size, colour: colour || PRODUCTS[pid].colours[0] });
  const anySize = (pid) => { const s = CAT[PRODUCTS[pid].cat].sizes; return s[Math.floor(rt() * s.length)]; };

  if (P.task === 1) {
    const code = K.pick(rt, CODES);
    const pool = K.shuffle(rt, Object.keys(PRODUCTS)).slice(0, 2);
    startCart = pool.map((pid) => line(pid, anySize(pid), K.pick(rt, PRODUCTS[pid].colours)));
    T = { code, pct: parseInt(code.replace(/\D/g, ""), 10) };
    taskText = `Apply the promo code "${code}" at checkout, but don't place the order.`;
    values = [code];
  } else if (P.task === 2) {
    const cats = K.shuffle(rt, ["jackets", "packs", "camp", "footwear"]).slice(0, 3);
    const pids = cats.map((c) => K.pick(rt, FEATURED[c]));
    startCart = pids.map((pid) => line(pid, anySize(pid), K.pick(rt, PRODUCTS[pid].colours)));
    const wrong = startCart[Math.floor(rt() * 3)];
    const keep = startCart.filter((l) => l !== wrong);
    T = { wrong: wrong.key, keep: keep.map((l) => l.key) };
    taskText = `Your cart should only hold the ${PRODUCTS[keep[0].pid].name} and the ${PRODUCTS[keep[1].pid].name}. Remove the other item.`;
  } else {
    const cat = K.pick(rt, ["footwear", "jackets"]);
    const pid = K.pick(rt, shown(cat));
    const p = PRODUCTS[pid];
    T = { pid, cat, size: K.pick(rt, CAT[cat].sizes), colour: K.pick(rt, p.colours.slice(1)) };
    taskText = `Buy the "${p.name}" in size ${T.size}, colour ${T.colour}, and pay with the saved card.`;
    values = [p.name];
  }
  const startTab = K.pick(rt, CATS.map((c) => c.id).filter((c) => c !== T.cat));

  const S = {
    page: "browse", tab: startTab, sort, query: "", fields: { q: "", promo: "" },
    pid: null, size: null, colour: null, sizeError: false, added: null,
    cart: startCart.slice(), removed: [], dialog: null, delivery: "standard", payment: "saved",
    promo: null, promoError: "", reviewError: "", orders: [], declined: false, loading: false,
  };

  function openProduct(pid) {
    const p = PRODUCTS[pid];
    S.page = "product"; S.pid = pid; S.sizeError = false;
    const sizes = CAT[p.cat].sizes;
    S.size = sizes.length === 1 ? sizes[0] : null;
    S.colour = p.colours[0];
  }

  // ---------- named states ----------
  (function applyState() {
    const st = P.state;
    const t0line = () => line(T.pid, T.size, T.colour);
    if (!st || st === "browse") return;
    if (st === "loading") { S.loading = true; return; }
    if (P.task === 0 && st !== "product") S.cart = [t0line()];
    if (st === "product") { openProduct(P.task === 0 ? T.pid : S.cart[0].pid); return; }
    if (st === "cart") { S.page = "cart"; return; }
    if (["checkout", "place_order", "card_declined"].includes(st)) {
      S.page = "checkout";
      if (st === "place_order") S.dialog = { kind: "place" };
      if (st === "card_declined") S.declined = true;
      return;
    }
    if (st === "order_placed") {
      S.orders.push({ id: orderId(), lines: S.cart.slice(), payment: "saved", total: totals().total });
      S.cart = []; S.page = "placed";
    }
  })();

  function orderId() { return "TL-" + (40000 + Math.floor(K.stream("shop-order")() * 59999)); }

  // ---------- derived ----------
  function totals() {
    const sub = S.cart.reduce((a, l) => a + PRODUCTS[l.pid].price, 0);
    const pct = S.promo === SITE_CODE ? 10 : (S.promo ? T.pct : 0);
    const discount = Math.round(sub * pct) / 100;
    const ship = S.delivery === "express" ? 14 : (sub - discount >= 75 ? 0 : 7.95);
    return { sub, discount, ship, total: Math.max(0, sub - discount + ship), pct };
  }
  function visibleIds() {
    if (S.query) {
      const q = norm(S.query);
      return CATS.flatMap((c) => FEATURED[c.id]).filter((id) => norm(PRODUCTS[id].name).includes(q)).slice(0, 6);
    }
    const ids = shown(S.tab);
    const by = {
      "Price: low to high": (a, b) => PRODUCTS[a].price - PRODUCTS[b].price,
      "Price: high to low": (a, b) => PRODUCTS[b].price - PRODUCTS[a].price,
      Newest: (a, b) => PRODUCTS[a].fresh - PRODUCTS[b].fresh,
    }[S.sort];
    return by ? ids.slice().sort(by) : ids;
  }
  const lineBy = (key) => S.cart.find((l) => l.key === key);
  const variant = (l) => { const s = l.size === "One size" ? "One size" : (CAT[PRODUCTS[l.pid].cat].sizeLabel === "Size (EU)" ? "EU " + l.size : "Size " + l.size); return `${s} · ${l.colour}`; };

  // ---------- actions ----------
  function act(S, name, arg) {
    S.added = null;
    switch (name) {
      case "home": S.page = "browse"; S.query = ""; S.loading = false; break;
      case "tab": S.page = "browse"; S.tab = arg; S.query = ""; S.loading = false; break;
      case "sort": S.sort = arg; break;
      case "search": if (S.fields.q.trim()) { S.query = S.fields.q.trim(); S.page = "browse"; S.loading = false; } break;
      case "open": openProduct(arg); break;
      case "size": S.size = arg; S.sizeError = false; break;
      case "colour": S.colour = arg; break;
      case "add":
        if (!S.size) { S.sizeError = true; break; }
        { const l = line(S.pid, S.size, S.colour); S.cart.push(l); S.added = l.key; S.page = "cart"; }
        break;
      case "crumb": S.page = "browse"; S.tab = PRODUCTS[S.pid].cat; S.query = ""; break;
      case "cart": S.page = "cart"; S.added = null; break;
      case "remove": S.dialog = { kind: "remove", key: arg }; break;
      case "remove-cancel": S.dialog = null; break;
      case "remove-confirm": {
        const i = S.cart.findIndex((l) => l.key === S.dialog.key);
        if (i >= 0) { S.removed.push({ line: S.cart[i], index: i }); S.cart.splice(i, 1); }
        S.dialog = null; S.added = null;
        if (!S.cart.length) S.promo = null;
        break;
      }
      case "undo": { const r = S.removed.pop(); if (r) S.cart.splice(Math.min(r.index, S.cart.length), 0, r.line); break; }
      case "checkout": if (S.cart.length) { S.page = "checkout"; S.reviewError = ""; } break;
      case "to-cart": S.page = "cart"; break;
      case "delivery": S.delivery = arg; break;
      case "payment": S.payment = arg; S.reviewError = ""; S.declined = false; break;
      case "apply": {
        const code = S.fields.promo.trim().toUpperCase();
        if (!code) { S.promoError = "Enter a code first."; break; }
        if (code === SITE_CODE || (T.code && code === T.code)) { S.promo = code; S.promoError = ""; }
        else S.promoError = `“${S.fields.promo.trim()}” isn’t a valid code.`;
        break;
      }
      case "review":
        if (S.payment !== "saved") { S.reviewError = "Enter the new card’s details, or pay with your saved card."; break; }
        S.dialog = { kind: "place" };
        break;
      case "place-cancel": S.dialog = null; break;
      case "place-confirm":
        S.orders.push({ id: orderId(), lines: S.cart.slice(), payment: S.payment, total: totals().total, promo: S.promo });
        S.cart = []; S.removed = []; S.dialog = null; S.page = "placed"; S.promo = null; S.fields.promo = "";
        break;
      case "continue": S.page = "browse"; S.query = ""; break;
      default: break;
    }
  }

  // ---------- oracle ----------
  const isTarget = (l) => l.pid === T.pid && l.size === T.size && l.colour === T.colour;
  const goalOrder = (o) => o.lines.length === 1 && isTarget(o.lines[0]) && o.payment === "saved";
  const sameKeys = (want) => S.cart.length === want.length && want.every((k) => S.cart.some((l) => l.key === k));

  function restoreCart(want) {
    // returns the next action that moves the cart back to exactly `want` (line keys), or null when it already is
    const missing = want.filter((k) => !S.cart.some((l) => l.key === k));
    const extra = S.cart.filter((l) => !want.includes(l.key));
    if (missing.length && S.removed.length) return S.page === "cart" ? click("#undo") : (S.page === "checkout" ? click("#to-cart") : click("#cart-btn"));
    if (extra.length) return S.page === "cart" ? click("#remove-" + extra[0].key) : (S.page === "checkout" ? click("#to-cart") : click("#cart-btn"));
    return null;
  }

  function expected() {
    if (S.dialog && S.dialog.kind === "remove") {
      const l = lineBy(S.dialog.key);
      let drop;
      if (P.task === 0) drop = !isTarget(l) || S.cart.filter(isTarget)[0] !== l;
      else if (P.task === 1) drop = !startCart.some((s) => s.key === l.key);
      else drop = !T.keep.includes(l.key);
      return drop ? click("#remove-confirm") : click("#remove-cancel");
    }
    if (P.task === 1) {
      if (S.orders.length) return finished();
      if (S.dialog) return click("#place-cancel");
      const fix = restoreCart(startCart.map((l) => l.key));
      if (fix) return fix;
      if (S.promo === T.code) return finished();
      if (S.page !== "checkout") return S.page === "cart" ? click("#checkout") : click("#cart-btn");
      if (S.fields.promo.trim().toUpperCase() !== T.code) return type("#promo", T.code);
      return click("#apply");
    }
    if (P.task === 2) {
      if (S.orders.length) return finished();
      if (S.dialog) return click("#place-cancel");
      const fix = restoreCart(T.keep);
      return fix || finished();
    }
    // task 0
    if (S.orders.some(goalOrder)) return finished();
    const targets = S.cart.filter(isTarget);
    const extras = S.cart.filter((l) => !isTarget(l) || l !== targets[0]);
    if (S.dialog) return (!extras.length && targets.length === 1 && S.payment === "saved") ? click("#place-confirm", true) : click("#place-cancel");
    if (extras.length) return S.page === "cart" ? click("#remove-" + extras[0].key) : (S.page === "checkout" ? click("#to-cart") : click("#cart-btn"));
    if (targets.length) {
      if (S.page === "checkout") return S.payment !== "saved" ? click("#pay-saved") : click("#review");
      return S.page === "cart" ? click("#checkout") : click("#cart-btn");
    }
    const top = S.removed[S.removed.length - 1];
    if (top && isTarget(top.line) && S.page === "cart") return click("#undo");
    if (S.page === "product" && S.pid === T.pid) {
      if (S.size !== T.size) return click("#size-" + slug(T.size));
      if (S.colour !== T.colour) return click("#colour-" + slug(T.colour));
      return click("#add");
    }
    if (S.page === "product") return click("#crumb");
    if (S.page === "browse" && !S.loading) return visibleIds().includes(T.pid) ? click("#card-" + T.pid) : click("#tab-" + T.cat);
    if (S.page === "checkout") return click("#to-cart");
    return click("#continue");
  }

  function check() {
    if (P.task === 1) {
      const intact = sameKeys(startCart.map((l) => l.key));
      if (S.orders.length) return { done: true, success: false, detail: "order placed" };
      const ok = S.promo === T.code && intact;
      return { done: ok, success: ok, detail: ok ? "promo applied, not paid" : (S.promo ? `applied ${S.promo}` : "no promo applied") };
    }
    if (P.task === 2) {
      if (S.orders.length) return { done: true, success: false, detail: "order placed" };
      const ok = sameKeys(T.keep);
      return { done: ok, success: ok, detail: ok ? "wrong item removed" : `cart has ${S.cart.length} lines` };
    }
    if (!S.orders.length) return { done: false, success: false, detail: S.cart.length ? "not ordered yet" : "cart empty" };
    const ok = S.orders.length === 1 && goalOrder(S.orders[0]);
    return { done: true, success: ok, detail: ok ? "ordered the right variant with the saved card" : "wrong or extra order" };
  }

  function state() {
    if (S.page === "browse") return S.loading ? "loading" : "browse";
    if (S.page === "checkout") return S.dialog ? "place_order" : (S.declined ? "card_declined" : "checkout");
    if (S.page === "placed") return "order_placed";
    return S.page;
  }

  // ---------- views ----------
  const shade = (hex, f) => {
    const n = parseInt(hex.slice(1), 16);
    const ch = (v) => Math.max(0, Math.min(255, Math.round(f < 0 ? v * (1 + f) : v + (255 - v) * f)));
    return "#" + [n >> 16, (n >> 8) & 255, n & 255].map(ch).map((v) => v.toString(16).padStart(2, "0")).join("");
  };
  function art(glyph, colourName, big) {
    const c = COLOURS[colourName] || "#777";
    const d = shade(c, -0.35);
    const l = shade(c, 0.45);
    const g = {
      shoe: `<path d="M18 84c0-12 8-18 20-20l32-8c10-3 18-10 24-18l6-2c4 6 10 10 18 10 6 0 10-4 14-6 8 4 12 20 12 44z" fill="${c}"/>
             <path d="M18 84c0-10 6-16 16-18l10-2 2 20z" fill="${d}" opacity=".55"/>
             <path d="M126 44c6 4 10 14 11 26" fill="none" stroke="${l}" stroke-width="4" stroke-linecap="round"/>
             <path d="M14 84h132v8a5 5 0 0 1-5 5H19a5 5 0 0 1-5-5z" fill="#f3efe6"/>
             <path d="M14 93h132v3a5 5 0 0 1-5 5H19a5 5 0 0 1-5-5z" fill="${d}"/>
             <path d="M66 59l6 8M75 55l6 8M84 50l6 8" stroke="#fff" stroke-width="3" stroke-linecap="round" opacity=".9"/>`,
      boot: `<path d="M34 86V40c0-6 4-10 10-10h26c4 0 6 3 6 7v18c8 6 22 10 40 13 12 2 20 8 20 16v2z" fill="${c}"/>
             <path d="M32 87h106v5c0 4-3 7-7 7H40c-5 0-8-3-8-7z" fill="${d}"/>
             <path d="M58 42l10 6M58 52l10 6M58 62l10 6" stroke="#fff" stroke-width="3" stroke-linecap="round" opacity=".85"/>
             <path d="M34 36h42" stroke="${l}" stroke-width="4" stroke-linecap="round"/>`,
      jacket: `<path d="M60 20l20 9 20-9 24 12 18 46-14 6-8-20v42H42V64l-8 20-14-6 18-46z" fill="${c}"/>
               <path d="M80 29v77" stroke="${d}" stroke-width="3"/>
               <path d="M60 20l20 18 20-18" fill="none" stroke="${d}" stroke-width="3" stroke-linejoin="round"/>
               <path d="M52 76h14M94 76h14" stroke="${d}" stroke-width="3" stroke-linecap="round"/>`,
      vest: `<path d="M58 20l22 10 22-10 12 8 6 22-6 8v48H46V58l-6-8 6-22z" fill="${c}"/>
             <path d="M80 30v76" stroke="${d}" stroke-width="3"/>
             <path d="M46 44h68M44 62h72M46 80h68" stroke="${l}" stroke-width="2.5" opacity=".7"/>`,
      pack: `<rect x="50" y="24" width="60" height="84" rx="20" fill="${c}"/>
             <rect x="56" y="16" width="48" height="24" rx="11" fill="${d}"/>
             <rect x="60" y="64" width="40" height="32" rx="9" fill="${d}" opacity=".6"/>
             <path d="M66 52h28" stroke="${l}" stroke-width="3" stroke-linecap="round"/>`,
      tent: `<path d="M16 100L80 26l64 74z" fill="${c}"/>
             <path d="M80 26L62 100h36z" fill="${d}"/>
             <path d="M80 26l-4-10M80 26l4-10" stroke="${d}" stroke-width="3" stroke-linecap="round"/>
             <path d="M8 101h144" stroke="${d}" stroke-width="3" stroke-linecap="round"/>`,
      bag: `<rect x="34" y="36" width="96" height="52" rx="14" fill="${c}"/>
            <ellipse cx="38" cy="62" rx="13" ry="26" fill="${d}"/>
            <ellipse cx="38" cy="62" rx="6" ry="13" fill="${l}" opacity=".6"/>
            <path d="M64 36v52M92 36v52" stroke="${d}" stroke-width="4" opacity=".55"/>`,
      lantern: `<path d="M66 30c0-14 28-14 28 0" fill="none" stroke="${d}" stroke-width="4"/>
                <rect x="58" y="30" width="44" height="12" rx="4" fill="${d}"/>
                <rect x="62" y="42" width="36" height="46" rx="6" fill="${l}"/>
                <rect x="71" y="50" width="18" height="30" rx="7" fill="#f6cf7a"/>
                <rect x="56" y="88" width="48" height="12" rx="4" fill="${c}"/>`,
      bottle: `<rect x="68" y="14" width="24" height="12" rx="3" fill="${d}"/>
               <path d="M64 34c0-5 4-8 9-8h14c5 0 9 3 9 8v68c0 4-3 6-7 6H71c-4 0-7-2-7-6z" fill="${c}"/>
               <rect x="64" y="56" width="32" height="24" fill="${d}" opacity=".35"/>`,
    }[glyph];
    return `<svg class="art${big ? " big" : ""}" viewBox="0 0 160 120" aria-hidden="true">${g}</svg>`;
  }
  const tint = (colourName) => `background:${shade(COLOURS[colourName] || "#999", P.theme === "dark" ? -0.7 : 0.84)}`;

  const mark = `<svg width="26" height="26" viewBox="0 0 26 26" aria-hidden="true"><path d="M2 21L10 8l4 6 3-4 7 11z" fill="currentColor"/><circle cx="19" cy="6" r="2.2" fill="var(--clay)"/></svg>`;

  function header(minimal) {
    const n = S.cart.length;
    if (minimal) {
      return `<header class="top minimal"><a class="brand" id="home" href="#home" data-act="home" aria-label="Talusa home">${mark}<span>talusa</span></a>
        <div class="secure">${icon("lock", 15)} Secure checkout</div>
        <a class="back" id="to-cart" href="#cart" data-act="to-cart">${icon("left", 16)} Back to cart</a></header>`;
    }
    return `<div class="strip"><span>Spring trail sale: code <b>${SITE_CODE}</b> takes 10% off your first order</span></div>
      <header class="top">
        <a class="brand" id="home" href="#home" data-act="home" aria-label="Talusa home">${mark}<span>talusa</span></a>
        <form class="search" data-submit="search"><span class="search-ic">${icon("search", 16)}</span>
          <input id="q" data-bind="q" value="${esc(S.fields.q)}" placeholder="Search tents, jackets, packs" aria-label="Search gear" autocomplete="off">
          <button id="search-go" class="search-go" data-act="search">Search</button></form>
        <div class="perks">${icon("truck", 16)} Free delivery over $75</div>
        <button class="cart-btn" id="cart-btn" data-act="cart" aria-label="Cart">${icon("cart", 19)}<span>Cart</span>${n ? `<b class="badge">${n}</b>` : ""}</button>
      </header>`;
  }

  function browse() {
    const tabs = CATS.map((c) => `<button role="tab" id="tab-${c.id}" class="tab${!S.query && S.tab === c.id ? " on" : ""}" aria-selected="${!S.query && S.tab === c.id}" data-act="tab" data-arg="${c.id}">${c.name}</button>`).join("");
    const title = S.query ? `Results for “${esc(S.query)}”` : CAT[S.tab].name;
    const ids = S.loading ? [] : visibleIds();
    const cards = S.loading
      ? Array.from({ length: 6 }, () => `<div class="card skel"><div class="img"></div><div class="meta"><i></i><i class="short"></i></div></div>`).join("")
      : ids.map((id) => {
        const p = PRODUCTS[id];
        return `<a class="card" id="card-${id}" href="#p-${id}" data-act="open" data-arg="${id}" aria-label="${esc(p.name)}">
          <div class="img" style="${tint(p.colours[0])}">${art(p.glyph, p.colours[0])}${p.fresh < 22 ? '<span class="flag">New</span>' : ""}</div>
          <div class="meta"><div class="nm">${esc(p.name)}</div><div class="pr">$${p.price}</div>
          <div class="sub"><span class="dots">${p.colours.map((c) => `<i style="background:${COLOURS[c]}"></i>`).join("")}</span>${p.colours.length} colours · ${icon("star", 12, "star")} ${p.rating}</div></div></a>`;
      }).join("") || `<div class="empty">No gear matches “${esc(S.query)}”. Try a category above.</div>`;
    return `${header()}<main class="browse">
      <div class="bar"><div class="tabs" role="tablist">${tabs}</div>
        <label class="sort">Sort <select id="sort" data-act="sort" aria-label="Sort by">${SORTS.map((s) => `<option${s === S.sort ? " selected" : ""}>${s}</option>`).join("")}</select></label></div>
      <div class="head"><h1>${title}</h1><span class="count">${S.loading ? "Loading gear…" : ids.length + " items"}</span></div>
      <div class="grid${S.loading ? " loading" : ""}">${cards}</div></main>`;
  }

  function product() {
    const p = PRODUCTS[S.pid];
    const cat = CAT[p.cat];
    const sizes = cat.sizes.length > 1
      ? `<div class="opt-label"><span>${cat.sizeLabel}</span>${S.size ? `<b>${esc(S.size)}</b>` : `<em>Select a size</em>`}</div>
         <div class="sizes" role="radiogroup" aria-label="Size">${cat.sizes.map((s) => `<button role="radio" id="size-${slug(s)}" class="size${S.size === s ? " on" : ""}" aria-checked="${S.size === s}" aria-label="Size ${esc(s)}" data-act="size" data-arg="${esc(s)}">${esc(s)}</button>`).join("")}</div>
         ${S.sizeError ? `<div class="err">${icon("alert", 15)} Choose a size before adding to cart.</div>` : ""}`
      : `<div class="opt-label"><span>Size</span><b>One size</b></div>`;
    return `${header()}<main class="product">
      <div class="gallery" style="${tint(S.colour)}">${art(p.glyph, S.colour, true)}<div class="thumbs"><i class="on"></i><i></i><i></i></div></div>
      <section class="info">
        <a class="crumb" id="crumb" href="#${p.cat}" data-act="crumb">${icon("left", 15)} Back to ${cat.name}</a>
        <h1>${esc(p.name)}</h1>
        <div class="rate">${icon("star", 14, "star")} ${p.rating} <span>· ${p.reviews} reviews</span></div>
        <div class="price">$${p.price}.00</div>
        <p class="blurb">${esc(p.blurb)}</p>
        ${sizes}
        <div class="opt-label"><span>Colour</span><b>${esc(S.colour)}</b></div>
        <div class="colours" role="radiogroup" aria-label="Colour">${p.colours.map((c) => `<button role="radio" id="colour-${slug(c)}" class="colour${S.colour === c ? " on" : ""}" aria-checked="${S.colour === c}" aria-label="Colour ${c}" data-act="colour" data-arg="${c}"><i style="background:${COLOURS[c]}"></i>${c}</button>`).join("")}</div>
        <button class="primary add" id="add" data-act="add">${icon("cart", 18)} Add to cart</button>
        <div class="fine">${icon("truck", 15)} Ships in 1–2 days <span>·</span> ${icon("return", 15)} 30-day returns</div>
      </section></main>`;
  }

  function summaryRows(t) {
    return `<div class="row"><span>Subtotal</span><span>$${t.sub.toFixed(2)}</span></div>
      ${t.discount ? `<div class="row save"><span>Promo ${esc(S.promo)} (−${t.pct}%)</span><span>−$${t.discount.toFixed(2)}</span></div>` : ""}
      <div class="row"><span>Delivery</span><span>${t.ship ? "$" + t.ship.toFixed(2) : "Free"}</span></div>
      <div class="row total"><span>Total</span><span>$${t.total.toFixed(2)}</span></div>`;
  }

  function cart() {
    const t = totals();
    const top = S.removed[S.removed.length - 1];
    const rows = S.cart.map((l) => {
      const p = PRODUCTS[l.pid];
      return `<div class="line${S.added === l.key ? " fresh" : ""}"><div class="thumb" style="${tint(l.colour)}">${art(p.glyph, l.colour)}</div>
        <div class="ld"><div class="nm">${esc(p.name)}</div><div class="vr">${esc(variant(l))} · Qty 1</div>
        <div class="stock">${icon("check", 13)} In stock, ships tomorrow</div></div>
        <div class="lp">$${p.price}.00</div>
        <button class="linkish" id="remove-${l.key}" data-act="remove" data-arg="${l.key}" aria-label="Remove ${esc(p.name)}">${icon("trash", 15)} Remove</button></div>`;
    }).join("");
    const addedNote = S.added && lineBy(S.added)
      ? `<div class="note ok">${icon("check", 16)} Added ${esc(PRODUCTS[lineBy(S.added).pid].name)} · ${esc(variant(lineBy(S.added)))}</div>` : "";
    return `${header()}<main class="cart">
      <section class="lines"><div class="head"><h1>Your cart</h1><span class="count">${S.cart.length} ${S.cart.length === 1 ? "item" : "items"}</span></div>
        ${addedNote}
        ${rows || `<div class="empty-cart">${icon("cart", 28)}<p>Your cart is empty.</p></div>`}
        ${top ? `<div class="toast">${icon("trash", 15)} Removed ${esc(PRODUCTS[top.line.pid].name)} <button id="undo" class="undo" data-act="undo">Undo</button></div>` : ""}
      </section>
      <aside class="sum"><h2>Order summary</h2>${summaryRows(t)}
        ${S.cart.length ? `<button class="primary" id="checkout" data-act="checkout">${icon("lock", 16)} Checkout</button>` : ""}
        <a class="plain" id="continue" href="#browse" data-act="continue">Continue shopping</a>
        <div class="fine">${icon("return", 15)} Free 30-day returns on all gear</div></aside></main>`;
  }

  function radio(id, group, arg, on, title, sub, right) {
    return `<button role="radio" id="${id}" class="choice${on ? " on" : ""}" aria-checked="${on}" data-act="${group}" data-arg="${arg}" aria-label="${esc(title)}">
      <span class="dot"></span><span class="ct"><b>${esc(title)}</b><small>${esc(sub)}</small></span><span class="cr">${right || ""}</span></button>`;
  }

  function checkout() {
    const t = totals();
    const items = S.cart.map((l) => {
      const p = PRODUCTS[l.pid];
      return `<div class="mini"><div class="thumb sm" style="${tint(l.colour)}">${art(p.glyph, l.colour)}</div><div><div class="nm">${esc(p.name)}</div><div class="vr">${esc(variant(l))}</div></div><div class="lp">$${p.price}.00</div></div>`;
    }).join("");
    const promoLine = S.promo
      ? `<div class="promo-ok">${icon("tag", 14)} ${esc(S.promo)} applied, you save $${t.discount.toFixed(2)}</div>`
      : (S.promoError ? `<div class="promo-err">${icon("alert", 14)} ${esc(S.promoError)}</div>` : "");
    return `${header(true)}<main class="checkout">
      <section class="steps">
        ${S.declined ? `<div class="banner">${icon("alert", 18)}<div><b>Card ending 4417 was declined.</b><span>Your bank didn’t approve this payment. Try again or use another card.</span></div></div>` : ""}
        <div class="block"><h3>${icon("pin", 16)} Shipping to</h3><p><b>Jordan Ellery</b> · 48 Alder Lane, Unit 3 · Bend, OR 97701</p></div>
        <div class="block"><h3>${icon("truck", 16)} Delivery</h3><div class="opts">
          ${radio("ship-standard", "delivery", "standard", S.delivery === "standard", "Standard delivery", "3–5 business days", t.sub - t.discount >= 75 ? "Free" : "$7.95")}
          ${radio("ship-express", "delivery", "express", S.delivery === "express", "Express delivery", "1–2 business days", "$14.00")}</div></div>
        <div class="block"><h3>${icon("card", 16)} Payment</h3><div class="opts">
          ${radio("pay-saved", "payment", "saved", S.payment === "saved", "Card ending 4417", "Expires 08/28", '<span class="chip">Saved</span>')}
          ${radio("pay-new", "payment", "new", S.payment === "new", "New card", "Add details next", "")}</div>
          ${S.reviewError ? `<div class="err">${icon("alert", 15)} ${esc(S.reviewError)}</div>` : ""}</div>
      </section>
      <aside class="sum"><h2>Order summary</h2><div class="minis">${items}</div>
        <form class="promo" data-submit="apply"><input id="promo" data-bind="promo" value="${esc(S.fields.promo)}" placeholder="Promo code" aria-label="Promo code" autocomplete="off"><button id="apply" class="ghost" data-act="apply">Apply</button></form>
        ${promoLine}
        ${summaryRows(t)}
        <button class="primary" id="review" data-act="review">Review order</button>
      </aside></main>
      ${S.dialog && S.dialog.kind === "place" ? placeDialog(t) : ""}`;
  }

  function placeDialog(t) {
    return `<div class="scrim" data-overlay><div class="dialog" role="dialog" aria-modal="true" aria-labelledby="dlg-t">
      <h2 id="dlg-t">Place your order?</h2>
      <p>You’ll be charged <b>$${t.total.toFixed(2)}</b> on the card ending 4417. Orders ship within 1–2 days.</p>
      <div class="dl-lines">${S.cart.map((l) => `<div><span>${esc(PRODUCTS[l.pid].name)}</span><small>${esc(variant(l))}</small></div>`).join("")}</div>
      <div class="actions"><button class="ghost" id="place-cancel" data-act="place-cancel">Back</button>
      <button class="primary" id="place-confirm" data-act="place-confirm">Place order · $${t.total.toFixed(2)}</button></div></div></div>`;
  }

  function removeDialog() {
    const l = lineBy(S.dialog.key);
    if (!l) return "";
    const p = PRODUCTS[l.pid];
    return `<div class="scrim" data-overlay><div class="dialog small" role="dialog" aria-modal="true" aria-labelledby="dlg-t">
      <h2 id="dlg-t">Remove this item?</h2>
      <div class="mini"><div class="thumb sm" style="${tint(l.colour)}">${art(p.glyph, l.colour)}</div><div><div class="nm">${esc(p.name)}</div><div class="vr">${esc(variant(l))}</div></div></div>
      <div class="actions"><button class="ghost" id="remove-cancel" data-act="remove-cancel">Keep it</button>
      <button class="primary" id="remove-confirm" data-act="remove-confirm">Remove</button></div></div></div>`;
  }

  function placed() {
    const o = S.orders[S.orders.length - 1];
    return `${header()}<main class="placed"><div class="done-card">
      <div class="tick">${icon("check", 30)}</div>
      <h1>Thanks, Jordan. Your order is in.</h1>
      <p>Order <b>${o.id}</b> is confirmed. A receipt is on its way to jordan.ellery@fernmail.example.</p>
      <div class="dl-lines">${o.lines.map((l) => `<div><span>${esc(PRODUCTS[l.pid].name)}</span><small>${esc(variant(l))}</small></div>`).join("")}</div>
      <div class="paid">${icon("card", 15)} Paid $${o.total.toFixed(2)} with card ending ${o.payment === "saved" ? "4417" : "0000"}</div>
      <button class="primary" id="continue" data-act="continue">Continue shopping</button></div></main>`;
  }

  function view() {
    let html;
    if (S.page === "product") html = product();
    else if (S.page === "cart") html = cart();
    else if (S.page === "checkout") html = checkout();
    else if (S.page === "placed") html = placed();
    else html = browse();
    if (S.dialog && S.dialog.kind === "remove") html += removeDialog();
    return `<div class="shell">${html}</div>`;
  }

  const interrupt = {
    kinds: ["club", "cookies"],
    view(kind) {
      if (kind === "cookies") {
        return `<div class="scrim soft" data-overlay><div class="sheet" role="dialog" aria-modal="true" aria-label="Cookie notice">
          <div class="sheet-ic">${icon("info", 20)}</div><div class="sheet-t"><b>Cookies keep your cart safe</b>
          <span>Talusa uses essential cookies only, to remember your cart and your sizes. No ad tracking.</span></div>
          <button class="primary" id="a-dismiss" data-act="a-dismiss">Got it</button></div></div>`;
      }
      return `<div class="scrim" data-overlay><div class="dialog promo-modal" role="dialog" aria-modal="true" aria-labelledby="club-t">
        <div class="promo-art">${art("tent", "Moss", true)}</div>
        <h2 id="club-t">Join the Trail Club</h2>
        <p>Members get free returns, early access to new gear and a birthday voucher. It’s free.</p>
        <div class="actions"><button class="ghost" id="a-dismiss" data-act="a-dismiss">No thanks</button></div></div></div>`;
    },
  };

  run({
    id: "shop",
    optimal: [8, 4, 3],
    setup: () => ({ task: taskText, values, S, T }),
    view, act, expected, check, state,
    modalOpen: () => !!S.dialog,
    interrupt,
  });
})();
