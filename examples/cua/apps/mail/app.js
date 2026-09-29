(function () {
  "use strict";
  const { P, K, esc, icon, norm, initials, run, click, type, finished } = window.AKit;

  const TODAY = "Tue, Sep 29, 2026";
  const DATE_BY_LABEL = {
    "Now": `${TODAY}, 10:30 AM`,
    "10:06 AM": `${TODAY}, 10:06 AM`,
    "9:42 AM": `${TODAY}, 9:42 AM`,
    "9:18 AM": `${TODAY}, 9:18 AM`,
    "8:55 AM": `${TODAY}, 8:55 AM`,
    "8:41 AM": `${TODAY}, 8:41 AM`,
    "Yesterday": "Mon, Sep 28, 2026",
    "Wed": "Wed, Sep 23, 2026",
    "Tue": "Tue, Sep 22, 2026",
    "Mon": "Mon, Sep 21, 2026",
    "Apr 11": "Sat, Apr 11, 2026"
  };
  const ORDER_BY_LABEL = { "Now": 0, "10:06 AM": 1, "9:42 AM": 2, "9:18 AM": 3, "8:55 AM": 4, "8:41 AM": 5, "Yesterday": 6, "Wed": 7, "Tue": 8, "Mon": 9, "Apr 11": 10 };
  const dateFor = (label) => DATE_BY_LABEL[label] || `${TODAY}, ${label}`;
  const USER = { name: "Mira Sato", email: "mira@northstar.example", company: "Northstar Works" };
  const ADDRS = ["receipts@lumenbay.example", "ops@marblefox.example", "payables@cedarspan.example", "invoices@rillpoint.example", "ap@brighthollow.example"];
  const REPLIES = [
    "Thanks, I'll review it by Friday.",
    "Got it, I'll send notes this afternoon.",
    "Looks good. I'll take the next pass.",
    "Thanks, I'll confirm once I've checked it."
  ];
  const AV = ["#7c3aed", "#0ea5e9", "#059669", "#dc6b21", "#be185d", "#4f46e5", "#6d7d20", "#0891b2", "#b45309"];
  const people = [
    ["Ari Vale", "ari@northstar.example"], ["Nina Park", "nina@northstar.example"], ["Owen Bell", "owen@northstar.example"],
    ["Leah Stone", "leah@northstar.example"], ["Theo Marin", "theo@northstar.example"], ["Priya Nair", "priya@northstar.example"],
    ["Marco Dunn", "marco@northstar.example"], ["Elena Cruz", "elena@northstar.example"]
  ];
  const clients = [
    ["Jon Bell", "jon@kindrill.example"], ["Rhea Holt", "rhea@silverpine.example"], ["Cam Noor", "cam@pebbleyard.example"],
    ["Tessa Quinn", "tessa@halcyonlane.example"], ["Milo Chen", "milo@draftwell.example"]
  ];
  const newsletters = [
    ["The Weekly Grain", "hello@weeklygrain.example", "Five small systems that make teams calmer", "View in browser · This week: tidy handoffs, better notes, and a quieter Monday board.", "You are receiving this because you subscribed to The Weekly Grain. View in browser or unsubscribe from the footer."],
    ["Fieldnotes Digest", "dispatch@fieldnotes.example", "New field guide: meetings that end early", "A practical digest for operators, with templates you can copy and a short unsubscribe link.", "Fieldnotes Digest collects practical operating notes. View in browser, manage preferences, or unsubscribe anytime."],
    ["Orbit Product Letter", "updates@orbitletter.example", "Release notes for teams that ship in public", "View in browser · A round-up of product rituals, tiny launches, and reader notes.", "You signed up for Orbit Product Letter. View in browser, forward to a teammate, or unsubscribe."],
    ["Tidepool Travel Deals", "deals@tidepool.example", "Coastal fares are dipping for late spring", "This week’s subscriber fares, flexible dates, and an unsubscribe link at the bottom.", "Tidepool Travel Deals sends member-only fare ideas. View in browser or unsubscribe in one click."]
  ];
  const times = ["9:42 AM", "9:18 AM", "8:55 AM", "Yesterday", "Wed", "Tue", "Mon", "Apr 11"];
  const slug = (s) => String(s).toLowerCase().replace(/[^a-z0-9]+/g, "-").replace(/^-|-$/g, "");
  const validEmail = (s) => /^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(String(s || "").trim());

  const r = K.stream("mail-data");
  const taskR = K.stream("mail-task-" + P.task);
  const forwardPerson = K.pick(taskR, clients);
  const otherInvoicePerson = K.pick(taskR, clients.filter((p) => p[0] !== forwardPerson[0]));
  const teammate = K.pick(taskR, people);
  const otherTeammates = K.shuffle(taskR, people.filter((p) => p[0] !== teammate[0])).slice(0, 2);
  const address = K.pick(taskR, ADDRS);
  const replyText = K.pick(taskR, REPLIES);
  const newsletterCount = K.between(K.stream("mail-newsletter-count"), 3, 4);

  let seq = 0;
  function msg(kind, from, email, subject, preview, body, opts) {
    opts = opts || {};
    return {
      id: "m" + (++seq), kind, from, email, subject, preview, body,
      folder: opts.folder || "inbox", unread: opts.unread !== false, time: opts.time || times[(seq - 1) % times.length],
      date: opts.date || dateFor(opts.time || times[(seq - 1) % times.length]),
      order: opts.order ?? ORDER_BY_LABEL[opts.time || times[(seq - 1) % times.length]] ?? seq,
      attachment: opts.attachment || null, invoice: !!opts.invoice, newsletter: kind === "newsletter", target: opts.target || "",
      archivedByUser: false, originalId: opts.originalId || null, to: opts.to || USER.email, colour: AV[(seq - 1) % AV.length]
    };
  }

  const base = [];
  const chosenNews = K.shuffle(r, newsletters).slice(0, newsletterCount);
  chosenNews.forEach((n, i) => base.push(msg("newsletter", n[0], n[1], n[2], n[3],
    `<p>Hi there,</p><p>${n[3]} We gathered a few short pieces for a slower, clearer week.</p><p>${n[4]}</p>`, { time: times[i], unread: i % 2 === 0 })));
  base.push(msg("real", forwardPerson[0], forwardPerson[1], "Invoice INV-2291 for April services", "Attached is the April invoice; please forward it to the right mailbox when you have a moment.",
    `<p>Hi Mira,</p><p>Attached is invoice INV-2291 for the April implementation work. The due date is May 6 and the line items match the statement we reviewed.</p><p>Thanks for routing this to the right mailbox.</p>`,
    { time: "8:41 AM", attachment: { name: "INV-2291.pdf", size: "84 KB" }, invoice: true, target: "forward-invoice", unread: true }));
  base.push(msg("real", forwardPerson[0], forwardPerson[1], "Quick note on the portal access", "No invoice here — just confirming the guest portal role looks correct now.",
    `<p>Hi Mira,</p><p>No invoice in this one. I just wanted to confirm the guest portal role is working on our side.</p><p>Appreciate the quick help.</p>`, { time: "Yesterday", unread: false }));
  base.push(msg("real", otherInvoicePerson[0], otherInvoicePerson[1], "Invoice INV-1078 for studio rental", "The studio rental invoice is attached for your records.",
    `<p>Hello,</p><p>I attached invoice INV-1078 for the studio rental. This is separate from the implementation invoice.</p>`,
    { time: "Wed", attachment: { name: "INV-1078.pdf", size: "71 KB" }, invoice: true, unread: false }));
  base.push(msg("real", teammate[0], teammate[1], "Latest draft of the launch brief", "This is the latest version; could you review the positioning section?",
    `<p>Hi Mira,</p><p>This is the latest draft of the launch brief. Could you review the positioning section before our Friday readout?</p><p>I left two comments where the customer quote may need trimming.</p>`,
    { time: "10:06 AM", target: "latest-reply", unread: true }));
  base.push(msg("real", teammate[0], teammate[1], "Older notes from the launch brief", "Keeping the earlier notes here so the thread history is easy to compare.",
    `<p>Hi Mira,</p><p>These are the older notes from yesterday’s version. The latest draft is in a newer message.</p>`, { time: "Yesterday", unread: false }));
  base.push(msg("real", otherTeammates[0][0], otherTeammates[0][1], "Calendar note: partner sync moved", "The partner sync moved to 2:30 PM, same room and dial-in.",
    `<p>Quick update: the partner sync moved to 2:30 PM. Same room, same dial-in.</p>`, { time: "Tue", unread: false }));
  base.push(msg("real", otherTeammates[1][0], otherTeammates[1][1], "Can you sanity-check the metrics slide?", "I marked the chart that needs a second set of eyes before noon.",
    `<p>Can you sanity-check the metrics slide before noon? I marked the chart that needs another look.</p>`, { time: "Mon", unread: false }));

  const by = (pred) => base.filter(pred);
  let required;
  if (P.task === 1) {
    required = by((m) => m.from === forwardPerson[0] || (m.invoice && m.from === otherInvoicePerson[0]));
    required = required.concat(K.shuffle(K.stream("mail-fill-forward"), base.filter((m) => !required.includes(m))).slice(0, 7 - required.length));
  } else if (P.task === 2) {
    required = by((m) => m.from === teammate[0] || otherTeammates.some((p) => p[0] === m.from));
    required = required.concat(K.shuffle(K.stream("mail-fill-reply"), base.filter((m) => !required.includes(m))).slice(0, 7 - required.length));
  } else {
    required = by((m) => m.newsletter).concat(K.shuffle(K.stream("mail-fill-archive"), base.filter((m) => !m.newsletter)).slice(0, 7 - newsletterCount));
  }
  const messages = K.shuffle(K.stream("mail-final-order"), required).slice(0, 7);
  const NEWS_IDS = new Set(messages.filter((m) => m.newsletter).map((m) => m.id));
  const REAL_IDS = new Set(messages.filter((m) => !m.newsletter).map((m) => m.id));
  const invoiceTarget = messages.find((m) => m.target === "forward-invoice");
  const latestTarget = messages.find((m) => m.target === "latest-reply");

  let taskText = `Archive every newsletter in your inbox.`;
  let values = [];
  let T = { newsIds: Array.from(NEWS_IDS), realIds: Array.from(REAL_IDS) };
  if (P.task === 1) {
    taskText = `Forward the invoice from ${forwardPerson[0]} to "${address}".`;
    values = [address];
    T = { invoiceId: invoiceTarget.id, person: forwardPerson[0], address };
  } else if (P.task === 2) {
    taskText = `Reply to the latest message from ${teammate[0]} with "${replyText}"`;
    values = [replyText];
    T = { latestId: latestTarget.id, teammate: teammate[0], text: replyText };
  }

  const S = {
    folder: "inbox", selected: null, messages: messages.map((m) => ({ ...m })), sent: [],
    fields: { to: "", subject: "", body: "" }, compose: null, error: "", toast: null,
    login: false, loading: false, lastArchived: null, focus: ""
  };

  function find(id) { return S.messages.find((m) => m.id === id) || S.sent.find((m) => m.id === id); }
  function folderMessages(folder) {
    const arr = folder === "sent" ? S.sent : S.messages.filter((m) => m.folder === folder);
    return arr.slice().sort((a, b) => (a.order - b.order) || a.subject.localeCompare(b.subject));
  }
  function selectedMessage() { return S.selected ? find(S.selected) : null; }
  function openMessage(id) { S.selected = id; S.toast = null; const m = find(id); if (m) m.unread = false; }
  function openComposer(kind, m) {
    if (kind === "forward") {
      S.compose = { kind, sourceId: m.id };
      S.fields = { to: "", subject: `Fwd: ${m.subject}`, body: `Forwarded message\nFrom: ${m.from} <${m.email}>\nDate: ${m.date}\nSubject: ${m.subject}\n\n${m.body.replace(/<[^>]+>/g, " ").replace(/\s+/g, " ").trim()}` };
    } else if (kind === "reply") {
      S.compose = { kind, sourceId: m.id };
      S.fields = { to: m.email, subject: `Re: ${m.subject}`, body: "" };
    } else {
      S.compose = { kind: "new", sourceId: null };
      S.fields = { to: "", subject: "", body: "" };
    }
    S.error = "";
  }
  function sentRecord() {
    const src = S.compose && S.compose.sourceId ? find(S.compose.sourceId) : null;
    return msg("sent", USER.name, USER.email, S.fields.subject.trim(), S.fields.body.trim().slice(0, 96),
      `<p>${esc(S.fields.body.trim()).replace(/\n/g, "<br>")}</p>`, {
        folder: "sent", time: "Now", date: dateFor("Now"), unread: false, originalId: src && src.id,
        to: S.fields.to.trim(), attachment: S.compose && S.compose.kind === "forward" && src ? src.attachment : null,
        target: S.compose ? S.compose.kind : "new"
      });
  }

  function act(S, name, arg) {
    switch (name) {
      case "folder": S.folder = arg; S.selected = null; S.toast = null; break;
      case "compose": openComposer("new"); break;
      case "open": openMessage(arg); break;
      case "reply": { const m = selectedMessage(); if (m) openComposer("reply", m); break; }
      case "forward": { const m = selectedMessage(); if (m) openComposer("forward", m); break; }
      case "archive": {
        const m = selectedMessage();
        if (m && m.folder === "inbox") { m.folder = "archive"; m.archivedByUser = true; S.lastArchived = m.id; S.selected = null; S.toast = "Conversation archived"; }
        break;
      }
      case "move-inbox": { const m = selectedMessage(); if (m && m.folder === "archive") { m.folder = "inbox"; m.archivedByUser = false; S.folder = "inbox"; S.selected = null; S.toast = "Moved to Inbox"; } break; }
      case "undo": { const m = find(S.lastArchived); if (m) { m.folder = "inbox"; m.archivedByUser = false; S.folder = "inbox"; S.selected = null; } S.lastArchived = null; S.toast = null; break; }
      case "discard": S.compose = null; S.error = ""; break;
      case "send": {
        if (!validEmail(S.fields.to)) { S.error = "Add a valid recipient"; break; }
        const out = sentRecord(); S.sent.push(out); S.compose = null; S.error = ""; S.folder = "sent"; S.selected = out.id; S.toast = "Message sent";
        break;
      }
      default: break;
    }
  }

  const canUndo = (id) => S.toast === "Conversation archived" && S.lastArchived === id;
  // the task's source message was archived by mistake: undo, or open it from Archive (reply/forward work there)
  function reachArchived(id, then) {
    const m = find(id);
    if (!m || m.folder !== "archive") return null;
    if (canUndo(id)) return click("#undo");
    if (S.folder !== "archive") return click("#folder-archive");
    if (S.selected !== id) return click("#msg-" + id);
    return click(then);
  }
  function restoreReal() {
    const bad = S.messages.find((m) => REAL_IDS.has(m.id) && m.folder === "archive");
    if (!bad) return null;
    if (canUndo(bad.id)) return click("#undo");
    if (S.folder !== "archive") return click("#folder-archive");
    if (S.selected !== bad.id) return click("#msg-" + bad.id);
    return click("#move-inbox");
  }

  function expected() {
    if (S.login || S.loading) return finished();
    if (S.compose && P.task !== 1 && P.task !== 2) return click("#discard");
    if (P.task === 0) {
      const fix = restoreReal(); if (fix) return fix;
      const left = S.messages.filter((m) => NEWS_IDS.has(m.id) && m.folder === "inbox");
      if (!left.length) return finished();
      if (S.folder !== "inbox") return click("#folder-inbox");
      if (S.selected !== left[0].id) return click("#msg-" + left[0].id);
      return click("#archive");
    }
    if (P.task === 1) {
      if (S.sent.some((s) => s.target === "forward" && s.originalId === T.invoiceId && s.to === T.address)) return finished();
      if (S.compose) {
        if (S.compose.kind !== "forward" || S.compose.sourceId !== T.invoiceId) return click("#discard");
        if (S.fields.to.trim() !== T.address) return type("#to", T.address);
        return click("#send", true);
      }
      const away1 = reachArchived(T.invoiceId, "#forward"); if (away1) return away1;
      if (S.folder !== "inbox") return click("#folder-inbox");
      if (S.selected !== T.invoiceId) return click("#msg-" + T.invoiceId);
      return click("#forward");
    }
    if (P.task === 2) {
      if (S.sent.some((s) => s.target === "reply" && s.originalId === T.latestId && norm(s.preview) === norm(T.text))) return finished();
      if (S.compose) {
        if (S.compose.kind !== "reply" || S.compose.sourceId !== T.latestId) return click("#discard");
        if (S.fields.body.trim() !== T.text) return type("#body", T.text);
        return click("#send", true);
      }
      const away2 = reachArchived(T.latestId, "#reply"); if (away2) return away2;
      if (S.folder !== "inbox") return click("#folder-inbox");
      if (S.selected !== T.latestId) return click("#msg-" + T.latestId);
      return click("#reply");
    }
    return finished();
  }

  function check() {
    try {
      if (P.task === 0) {
        const newsLeft = S.messages.filter((m) => NEWS_IDS.has(m.id) && m.folder === "inbox").length;
        const realArchived = S.messages.filter((m) => REAL_IDS.has(m.id) && m.folder === "archive").map((m) => m.subject);
        const ok = newsLeft === 0 && realArchived.length === 0;
        return { done: newsLeft === 0, success: ok, detail: ok ? "all newsletters archived" : (realArchived.length ? "real message archived" : `${newsLeft} newsletters remain`) };
      }
      if (P.task === 1) {
        if (!S.sent.length) return { done: false, success: false, detail: "nothing sent" };
        const s = S.sent[0];
        const ok = S.sent.length === 1 && s.target === "forward" && s.originalId === T.invoiceId && s.to === T.address && !!s.attachment;
        return { done: true, success: ok, detail: ok ? "forwarded the invoice" : "wrong sent message" };
      }
      if (!S.sent.length) return { done: false, success: false, detail: "nothing sent" };
      const s = S.sent[0];
      const src = find(T.latestId);
      const ok = S.sent.length === 1 && s.target === "reply" && s.originalId === T.latestId && s.to === src.email && norm(s.preview) === norm(T.text);
      return { done: true, success: ok, detail: ok ? "replied to latest message" : "wrong reply sent" };
    } catch (e) { return { done: false, success: false, detail: "check error" }; }
  }

  function state() {
    if (S.login) return "login";
    if (S.loading) return "loading";
    if (S.compose) return S.error ? "recipient_error" : "compose";
    if (S.folder === "sent") return "sent";
    return S.selected ? "message" : "inbox";
  }

  (function applyState() {
    const st = P.state;
    if (!st || st === "inbox") return;
    if (st === "login") { S.login = true; return; }
    if (st === "loading") { S.loading = true; return; }
    if (st === "message") { S.selected = folderMessages("inbox")[0] && folderMessages("inbox")[0].id; return; }
    if (st === "compose") {
      if (P.task === 1 && invoiceTarget) openComposer("forward", invoiceTarget);
      else if (P.task === 2 && latestTarget) openComposer("reply", latestTarget);
      else openComposer("new");
      return;
    }
    if (st === "recipient_error") {
      if (P.task === 1 && invoiceTarget) openComposer("forward", invoiceTarget);
      else if (P.task === 2 && latestTarget) openComposer("reply", latestTarget);
      else openComposer("new");
      S.error = "Add a valid recipient";
      return;
    }
    if (st === "sent") { const out = msg("sent", USER.name, USER.email, "Re: Project brief", "Thanks, I’ll review it.", "<p>Thanks, I’ll review it.</p>", { folder: "sent", time: "Now", date: dateFor("Now"), unread: false, to: "ari@northstar.example", target: "reply" }); S.sent.push(out); S.folder = "sent"; S.selected = out.id; }
  })();

  function avatar(m, small) { return `<span class="avatar${small ? " sm" : ""}" style="--av:${m.colour}">${esc(initials(m.from))}</span>`; }
  function sidebar() {
    const unread = S.messages.filter((m) => m.folder === "inbox" && m.unread).length;
    const count = (f) => f === "sent" ? S.sent.length : S.messages.filter((m) => m.folder === f).length;
    const item = (id, label, ic, c) => `<button id="folder-${id}" class="navitem${S.folder === id ? " on" : ""}" data-act="folder" data-arg="${id}" aria-label="${label} folder">${icon(ic, 17)}<span>${label}</span>${c ? `<b>${c}</b>` : ""}</button>`;
    return `<aside class="side"><div class="brand"><svg width="28" height="28" viewBox="0 0 28 28" aria-hidden="true"><path d="M4 8.5A4.5 4.5 0 0 1 8.5 4h11A4.5 4.5 0 0 1 24 8.5v11a4.5 4.5 0 0 1-4.5 4.5h-11A4.5 4.5 0 0 1 4 19.5z" fill="currentColor" opacity=".22"/><path d="M7 9l7 5.2L21 9" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/><path d="M7 19h14" stroke="currentColor" stroke-width="2" stroke-linecap="round"/></svg><span>Tidings</span></div>
      <button id="compose" class="compose" data-act="compose">${icon("edit", 17)} Compose</button>
      <nav>${item("inbox", "Inbox", "inbox", unread)}${item("sent", "Sent", "send", count("sent"))}${item("archive", "Archive", "archive", count("archive"))}</nav>
      <div class="account"><b>${esc(USER.name)}</b><span>${esc(USER.email)}</span></div></aside>`;
  }
  function listPane() {
    const rows = S.loading ? Array.from({ length: 6 }, (_, i) => `<div class="mrow skel"><i></i><p><b></b><span></span></p></div>`).join("") : folderMessages(S.folder).slice(0, 7).map((m) =>
      `<button id="msg-${m.id}" class="mrow${S.selected === m.id ? " sel" : ""}${m.unread ? " unread" : ""}" data-act="open" data-arg="${m.id}" aria-label="${esc(m.from)} ${esc(m.subject)}">
        ${avatar(m, true)}<span class="mtext"><span class="mtop"><b>${esc(m.from)}</b><time>${esc(m.time)}</time></span><span class="subj">${esc(m.subject)}</span><span class="prev">${esc(m.preview)}</span></span></button>`).join("");
    const title = S.folder === "inbox" ? "Inbox" : S.folder === "sent" ? "Sent" : "Archive";
    return `<section class="list"><div class="listhead"><div><h1>${title}</h1><span>${folderMessages(S.folder).length} conversations</span></div></div><div class="rows">${rows || `<div class="empty">No messages in ${title.toLowerCase()}.</div>`}</div></section>`;
  }
  function readPane() {
    const m = selectedMessage();
    if (!m) return `<section class="reader empty-reader"><div class="paper">${icon("mail", 34)}<h2>Select a message</h2><p>Open a conversation from the list to read, reply, forward or archive it.</p></div></section>`;
    const inArchive = m.folder === "archive";
    const tools = m.folder !== "sent" ? `<div class="tools"><button id="reply" data-act="reply">${icon("reply", 15)} Reply</button><button id="forward" data-act="forward">${icon("forward", 15)} Forward</button>${inArchive ? `<button id="move-inbox" data-act="move-inbox">${icon("inbox", 15)} Move to Inbox</button>` : `<button id="archive" data-act="archive">${icon("archive", 15)} Archive</button>`}</div>` : `<div class="tools sentnote">Sent to ${esc(m.to)}</div>`;
    return `<section class="reader"><article class="message"><div class="subject"><h2>${esc(m.subject)}</h2></div>
      <div class="fromline">${avatar(m)}<div><b>${esc(m.from)}</b><span>${esc(m.email)} · ${esc(m.date)}</span></div></div>${tools}
      <div class="body">${m.body}</div>${m.attachment ? `<div class="attach">${icon("paperclip", 16)}<b>${esc(m.attachment.name)}</b><span>${esc(m.attachment.size)}</span></div>` : ""}</article></section>`;
  }
  function toast() { return S.toast ? `<div class="toast">${icon(S.toast.includes("sent") ? "send" : "archive", 15)} ${esc(S.toast)}${S.toast === "Conversation archived" ? `<button id="undo" data-act="undo">Undo</button>` : ""}</div>` : ""; }
  function composer() {
    if (!S.compose) return "";
    const src = S.compose.sourceId ? find(S.compose.sourceId) : null;
    const isReply = S.compose.kind === "reply";
    return `<div class="scrim" data-overlay><form class="composer" data-submit="send" aria-label="Compose message"><div class="chead"><h2>${S.compose.kind === "forward" ? "Forward message" : isReply ? "Reply" : "New message"}</h2></div>
      <label>To<input id="to" data-bind="to" value="${esc(S.fields.to)}" autocomplete="off" aria-label="To"></label>
      ${S.error ? `<div class="field-error">${icon("alert", 14)} ${esc(S.error)}</div>` : ""}
      <label>Subject<input id="subject" data-bind="subject" value="${esc(S.fields.subject)}" autocomplete="off" aria-label="Subject"></label>
      <label class="bodylabel">Message<textarea id="body" data-bind="body" aria-label="Message body">${esc(S.fields.body)}</textarea></label>
      ${isReply && src ? `<div class="quote"><b>Quoted original</b><p>${esc(src.preview)}</p></div>` : ""}
      ${S.compose.kind === "forward" && src && src.attachment ? `<div class="attach carry">${icon("paperclip", 15)}<b>${esc(src.attachment.name)}</b><span>${esc(src.attachment.size)}</span></div>` : ""}
      <div class="cactions"><button id="discard" class="ghost" data-act="discard" type="button">Discard</button><button id="send" class="primary" data-act="send" type="submit">${icon("send", 15)} Send</button></div></form></div>`;
  }
  function login() { return `<div class="login"><div class="login-card">${icon("lock", 34)}<h1>Session expired</h1><p>Sign in again to continue to Tidings for ${esc(USER.company)}.</p><button class="primary" disabled>Sign in</button></div></div>`; }
  function loading() { return `<div class="shell">${sidebar()}<main class="main">${listPane()}<section class="reader empty-reader"><div class="paper"><h2>Loading mail…</h2><p>Fetching conversations securely.</p></div></section></main></div>`; }
  function view() {
    if (S.login) return login();
    if (S.loading) return loading();
    return `<div class="shell">${sidebar()}<main class="main">${listPane()}${readPane()}</main>${toast()}${composer()}</div>`;
  }
  const interrupt = {
    kinds: ["notify", "focus"],
    view(kind) {
      if (kind === "notify") return `<div class="scrim soft" data-overlay><div class="prompt">${icon("bell", 20)}<div><b>Turn on desktop notifications?</b><span>Tidings can alert you when new mail arrives for ${esc(USER.company)}.</span></div><button id="a-dismiss" data-act="a-dismiss">Not now</button></div></div>`;
      return `<div class="scrim" data-overlay><div class="whats" role="dialog" aria-modal="true"><div class="spark">${icon("sparkle", 30)}</div><h2>Try Focused Inbox</h2><p>We can separate newsletters from teammate mail and keep the important threads up top.</p><button id="a-dismiss" data-act="a-dismiss">Maybe later</button></div></div>`;
    }
  };
  run({ id: "mail", optimal: [8, 4, 4], setup: () => ({ task: taskText, values, S, T }), view, act, expected, check, state, modalOpen: () => false, interrupt });
})();
