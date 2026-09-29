(function () {
  "use strict";
  const { K, esc, icon, initials, run, click, type, choose, finished } = window.AKit;
  const P = K.params;

  const TABS = ["Profile", "Security", "Notifications", "API keys", "Billing"];
  const TIMEOUTS = ["15 minutes", "30 minutes", "1 hour", "4 hours", "8 hours", "24 hours"];
  const FREQS = ["Immediately", "Daily", "Weekly", "Monthly", "Never"];
  const TODAY = "September 29, 2026";
  const DEVICES = ["Personal phone", "Work phone", "Backup phone", "Travel tablet"];
  const PEOPLE = ["Mira Chen", "Theo Patel", "Nina Okafor", "Elias Rowe", "Iris Kline"];
  const DOMAINS = ["harborview.example", "plinthlabs.example", "northquay.example"];
  const KEY_NAMES = ["ci-deploy", "ci-deploy-staging", "cicd-deploy", "data-sync", "data-sync-preview", "billing-import", "billing-import-old", "notebook-lab"];
  const PREFIX = ["pl_live_4F9A", "pl_live_7C2D", "pl_test_81BE", "pl_live_0AA6", "pl_test_C910", "pl_live_DD31"];

  const slug = (s) => String(s).toLowerCase().replace(/[^a-z0-9]+/g, "-").replace(/^-|-$/g, "");
  const clone = (x) => JSON.parse(JSON.stringify(x));
  const same = (a, b) => JSON.stringify(a) === JSON.stringify(b);

  const rt = K.stream("settings-task-" + P.task);
  const rl = K.stream("settings-layout");
  const dataR = K.stream("settings-data");
  const density = K.pick(rl, ["comfortable", "compact"]);
  const collapsed = K.pick(rl, [false, false, true]);
  const user = K.pick(dataR, PEOPLE);
  const domain = K.pick(dataR, DOMAINS);
  const device = K.pick(dataR, DEVICES);
  const workspace = "Harborview Labs";
  const email = `${user.toLowerCase().replace(/ /g, ".")}@${domain}`;

  const startTimeout = K.pick(rt, TIMEOUTS.slice(0, 4));
  let targetTimeout = K.pick(rt, TIMEOUTS.filter((x) => x !== startTimeout));
  const startFreq = K.pick(rt, FREQS.filter((x) => x !== "Weekly"));
  const keyPool = K.shuffle(rt, ["ci-deploy", "ci-deploy-staging"].concat(K.shuffle(rt, KEY_NAMES.filter((n) => n !== "ci-deploy" && n !== "ci-deploy-staging")).slice(0, 3)));
  const keys = K.shuffle(rt, keyPool).map((name, i) => ({
    id: slug(name) + "-" + i,
    name,
    prefix: PREFIX[i % PREFIX.length] + "••••",
    created: ["Jan 14, 2026", "Mar 2, 2026", "May 18, 2026", "Jul 9, 2026", "Sep 4, 2026"][i],
    used: ["Today", "Yesterday", "6 days ago", "Aug 28", "Never"][i],
  }));
  const targetKey = K.pick(rt, keys).name;

  let taskText, values = [];
  if (P.task === 1) taskText = "Change the notification email frequency to Weekly and save.";
  else if (P.task === 2) { taskText = `Delete the API key named "${targetKey}".`; values = [targetKey]; }
  else taskText = `Turn on two-factor authentication, set the session timeout to ${targetTimeout}, and save.`;
  const T = { timeout: targetTimeout, key: targetKey, device };

  const original = {
    security: { twoFactor: false, timeout: startTimeout, signAlerts: K.pick(dataR, [true, false]), requireSso: false },
    notifications: { digest: startFreq, mentions: true, productNews: K.pick(dataR, [true, false]), weeklyReport: K.pick(dataR, [true, false]) },
  };
  const S = {
    page: "Profile",
    density, collapsed, user, workspace, domain,
    saved: clone(original),
    draft: clone(original),
    keys: clone(keys),
    dialog: null,
    fields: { deleteName: "" },
    toast: "", error: "", savedAt: "Last saved Sep 24",
    deleted: [], saves: [], committedOtherSessions: false, signedIn: true,
  };

  (function applyState() {
    const st = P.state;
    if (!st || st === "profile") return;
    if (st === "security") { S.page = "Security"; return; }
    if (st === "notifications") { S.page = "Notifications"; return; }
    if (st === "api_keys") { S.page = "API keys"; return; }
    if (st === "enable_2fa") { S.page = "Security"; S.dialog = { kind: "2fa" }; return; }
    if (st === "delete_key") { S.page = "API keys"; S.dialog = { kind: "delete", name: P.task === 2 ? targetKey : keys[0].name }; return; }
    if (st === "save_failed") { S.page = "Security"; S.error = "Couldn't save changes — check required security fields and try again."; return; }
    if (st === "signed_out") { S.signedIn = false; return; }
  })();

  function act(S, name, arg) {
    S.toast = ""; S.error = "";
    switch (name) {
      case "tab": S.page = arg; break;
      case "toggle-2fa":
        if (!S.draft.security.twoFactor) S.dialog = { kind: "2fa" };
        else S.draft.security.twoFactor = false;
        break;
      case "confirm-2fa": S.draft.security.twoFactor = true; S.dialog = null; break;
      case "cancel-2fa": S.dialog = null; break;
      case "security-timeout": S.draft.security.timeout = arg; break;
      case "toggle-security": S.draft.security[arg] = !S.draft.security[arg]; break;
      case "save-security":
        S.saved.security = clone(S.draft.security); S.saves.push({ kind: "security", value: clone(S.saved.security) });
        S.toast = "Security settings saved"; S.savedAt = "Saved just now"; break;
      case "notify-digest": S.draft.notifications.digest = arg; break;
      case "toggle-notify": S.draft.notifications[arg] = !S.draft.notifications[arg]; break;
      case "save-notifications":
        S.saved.notifications = clone(S.draft.notifications); S.saves.push({ kind: "notifications", value: clone(S.saved.notifications) });
        S.toast = "Notification settings saved"; S.savedAt = "Saved just now"; break;
      case "delete-open": S.dialog = { kind: "delete", name: arg, err: "" }; S.fields.deleteName = ""; break;
      case "delete-cancel": S.dialog = null; S.fields.deleteName = ""; break;
      case "delete-confirm":
        if (!S.dialog || S.fields.deleteName !== S.dialog.name) { if (S.dialog) S.dialog.err = "Type the key name exactly to continue."; break; }
        S.deleted.push(S.dialog.name); S.keys = S.keys.filter((k) => k.name !== S.dialog.name); S.dialog = null; S.fields.deleteName = ""; S.toast = "API key deleted"; break;
      case "signout-open": S.dialog = { kind: "signout" }; break;
      case "signout-cancel": S.dialog = null; break;
      case "signout-confirm": S.committedOtherSessions = true; S.dialog = null; S.toast = "Other sessions signed out"; break;
      case "sign-out": S.signedIn = false; break;
      default: break;
    }
  }
  function bind(S, key, value) { S.fields[key] = value; }

  function securityOk() {
    const s = S.saved.security;
    return s.twoFactor === true && s.timeout === T.timeout && s.signAlerts === original.security.signAlerts && s.requireSso === original.security.requireSso && !S.committedOtherSessions;
  }
  function notificationsOk() {
    const n = S.saved.notifications;
    return n.digest === "Weekly" && n.mentions === original.notifications.mentions && n.productNews === original.notifications.productNews && n.weeklyReport === original.notifications.weeklyReport;
  }
  function check() {
    if (P.task === 2) {
      const wrong = S.deleted.some((n) => n !== T.key);
      const ok = S.deleted.length === 1 && S.deleted[0] === T.key && !S.keys.some((k) => k.name === T.key);
      return { done: ok || wrong, success: ok && !wrong, detail: ok ? `deleted ${T.key}` : (wrong ? `wrong key deleted: ${S.deleted.join(", ")}` : "key still present") };
    }
    if (P.task === 1) {
      const saved = S.saves.some((x) => x.kind === "notifications");
      const ok = notificationsOk();
      // a wrong save can be corrected by saving again, so only the goal ends this task
      return { done: ok, success: ok, detail: ok ? "weekly digest saved" : (saved ? "wrong notification settings saved" : "notifications not saved") };
    }
    const saved = S.saves.some((x) => x.kind === "security");
    const ok = securityOk();
    return { done: ok || S.committedOtherSessions, success: ok, detail: ok ? "2FA and timeout saved" : (S.committedOtherSessions ? "signed out other sessions" : (saved ? "wrong security settings saved" : "security not saved")) };
  }

  function expected() {
    const chk = check();
    if (chk.success) return finished();
    if (!S.signedIn) return finished();
    if (S.dialog) {
      if (S.dialog.kind === "2fa") return P.task === 0 ? click("#turn-on-2fa") : click("#cancel-2fa");
      if (S.dialog.kind === "signout") return click("#signout-cancel");
      if (S.dialog.kind === "delete") {
        if (P.task !== 2 || S.dialog.name !== T.key) return click("#delete-cancel");
        if (S.fields.deleteName !== T.key) return type("#delete-name", T.key);
        return click("#delete-confirm", true);
      }
    }
    if (P.task === 1) {
      if (S.page !== "Notifications") return click("#nav-notifications");
      if (S.draft.notifications.digest !== "Weekly") return choose("#email-digest", "Weekly");
      for (const k of ["mentions", "productNews", "weeklyReport"]) if (S.draft.notifications[k] !== original.notifications[k]) return click("#notify-" + slug(k));
      return click("#save-notifications");
    }
    if (P.task === 2) {
      if (S.page !== "API keys") return click("#nav-api-keys");
      return click("#delete-" + slug(T.key));
    }
    if (S.page !== "Security") return click("#nav-security");
    if (!S.draft.security.twoFactor) return click("#two-factor");
    if (S.draft.security.timeout !== T.timeout) return choose("#session-timeout", T.timeout);
    const secId = { signAlerts: "#sec-sign-alerts", requireSso: "#sec-require-sso" };
    for (const k of ["signAlerts", "requireSso"]) if (S.draft.security[k] !== original.security[k]) return click(secId[k]);
    return click("#save-security");
  }

  function state() {
    if (!S.signedIn) return "signed_out";
    if (S.error) return "save_failed";
    if (S.dialog && S.dialog.kind === "2fa") return "enable_2fa";
    if (S.dialog && S.dialog.kind === "delete") return "delete_key";
    if (S.page === "Security") return "security";
    if (S.page === "Notifications") return "notifications";
    if (S.page === "API keys") return "api_keys";
    return "profile";
  }
  const modalOpen = () => !!S.dialog;

  function sw(id, checked, label, desc, actName, arg) {
    return `<div class="setting-row"><div><b>${esc(label)}</b><p>${esc(desc)}</p></div><button class="switch" id="${id}" role="switch" aria-checked="${checked}" aria-label="${esc(label)}" data-act="${actName}"${arg ? ` data-arg="${arg}"` : ""}><span></span><em>${checked ? "On" : "Off"}</em></button></div>`;
  }
  function selectRow(label, desc, id, action, value, opts) {
    return `<div class="setting-row"><div><b>${esc(label)}</b><p>${esc(desc)}</p></div><select id="${id}" data-act="${action}" aria-label="${esc(label)}">${opts.map((o) => `<option value="${esc(o)}"${o === value ? " selected" : ""}>${esc(o)}</option>`).join("")}</select></div>`;
  }
  function shell(content) {
    const nav = TABS.map((t) => `<button id="nav-${slug(t)}" class="nav ${S.page === t ? "on" : ""}" data-act="tab" data-arg="${esc(t)}" aria-label="${esc(t)}">${navIcon(t)}<span>${esc(t)}</span></button>`).join("");
    const body = !S.signedIn ? login() : `<div class="shell ${S.collapsed ? "collapsed" : ""} ${S.density}">
      <aside class="side"><div class="brand"><span class="logo">P</span><div><b>Plinth</b><small>${esc(workspace)}</small></div></div><nav>${nav}</nav><div class="side-foot">${icon("shield", 16)}<span>Audit ready</span></div></aside>
      <main class="main"><header class="top"><div><div class="crumb">Settings / ${esc(S.page === "Profile" ? "Profile" : S.page)}</div><h1>${esc(S.page)}</h1></div><div class="date">${TODAY}</div><div class="avatar" aria-label="${esc(user)}">${initials(user)}</div></header>${S.error ? `<div class="banner error">${icon("alert", 17)} ${esc(S.error)}</div>` : ""}${S.toast ? `<div class="toast">${icon("check", 16)} <b>${esc(S.toast)}</b><span>${esc(S.savedAt)}</span></div>` : ""}${content}</main>
    </div>`;
    return body + dialogs();
  }
  function navIcon(t) { return icon({ Profile: "user", Security: "shield", Notifications: "bell", "API keys": "key", Billing: "card" }[t], 17); }
  function profile() {
    return `<section class="card profile-card"><div class="big-avatar">${initials(user)}</div><div><h2>${esc(user)}</h2><p>${esc(email)}</p><span>${esc(workspace)} · Owner</span></div></section>
      <div class="profile-lower"><section class="card overview-card"><h2>Workspace overview</h2><div class="stat-grid"><div><b>18</b><span>members</span></div><div><b>4</b><span>teams</span></div><div><b>SSO off</b><span>optional login</span></div></div></section>
      <section class="card details-card"><div class="section-title"><h2>Personal details</h2><span>Read only</span></div><dl class="detail-list"><div><dt>Full name</dt><dd>${esc(user)}</dd></div><div><dt>Email</dt><dd>${esc(email)}</dd></div><div><dt>Role</dt><dd>Workspace owner</dd></div><div><dt>Time zone</dt><dd>Europe/Lisbon (UTC+1)</dd></div><div><dt>Language</dt><dd>English</dd></div></dl></section></div>`;
  }
  function security() {
    const s = S.draft.security;
    return `<section class="card"><div class="card-head"><div><h2>Account protection</h2><p>Changes apply to ${esc(workspace)} when saved.</p></div><button class="primary" id="save-security" data-act="save-security">Save changes</button></div>
      ${sw("two-factor", s.twoFactor, "Two-factor authentication", `Authenticator already registered on ${device}.`, "toggle-2fa")}
      ${selectRow("Session timeout", "Automatically lock inactive browser sessions.", "session-timeout", "security-timeout", s.timeout, TIMEOUTS)}
      ${sw("sec-sign-alerts", s.signAlerts, "Sign-in alerts", "Email admins when a new device signs in.", "toggle-security", "signAlerts")}
      ${sw("sec-require-sso", s.requireSso, "Require SSO", "Force members to use the verified identity provider.", "toggle-security", "requireSso")}
      <div class="setting-row danger"><div><b>Other active sessions</b><p>End all sessions except this browser.</p></div><button class="secondary" id="signout-other" data-act="signout-open">Sign out all other sessions</button></div></section>`;
  }
  function notifications() {
    const n = S.draft.notifications;
    return `<section class="card notify-card"><div class="card-head"><div><h2>Email notifications</h2><p>Choose what Plinth sends to ${esc(email)}.</p></div><button class="primary" id="save-notifications" data-act="save-notifications">Save changes</button></div>
      ${selectRow("Email digest", "Summary of important workspace activity.", "email-digest", "notify-digest", n.digest, FREQS)}
      ${sw("notify-mentions", n.mentions, "Mentions", "Notify when teammates mention you.", "toggle-notify", "mentions")}
      ${sw("notify-productnews", n.productNews, "Product updates", "Release notes and owner webinars only.", "toggle-notify", "productNews")}
      ${sw("notify-weeklyreport", n.weeklyReport, "Weekly report", "Monday metrics snapshot for owners.", "toggle-notify", "weeklyReport")}
      <div class="notify-preview"><b>Delivery preview</b><span>Critical alerts always arrive immediately.</span><span>Digest messages are batched before 09:00 local time.</span></div></section>`;
  }
  function apiKeys() {
    const rows = S.keys.map((k) => `<div class="key-row"><div><b>${esc(k.name)}</b><p><code>${esc(k.prefix)}</code> · Created ${esc(k.created)} · Last used ${esc(k.used)}</p></div><button class="danger-btn" id="delete-${slug(k.name)}" aria-label="Delete ${esc(k.name)}" data-act="delete-open" data-arg="${esc(k.name)}">Delete</button></div>`).join("");
    return `<section class="card"><div class="card-head"><div><h2>API keys</h2><p>Keys can access Plinth data for ${esc(workspace)}.</p></div><span class="pill">${S.keys.length} active</span></div>${rows || `<div class="empty">No API keys remain.</div>`}</section>`;
  }
  function billing() { return `<section class="card"><h2>Billing</h2><div class="setting-row"><div><b>Plan</b><p>Scale plan renews on October 15, 2026.</p></div><span class="pill">Scale</span></div><div class="setting-row"><div><b>Payment method</b><p>Card ending 1830, managed by finance.</p></div><span class="muted">Current</span></div></section>`; }
  function login() { return `<main class="login"><div class="login-card"><span class="logo">P</span><h1>Signed out</h1><p>Your Plinth session for ${esc(workspace)} has ended. Reopen the secure sign-in link to continue.</p><button class="primary" id="signed-out-button" disabled>Sign in unavailable</button></div></main>`; }
  function dialogs() {
    if (!S.dialog) return "";
    if (S.dialog.kind === "2fa") return `<div class="scrim" data-overlay><div class="dialog" role="dialog" aria-modal="true" aria-labelledby="d2fa"><span class="dialog-kicker">Security step</span><h2 id="d2fa">Turn on two-factor authentication?</h2><p>Plinth will require codes from the authenticator already registered on <b>${esc(device)}</b> for ${esc(user)}.</p><div class="actions"><button class="secondary" id="cancel-2fa" data-act="cancel-2fa">Cancel</button><button class="primary" id="turn-on-2fa" data-act="confirm-2fa">Turn on</button></div></div></div>`;
    if (S.dialog.kind === "signout") return `<div class="scrim" data-overlay><div class="dialog small" role="dialog" aria-modal="true"><h2>Sign out other sessions?</h2><p>This ends active Plinth sessions on other devices.</p><div class="actions"><button class="secondary" id="signout-cancel" data-act="signout-cancel">Cancel</button><button class="primary" id="signout-confirm" data-act="signout-confirm">Sign out sessions</button></div></div></div>`;
    return `<div class="scrim" data-overlay><div class="dialog" role="dialog" aria-modal="true" aria-labelledby="del-title"><span class="dialog-kicker danger-kicker">Permanent action</span><h2 id="del-title">Delete API key?</h2><p>Type <b>${esc(S.dialog.name)}</b> to permanently delete this key. Integrations using it will stop working.</p><label class="field-label" for="delete-name">Key name</label><input id="delete-name" data-bind="deleteName" value="${esc(S.fields.deleteName)}" aria-label="API key name" autocomplete="off">${S.dialog.err ? `<div class="form-err">${icon("alert", 14)} ${esc(S.dialog.err)}</div>` : ""}<div class="actions"><button class="secondary" id="delete-cancel" data-act="delete-cancel">Cancel</button><button class="danger-primary" id="delete-confirm" data-act="delete-confirm">Delete key</button></div></div></div>`;
  }
  function view() {
    let content = profile();
    if (S.page === "Security") content = security();
    else if (S.page === "Notifications") content = notifications();
    else if (S.page === "API keys") content = apiKeys();
    else if (S.page === "Billing") content = billing();
    return shell(content);
  }

  const interrupt = {
    kinds: ["whats-new", "cookies"],
    view(kind) {
      if (kind === "cookies") return `<div class="scrim soft" data-overlay><div class="notice" role="dialog" aria-modal="true" aria-label="Cookie preferences"><div>${icon("info", 19)}</div><p><b>Cookie preferences</b><span>Plinth uses essential cookies for session security and saved preferences.</span></p><button class="primary" id="a-dismiss" data-act="a-dismiss">OK</button></div></div>`;
      return `<div class="scrim soft" data-overlay><div class="whats" role="dialog" aria-modal="true" aria-label="What's new"><div class="spark">${icon("sparkle", 24)}</div><h2>What's new in Plinth</h2><p>Security center now shows authenticator devices, API key activity, and workspace policy drift in one place.</p><button class="primary" id="a-dismiss" data-act="a-dismiss">Dismiss</button></div></div>`;
    },
  };

  run({ id: "settings", optimal: [5, 3, 4], setup: () => ({ task: taskText, values, S, T }), view, act, bind, expected, check, state, modalOpen: () => false, interrupt });
})();
