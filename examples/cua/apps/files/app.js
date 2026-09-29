// Skyvault: a fictional cloud file manager for the files CUA scenario.
(function () {
  "use strict";
  const { K, esc, icon, initials, norm, run, click, type, choose, finished } = window.AKit;
  const P = K.params;

  const PEOPLE = ["Mira Voss", "Theo Penn", "Lena Ort", "Nico Vale", "Iris Chen", "Owen Park"];
  const DOMAINS = ["northstar.example", "oakbridge.example", "bluekite.example", "verdant.example"];
  const TODAY = "2026-09-29";
  const NAV = [
    ["my", "My files", "folder"],
    ["shared", "Shared with me", "users"],
    ["recent", "Recent", "history"],
    ["trash", "Trash", "trash"],
  ];
  const FILE_SETS = [
    { target: "Q3 budget.xlsx", near: "Q3 budget (old).xlsx", folder: "Finance", nearFolder: "Finance archive", kind: "sheet" },
    { target: "Nimbus launch plan.pdf", near: "Nimbus launch plan draft.pdf", folder: "Launch", nearFolder: "Launch archive", kind: "pdf" },
    { target: "Client mural notes.docx", near: "Client mural notes copy.docx", folder: "Studio", nearFolder: "Studio archive", kind: "doc" },
    { target: "Roadshow slides.pptx", near: "Roadshow slides old.pptx", folder: "Events", nearFolder: "Events archive", kind: "slides" },
  ];
  const DELETE_SETS = [
    ["Vendor receipts.pdf", "Vendor receipts backup.pdf", "pdf"],
    ["Hiring notes.docx", "Hiring notes old.docx", "doc"],
    ["Office layout.png", "Office layout marked.png", "image"],
    ["Renewal tracker.xlsx", "Renewal tracker copy.xlsx", "sheet"],
  ];
  const SHARE_SETS = [
    ["Board packet.docx", "Board packet notes.docx", "doc"],
    ["Partner brief.pdf", "Partner brief draft.pdf", "pdf"],
    ["Design readout.pptx", "Design readout backup.pptx", "slides"],
    ["Usage summary.xlsx", "Usage summary old.xlsx", "sheet"],
  ];
  const EXTRA = [
    ["Product roadmap.docx", "doc"], ["Team photo.png", "image"], ["Metrics snapshot.xlsx", "sheet"],
    ["Brand refresh deck.pptx", "slides"], ["Contract scan.pdf", "pdf"], ["Research clips", "folder"]
  ];
  const OWNERS = ["Ari Lane", "Mika Noor", "Tessa Wynn", "Rory Lake", "Sage Collin", "Eli Stone"];
  const DATES = ["Today", "Yesterday", "Sep 27", "Sep 25", "Sep 20", "Sep 18", "Sep 12"];
  function sizeFor(kind, r) {
    if (kind === "doc") return `${K.between(r, 18, 240)} KB`;
    if (kind === "sheet") return `${K.between(r, 30, 400)} KB`;
    if (kind === "slides") return `${K.between(r, 1, 9)}.${K.between(r, 0, 9)} MB`;
    if (kind === "pdf") return `${K.between(r, 2, 40) / 10} MB`;
    if (kind === "image") return `${K.between(r, 8, 60) / 10} MB`;
    return "—";
  }
  function ownerEmail(name) { return `${slug(name).replace(/-/g, ".")}@skyvault.example`; }
  const slug = (s) => String(s).toLowerCase().replace(/[^a-z0-9]+/g, "-").replace(/^-|-$/g, "");

  const rt = K.stream("files-task-" + P.task);
  const rl = K.stream("files-layout");
  const collapsed = K.pick(rl, [false, true, false]);
  const sort = K.pick(rl, ["name", "modified"]);
  const density = K.pick(rl, ["cozy", "compact"]);
  const chosenMove = K.pick(rt, FILE_SETS);
  const chosenDelete = K.pick(rt, DELETE_SETS);
  const chosenShare = K.pick(rt, SHARE_SETS);
  const emailName = K.pick(rt, ["avery", "jamie", "marin", "luca", "selene", "tobin"]);
  const email = `${emailName}.${K.pick(rt, ["lin", "cole", "okoye", "berg", "ward", "reyes", "novak", "hale"])}@${K.pick(rt, DOMAINS)}`;

  let seq = 0;
  function file(name, kind, extra) {
    const r = K.stream("files-row-" + name);
    return Object.assign({
      id: "f" + (++seq), type: "file", name, kind,
      owner: K.pick(r, OWNERS), modified: K.pick(r, DATES), size: sizeFor(kind, r), loc: "projects", shares: []
    }, extra || {});
  }
  function folder(name, extra) {
    return Object.assign({ id: "d" + (++seq), type: "folder", name, kind: "folder", owner: "Me", modified: "Sep 24", size: "—", loc: "projects" }, extra || {});
  }

  const targetFolder = folder(chosenMove.folder);
  const nearFolder = folder(chosenMove.nearFolder);
  const moveFile = file(chosenMove.target, chosenMove.kind, { tag: "move-target" });
  const moveNear = file(chosenMove.near, chosenMove.kind, { tag: "move-near" });
  const deleteFile = file(chosenDelete[0], chosenDelete[2], { tag: "delete-target" });
  const deleteNear = file(chosenDelete[1], chosenDelete[2], { tag: "delete-near" });
  const shareFile = file(chosenShare[0], chosenShare[2], { tag: "share-target", shares: [{ name: "Mira Voss", email: "mira.voss@skyvault.example", role: "Editor" }] });
  const shareNear = file(chosenShare[1], chosenShare[2], { tag: "share-near" });
  const filler = K.shuffle(rt, EXTRA).slice(0, 3).map(([n, k]) => k === "folder" ? folder(n) : file(n, k));
  const ITEMS = [targetFolder, nearFolder, moveFile, moveNear, deleteFile, deleteNear, shareFile, shareNear].concat(filler);
  const byId = (id) => ITEMS.find((x) => x.id === id);

  let T;
  let taskText;
  let values = [];
  if (P.task === 1) {
    [moveFile, moveNear, shareFile, shareNear].forEach((x) => { x.loc = "hidden"; });
    T = { deleteFile: deleteFile.id, deleteDistractor: deleteNear.id };
    taskText = `Delete ${deleteFile.name}.`;
  } else if (P.task === 2) {
    [moveFile, moveNear, deleteFile, deleteNear].forEach((x) => { x.loc = "hidden"; });
    T = { shareFile: shareFile.id, shareDistractor: shareNear.id, email };
    taskText = `Share ${shareFile.name} with "${email}" as a viewer.`;
    values = [email];
  } else {
    [deleteFile, deleteNear, shareFile, shareNear].forEach((x) => { x.loc = "hidden"; });
    T = { moveFile: moveFile.id, moveDistractor: moveNear.id, folder: targetFolder.id, folderDistractor: nearFolder.id };
    taskText = `Move ${moveFile.name} into the ${targetFolder.name} folder.`;
  }

  const S = {
    nav: "my", folder: "projects", selected: null, dialog: null, fields: { email: "" }, role: "Editor",
    pickedFolder: null, toast: "", deleted: [], moves: [], shareLog: [], loading: false,
    collapsed, sort, density, today: TODAY,
  };

  function applyState() {
    const st = P.state;
    if (!st || st === "files") return;
    if (st === "loading") { S.loading = true; return; }
    const target = P.task === 1 ? T.deleteFile : (P.task === 2 ? T.shareFile : T.moveFile);
    if (st === "selected") { S.selected = target; return; }
    if (st === "move_dialog") { S.selected = P.task === 0 ? T.moveFile : target; S.dialog = "move"; S.pickedFolder = targetFolder.id; return; }
    if (st === "delete_confirm") { S.selected = P.task === 1 ? T.deleteFile : target; S.dialog = "delete"; return; }
    if (st === "share_dialog" || st === "share_error") {
      S.selected = P.task === 2 ? T.shareFile : target; S.dialog = "share"; S.role = st === "share_error" ? "Editor" : "Viewer";
      if (st === "share_error") { S.fields.email = "not-an-email"; S.shareError = "Enter a valid email address"; }
    }
  }
  applyState();

  function visibleRows() {
    if (S.loading) return [];
    let rows = ITEMS.filter((x) => !x.deleted && x.loc === S.folder && (S.folder === "projects" ? true : x.type === "file"));
    if (S.folder !== "projects") rows = rows.filter((x) => x.type === "file");
    rows = rows.slice().sort((a, b) => {
      if (S.sort === "modified") return DATES.indexOf(a.modified) - DATES.indexOf(b.modified) || a.name.localeCompare(b.name);
      if (a.type !== b.type) return a.type === "folder" ? -1 : 1;
      return a.name.localeCompare(b.name);
    });
    return rows.slice(0, 7);
  }
  function selectedItem() { return byId(S.selected); }
  const fileInFolder = (fid, did) => byId(fid) && byId(fid).loc === did;

  function act(S, name, arg) {
    S.toast = ""; S.shareError = "";
    switch (name) {
      case "nav": S.nav = arg; S.folder = "projects"; S.selected = null; S.dialog = null; break;
      case "new": S.toast = "New menu is ready"; break;
      case "open-folder": S.folder = arg; S.selected = null; break;
      case "crumb-projects": S.folder = "projects"; S.selected = null; break;
      case "select": S.selected = arg; break;
      case "share": if (selectedItem() && selectedItem().type === "file") { S.dialog = "share"; S.role = "Editor"; S.fields.email = ""; } break;
      case "move": if (selectedItem() && selectedItem().type === "file") { S.dialog = "move"; S.pickedFolder = null; } break;
      case "delete": if (selectedItem() && selectedItem().type === "file") S.dialog = "delete"; break;
      case "cancel": S.dialog = null; break;
      case "pick-folder": S.pickedFolder = arg; break;
      case "move-here": {
        const it = selectedItem();
        if (it && S.pickedFolder) {
          S.moves.push({ file: it.id, from: it.loc, to: S.pickedFolder });
          it.loc = S.pickedFolder; S.toast = `Moved to ${byId(S.pickedFolder).name}`; S.selected = null; S.dialog = null; S.pickedFolder = null;
        }
        break;
      }
      case "delete-confirm": {
        const it = selectedItem();
        if (it) { it.deleted = true; S.deleted.push(it.id); S.toast = `Deleted ${it.name}`; S.selected = null; S.dialog = null; }
        break;
      }
      case "role": S.role = arg; break;
      case "share-submit": {
        const it = selectedItem();
        const to = S.fields.email.trim();
        if (!/^[-a-z0-9_.]+@[-a-z0-9.]+\.[a-z]{2,}$/i.test(to)) { S.shareError = "Enter a valid email address"; break; }
        if (it) {
          const ev = { file: it.id, email: to, role: S.role };
          it.shares.push({ email: to, role: S.role }); S.shareLog.push(ev); S.toast = `Shared with ${to}`;
          S.selected = null; S.dialog = null;
        }
        break;
      }
      default: break;
    }
  }

  function expected() {
    if (P.task === 0) return expectedMove();
    if (P.task === 1) return expectedDelete();
    return expectedShare();
  }
  function closeWrongDialog(want) {
    if (!S.dialog) return null;
    if (S.dialog !== want) return click("#dlg-cancel");
    return null;
  }
  function ensureVisibleFile(fid) {
    const it = byId(fid);
    if (!it || it.deleted) return null;
    if (S.folder !== it.loc) {
      if (S.folder !== "projects") return click("#crumb-projects");
      const folderRow = byId(it.loc);
      if (folderRow) return click("#row-" + folderRow.id);
    }
    if (S.selected !== fid) return click("#row-" + fid);
    return null;
  }
  function expectedMove() {
    if (fileInFolder(T.moveFile, T.folder) && !byId(T.moveFile).deleted) return finished();
    const wrong = closeWrongDialog("move"); if (wrong) return wrong;
    if (S.dialog === "move") {
      if (S.selected !== T.moveFile) return click("#dlg-cancel");
      if (S.pickedFolder !== T.folder) return click("#pick-" + T.folder);
      return click("#move-here");
    }
    const vis = ensureVisibleFile(T.moveFile); if (vis) return vis;
    return click("#move");
  }
  function expectedDelete() {
    if (S.deleted.includes(T.deleteFile)) return finished();
    const wrong = closeWrongDialog("delete"); if (wrong) return wrong;
    if (S.dialog === "delete") return S.selected === T.deleteFile ? click("#delete-confirm", true) : click("#dlg-cancel");
    if (S.folder !== "projects") return click("#crumb-projects");
    if (S.selected !== T.deleteFile) return click("#row-" + T.deleteFile);
    return click("#delete");
  }
  function expectedShare() {
    if (S.shareLog.some((e) => e.file === T.shareFile && e.email === T.email && e.role === "Viewer")) return finished();
    const wrong = closeWrongDialog("share"); if (wrong) return wrong;
    if (S.dialog === "share") {
      if (S.selected !== T.shareFile) return click("#dlg-cancel");
      if (S.fields.email.trim() !== T.email) return type("#share-email", T.email);
      if (S.role !== "Viewer") return choose("#share-role", "Viewer");
      return click("#share-submit", true);
    }
    if (S.folder !== "projects") return click("#crumb-projects");
    if (S.selected !== T.shareFile) return click("#row-" + T.shareFile);
    return click("#share");
  }

  function check() {
    try {
      if (P.task === 0) {
        const forbidden = ITEMS.some((x) => x.id !== T.moveFile && x.loc !== "projects" && x.loc !== "hidden" && x.type === "file") || !!(byId(T.moveFile) && byId(T.moveFile).deleted);
        const ok = fileInFolder(T.moveFile, T.folder) && !forbidden;
        return { done: ok || forbidden, success: ok, detail: ok ? `moved to ${byId(T.folder).name}` : (forbidden ? "wrong file moved or deleted" : "not moved yet") };
      }
      if (P.task === 1) {
        const wrong = S.deleted.some((id) => id !== T.deleteFile);
        const ok = S.deleted.length === 1 && S.deleted[0] === T.deleteFile;
        return { done: ok || wrong, success: ok, detail: ok ? `deleted ${byId(T.deleteFile).name}` : (wrong ? "wrong file deleted" : "not deleted yet") };
      }
      const wrongShare = S.shareLog.some((e) => e.file !== T.shareFile || e.email !== T.email || e.role !== "Viewer");
      const ok = S.shareLog.length === 1 && !wrongShare;
      return { done: ok || wrongShare, success: ok, detail: ok ? `shared with ${T.email}` : (wrongShare ? "wrong share created" : "not shared yet") };
    } catch (e) { return { done: false, success: false, detail: String(e) }; }
  }

  function state() {
    if (S.loading) return "loading";
    if (S.dialog === "move") return "move_dialog";
    if (S.dialog === "delete") return "delete_confirm";
    if (S.dialog === "share") return S.shareError ? "share_error" : "share_dialog";
    return S.selected ? "selected" : "files";
  }

  function folderIcon(cls) { return `<svg class="${cls}" viewBox="0 0 42 34" aria-hidden="true"><path class="folder-tab" d="M3 9.5c0-2.2 1.8-4 4-4h9.2l3.7 4H35c2.2 0 4 1.8 4 4v1.4H3z"/><path class="folder-body" d="M3 13h36v12.5c0 2.5-2 4.5-4.5 4.5h-27C5 30 3 28 3 25.5z"/></svg>`; }
  function glyph(kind) {
    const map = { doc: ["DOC", "blue"], sheet: ["XLS", "green"], slides: ["PPT", "amber"], pdf: ["PDF", "red"], image: ["IMG", "violet"], folder: ["", "folder"] };
    const [txt, cls] = map[kind] || map.doc;
    if (kind === "folder") return `<span class="tile foldertile">${folderIcon("folder-svg")}</span>`;
    return `<span class="tile ${cls}"><svg viewBox="0 0 42 48" aria-hidden="true"><path class="page" d="M9 3.5h18.5L35 11v29.5c0 2.2-1.8 4-4 4H9c-2.2 0-4-1.8-4-4v-33c0-2.2 1.8-4 4-4z"/><path class="fold" d="M27 4v7c0 1.2 1 2.2 2.2 2.2H35"/><path class="line" d="M12 20h17M12 25h12"/></svg><b>${txt}</b></span>`;
  }
  const logo = `<svg class="logo" viewBox="0 0 34 34" aria-hidden="true"><defs><linearGradient id="g" x1="4" y1="4" x2="30" y2="30"><stop stop-color="#38bdf8"/><stop offset="1" stop-color="#2563eb"/></linearGradient></defs><path fill="url(#g)" d="M17 3l11 6.4v12.8L17 29 6 22.2V9.4z"/><path fill="#fff" opacity=".9" d="M11 13.2l6-3.5 6 3.5-6 3.5zM11 17.5l6 3.5 6-3.5v3.1l-6 3.6-6-3.6z"/></svg>`;

  function side() {
    return `<aside class="side ${S.collapsed ? "collapsed" : ""}">
      <div class="brand">${logo}${S.collapsed ? "" : "<span>Skyvault</span>"}</div>
      <nav>${NAV.map(([id, label, ic]) => `<button id="nav-${id}" class="nav ${S.nav === id ? "on" : ""}" data-act="nav" data-arg="${id}" aria-label="${label}">${icon(ic, 18)}${S.collapsed ? "" : `<span>${label}</span>`}</button>`).join("")}</nav>
      <div class="meter" aria-hidden="true"><div><b>${S.collapsed ? "68%" : "68.2 GB"}</b>${S.collapsed ? "" : " of 100 GB"}</div><i><em style="width:68%"></em></i>${S.collapsed ? "" : "<small>Storage used</small>"}</div>
    </aside>`;
  }
  function toolbar() {
    const it = selectedItem();
    if (it && it.type === "file") return `<div class="selbar"><b>1 selected</b><button id="share" data-act="share">${icon("share", 16)} Share</button><button id="move" data-act="move">${icon("move", 16)} Move</button><button id="delete" data-act="delete">${icon("trash", 16)} Delete</button></div>`;
    return `<div class="tools"><label class="search"><span>${icon("search", 16)}</span><input id="search" aria-label="Search files" placeholder="Search files" readonly></label><button id="new" data-act="new" class="new">${icon("plus", 17)} New</button></div>`;
  }
  function crumbs() {
    const here = S.folder === "projects" ? "Projects" : byId(S.folder).name;
    return `<div class="crumbs"><span>My files</span><span>/</span>${S.folder === "projects" ? `<b>Projects</b>` : `<button id="crumb-projects" data-act="crumb-projects">Projects</button><span>/</span><b>${esc(here)}</b>`}</div>`;
  }
  function row(x) {
    const selected = S.selected === x.id;
    if (x.type === "folder") {
      return `<button class="row folder ${selected ? "selected" : ""}" id="row-${x.id}" data-act="open-folder" data-arg="${x.id}" aria-label="${esc(x.name)}"><span class="namecell">${glyph("folder")}<span><b>${esc(x.name)}</b><small>${x.name.includes("archive") ? "Archived files" : "Folder"}</small></span></span><span>${esc(x.owner)}</span><span>${esc(x.modified)}</span><span>${esc(x.size)}</span></button>`;
    }
    return `<button class="row ${selected ? "selected" : ""}" id="row-${x.id}" data-act="select" data-arg="${x.id}" aria-label="${esc(x.name)}" aria-pressed="${selected}"><span class="namecell">${glyph(x.kind)}<span><b>${esc(x.name)}</b><small>${kindLabel(x.kind)}</small></span></span><span>${esc(x.owner)}</span><span>${esc(x.modified)}</span><span>${esc(x.size)}</span></button>`;
  }
  function kindLabel(k) { return ({ doc: "Document", sheet: "Spreadsheet", slides: "Presentation", pdf: "PDF", image: "Image" })[k] || "File"; }
  function list() {
    if (S.loading) return `<div class="skeleton">${Array.from({ length: 6 }, (_, i) => `<div class="sk"><i></i><b></b><span></span><em></em></div>`).join("")}</div>`;
    const rows = visibleRows().map(row).join("");
    return `<div class="list"><div class="headrow"><span>Name</span><span>Owner</span><span>Modified</span><span>Size</span></div>${rows || `<div class="empty">${icon("folder", 34)}<b>This folder is empty</b><span>Moved files will appear here.</span></div>`}</div>`;
  }
  function main() {
    const folderName = S.folder === "projects" ? "Projects" : byId(S.folder).name;
    const meta = S.loading ? "Syncing changes…" : `${visibleRows().length} items · Sorted by ${S.sort === "name" ? "Name" : "Modified"}`;
    return `<section class="main"><header class="top"><div class="title">${crumbs()}<h1>${esc(folderName)}</h1></div>${toolbar()}</header><div class="context"><span>${meta}</span></div>${list()}${S.toast ? `<div class="toast">${icon("check", 16)} ${esc(S.toast)}</div>` : ""}</section>`;
  }

  function moveDialog() {
    const it = selectedItem();
    const opts = [targetFolder, nearFolder].map((f) => `<button role="option" id="pick-${f.id}" class="folderpick ${S.pickedFolder === f.id ? "on" : ""}" aria-selected="${S.pickedFolder === f.id}" data-act="pick-folder" data-arg="${f.id}">${folderIcon("pick-folder-icon")}<span>${esc(f.name)}</span></button>`).join("");
    return `<div class="scrim" data-overlay><div class="dialog" role="dialog" aria-modal="true" aria-labelledby="move-t"><h2 id="move-t">Move ${esc(it ? it.name : "file")}</h2><p>Choose a destination folder in My files.</p><div class="picklist">${opts}</div><div class="actions"><button id="dlg-cancel" class="ghost" data-act="cancel">Cancel</button><button id="move-here" class="primary" data-act="move-here" ${S.pickedFolder ? "" : "disabled"}>Move here</button></div></div></div>`;
  }
  function deleteDialog() {
    const it = selectedItem();
    return `<div class="scrim" data-overlay><div class="dialog small" role="dialog" aria-modal="true" aria-labelledby="del-t"><h2 id="del-t">Delete ${esc(it ? it.name : "file")}?</h2><p>This moves the file to Trash. It can be restored by an admin for 30 days.</p><div class="actions"><button id="dlg-cancel" class="ghost" data-act="cancel">Cancel</button><button id="delete-confirm" class="danger" data-act="delete-confirm">Delete</button></div></div></div>`;
  }
  function shareDialog() {
    const it = selectedItem();
    const owner = it ? it.owner : "Ari Lane";
    const access = [{ name: owner, email: ownerEmail(owner), role: "Owner" }].concat((it ? it.shares : []).map((p) => ({ name: p.name || p.email.split("@")[0].replace(/[._-]/g, " "), email: p.email, role: p.role })));
    const people = access.map((p) => `<div class="person"><span>${initials(p.name)}</span><b>${esc(p.name)}<small>${esc(p.email)}</small></b><em>${esc(p.role)}</em></div>`).join("");
    return `<div class="scrim" data-overlay><div class="dialog sharebox" role="dialog" aria-modal="true" aria-labelledby="share-t"><h2 id="share-t">Share ${esc(it ? it.name : "document")}</h2><p class="subhead">People with access</p><div class="people">${people}</div><div class="sharegrid"><label class="field">Email<input id="share-email" data-bind="email" value="${esc(S.fields.email)}" aria-label="Email address" autocomplete="off"></label><label class="field rolefield">Role<select id="share-role" data-act="role" aria-label="Role"><option${S.role === "Editor" ? " selected" : ""}>Editor</option><option${S.role === "Viewer" ? " selected" : ""}>Viewer</option></select></label></div>${S.shareError ? `<div class="error" role="alert">${esc(S.shareError)}</div>` : ""}<div class="actions"><button id="dlg-cancel" class="ghost" data-act="cancel">Cancel</button><button id="share-submit" class="primary" data-act="share-submit">Share</button></div></div></div>`;
  }

  function view() {
    let html = `<div class="shell ${S.density}">${side()}${main()}</div>`;
    if (S.dialog === "move") html += moveDialog();
    if (S.dialog === "delete") html += deleteDialog();
    if (S.dialog === "share") html += shareDialog();
    return html;
  }

  const interrupt = {
    kinds: ["storage", "sync"],
    view(kind) {
      if (kind === "sync") return `<div class="scrim soft" data-overlay><div class="nag sync" role="dialog" aria-modal="true" aria-label="Install desktop sync"><div class="nagicon">${icon("download", 24)}</div><div><b>Install desktop sync</b><span>Keep Skyvault folders available from Finder and Explorer.</span></div><button id="a-dismiss" data-act="a-dismiss">Maybe later</button></div></div>`;
      return `<div class="scrim soft" data-overlay><div class="nag" role="dialog" aria-modal="true" aria-label="Storage almost full"><div class="nagicon">${icon("cloud", 24)}</div><div><b>Storage almost full</b><span>You have used 68.2 GB of 100 GB. Upgrade for more shared space.</span></div><button id="a-dismiss" data-act="a-dismiss">Not now</button></div></div>`;
    },
  };

  run({
    id: "files", optimal: [4, 3, 5],
    setup: () => ({ task: taskText, values, S, T }), view, act, expected, check, state,
    bind: (S, key, value) => { S.fields[key] = value; if (key === "email") S.shareError = ""; },
    modalOpen: () => false,
    interrupt,
  });
})();
