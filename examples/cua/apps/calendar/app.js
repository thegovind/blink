// Quarterhour: seeded week calendar and room booking scenario.
(function () {
  "use strict";
  const { K, esc, icon, run, click, type, choose, finished, norm } = window.AKit;
  const P = K.params;

  const DAYS = [
    { key: "mon", short: "Mon", name: "Monday", label: "Monday, 5 Oct", date: "5" },
    { key: "tue", short: "Tue", name: "Tuesday", label: "Tuesday, 6 Oct", date: "6" },
    { key: "wed", short: "Wed", name: "Wednesday", label: "Wednesday, 7 Oct", date: "7" },
    { key: "thu", short: "Thu", name: "Thursday", label: "Thursday, 8 Oct", date: "8" },
    { key: "fri", short: "Fri", name: "Friday", label: "Friday, 9 Oct", date: "9" },
  ];
  const DAY_BY_KEY = Object.fromEntries(DAYS.map((d) => [d.key, d]));
  const DAY_BY_NAME = Object.fromEntries(DAYS.map((d) => [d.name, d]));
  const ROOMS = [
    { name: "Birch", color: "#4fc3a1", soft: "mint" },
    { name: "Cedar", color: "#58aee9", soft: "sky" },
    { name: "Juniper", color: "#a68be8", soft: "lilac" },
    { name: "Maple", color: "#f2a36d", soft: "peach" },
    { name: "Alder", color: "#d8b768", soft: "sand" },
  ];
  const PEOPLE = ["Mara Vale", "Ivo Chen", "Nell Arora", "Theo Park", "Sana Imani", "Owen Slate", "Lina Moss", "Bea Calder"];
  const TITLES = ["Roadmap sync", "Design review", "Customer council", "Launch retro", "Metrics clinic", "Ops standup", "Prototype share", "Hiring panel", "Risk review", "Studio critique"];
  const BOOK_TITLES = ["Partner kickoff", "Budget readout", "Sprint planning", "Research playback", "Content review", "Launch rehearsal"];
  const STARTS = ["08:00", "08:30", "09:00", "09:30", "10:00", "10:30", "11:00", "11:30", "12:00", "12:30", "13:00", "13:30", "14:00", "14:30", "15:00", "15:30", "16:00"];
  const GRID_TIMES = ["08:00", "09:00", "10:00", "11:00", "12:00", "13:00", "14:00", "15:00", "16:00"];
  const TASK_SLOTS = [
    { day: "thu", time: "14:00", room: "Maple", title: "Partner kickoff" },
    { day: "wed", time: "15:00", room: "Alder", title: "Sprint planning" },
    { day: "fri", time: "09:30", room: "Birch", title: "Research playback" },
    { day: "tue", time: "11:00", room: "Juniper", title: "Launch rehearsal" },
    { day: "thu", time: "08:30", room: "Cedar", title: "Content review" },
  ];
  const slug = (s) => String(s).toLowerCase().replace(/[^a-z0-9]+/g, "-").replace(/^-|-$/g, "");
  const room = (name) => ROOMS.find((r) => r.name === name) || ROOMS[0];
  const dayLabel = (key) => DAY_BY_KEY[key].label;
  const shortDayTime = (key, time) => `${DAY_BY_KEY[key].short} ${time}`;
  const mins = (t) => { const [h, m] = t.split(":").map(Number); return h * 60 + m; };
  const hm = (m) => String(Math.floor(m / 60)).padStart(2, "0") + ":" + String(m % 60).padStart(2, "0");
  const timeTop = (t) => 38 + ((mins(t) - 8 * 60) / 60) * 48;
  const eventHeight = (e) => Math.max(30, (e.dur || 60) / 60 * 48 - 5);

  const rl = K.stream("calendar-layout");
  const sideCardMode = K.pick(rl, ["up-next", "free-now"]);
  const taskSlot = TASK_SLOTS[(P.seed - 1) % TASK_SLOTS.length];
  const otherRoomFree = ROOMS.map((r) => r.name).find((r) => r !== taskSlot.room && !["Birch 08:00 mon"].includes(`${r} ${taskSlot.time} ${taskSlot.day}`)) || ROOMS.find((r) => r.name !== taskSlot.room).name;

  function seedEvents() {
    const base = [
      { id: "ev-design-thu", title: "Design review", day: "thu", start: "10:00", dur: 60, room: "Cedar", organiser: "Mara Vale", attendees: ["Ivo Chen", "Nell Arora"] },
      { id: "ev-design-tue", title: "Design review", day: "tue", start: "15:00", dur: 60, room: "Alder", organiser: "Sana Imani", attendees: ["Theo Park", "Bea Calder"] },
      { id: "ev-roadmap-mon", title: "Roadmap sync", day: "mon", start: "09:00", dur: 60, room: "Birch", organiser: "Ivo Chen", attendees: ["Mara Vale", "Owen Slate"] },
      { id: "ev-metrics-wed", title: "Metrics clinic", day: "wed", start: "11:00", dur: 60, room: "Juniper", organiser: "Lina Moss", attendees: ["Nell Arora", "Theo Park"] },
      { id: "ev-retro-fri", title: "Launch retro", day: "fri", start: "13:00", dur: 60, room: "Maple", organiser: "Owen Slate", attendees: ["Sana Imani", "Bea Calder"] },
      { id: "ev-ops-mon", title: "Ops standup", day: "mon", start: "14:30", dur: 60, room: "Alder", organiser: "Theo Park", attendees: ["Mara Vale"] },
      { id: "ev-council-wed", title: "Customer council", day: "wed", start: "09:30", dur: 60, room: "Maple", organiser: "Bea Calder", attendees: ["Ivo Chen", "Lina Moss"] },
      { id: "ev-proto-thu", title: "Prototype share", day: "thu", start: "15:30", dur: 60, room: "Birch", organiser: "Nell Arora", attendees: ["Sana Imani"] },
      { id: "ev-risk-fri", title: "Risk review", day: "fri", start: "10:30", dur: 60, room: "Juniper", organiser: "Mara Vale", attendees: ["Owen Slate"] },
      { id: "ev-hiring-tue", title: "Hiring panel", day: "tue", start: "09:00", dur: 60, room: "Cedar", organiser: "Lina Moss", attendees: ["Ivo Chen", "Theo Park"] },
    ];
    return K.shuffle(K.stream("calendar-events"), base).slice(0, 10).sort((a, b) => DAYS.findIndex((d) => d.key === a.day) - DAYS.findIndex((d) => d.key === b.day) || mins(a.start) - mins(b.start));
  }

  let T;
  let taskText;
  let values = [];
  if (P.task === 1) {
    T = { meeting: "Design review", day: "thu", eventId: "ev-design-thu" };
    taskText = `Cancel the ${T.meeting} on ${DAY_BY_KEY[T.day].name}.`;
  } else {
    const chosen = taskSlot;
    T = { title: chosen.title, day: chosen.day, dayLabel: dayLabel(chosen.day), time: chosen.time, room: chosen.room, wrongRoom: otherRoomFree };
    taskText = `Book the ${T.room} room on ${DAY_BY_KEY[T.day].name} at ${T.time} for "${T.title}".`;
    values = [T.title];
  }

  const defaultDay = DAYS.find((d) => d.key !== (T.day || "thu")).label;
  const defaultTime = STARTS.find((t) => t !== (T.time || "10:00"));
  const defaultRoom = P.task === 0 ? T.wrongRoom : "Birch";
  const S = {
    weekOffset: 0, loading: false, panel: null, selected: null, confirm: false, toast: "", conflict: false,
    fields: { title: "" }, form: { day: defaultDay, time: defaultTime, room: defaultRoom },
    events: seedEvents(), bookings: [], cancellations: [], cancelledIds: [],
  };

  function formDayKey() { const d = DAYS.find((x) => x.label === S.form.day); return d ? d.key : "mon"; }
  function eventById(id) { return S.events.find((e) => e.id === id) || S.bookings.find((e) => e.id === id); }
  function isBusy(day, time, rm, ignoreId) {
    const a = mins(time), b = a + 60;
    return S.events.concat(S.bookings).some((e) => e.id !== ignoreId && !S.cancelledIds.includes(e.id) && e.day === day && e.room === rm && a < mins(e.start) + (e.dur || 60) && b > mins(e.start));
  }
  function targetBooked() {
    return S.bookings.filter((e) => e.title === T.title && e.day === T.day && e.start === T.time && e.room === T.room).length === 1;
  }

  (function applyState() {
    const st = P.state;
    if (!st || st === "week") return;
    if (st === "loading") { S.loading = true; return; }
    if (st === "event_detail") { S.selected = P.task === 1 ? T.eventId : "ev-roadmap-mon"; return; }
    if (st === "cancel_confirm") { S.selected = P.task === 1 ? T.eventId : "ev-roadmap-mon"; S.confirm = true; return; }
    if (st === "new_booking" || st === "conflict") { S.panel = "new"; if (st === "conflict") { S.fields.title = "Studio critique"; S.form.day = "Monday, 5 Oct"; S.form.time = "09:00"; S.form.room = "Birch"; S.conflict = true; } return; }
    if (st === "booked") {
      const ev = { id: "bk-state", title: P.task === 0 ? T.title : "Partner kickoff", day: P.task === 0 ? T.day : "thu", start: P.task === 0 ? T.time : "14:00", dur: 60, room: P.task === 0 ? T.room : "Maple", organiser: "Mara Vale", attendees: ["Nell Arora"] };
      S.bookings.push(ev); S.toast = `Booked ${ev.room}, ${shortDayTime(ev.day, ev.start)}`; return;
    }
  })();

  function act(S, name, arg) {
    if (!["a-dismiss"].includes(name)) S.toast = S.toast && name === "today" ? S.toast : S.toast;
    switch (name) {
      case "prev": S.weekOffset -= 1; S.selected = null; S.confirm = false; S.panel = null; break;
      case "next": S.weekOffset += 1; S.selected = null; S.confirm = false; S.panel = null; break;
      case "today": S.weekOffset = 0; break;
      case "new": S.panel = "new"; S.selected = null; S.confirm = false; S.conflict = false; break;
      case "close-panel": S.panel = null; S.conflict = false; break;
      case "book-day": S.form.day = arg; S.conflict = false; break;
      case "book-time": S.form.time = arg; S.conflict = false; break;
      case "book-room": S.form.room = arg; S.conflict = false; break;
      case "book-submit": {
        const day = formDayKey();
        if (!S.fields.title.trim()) break;
        if (isBusy(day, S.form.time, S.form.room)) { S.conflict = true; break; }
        const ev = { id: "bk-" + (S.bookings.length + 1), title: S.fields.title.trim(), day, start: S.form.time, dur: 60, room: S.form.room, organiser: "Mara Vale", attendees: ["Nell Arora", "Theo Park"] };
        S.bookings.push(ev); S.panel = null; S.conflict = false; S.toast = `Booked ${ev.room}, ${shortDayTime(ev.day, ev.start)}`;
        break;
      }
      case "open-event": S.selected = arg; S.confirm = false; S.panel = null; break;
      case "close-detail": S.selected = null; S.confirm = false; break;
      case "cancel-open": S.confirm = true; break;
      case "cancel-keep": S.confirm = false; break;
      case "cancel-confirm": {
        const ev = eventById(S.selected);
        if (ev && !S.cancelledIds.includes(ev.id)) { S.cancelledIds.push(ev.id); S.cancellations.push({ id: ev.id, title: ev.title, day: ev.day }); S.toast = "Meeting cancelled"; }
        S.confirm = false; S.selected = null;
        break;
      }
      default: break;
    }
  }

  function bind(S, key, value) { S.fields[key] = value; S.conflict = false; }

  function expected() {
    // finish only when the goal itself holds; after a wrong commit keep pursuing the goal
    if (P.task === 0 ? targetBooked() : S.cancellations.some((c) => c.id === T.eventId)) return finished();
    if (S.weekOffset !== 0) return click("#today");
    if (P.task === 0) {
      if (S.confirm) return click("#cancel-keep");
      if (S.selected) return click("#close-detail");
      if (S.panel === "new") {
        if (norm(S.fields.title) !== norm(T.title)) return type("#book-title", T.title);
        if (S.form.day !== T.dayLabel) return choose("#book-day", T.dayLabel);
        if (S.form.time !== T.time) return choose("#book-time", T.time);
        if (S.form.room !== T.room) return choose("#book-room", T.room);
        if (S.conflict) return choose("#book-room", T.room);
        return click("#book-submit");
      }
      return click("#new-booking");
    }
    if (S.panel) return click("#close-panel");
    if (S.confirm) return S.selected === T.eventId ? click("#cancel-confirm", true) : click("#cancel-keep");
    if (S.selected) return S.selected === T.eventId ? click("#cancel-open") : click("#close-detail");
    return click("#event-" + T.eventId);
  }

  function check() {
    if (P.task === 0) {
      const ok = targetBooked() && S.bookings.length === 1 && S.cancellations.length === 0;
      if (ok) return { done: true, success: true, detail: "booked exact room and slot" };
      if (S.bookings.length || S.cancellations.length) return { done: true, success: false, detail: S.bookings.length ? "wrong booking made" : "meeting cancelled" };
      return { done: false, success: false, detail: "not booked" };
    }
    const target = S.cancellations.some((c) => c.id === T.eventId);
    if (target && S.cancellations.length === 1 && S.bookings.length === 0) return { done: true, success: true, detail: "target meeting cancelled" };
    if (S.cancellations.length || S.bookings.length) return { done: true, success: false, detail: S.cancellations.length ? "wrong meeting cancelled" : "booking made" };
    return { done: false, success: false, detail: "meeting still on calendar" };
  }

  function state() {
    if (S.loading) return "loading";
    if (S.panel === "new") return S.conflict ? "conflict" : "new_booking";
    if (S.confirm) return "cancel_confirm";
    if (S.selected) return "event_detail";
    if (S.toast && (targetBooked() || S.bookings.length)) return "booked";
    return "week";
  }
  function modalOpen() { return false; }

  const mark = `<svg class="brand-mark" viewBox="0 0 28 28" aria-hidden="true"><rect x="3" y="5" width="22" height="19" rx="6" fill="currentColor"/><path d="M8 11h12M8 16h7" stroke="var(--mark-line)" stroke-width="2" stroke-linecap="round"/><circle cx="20" cy="8" r="4" fill="var(--accent)"/></svg>`;
  function topbar() {
    const range = S.weekOffset === 0 ? "5–9 Oct 2026" : (S.weekOffset < 0 ? "28 Sep–2 Oct 2026" : "12–16 Oct 2026");
    return `<header class="topbar"><div class="brand">${mark}<span>Quarterhour</span></div><nav class="nav" aria-label="Week controls">
      <button id="today" class="quiet" data-act="today">Today</button>
      <button id="prev-week" class="iconbtn" data-act="prev" aria-label="Previous week">${icon("left", 16)}</button>
      <button id="next-week" class="iconbtn" data-act="next" aria-label="Next week">${icon("right", 16)}</button>
      <h1>${range}</h1></nav><button id="new-booking" class="primary" data-act="new">${icon("plus", 16)} New booking</button></header>`;
  }
  function sidebar() {
    const week = ["", "", "", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12", "13", "14", "15", "16", "17", "18"];
    const liveEvents = S.events.concat(S.bookings).filter((e) => !S.cancelledIds.includes(e.id));
    const upNext = liveEvents.filter((e) => e.day === "mon").sort((a, b) => mins(a.start) - mins(b.start))[0] || liveEvents[0];
    const busyNow = liveEvents.filter((e) => e.day === "mon" && mins(e.start) <= 8 * 60 && mins(e.start) + (e.dur || 60) > 8 * 60).map((e) => e.room);
    const freeRooms = ROOMS.map((r) => r.name).filter((name) => !busyNow.includes(name)).slice(0, 3).join(", ");
    const sideCard = sideCardMode === "free-now"
      ? `<section class="hint"><b>Free now</b><span>${esc(freeRooms)} open at 08:00.</span></section>`
      : `<section class="hint"><b>Up next</b><span>${esc(upNext.start)} · ${esc(upNext.title)} in ${esc(upNext.room)}.</span></section>`;
    return `<aside class="side"><section class="mini"><div class="mini-head"><b>October 2026</b><span>Week 41</span></div><div class="dow">M T W T F S S</div><div class="dates">${week.map((d) => `<span class="${["5","6","7","8","9"].includes(d) ? "on" : ""}">${d}</span>`).join("")}</div></section>
      <section class="rooms"><h2>Rooms</h2>${ROOMS.map((r) => `<div><i style="background:${r.color}"></i><span>${r.name}</span></div>`).join("")}</section>
      ${sideCard}</aside>`;
  }
  function skeleton() {
    const ghosts = [
      { col: 2, top: timeTop("09:00"), w: "70%" },
      { col: 4, top: timeTop("10:30"), w: "62%" },
      { col: 6, top: timeTop("13:00"), w: "66%" },
    ].map((g) => `<div class="event ghost" style="grid-column:${g.col}; top:${g.top}px; width:${g.w}"><b></b><span></span></div>`).join("");
    return `<main class="content"><div class="calendar loading-grid"><div class="time-head"></div>${DAYS.map((d) => `<div class="day-head"><b>${d.short}</b><span>${d.date}</span></div>`).join("")}${GRID_TIMES.map((t) => `<div class="time-label">${t}</div>${DAYS.map(() => `<div class="cell skel"></div>`).join("")}`).join("")}${ghosts}<div class="load-note">Loading room availability</div></div></main>`;
  }
  function calendar() {
    if (S.loading) return skeleton();
    const evs = S.events.concat(S.bookings).filter((e) => !S.cancelledIds.includes(e.id));
    const events = evs.map((e) => {
      const d = DAYS.findIndex((x) => x.key === e.day) + 2;
      const r = room(e.room);
      return `<button id="event-${e.id}" class="event ${r.soft}" style="grid-column:${d}; top:${timeTop(e.start)}px; height:${eventHeight(e)}px; --edge:${r.color}" data-act="open-event" data-arg="${esc(e.id)}" aria-label="${esc(e.title)}, ${DAY_BY_KEY[e.day].short} ${e.start}"><b>${esc(e.title)}</b><span>${esc(e.start)} · ${esc(e.room)}</span></button>`;
    }).join("");
    return `<main class="content"><div class="calendar"><div class="time-head"></div>${DAYS.map((d) => `<div class="day-head"><b>${d.short}</b><span>${d.date}</span></div>`).join("")}${GRID_TIMES.map((t) => `<div class="time-label">${t}</div>${DAYS.map(() => `<div class="cell"></div>`).join("")}`).join("")}${events}</div></main>`;
  }
  function detail() {
    const e = eventById(S.selected);
    if (!e) return "";
    return `<div class="scrim" data-overlay><section class="detail drawer small" role="dialog" aria-modal="true" aria-labelledby="detail-title"><button id="close-detail" class="close" data-act="close-detail" aria-label="Close event detail">${icon("close", 17)}</button><p class="eyebrow">Meeting detail</p><h2 id="detail-title">${esc(e.title)}</h2><div class="kv"><span>When</span><b>${DAY_BY_KEY[e.day].name}, ${e.start}–${hm(mins(e.start) + (e.dur || 60))}</b></div><div class="kv"><span>Room</span><b>${esc(e.room)}</b></div><div class="kv"><span>Organiser</span><b>${esc(e.organiser)}</b></div><div class="att"><span>Attendees</span><p>${e.attendees.map(esc).join(", ")}</p></div><button id="cancel-open" class="danger" data-act="cancel-open">Cancel meeting</button></section></div>`;
  }
  function confirm() {
    return `<div class="scrim" data-overlay><section class="confirm" role="dialog" aria-modal="true" aria-labelledby="confirm-title"><h2 id="confirm-title">Cancel this meeting?</h2><p>Attendees will be notified.</p><div class="actions"><button id="cancel-keep" class="secondary" data-act="cancel-keep">Keep meeting</button><button id="cancel-confirm" class="danger" data-act="cancel-confirm">Cancel meeting</button></div></section></div>`;
  }
  function bookingPanel() {
    return `<div class="scrim right" data-overlay><form class="drawer" data-submit="book-submit" aria-label="New room booking"><button id="close-panel" class="close" data-act="close-panel" aria-label="Close booking panel">${icon("close", 17)}</button><p class="eyebrow">Room booking</p><h2>New booking</h2><label>Title<input id="book-title" data-bind="title" value="${esc(S.fields.title)}" aria-label="Booking title" autocomplete="off"></label><label>Day<select id="book-day" data-act="book-day" aria-label="Booking day">${DAYS.map((d) => `<option value="${d.label}"${S.form.day === d.label ? " selected" : ""}>${d.label}</option>`).join("")}</select></label><label>Start<select id="book-time" data-act="book-time" aria-label="Booking start time">${STARTS.map((t) => `<option${S.form.time === t ? " selected" : ""}>${t}</option>`).join("")}</select></label><label>Room<select id="book-room" data-act="book-room" aria-label="Booking room">${ROOMS.map((r) => `<option${S.form.room === r.name ? " selected" : ""}>${r.name}</option>`).join("")}</select></label><div class="duration">${icon("clock", 15)} Duration fixed at 1 hour</div>${S.conflict ? `<div class="error" role="status">${icon("alert", 15)} ${esc(S.form.room)} is already booked at ${esc(S.form.time)}.</div>` : ""}<button id="book-submit" class="primary wide" data-act="book-submit">Book room</button></form></div>`;
  }
  function toast() { return S.toast ? `<div class="toast">${icon("check", 16)} ${esc(S.toast)}</div>` : ""; }
  function view() {
    let html = `<div class="shell">${topbar()}<div class="workspace">${sidebar()}${calendar()}</div>${toast()}</div>`;
    if (S.panel === "new") html += bookingPanel();
    if (S.selected && !S.confirm) html += detail();
    if (S.confirm) html += confirm();
    return html;
  }

  const interrupt = {
    kinds: ["phone", "hours"],
    view(kind) {
      if (kind === "phone") return `<div class="scrim soft" data-overlay><section class="notice card" role="dialog" aria-modal="true" aria-label="Connect calendar"><div class="notice-ic">${icon("calendar", 22)}</div><div><b>Connect your phone calendar</b><span>Quarterhour can compare mobile holds before you book a room.</span></div><button id="a-dismiss" class="primary" data-act="a-dismiss">Maybe later</button></section></div>`;
      return `<div class="scrim soft" data-overlay><section class="notice" role="dialog" aria-modal="true" aria-label="Working hours updated"><div class="notice-ic">${icon("clock", 22)}</div><div><b>Working hours updated</b><span>The week now shows the team’s 08:00–17:00 planning window.</span></div><button id="a-dismiss" class="primary" data-act="a-dismiss">Got it</button></section></div>`;
    },
  };

  run({ id: "calendar", optimal: [6, 3], setup: () => ({ task: taskText, values, S, T }), view, act, bind, expected, check, state, modalOpen, interrupt });
})();
