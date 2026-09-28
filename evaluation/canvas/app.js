// Paint the entire canvas teller interface as a branch workstation. The page
// contains no text, form, or button that a reader can find.
'use strict';

const canvas = document.getElementById('screen');
const paint = canvas.getContext('2d');

const FACE = '#d4d0c8';
const LIGHT = '#ffffff';
const SHADE = '#808080';
const DARK = '#404040';
const INK = '#000000';
const MUTED = '#4a4a4a';
const ALERT = '#a00000';
const ALERT_WASH = '#fbe4e1';
const NAVY = '#0a246a';
const SKY = '#a6caf0';
const PAPER = '#ffffff';
const STRIPE = '#f3f1ec';
const MARK = '#fff1a8';
const HOVER = '#e3ebf7';

const FAMILY = 'Tahoma, Verdana, Geneva, sans-serif';
const UI = `15px ${FAMILY}`;
const UI_BOLD = `bold 15px ${FAMILY}`;
const HEADING = `bold 21px ${FAMILY}`;
const TITLE = `bold 14px ${FAMILY}`;

const TITLE_HEIGHT = 30;
const TOOLBAR_HEIGHT = 46;
const STATUS_HEIGHT = 28;
const MESSAGE_HEIGHT = 34;
const LEFT = 24;
const TOP = TITLE_HEIGHT + TOOLBAR_HEIGHT + 34;
const ROW = 28;

const state = {
  screen: 'loading',
  institution: '',
  legalName: '',
  timeZone: 'UTC',
  businessDate: '',
  operator: null,
  products: [],
  fields: { operator: '', member: '', nickname: '', query: '' },
  focus: '',
  product: '',
  delivery: '',
  message: '',
  busy: false,
  member: null,
  memberPage: 0,
  review: null,
  receipt: null,
  account: null,
  accountFrom: 'member',
  register: null,
  opened: '',
  openedAt: null,
  hover: '',
  row: -1,
  caps: false,
};
let regions = [];
let listing = null;

async function call(method, path, payload) {
  state.busy = true;
  canvas.style.cursor = 'wait';
  draw();
  try {
    const response = await fetch(path, {
      method,
      headers: { 'content-type': 'application/json' },
      body: payload === undefined ? undefined : JSON.stringify(payload),
    });
    let body = {};
    try { body = await response.json(); } catch { body = { message: 'The application answered with an error.' }; }
    if (response.status === 401) {
      state.operator = null;
      go('signon', body.message);
    }
    return { status: response.status, body };
  } catch {
    return { status: 0, body: { message: 'The teller server did not answer. Try again.' } };
  } finally {
    state.busy = false;
    canvas.style.cursor = 'default';
    draw();
  }
}

const FOCUS = { signon: 'operator', home: 'member', open: 'nickname', register: 'query' };

function go(screen, message = '') {
  state.screen = screen;
  state.message = message;
  state.focus = FOCUS[screen] ?? '';
  state.row = -1;
  state.hover = '';
  draw();
}

async function load() {
  const { body } = await call('GET', '/api/session');
  state.institution = body.institution ?? '';
  state.legalName = body.legalName ?? '';
  state.timeZone = body.timeZone ?? 'UTC';
  state.businessDate = body.businessDate ?? '';
  state.operator = body.operator;
  state.products = body.products ?? [];
  go(state.operator ? 'home' : 'signon');
}

const actions = {
  async signon() {
    const { status, body } = await call('POST', '/api/signon', { operator: state.fields.operator });
    if (status !== 200) return go('signon', body.message);
    state.operator = body;
    state.fields.member = '';
    go('home');
  },
  signout() {
    document.cookie = 'sid=; Max-Age=0; Path=/';
    Object.assign(state, { operator: null, member: null, account: null, register: null, opened: '' });
    state.fields.operator = '';
    go('signon');
  },
  async find() {
    const { status, body } = await call('POST', '/api/member', { number: state.fields.member });
    if (status === 401) return undefined;
    if (status !== 200) return go('home', body.message);
    if (state.member?.number !== body.number) state.memberPage = 0;
    state.member = body;
    go('member');
  },
  async viewMember() {
    const number = state.screen === 'account' ? state.account.member : state.receipt.member;
    state.fields.member = number;
    await actions.find();
  },
  open() {
    state.fields.nickname = '';
    state.product = '';
    state.delivery = '';
    go('open');
  },
  search() { state.fields.member = ''; go('home'); },
  async review() {
    const { status, body } = await call('POST', '/api/open/review', form());
    if (status === 401) return undefined;
    if (status !== 200) return go('open', refusal(body));
    state.review = body;
    go('review');
  },
  async commit() {
    const { status, body } = await call('POST', '/api/open/commit', { ...form(), key: state.review.key });
    if (status === 401) return undefined;
    if (status === 0) {
      return go('review', 'The teller server did not answer, so the account may already be open. Look up the member before you commit again.');
    }
    if (status !== 200) return go('open', refusal(body));
    state.receipt = body;
    state.opened = body.account;
    state.openedAt = new Date();
    state.memberPage = 0;
    go('receipt');
  },
  back() { go(state.screen === 'review' ? 'open' : 'member'); },
  async account(number, from) {
    const { status, body } = await call('GET', `/api/account?number=${encodeURIComponent(number)}`);
    if (status === 401) return undefined;
    if (status !== 200) return go(state.screen, body.message);
    state.account = body;
    state.accountFrom = from;
    go('account');
  },
  leaveAccount() { go(state.accountFrom); },
  async register(page = 0) {
    const query = encodeURIComponent(state.fields.query.trim());
    const { status, body } = await call('GET', `/api/accounts?query=${query}&page=${page}`);
    if (status === 401) return undefined;
    if (status !== 200) return go('register', body.message);
    state.register = body;
    go('register');
  },
  registerSearch() { return actions.register(0); },
  browse() { state.fields.query = ''; return actions.register(0); },
};

function act(run) {
  return () => { if (!state.busy) run(); };
}

function form() {
  return {
    member: state.member ? state.member.number : '',
    product: state.product,
    nickname: state.fields.nickname,
    delivery: state.delivery,
  };
}

function refusal(body) {
  const lead = {
    not_authorized: 'You are not allowed to do this',
    ineligible: 'This member cannot open an account',
    validation: 'Check the form',
    not_found: 'Not found',
  }[body.code] ?? 'Not completed';
  return `${lead}: ${body.message}`;
}

function money(minor, exponent, currency) {
  const amount = (minor / 10 ** exponent).toLocaleString('en-US', {
    minimumFractionDigits: exponent,
    maximumFractionDigits: exponent,
  });
  return `${amount} ${currency}`;
}

function masked(email) {
  if (!email) return 'none on file';
  const [name, domain] = email.split('@');
  return `${name.slice(0, 1)}•••@${domain ?? ''}`;
}

function phone(value) {
  if (!value) return 'none on file';
  return `•••-•••-${value.replace(/\D/g, '').slice(-4)}`;
}

function clock(at = new Date()) {
  return at.toLocaleTimeString('en-US', { timeZone: state.timeZone, hour: 'numeric', minute: '2-digit', timeZoneName: 'short' });
}

function day(iso) {
  const [year, month, date] = String(iso ?? '').slice(0, 10).split('-');
  return year && month && date ? `${month}/${date}/${year}` : '';
}

// Painting.

function text(value, x, y, { font = UI, color = INK, align = 'left' } = {}) {
  paint.font = font;
  paint.fillStyle = color;
  paint.textAlign = align;
  paint.fillText(value, x, y);
  paint.textAlign = 'left';
}

function measure(value, font = UI) {
  paint.font = font;
  return paint.measureText(value).width;
}

function fit(value, font, room) {
  if (measure(value, font) <= room) return value;
  let shown = value;
  while (shown.length > 1 && measure(`${shown}…`, font) > room) shown = shown.slice(0, -1);
  return `${shown}…`;
}

function edge(x, y, w, h, topLeft, bottomRight) {
  paint.fillStyle = topLeft;
  paint.fillRect(x, y, w - 1, 1);
  paint.fillRect(x, y, 1, h - 1);
  paint.fillStyle = bottomRight;
  paint.fillRect(x, y + h - 1, w, 1);
  paint.fillRect(x + w - 1, y, 1, h);
}

function raised(x, y, w, h) {
  edge(x, y, w, h, LIGHT, DARK);
  edge(x + 1, y + 1, w - 2, h - 2, FACE, SHADE);
}

function sunken(x, y, w, h) {
  edge(x, y, w, h, SHADE, LIGHT);
  edge(x + 1, y + 1, w - 2, h - 2, DARK, FACE);
}

function button(label, x, y, run, { primary = false, pressed = false, height = 32 } = {}) {
  const w = Math.max(112, measure(label) + 36);
  paint.fillStyle = FACE;
  paint.fillRect(x, y, w, height);
  if (primary) {
    paint.fillStyle = INK;
    paint.fillRect(x - 1, y - 1, w + 2, height + 2);
    paint.fillStyle = FACE;
    paint.fillRect(x, y, w, height);
  }
  if (pressed) sunken(x, y, w, height);
  else raised(x, y, w, height);
  const shift = pressed ? 1 : 0;
  text(label, x + w / 2 + shift, y + height / 2 + 5 + shift, { align: 'center', color: state.busy ? SHADE : INK });
  regions.push({ x, y, width: w, height, cursor: 'pointer', action: act(run) });
  return w;
}

function field(name, label, x, y, w = 340) {
  text(label, x, y);
  const top = y + 8;
  const h = 32;
  const focused = state.focus === name;
  paint.fillStyle = PAPER;
  paint.fillRect(x, top, w, h);
  sunken(x, top, w, h);
  if (focused) {
    paint.strokeStyle = NAVY;
    paint.lineWidth = 1;
    paint.strokeRect(x - 2.5, top - 2.5, w + 5, h + 5);
  }
  const value = fit(state.fields[name], UI, w - 20);
  text(value, x + 8, top + 21);
  if (focused) {
    paint.fillStyle = INK;
    paint.fillRect(x + 9 + measure(value), top + 7, 1.5, 19);
  }
  regions.push({ x, y: top, width: w, height: h, cursor: 'text', action: () => { state.focus = name; draw(); } });
  return top + h;
}

function option(label, x, y, chosen, pick) {
  const cx = x + 8;
  const cy = y - 5;
  paint.fillStyle = PAPER;
  paint.beginPath();
  paint.arc(cx, cy, 7, 0, Math.PI * 2);
  paint.fill();
  paint.strokeStyle = SHADE;
  paint.lineWidth = 1.5;
  paint.stroke();
  if (chosen) {
    paint.fillStyle = INK;
    paint.beginPath();
    paint.arc(cx, cy, 3.5, 0, Math.PI * 2);
    paint.fill();
  }
  text(label, x + 24, y);
  const width = 34 + measure(label);
  regions.push({ x, y: y - 18, width, height: 26, cursor: 'pointer', action: () => { pick(); draw(); } });
}

function group(title, x, y, w, h) {
  paint.strokeStyle = SHADE;
  paint.lineWidth = 1;
  paint.strokeRect(x + 0.5, y + 0.5, w - 1, h - 1);
  paint.strokeStyle = LIGHT;
  paint.strokeRect(x + 1.5, y + 1.5, w - 1, h - 1);
  const room = measure(title, UI_BOLD) + 12;
  paint.fillStyle = FACE;
  paint.fillRect(x + 10, y - 9, room, 18);
  text(title, x + 16, y + 5, { font: UI_BOLD });
}

function lines(entries, x, y, spacing = ROW) {
  entries.forEach((entry, index) => text(entry, x, y + index * spacing));
  return y + (entries.length - 1) * spacing;
}

function grid(x, y, w, columns, rows, { open, marked, empty }) {
  if (open) listing = { rows, open };
  const head = 28;
  const h = head + Math.max(1, rows.length) * ROW + 4;
  paint.fillStyle = PAPER;
  paint.fillRect(x, y, w, h);
  sunken(x, y, w, h);
  const inner = w - 4;
  const total = columns.reduce((sum, column) => sum + column.share, 0);
  const widths = columns.map((column) => Math.floor((column.share / total) * inner));
  let left = x + 2;
  columns.forEach((column, index) => {
    paint.fillStyle = FACE;
    paint.fillRect(left, y + 2, widths[index], head);
    raised(left, y + 2, widths[index], head);
    const title = fit(column.title, UI, widths[index] - 16);
    if (column.align === 'right') text(title, left + widths[index] - 8, y + 21, { align: 'right' });
    else text(title, left + 8, y + 21);
    left += widths[index];
  });
  rows.forEach((row, index) => {
    const top = y + 2 + head + index * ROW;
    const selected = open && state.row === index;
    const hovered = open && state.hover === `row:${index}`;
    paint.fillStyle = selected ? NAVY : hovered ? HOVER : marked?.(row) ? MARK : index % 2 ? STRIPE : PAPER;
    paint.fillRect(x + 2, top, inner, ROW);
    left = x + 2;
    columns.forEach((column, at) => {
      const shown = fit(String(column.value(row)), UI, widths[at] - 16);
      const color = selected ? LIGHT : INK;
      if (column.align === 'right') text(shown, left + widths[at] - 8, top + 19, { align: 'right', color });
      else text(shown, left + 8, top + 19, { color });
      left += widths[at];
    });
    if (open) regions.push({ x: x + 2, y: top, width: inner, height: ROW, cursor: 'pointer', hover: `row:${index}`, action: act(() => open(row)) });
  });
  if (!rows.length) text(empty, x + 10, y + 2 + head + 19, { color: MUTED });
  return y + h;
}

function pager(x, y, page, pages, turn, { size, total }) {
  const first = total === 0 ? 0 : page * size + 1;
  const last = Math.min(total, (page + 1) * size);
  const shown = `Showing ${first} to ${last} of ${total}`;
  let at = x;
  if (pages > 1 && page > 0) at += button('Previous page', at, y, () => turn(page - 1)) + 12;
  if (pages > 1) {
    text(`Page ${page + 1} of ${pages}`, at, y + 21, { color: MUTED });
    at += measure(`Page ${page + 1} of ${pages}`) + 16;
  }
  if (pages > 1 && page < pages - 1) at += button('Next page', at, y, () => turn(page + 1)) + 16;
  text(shown, at, y + 21, { color: MUTED });
}

function heading(value, note = '') {
  text(value, LEFT, TOP, { font: HEADING });
  if (note) text(note, LEFT, TOP + 26, { color: MUTED });
}

function ledgerColumns(withMember) {
  const columns = [
    { title: 'Opened', share: 1.1, value: (row) => (row.number === state.opened ? 'Just opened' : day(row.opened)) },
    { title: 'Account number', share: 1.3, value: (row) => row.number },
  ];
  if (withMember) columns.push({ title: 'Member', share: 1.1, value: (row) => row.member });
  columns.push(
    { title: 'Product', share: 1.3, value: (row) => row.product },
    { title: 'Nickname', share: 1.5, value: (row) => row.nickname },
    { title: 'Statements', share: 1, value: (row) => row.delivery },
    { title: 'Status', share: 0.9, value: (row) => row.status },
    { title: 'Balance', share: 1.5, align: 'right', value: (row) => money(row.posted, row.exponent, row.currency) },
  );
  return columns;
}

function chrome(W, H) {
  paint.fillStyle = FACE;
  paint.fillRect(0, 0, W, H);
  const band = paint.createLinearGradient(0, 0, W, 0);
  band.addColorStop(0, NAVY);
  band.addColorStop(1, SKY);
  paint.fillStyle = band;
  paint.fillRect(0, 0, W, TITLE_HEIGHT);
  text(`${state.institution} Teller Workstation`, 12, 20, { font: TITLE, color: LIGHT });

  const bar = TITLE_HEIGHT;
  edge(0, bar, W, TOOLBAR_HEIGHT, LIGHT, SHADE);
  if (state.operator) {
    const onMember = ['home', 'member', 'open', 'review', 'receipt'].includes(state.screen)
      || (state.screen === 'account' && state.accountFrom === 'member');
    const onRegister = state.screen === 'register' || (state.screen === 'account' && state.accountFrom === 'register');
    let x = 10;
    x += button('Member search (F2)', x, bar + 7, actions.search, { pressed: onMember }) + 8;
    button('Accounts register (F4)', x, bar + 7, actions.browse, { pressed: onRegister });
    const out = measure('Sign out') + 36;
    const outX = W - 10 - Math.max(112, out);
    button('Sign out', outX, bar + 7, actions.signout);
    const role = state.operator.role ? `, ${state.operator.role}` : '';
    const who = `${state.operator.name}${role}`;
    text(who, outX - 16, bar + 28, { align: 'right' });
    text(`Operator: ${state.operator.number}`, outX - 16 - measure(who) - 48, bar + 28, { align: 'right' });
  }

  const foot = H - STATUS_HEIGHT;
  const small = `13px ${FAMILY}`;
  edge(0, foot, W, STATUS_HEIGHT, LIGHT, SHADE);
  const panels = [
    { width: 150, value: state.busy ? 'Working…' : 'Ready' },
    { width: 0, value: state.operator ? 'F2 Member search    F4 Accounts register    ↑↓ Select row    Enter Continue    Esc Back' : 'Enter Continue', color: MUTED },
    { width: 210, value: state.businessDate ? `Business date ${day(state.businessDate)}` : '' },
    { width: 56, value: 'CAPS', align: 'center', color: state.caps ? INK : SHADE },
    { width: 130, value: clock(), align: 'center' },
  ];
  const fixed = panels.reduce((sum, panel) => sum + panel.width + 4, 4);
  let at = 4;
  for (const panel of panels) {
    const width = panel.width || W - fixed;
    sunken(at, foot + 3, width, STATUS_HEIGHT - 6);
    const shown = fit(panel.value, small, width - 12);
    if (panel.align === 'center') text(shown, at + width / 2, foot + 19, { font: small, align: 'center', color: panel.color ?? INK });
    else text(shown, at + 8, foot + 19, { font: small, color: panel.color ?? INK });
    at += width + 4;
  }
}

function message(W, H) {
  if (!state.message) return;
  const y = H - STATUS_HEIGHT - MESSAGE_HEIGHT - 6;
  paint.fillStyle = ALERT_WASH;
  paint.fillRect(LEFT, y, W - LEFT * 2, MESSAGE_HEIGHT);
  paint.strokeStyle = ALERT;
  paint.lineWidth = 1;
  paint.strokeRect(LEFT + 0.5, y + 0.5, W - LEFT * 2 - 1, MESSAGE_HEIGHT - 1);
  text(fit(state.message, UI_BOLD, W - LEFT * 2 - 24), LEFT + 12, y + 22, { font: UI_BOLD, color: ALERT });
}

const screens = {
  loading() { text('Loading…', LEFT, TOP); },
  signon() {
    heading('Operator sign-on', state.legalName || 'Enter your operator number to open the workstation.');
    group('Operator', LEFT, TOP + 52, 380, 140);
    field('operator', 'Operator number', LEFT + 18, TOP + 90);
    button('Sign on', LEFT + 18, TOP + 146, actions.signon, { primary: true });
  },
  home() {
    heading('Find a member', 'Look up a member by number to see their record and accounts.');
    group('Member lookup', LEFT, TOP + 52, 380, 140);
    field('member', 'Member number', LEFT + 18, TOP + 90);
    button('Find member', LEFT + 18, TOP + 146, actions.find, { primary: true });
    text('Every account, newest first, is in the Accounts register (F4).', LEFT, TOP + 226, { color: MUTED });
  },
  member(W, H) {
    const found = state.member;
    heading('Member', `As of ${clock()}`);
    group('Member record', LEFT, TOP + 52, 860, 164);
    lines([
      `Member number: ${found.number}`,
      `Name: ${found.name}`,
      `Status: ${found.status}`,
      `Open accounts: ${found.openAccounts}`,
      `Member since: ${day(found.joined)}`,
    ], LEFT + 18, TOP + 82);
    lines([
      `Type: ${found.kind}`,
      `Email: ${masked(found.email)}`,
      `Phone: ${phone(found.phone)}`,
      `Address: ${fit(found.address ?? 'none on file', UI, 380)}`,
    ], LEFT + 420, TOP + 82);
    const width = button('Open account', LEFT, TOP + 232, actions.open, { primary: true });
    button('New search', LEFT + width + 16, TOP + 232, actions.search);

    const top = TOP + 298;
    const room = H - STATUS_HEIGHT - MESSAGE_HEIGHT - 70 - top;
    const size = Math.max(3, Math.floor((room - 34) / ROW));
    const pages = Math.max(1, Math.ceil(found.accounts.length / size));
    state.memberPage = Math.min(state.memberPage, pages - 1);
    const rows = found.accounts.slice(state.memberPage * size, (state.memberPage + 1) * size);
    const title = `All accounts (${found.accounts.length})`;
    text(title, LEFT, top, { font: UI_BOLD });
    text('Select an account to see its balance and ledger.', LEFT + measure(title, UI_BOLD) + 16, top, { color: MUTED });
    const bottom = grid(LEFT, top + 12, Math.min(W - LEFT * 2, 1180), ledgerColumns(false), rows, {
      open: (row) => actions.account(row.number, 'member'),
      marked: (row) => row.number === state.opened,
      empty: 'This member has no accounts yet.',
    });
    pager(LEFT, bottom + 12, state.memberPage, pages, (page) => { state.memberPage = page; state.row = -1; draw(); }, { size, total: found.accounts.length });
  },
  open() {
    heading('Open an account', `For member ${state.member.number}`);
    group('Product', LEFT, TOP + 52, 700, 26 + Math.ceil(state.products.length / 3) * 34);
    state.products.forEach((name, index) => {
      option(name, LEFT + 18 + (index % 3) * 220, TOP + 88 + Math.floor(index / 3) * 34, state.product === name, () => { state.product = name; });
    });
    const below = TOP + 52 + 26 + Math.ceil(state.products.length / 3) * 34;
    field('nickname', 'Nickname', LEFT, below + 36);
    text('Statement delivery', LEFT, below + 112);
    option('Paper', LEFT, below + 144, state.delivery === 'paper', () => { state.delivery = 'paper'; });
    option('Electronic', LEFT + 160, below + 144, state.delivery === 'electronic', () => { state.delivery = 'electronic'; });
    const width = button('Review', LEFT, below + 172, actions.review, { primary: true });
    button('Cancel', LEFT + width + 16, below + 172, actions.back);
  },
  review() {
    heading('Review', `Opening ${state.product || 'an account'} for member ${state.member.number}. Nothing is opened until you commit.`);
    const terms = state.review.terms.length ? state.review.terms : ['Nothing to review.'];
    group('Terms', LEFT, TOP + 52, 760, 30 + terms.length * ROW + 2 * ROW);
    const at = lines(terms, LEFT + 18, TOP + 84) + ROW;
    lines([`Nickname: ${state.fields.nickname}`, `Statement delivery: ${state.delivery}`], LEFT + 18, at);
    const top = TOP + 52 + 30 + terms.length * ROW + 2 * ROW + 20;
    const width = button('Commit', LEFT, top, actions.commit, { primary: true });
    button('Back', LEFT + width + 16, top, actions.back);
  },
  receipt() {
    const done = state.receipt;
    heading(done.repeated ? 'Already opened' : 'Account opened');
    group('Receipt', LEFT, TOP + 24, 560, 264);
    lines([
      `Account number: ${done.account}`,
      `Member number: ${done.member}`,
      `Product: ${state.product}`,
      `Nickname: ${state.fields.nickname}`,
      `Statement delivery: ${state.delivery}`,
    ], LEFT + 18, TOP + 54);
    const when = state.openedAt ? `, ${clock(state.openedAt)}` : '';
    lines([
      `Receipt: ${done.receipt}`,
      `Opened by: ${state.operator.number}`,
      `Business date: ${day(state.businessDate)}${when}`,
    ], LEFT + 18, TOP + 194);
    const width = button('View member', LEFT, TOP + 294, actions.viewMember, { primary: true });
    button('New search', LEFT + width + 16, TOP + 294, actions.search);
  },
  account(W) {
    const shown = state.account;
    heading('Account', `${shown.product} held by ${shown.owner}`);
    group('Account record', LEFT, TOP + 52, 860, 164);
    lines([
      `Account number: ${shown.number}`,
      `Member number: ${shown.member}`,
      `Nickname: ${shown.nickname}`,
      `Statement delivery: ${shown.delivery}`,
      `Status: ${shown.status}`,
    ], LEFT + 18, TOP + 82);
    lines([
      `Opened: ${day(shown.opened)}`,
      `Posted balance: ${money(shown.posted, shown.exponent, shown.currency)}`,
      `Available balance: ${money(shown.available, shown.exponent, shown.currency)}`,
      `Holds: ${shown.holds}    Restrictions: ${shown.restrictions}`,
      `Relationship: ${shown.role || 'none'}`,
    ], LEFT + 420, TOP + 82);
    const width = button('View member', LEFT, TOP + 232, actions.viewMember, { primary: true });
    button(state.accountFrom === 'register' ? 'Back to register' : 'Back to member', LEFT + width + 16, TOP + 232, actions.leaveAccount);
    text('Recent ledger entries', LEFT, TOP + 298, { font: UI_BOLD });
    grid(LEFT, TOP + 310, Math.min(W - LEFT * 2, 1180), [
      { title: 'Date', share: 1, value: (row) => day(row.date) },
      { title: 'Reference', share: 1.6, value: (row) => row.reference },
      { title: 'Description', share: 2.6, value: (row) => row.description },
      { title: 'Amount', share: 1.3, align: 'right', value: (row) => money(row.amount, shown.exponent, shown.currency) },
      { title: 'Balance', share: 1.4, align: 'right', value: (row) => money(row.balance, shown.exponent, shown.currency) },
    ], shown.entries, { empty: 'No ledger entries yet.' });
  },
  register(W) {
    const listed = state.register ?? { items: [], total: 0, page: 0, pages: 1 };
    heading('Accounts register', `Every account at ${state.institution}, newest first. As of ${clock()}.`);
    const bottom = field('query', 'Account number, member number, or nickname', LEFT, TOP + 64, 420);
    button('Search', LEFT + 436, bottom - 32, actions.registerSearch, { primary: true });
    const end = grid(LEFT, bottom + 20, Math.min(W - LEFT * 2, 1180), ledgerColumns(true), listed.items, {
      open: (row) => actions.account(row.number, 'register'),
      marked: (row) => row.number === state.opened,
      empty: 'No account matches this search.',
    });
    pager(LEFT, end + 12, listed.page, listed.pages, (page) => actions.register(page), { size: 12, total: listed.total });
  },
};

function draw() {
  const ratio = devicePixelRatio || 1;
  const W = innerWidth;
  const H = innerHeight;
  canvas.width = Math.floor(W * ratio);
  canvas.height = Math.floor(H * ratio);
  canvas.style.width = `${W}px`;
  canvas.style.height = `${H}px`;
  paint.setTransform(ratio, 0, 0, ratio, 0, 0);
  regions = [];
  listing = null;
  chrome(W, H);
  screens[state.screen](W, H);
  message(W, H);
}

function hit(event) {
  return regions.find((r) => event.offsetX >= r.x && event.offsetX <= r.x + r.width
    && event.offsetY >= r.y && event.offsetY <= r.y + r.height);
}

canvas.addEventListener('click', (event) => {
  canvas.focus();
  hit(event)?.action();
});

canvas.addEventListener('mousemove', (event) => {
  const found = hit(event);
  if (!state.busy) canvas.style.cursor = found?.cursor ?? 'default';
  const hover = found?.hover ?? '';
  if (hover !== state.hover) {
    state.hover = hover;
    draw();
  }
});

canvas.addEventListener('mouseleave', () => {
  if (state.hover) {
    state.hover = '';
    draw();
  }
});

const SUBMIT = { signon: 'signon', home: 'find', open: 'review', review: 'commit', register: 'registerSearch' };
const ESCAPE = { open: 'back', review: 'back', account: 'leaveAccount', member: 'search' };

function choose(step) {
  if (!listing || !listing.rows.length) return;
  const last = listing.rows.length - 1;
  state.row = state.row < 0 ? (step > 0 ? 0 : last) : Math.min(last, Math.max(0, state.row + step));
  draw();
}

canvas.addEventListener('keydown', (event) => {
  const caps = event.getModifierState?.('CapsLock') ?? false;
  if (caps !== state.caps) {
    state.caps = caps;
    draw();
  }
  const name = state.focus;
  if (state.operator && (event.key === 'F2' || event.key === 'F4')) {
    event.preventDefault();
    if (!state.busy) (event.key === 'F2' ? actions.search : actions.browse)();
    return;
  }
  if (event.key === 'ArrowDown' || event.key === 'ArrowUp') {
    event.preventDefault();
    choose(event.key === 'ArrowDown' ? 1 : -1);
    return;
  }
  if (event.key === 'Enter' && listing && state.row >= 0 && listing.rows[state.row]) {
    event.preventDefault();
    if (!state.busy) listing.open(listing.rows[state.row]);
    return;
  }
  if (event.key === 'Escape' && ESCAPE[state.screen]) {
    event.preventDefault();
    if (!state.busy) actions[ESCAPE[state.screen]]();
    return;
  }
  if (event.key === 'Enter' && SUBMIT[state.screen]) {
    event.preventDefault();
    if (!state.busy) actions[SUBMIT[state.screen]]();
    return;
  }
  if (!name) return;
  if (event.key === 'Backspace') {
    state.fields[name] = state.fields[name].slice(0, -1);
  } else if (event.key.length === 1 && !event.ctrlKey && !event.metaKey) {
    if (state.fields[name].length < 60) state.fields[name] += event.key;
  } else {
    return;
  }
  state.row = -1;
  event.preventDefault();
  draw();
});

addEventListener('resize', draw);
setInterval(draw, 30_000);
canvas.focus();
load();
