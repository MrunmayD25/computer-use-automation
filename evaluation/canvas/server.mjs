// Serve a teller application drawn entirely on one canvas for the no-DOM case.
//
// This local stand-in is not a supplied fixture. Its only page element is a
// <canvas>, so the browser's accessibility tree and DOM reveal nothing on the
// screen. Every read and write uses the supplied financial database services.
// Opening an account therefore follows the same validation, permission, and
// receipt rules as the supplied sites.
//
//   node evaluation/canvas/server.mjs --database .runtime/evaluation.sqlite \
//     --services <environments>/database/financial/dist/services.js --port 4390

import { randomBytes, randomUUID } from 'node:crypto';
import { readFileSync } from 'node:fs';
import { createServer } from 'node:http';
import { dirname, join, resolve } from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';
import { parseArgs } from 'node:util';

const here = dirname(fileURLToPath(import.meta.url));
const { values } = parseArgs({
  options: {
    database: { type: 'string' },
    services: { type: 'string' },
    port: { type: 'string', default: '4390' },
    host: { type: 'string', default: '127.0.0.1' },
  },
  strict: true,
});
if (!values.database || !values.services) {
  throw new Error('--database and --services are required');
}
const services = await import(pathToFileURL(resolve(values.services)).href);
const store = services.openFinancialStore({ path: resolve(values.database) });

const INSTITUTION = 'northstar';
const WORLD = 'financial-demo-v1:northstar';
const binding = store.read((db) => {
  const row = db
    .prepare('SELECT id FROM datasets WHERE name=? AND institution_id=?')
    .get(WORLD, INSTITUTION);
  if (!row) throw new Error('the seeded world has no Northstar dataset');
  return { institutionId: INSTITUTION, datasetId: String(row.id) };
});
const bank = store.read((db) => {
  const row = db
    .prepare('SELECT i.display_name, i.legal_name, i.time_zone, d.as_of FROM institutions i JOIN datasets d ON d.institution_id=i.id WHERE d.id=?')
    .get(binding.datasetId);
  return {
    name: String(row?.display_name ?? 'Northstar'),
    legalName: String(row?.legal_name ?? ''),
    timeZone: String(row?.time_zone ?? 'UTC'),
    businessDate: String(row?.as_of ?? '').slice(0, 10),
  };
});
const institution = bank.name;

const page = readFileSync(join(here, 'index.html'));
const script = readFileSync(join(here, 'app.js'));
const sessions = new Map();

function json(response, status, body) {
  response.writeHead(status, { 'content-type': 'application/json', 'cache-control': 'no-store' });
  response.end(JSON.stringify(body));
}

function session(request) {
  const cookie = request.headers.cookie ?? '';
  const pair = cookie.split(';').map((part) => part.trim()).find((part) => part.startsWith('sid='));
  return pair ? sessions.get(pair.slice(4)) : undefined;
}

async function body(request) {
  let text = '';
  for await (const chunk of request) {
    text += chunk;
    if (text.length > 16_384) throw new Error('request too large');
  }
  return text ? JSON.parse(text) : {};
}

function operator(actorNumber) {
  return store.read((db) =>
    services.operatorProfile(db, binding.datasetId, binding.institutionId, actorNumber),
  );
}

function member(number) {
  return store.read((db) => {
    const found = services.searchMembers(db, binding.datasetId, {
      query: number, field: 'memberNumber', match: 'exact', limit: 2, offset: 0,
    });
    return found.items.length === 1 ? found.items[0] : undefined;
  });
}

function ledger(account) {
  return {
    number: account.accountNumber,
    member: account.memberNumber,
    owner: account.ownerName,
    product: account.productName,
    nickname: account.nickname,
    delivery: account.statementDelivery,
    status: account.status,
    currency: account.currency,
    exponent: account.exponent,
    posted: account.postedMinorUnits,
    available: account.availableMinorUnits,
    opened: account.openedAt,
  };
}

function newestFirst(left, right) {
  return right.opened.localeCompare(left.opened) || right.number.localeCompare(left.number);
}

function memberRecord(found) {
  return store.read((db) => {
    const detail = services.memberDetail(db, binding.datasetId, found.id);
    const address = detail?.address;
    return {
      number: found.memberNumber,
      name: found.displayName,
      kind: found.kind,
      status: found.status,
      joined: found.joinedAt,
      email: detail?.email?.value ?? null,
      phone: detail?.phone?.value ?? null,
      address: address ? `${address.line1}, ${address.city} ${address.region} ${address.postalCode}` : null,
      openAccounts: found.openAccounts,
      accounts: (detail?.accounts ?? []).map(ledger).sort(newestFirst),
    };
  });
}

function account(number) {
  return store.read((db) => {
    const found = services
      .searchAccounts(db, binding.datasetId, { query: number, limit: 5, offset: 0 })
      .items.find((item) => item.accountNumber === number);
    if (!found) return undefined;
    const entries = services.accountTransactions(db, binding.datasetId, found.id, { limit: 8, offset: 0, order: 'newest' });
    const detail = services.accountDetail(db, binding.datasetId, found.id);
    const active = (items) => (items ?? []).filter((item) => item.status === 'active').length;
    return {
      ...ledger(found),
      role: detail?.parties.find((party) => party.role === 'owner')?.role ?? found.memberRoles[0] ?? '',
      holds: active(detail?.holds),
      restrictions: active(detail?.restrictions),
      entries: entries.items.map((entry) => ({
        date: entry.effectiveDate,
        reference: entry.reference,
        description: entry.description,
        amount: entry.amountMinorUnits,
        balance: entry.runningBalanceMinorUnits,
      })),
    };
  });
}

const REGISTER_PAGE = 12;

function register(query, page) {
  return store.read((db) => {
    const found = services.searchAccounts(db, binding.datasetId, {
      query: query || undefined, sort: 'opened', direction: 'desc', limit: REGISTER_PAGE, offset: page * REGISTER_PAGE,
    });
    return {
      items: found.items.map(ledger).sort(newestFirst),
      total: found.total,
      page,
      pages: Math.max(1, Math.ceil(found.total / REGISTER_PAGE)),
    };
  });
}

function products() {
  return store.read((db) =>
    db
      .prepare("SELECT id, name FROM products WHERE dataset_id=? AND family='deposit' AND status='active' AND code NOT LIKE '%-TERM' ORDER BY name")
      .all(binding.datasetId)
      .map((row) => ({ id: String(row.id), name: String(row.name) })),
  );
}

function command(form, found) {
  const product = products().find((item) => item.name === form.product);
  return {
    kind: 'account.open',
    memberId: found.id,
    productId: product?.id ?? '',
    nickname: String(form.nickname ?? ''),
    statementDelivery: String(form.delivery ?? ''),
  };
}

function context(current) {
  return { binding, actorId: current.actorId, reference: `canvas-teller:${current.id.slice(0, 12)}` };
}

const routes = {
  'GET /': (request, response) => {
    response.writeHead(200, { 'content-type': 'text/html; charset=utf-8', 'cache-control': 'no-store' });
    response.end(page);
  },
  'GET /app.js': (request, response) => {
    response.writeHead(200, { 'content-type': 'text/javascript; charset=utf-8' });
    response.end(script);
  },
  'GET /api/session': (request, response) => {
    const current = session(request);
    json(response, 200, {
      institution,
      legalName: bank.legalName,
      timeZone: bank.timeZone,
      businessDate: bank.businessDate,
      operator: current ? { number: current.actorNumber, name: current.name, role: current.role } : null,
      products: products().map((item) => item.name),
    });
  },
  'POST /api/signon': async (request, response) => {
    const { operator: number } = await body(request);
    const profile = operator(String(number ?? '').trim());
    if (!profile || profile.status !== 'active') {
      json(response, 400, { message: `Operator ${number} cannot sign on here.` });
      return;
    }
    const id = randomBytes(16).toString('hex');
    const role = profile.roles[0] ?? '';
    sessions.set(id, { id, actorId: profile.actor.id, actorNumber: profile.actor.actorNumber, name: profile.actor.displayName, role });
    response.setHeader('set-cookie', `sid=${id}; Path=/; HttpOnly; SameSite=Strict`);
    json(response, 200, { number: profile.actor.actorNumber, name: profile.actor.displayName, role });
  },
  'POST /api/member': async (request, response) => {
    if (!session(request)) return json(response, 401, { message: 'Your session has ended. Sign on again.' });
    const { number } = await body(request);
    const found = member(String(number ?? '').trim());
    if (!found) return json(response, 404, { message: `No member with number ${number}.` });
    json(response, 200, memberRecord(found));
  },
  'GET /api/account': (request, response) => {
    if (!session(request)) return json(response, 401, { message: 'Your session has ended. Sign on again.' });
    const number = new URL(request.url, 'http://canvas.invalid').searchParams.get('number')?.trim() ?? '';
    const found = account(number);
    if (!found) return json(response, 404, { message: `No account with number ${number}.` });
    json(response, 200, found);
  },
  'GET /api/accounts': (request, response) => {
    if (!session(request)) return json(response, 401, { message: 'Your session has ended. Sign on again.' });
    const search = new URL(request.url, 'http://canvas.invalid').searchParams;
    const page = Math.max(0, Number.parseInt(search.get('page') ?? '0', 10) || 0);
    json(response, 200, register(search.get('query')?.trim() ?? '', page));
  },
  'POST /api/open/review': async (request, response) => {
    const current = session(request);
    if (!current) return json(response, 401, { message: 'Your session has ended. Sign on again.' });
    const form = await body(request);
    const found = member(String(form.member ?? '').trim());
    if (!found) return json(response, 404, { message: `No member with number ${form.member}.` });
    const outcome = services.reviewCommand(store, context(current), command(form, found));
    if (outcome.kind !== 'reviewable') return json(response, 422, { code: outcome.code ?? outcome.kind, message: outcome.message });
    json(response, 200, { terms: outcome.review.consequences ?? [], key: randomUUID() });
  },
  'POST /api/open/commit': async (request, response) => {
    const current = session(request);
    if (!current) return json(response, 401, { message: 'Your session has ended. Sign on again.' });
    const form = await body(request);
    const found = member(String(form.member ?? '').trim());
    if (!found) return json(response, 404, { message: `No member with number ${form.member}.` });
    const outcome = services.executeCommand(store, context(current), command(form, found), String(form.key ?? randomUUID()));
    if (outcome.kind !== 'committed' && outcome.kind !== 'duplicate') {
      return json(response, 422, { code: outcome.code ?? outcome.kind, message: outcome.message });
    }
    const receipt = outcome.receipt;
    json(response, 200, {
      receipt: receipt.receiptNumber,
      account: receipt.businessReference,
      member: found.memberNumber,
      repeated: outcome.kind === 'duplicate',
    });
  },
};

createServer(async (request, response) => {
  const path = new URL(request.url ?? '/', 'http://canvas.invalid').pathname;
  const handler = routes[`${request.method} ${path}`];
  try {
    if (handler) await handler(request, response);
    else json(response, 404, { message: 'Not found.' });
  } catch (error) {
    console.error(`${request.method} ${path} failed:`, error);
    json(response, 500, { message: 'The request could not be completed.' });
  }
}).listen(Number(values.port), values.host, () => {
  console.log(`canvas teller listening on http://${values.host}:${values.port}/`);
});
