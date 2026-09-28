"""Scripts for observing and targeting controls in one frame.

Keeping the scripts separate leaves the adapter as readable Python. They
return only roles, accessible names, a closed list of attributes, supported
interactions, enclosing rows, regions, forms, or tables, and context paths.
The enclosing structure distinguishes repeated labels without relying on
match counts. Context paths let the adapter detect when a control now belongs
to a different record.

Observation and targeting share every definition here. ``SCRIPT`` walks the
frame with ``eachCandidate`` and describes each element with ``roleOf``,
``nameOf``, ``scopeOf``, and ``attributeOf``. ``RESOLVE`` walks the same
elements in the same order and compares a target against the same functions.
This lets targeting find an observed control by exactly the role,
name, scope, attribute, or class the observation reported for it. A driver's
own role and name rules are never consulted, because they disagree with any
second implementation at the edges: hidden decoration, several label
references, long names, and names with quotes in them.

These scripts never return three categories of data:

- Secret values. They never read the value or ``value`` attribute of password
  fields, one-time-code fields, or fields that received a declared secret.
  The adapter tracks such a field by the element and the session's mark on it,
  never by its label. An editable region stores its value as text, so the
  scripts also withhold the region's text, all text inside it, and readings
  from elements that contain it.
- Content inside a closed shadow root. The scripts count and report the root
  instead of guessing what it contains.
- Pattern matches. Every test is an equality or membership check because a
  pattern would guess at page content.

``CONTEXT`` answers the context question for one element, which is what the
adapter asks immediately before it acts, so a decision made about one record
cannot land on another. ``EVIDENCE`` reads one piece of record evidence again
and reports whether it still sits in the target's row or panel. ``READ``
reports what one element shows, withholding a secret field. ``PROTECT`` and
``HELD`` place and check the mark on a field that received a declared secret.
"""

_HELPERS = """
  const TEXT = options.maxText;
  const NAME = options.maxName || options.maxText;
  const SECRETS = options.secretElements || [];
  const MARK = options.secretMark || null;
  const operatorOf = (el) => {
    const host = el && el.closest('[data-computeruse-ui]');
    return !!host && !!options.controlMark &&
      host.getAttribute('data-computeruse-ui') === options.controlMark;
  };
  const SECRET_AUTOCOMPLETE = ['current-password', 'new-password', 'one-time-code'];
  const WHITESPACE = [' ', '\\n', '\\t', '\\r', '\\f', '\\u00a0'];
  const squash = (value) => {
    let out = '';
    let gap = false;
    for (const ch of (value || '')) {
      if (WHITESPACE.includes(ch)) { gap = out.length > 0; continue; }
      if (gap) { out += ' '; }
      out += ch;
      gap = false;
    }
    return out;
  };
  const cut = (value, limit) => value.length > limit ? value.slice(0, limit) : value;

  const tagOf = (el) => el.tagName.toLowerCase();
  const typeOf = (el) => (el.getAttribute('type') || '').toLowerCase();
  const SKIPPED = ['script', 'style', 'noscript', 'template', 'head', 'title',
                   'meta', 'link', 'option', 'optgroup', 'html', 'body'];
  const FIELDS = ['input', 'select', 'textarea'];
  const INLINE = ['inline', 'contents'];
  const styleOf = (el) => {
    const view = el.ownerDocument.defaultView;
    return view ? view.getComputedStyle(el) : null;
  };

  // Return the text a reader sees inside an element. Content marked aria-hidden,
  // hidden, or not displayed is skipped, which is what keeps a decorative
  // icon out of a button's name. Form fields inside are skipped too, so a
  // label wrapped around a select is not named after every option in it. A
  // block element separates its words from its neighbours'.
  const shownText = (el) => {
    if (operatorOf(el) || secretOf(el) || withinSecret(el)) { return ''; }
    const parts = [];
    let size = 0;
    const walk = (node) => {
      if (size > NAME * 2) { return; }
      if (node.nodeType === 3) {
        parts.push(node.data);
        size += node.data.length;
        return;
      }
      if (node.nodeType !== 1) { return; }
      const tag = tagOf(node);
      if (operatorOf(node)) { return; }
      if (SKIPPED.includes(tag) || FIELDS.includes(tag)) { return; }
      if (secretOf(node)) { return; }
      if (node.getAttribute('aria-hidden') === 'true') { return; }
      if (node.hidden === true) { return; }
      const style = styleOf(node);
      if (style && (style.display === 'none' || style.visibility === 'hidden')) {
        return;
      }
      if (tag === 'img') {
        parts.push(' ' + (node.getAttribute('alt') || '') + ' ');
        return;
      }
      if (tag === 'br') { parts.push(' '); return; }
      const block = !style || !INLINE.includes(style.display);
      if (block) { parts.push(' '); }
      for (const child of node.childNodes) { walk(child); }
      if (block) { parts.push(' '); }
    };
    for (const child of el.childNodes) { walk(child); }
    return squash(parts.join(''));
  };
  // Context and slots are compared only by these scripts, so they may be
  // cut short. Names, scopes, and values identify a target, so they keep
  // up to NAME characters, and the adapter shortens them only for display.
  const text = (el) => cut(shownText(el), TEXT);
  const full = (el) => cut(shownText(el), NAME);
  const attributeOf = (el, name) => cut(squash(el.getAttribute(name) || ''), NAME);

  const CONTAINER = '[role="region"], section, article, fieldset';
  const HEADINGS = 'h1, h2, h3, h4, legend';

  // The heading a reader would say this control sits under: the last heading
  // before it, within its own container. A record swapped in at the same URL
  // changes this string, while moving the button under the same heading does
  // not.
  const headingAbove = (el) => {
    for (let anchor = el; anchor; anchor = anchor.getRootNode().host) {
      const scope = anchor.closest(CONTAINER) || anchor.getRootNode();
      let found = null;
      for (const heading of scope.querySelectorAll(HEADINGS)) {
        const position = heading.compareDocumentPosition(anchor);
        if (!(position & 1) && (position & 4)) { found = heading; }
      }
      if (found) { return found; }
    }
    return null;
  };

  const contextOf = (el) => {
    const path = [];
    const popup = el.closest('[role="listbox"], [role="menu"]');
    if (popup) { path.push('popup:' + popup.getAttribute('role')); }
    let page = null;
    for (let anchor = el; anchor; anchor = anchor.getRootNode().host) {
      page = anchor.getRootNode().querySelector('h1');
      if (page) { break; }
    }
    if (page) { path.push('page:' + text(page)); }
    const container = el.closest(CONTAINER);
    if (container) {
      const label = container.getAttribute('aria-label') || '';
      if (label) { path.push('region:' + cut(squash(label), TEXT)); }
    }
    const heading = headingAbove(el);
    if (heading) { path.push('heading:' + text(heading)); }
    const row = el.closest('tr');
    if (row && row.closest('table')) {
      const head = row.querySelector('th') || row.querySelector('td');
      if (head) { path.push('row:' + text(head)); }
    }
    const form = el.closest('form');
    if (form) {
      const named = form.getAttribute('name') || form.getAttribute('aria-label') || '';
      if (named) { path.push('form:' + cut(squash(named), TEXT)); }
    }
    return path;
  };

  // The row, region, form, or table a control is looked for inside. A
  // target's scope is compared with this, so every scope an observation
  // reports is one a target can name.
  const scopeOf = (el) => {
    const row = el.closest('tr, [role="row"]');
    if (row) {
      const cellSelector = 'th, td, [role="cell"], [role="gridcell"], ' +
        '[role="rowheader"]';
      const cells = Array.from(row.querySelectorAll(cellSelector));
      const siblings = row.parentElement ? Array.from(row.parentElement.children) : [];
      // Every cell that no other row repeats in its column names the row.
      // The first is the row's name; a target may name the row by any, so a
      // row keyed by a sequence number can be found by its member instead.
      const names = [];
      const columns = [];
      for (let index = 0; index < cells.length; index += 1) {
        const cell = cells[index];
        const interactive = 'button, input, select, textarea, [role="button"]';
        if (cell.querySelector(interactive)) { continue; }
        const label = full(cell);
        if (!label) { continue; }
        const unique = siblings.every(other => {
          if (other === row) { return true; }
          const peers = other.querySelectorAll(cellSelector);
          return !peers[index] || full(peers[index]) !== label;
        });
        if (unique && !names.includes(label)) {
          names.push(label);
          columns.push(index + 1);
        }
      }
      if (names.length) {
        return { kind: 'row', name: names[0], names: names, columns: columns };
      }
    }
    const region = el.closest(CONTAINER);
    if (region) {
      const titled = region.querySelector('h1, h2, h3, legend');
      const heading = attributeOf(region, 'aria-label') || (titled ? full(titled) : '');
      if (heading) { return { kind: 'region', name: heading }; }
    }
    const form = el.closest('form');
    if (form) {
      const named = attributeOf(form, 'name') || attributeOf(form, 'aria-label');
      if (named) { return { kind: 'form', name: named }; }
    }
    const table = el.closest('table');
    if (table) {
      const caption = table.querySelector('caption');
      const named = attributeOf(table, 'aria-label') || (caption ? full(caption) : '');
      if (named) { return { kind: 'table', name: named }; }
    }
    return null;
  };
  const sameScope = (seen, wanted) =>
    !!seen && seen.kind === wanted.kind &&
      (seen.name === wanted.name ||
        (seen.kind === 'row' && (seen.names || []).includes(wanted.name)));

  const ROW = 'tr, [role="row"]';

  // Decided before anything reads a value, because a field that holds a
  // credential must not have its value read at all, not even to discard it.
  // A field the run typed a declared secret into is known by the element
  // itself and by the session's mark on it, so renaming it changes nothing.
  const secretOf = (el) => {
    if (SECRETS.includes(el)) { return true; }
    if (MARK && el.getAttribute(MARK.name) === MARK.token) { return true; }
    if (el.tagName.toLowerCase() !== 'input') { return false; }
    const type = (el.getAttribute('type') || '').toLowerCase();
    if (type === 'password') { return true; }
    const hint = el.getAttribute('autocomplete') || '';
    return SECRET_AUTOCOMPLETE.includes(hint);
  };

  // An editable region keeps what was typed as text in the document, where
  // a name, an ancestor's text, or a heading would read it. So nothing
  // inside a secret editable region is read or described, and anything that
  // holds one is not read at all. Only editable content can sit inside an
  // editable region, which keeps the walk up to elements that are editable.
  const withinSecret = (el) => {
    if (!el.isContentEditable) { return false; }
    for (let at = el.parentElement; at; at = at.parentElement) {
      if (secretOf(at)) { return true; }
    }
    return false;
  };
  const holdsSecret = (el) => {
    for (const held of SECRETS) {
      if (held !== el && held.isContentEditable && el.contains(held)) { return true; }
    }
    for (const region of el.querySelectorAll('[contenteditable]')) {
      if (secretOf(region)) { return true; }
    }
    return false;
  };
  const concealed = (el) => secretOf(el) || withinSecret(el) || holdsSecret(el);

  // Every id in aria-labelledby, in order, each read as its shown text. A
  // referenced element is read even when it is itself hidden, as the page
  // asked for it by id.
  const labelledBy = (el) => {
    const ids = squash(el.getAttribute('aria-labelledby') || '').split(' ');
    const root = el.getRootNode();
    const parts = [];
    for (const id of ids) {
      if (!id) { continue; }
      const byRoot = typeof root.getElementById === 'function'
        ? root.getElementById(id) : null;
      const source = byRoot || el.ownerDocument.getElementById(id);
      if (source && source !== el) {
        const said = shownText(source) ||
          squash(source.getAttribute('aria-label') || '');
        if (said) { parts.push(said); }
      }
    }
    return cut(squash(parts.join(' ')), NAME);
  };

  const nameOf = (el, tag, type) => {
    const referenced = labelledBy(el);
    if (referenced) { return referenced; }
    const label = attributeOf(el, 'aria-label');
    if (label) { return label; }
    const labels = Array.from(el.labels || []).map(full).filter(Boolean);
    if (labels.length) { return cut(labels.join(' '), NAME); }
    if (tag === 'input' && ['submit', 'button', 'reset'].includes(type)) {
      return cut(squash(el.value || ''), NAME);
    }
    if (tag === 'input' && type === 'image') { return attributeOf(el, 'alt'); }
    const title = attributeOf(el, 'title');
    if (FIELDS.includes(tag) || tag === 'canvas') {
      return attributeOf(el, 'placeholder') || title;
    }
    if (tag === 'img') { return attributeOf(el, 'alt') || title; }
    return full(el) || title;
  };

  const IMPLICIT = {
    a: 'link', button: 'button', input: 'textbox', textarea: 'textbox',
    td: 'cell', li: 'listitem', h1: 'heading', h2: 'heading', h3: 'heading',
    h4: 'heading', h5: 'heading', h6: 'heading', label: 'label', output: 'status',
    canvas: 'canvas', dd: 'definition', dt: 'term', legend: 'legend',
    p: 'paragraph', caption: 'caption', summary: 'button', img: 'img'
  };
  const INPUT_ROLE = {
    submit: 'button', button: 'button', reset: 'button', image: 'button',
    checkbox: 'checkbox', radio: 'radio', search: 'searchbox', email: 'textbox',
    number: 'spinbutton', range: 'slider', tel: 'textbox', text: 'textbox',
    url: 'textbox', password: 'textbox', date: 'textbox'
  };

  // A declared role is the first token of the attribute, which is the one a
  // browser uses when it knows it. Anything with no role of its own is
  // generic, so a tag name is never offered as if it were a role.
  const roleOf = (el, tag, type) => {
    const declared = squash(el.getAttribute('role') || '').split(' ')[0];
    if (declared) { return declared; }
    if (tag === 'input') { return INPUT_ROLE[type] || 'textbox'; }
    if (tag === 'a') { return el.hasAttribute('href') ? 'link' : 'generic'; }
    if (tag === 'select') {
      return (el.multiple || el.size > 1) ? 'listbox' : 'combobox';
    }
    if (tag === 'th') {
      return (el.getAttribute('scope') || '').toLowerCase() === 'row'
        ? 'rowheader' : 'columnheader';
    }
    if (el.isContentEditable && !FIELDS.includes(tag)) { return 'textbox'; }
    return IMPLICIT[tag] || 'generic';
  };

  // Include controls, labelled or text-bearing elements, and top-level
  // editable regions. Include an element with its own text regardless of its
  // tag, which reaches a balance written into a plain div.
  const ownText = (el) => {
    for (const child of el.childNodes) {
      if (child.nodeType === 3 && squash(child.data)) { return true; }
    }
    return false;
  };
  const describable = (el) => {
    if (operatorOf(el)) { return false; }
    const tag = tagOf(el);
    if (SKIPPED.includes(tag)) { return false; }
    if (tag === 'input' && typeOf(el) === 'hidden') { return false; }
    if (withinSecret(el)) { return false; }
    if (el.matches(options.selector)) { return true; }
    if (el.isContentEditable &&
        !(el.parentElement && el.parentElement.isContentEditable)) { return true; }
    return ownText(el);
  };

  const visible = (el) => {
    if (typeof el.checkVisibility === 'function') {
      return el.checkVisibility({ checkOpacity: true, checkVisibilityCSS: true });
    }
    const box = el.getBoundingClientRect();
    return box.width > 0 && box.height > 0;
  };

  // Every describable element, in document order, then every open shadow
  // root in the order its host appears. A query never crosses a shadow
  // boundary, so each element is visited exactly once. ``visit`` returns
  // false to stop the walk.
  const eachCandidate = (root, visit, onHost) => {
    for (const el of root.querySelectorAll('*')) {
      if (describable(el) && visit(el, root) === false) { return false; }
    }
    for (const el of root.querySelectorAll('*')) {
      if (operatorOf(el)) { continue; }
      if (el.shadowRoot) {
        if (onHost) { onHost(el, true); }
        if (eachCandidate(el.shadowRoot, visit, onHost) === false) { return false; }
      } else if (onHost) {
        onHost(el, false);
      }
    }
    return true;
  };

  // Describe a control without its record context, using its tag, role,
  // and accessible name. Two Approve buttons are alike.
  const likeOf = (el) => {
    const tag = tagOf(el);
    return tag + '|' + roleOf(el, tag, typeOf(el)) + '|' + nameOf(el, tag, typeOf(el));
  };

  // Describe the kind of value an element holds, omitting that value: the
  // definition under the same term, the cell in the same column, or the field
  // with the same label. Two member numbers fill the same slot.
  // A field whose label is drawn beside it but not linked to it, as many
  // form builders do, is placed by that label: the one label in the nearest
  // container that holds this field and no other. The search crosses a
  // shadow root to its host, so a field inside a custom element is placed by
  // the label its host's container shows. The field's name stays empty.
  const FIELDISH = 'input, select, textarea';
  const fieldsIn = (container) =>
    Array.from(container.querySelectorAll('*')).filter(item =>
      item.matches(FIELDISH) || (item.tagName.includes('-') && item.shadowRoot &&
        item.shadowRoot.querySelector(FIELDISH))).length;
  const groupLabelOf = (el) => {
    let at = el;
    for (let depth = 0; depth < 6; depth += 1) {
      const parent = at.parentElement ||
        (at.parentNode && at.parentNode.host ? at.parentNode.host : null);
      if (!parent) { return ''; }
      const labels = Array.from(parent.querySelectorAll('label, legend'))
        .filter(label => !label.control && !label.hasAttribute('for'));
      if (labels.length > 1) { return ''; }
      if (labels.length === 1) {
        // The field counts once, as itself or as its custom element's host.
        return fieldsIn(parent) === 1 ? cut(full(labels[0]), NAME) : '';
      }
      at = parent;
    }
    return '';
  };

  const slotOf = (el) => {
    const tag = tagOf(el);
    // A link or a value that is all a table or definition cell holds, such
    // as a receipt number's link, sits in the slot of that cell.
    const cell = el.parentElement;
    if (!['td', 'th', 'dd'].includes(tag) && cell &&
        ['td', 'dd'].includes(tagOf(cell)) && cell.children.length === 1 &&
        text(cell) === text(el)) {
      const held = slotOf(cell);
      return tag + held.slice(tagOf(cell).length);
    }
    if (tag === 'dd') {
      let term = el.previousElementSibling;
      while (term && tagOf(term) !== 'dt') { term = term.previousElementSibling; }
      return 'dd|' + (term ? text(term) : '');
    }
    if (tag === 'td' || tag === 'th') {
      const table = el.closest('table');
      const head = table && table.tHead && table.tHead.rows[0];
      const column = head && head.cells[el.cellIndex];
      if (column) { return tag + '|' + text(column); }
      // A key and value table names each value by the header cell that
      // starts its row, as a person reads it.
      const row = el.parentElement;
      const first = row && row.cells && row.cells[0];
      const header = first && first !== el && tagOf(first) === 'th';
      if (tag === 'td' && header && text(first)) {
        return 'td|' + text(first);
      }
      // A column with no header is named by its position, marked with #
      // so no reader takes the number for text the page showed.
      return tag + '#' + el.cellIndex;
    }
    if (['gridcell', 'rowheader'].includes(el.getAttribute('role'))) {
      const grid = el.closest('[role="grid"], [role="treegrid"], [role="table"]');
      const index = el.getAttribute('aria-colindex');
      if (grid && index && Number.isInteger(Number(index)) && Number(index) > 0) {
        const columns = Array.from(grid.querySelectorAll('[role="columnheader"]'))
          .filter(header => header.getAttribute('aria-colindex') === index &&
            header.closest('[role="grid"], [role="treegrid"], [role="table"]')
              === grid);
        if (columns.length === 1 && text(columns[0])) {
          return tag + '|' + text(columns[0]);
        }
      }
    }
    if (FIELDS.includes(tag)) {
      return tag + '|' + (nameOf(el, tag, typeOf(el)) || groupLabelOf(el));
    }
    // Inline values often follow a separate label. This names the slot,
    // never the record or a permission, and the recorder still requires
    // explicit storage permission for the label.
    if (['strong', 'b', 'small', 'span'].includes(tag)) {
      const before = el.previousElementSibling;
      if (before && ['span', 'label'].includes(tagOf(before)) &&
          !before.querySelector('input, select, textarea, button, a')) {
        const label = text(before);
        if (label) { return tag + '|' + label; }
      }
    }
    const label = el.getAttribute('aria-label') || el.getAttribute('class') ||
      el.getAttribute('title') || '';
    return tag + '|' + cut(squash(label), TEXT);
  };

  // The element that bounds one record, for a row or a smallest shared
  // container. The body is never a record.
  const boundaryOf = (el, source, relation) => {
    if (relation === 'row') {
      const row = el.closest(ROW);
      return row && row.contains(source) ? row : null;
    }
    const doc = el.ownerDocument;
    let at = el.parentElement;
    while (at && at !== doc.body && at !== doc.documentElement) {
      if (at.contains(source)) { return at; }
      at = at.parentElement;
    }
    return null;
  };

  const countAlike = (boundary, el, describe) => {
    const wanted = describe(el);
    let found = 0;
    for (const other of boundary.querySelectorAll(tagOf(el))) {
      if (describe(other) === wanted) { found += 1; }
    }
    return found;
  };

  // The value a form field holds, or null for anything that is not a field.
  // A checkbox or radio holds its checked state, and a select its chosen
  // options. Callers decide secrecy first and never call this on a secret.
  const valueOf = (el, tag) => {
    if (tag === 'select') {
      return cut(Array.from(el.selectedOptions || []).map(full).join(', '), NAME);
    }
    if (tag === 'input' && ['checkbox', 'radio'].includes(typeOf(el))) {
      return el.checked ? 'checked' : 'not checked';
    }
    if (tag === 'input' || tag === 'textarea') {
      return cut(squash(el.value || ''), NAME);
    }
    return null;
  };

  // Return what an element shows, either a field value or its accessible name.
  // This is the one reading used by record evidence, by read, and by the
  // checks a finished run is verified against.
  const shownOf = (el) => {
    const tag = tagOf(el);
    const held = valueOf(el, tag);
    if (held !== null) { return held; }
    if (el.isContentEditable) { return full(el); }
    return nameOf(el, tag, typeOf(el));
  };

  // Read record evidence the way the collector reports it, and say whether
  // it still identifies the record the target belongs to.
  const evidenceOf = (el, source, relation) => {
    const none = { secret: false, value: null, related: false };
    if (!source || !source.isConnected) { return none; }
    if (concealed(source)) { return { secret: true, value: null, related: false }; }
    const value = shownOf(source);
    if (relation === 'labelled') {
      const named = squash((el.getAttribute('aria-describedby') || '') + ' ' +
                           (el.getAttribute('aria-labelledby') || '')).split(' ');
      const related = !!source.id && named.includes(source.id);
      return { secret: false, value: value, related: related };
    }
    const boundary = boundaryOf(el, source, relation);
    if (!boundary) { return { secret: false, value: value, related: false }; }
    const alone = countAlike(boundary, el, likeOf) === 1 &&
      countAlike(boundary, source, slotOf) === 1;
    return { secret: false, value: value, related: alone };
  };

  // Match an element to a target with the functions used by the observation.
  // Never compare a secret field's value attribute, so a guessed locator
  // cannot test what the field holds.
  // The cheap comparisons come first, so a page of many elements is not
  // named and scoped element by element.
  const namesElement = (el, target) => {
    const tag = tagOf(el);
    const type = typeOf(el);
    let named;
    if (target.kind === 'ax') {
      named = roleOf(el, tag, type) === target.role &&
        nameOf(el, tag, type) === target.name;
    } else if (tag !== target.tag) {
      named = false;
    } else if (target.attribute === 'text') {
      named = nameOf(el, tag, type) === target.value;
    } else if (target.attribute === 'slot') {
      named = slotOf(el) === target.value;
    } else if (target.attribute === 'css-class') {
      named = el.classList.contains(target.value);
    } else if (target.attribute === 'value' && secretOf(el)) {
      named = false;
    } else {
      named = attributeOf(el, target.attribute) === target.value;
    }
    return named && (!target.scope || sameScope(scopeOf(el), target.scope));
  };
  const matchesControl = (el, expected) => {
    if (!el.isConnected || concealed(el)) { return false; }
    const tag = tagOf(el), type = typeOf(el);
    if (tag !== expected.tag || roleOf(el, tag, type) !== expected.role ||
        nameOf(el, tag, type) !== expected.name) { return false; }
    const scope = scopeOf(el);
    if (Boolean(scope) !== Boolean(expected.scope)) { return false; }
    if (scope && (scope.kind !== expected.scope.kind ||
        scope.name !== expected.scope.name ||
        JSON.stringify(scope.names || []) !== JSON.stringify(expected.rowNames) ||
        JSON.stringify(scope.columns || []) !== JSON.stringify(expected.rowColumns))) {
      return false;
    }
    return slotOf(el) === expected.slot &&
      expected.attributes.every(([name, value]) => attributeOf(el, name) === value) &&
      expected.classes.every(name => el.classList.contains(name));
  };
"""

_FORMS = """
  // Native form submission, as the browser defines it. A submit button
  // submits its form owner. Enter in a text-like input activates the form's
  // default button, the first submit button among the form's elements; with
  // no default button, it submits only when the form has at most one such
  // input. Anything else a page does with Enter is its own script, which
  // nothing here can see.
  const TEXT_LIKE = ['text', 'search', 'url', 'tel', 'email', 'password', 'date',
                    'month', 'week', 'time', 'datetime-local', 'number'];
  const formOf = (el) =>
    ('form' in el && el.tagName !== 'FORM') ? el.form : el.closest('form');
  const submitter = (el) => {
    const tag = tagOf(el);
    if (tag === 'button') { return el.type === 'submit'; }
    return tag === 'input' && (el.type === 'submit' || el.type === 'image');
  };
  // The same submission described by what the page shows rather than by
  // element, so it can be recognised after a reload or a redraw. A form is
  // its name, id, label, action, and method, and the labels of its fields in
  // order. A submission is its form and the button that performs it.
  const formKeys = new Map();
  const formAs = (form) => {
    if (!form) { return ''; }
    if (!formKeys.has(form)) {
      const own = ['name', 'id', 'aria-label', 'action', 'method']
        .map((name) => cut(squash(form.getAttribute(name) || ''), TEXT));
      const fields = Array.from(form.elements)
        .filter((field) => !submitter(field))
        .map((field) =>
          tagOf(field) + ':' + nameOf(field, tagOf(field), typeOf(field)));
      formKeys.set(form, 'form|' + own.join('|') + '|' + fields.join(','));
    }
    return formKeys.get(form);
  };
  const submissionAs = (pair) => {
    if (!pair) { return ''; }
    const [form, button] = pair;
    return formAs(form) + '>' + (button ? likeOf(button) : '');
  };
  const submitsPair = (el) => {
    const form = formOf(el);
    return form && submitter(el) ? [form, el] : null;
  };
  const enterPair = (el) => {
    const form = formOf(el);
    if (!form) { return null; }
    if (submitter(el)) { return [form, el]; }
    if (tagOf(el) !== 'input' || !TEXT_LIKE.includes(el.type)) { return null; }
    const fields = Array.from(form.elements);
    const button = fields.find(submitter);
    if (button) { return button.disabled ? null : [form, button]; }
    const blocking = fields.filter(
      (f) => tagOf(f) === 'input' && TEXT_LIKE.includes(f.type));
    return blocking.length <= 1 ? [form, null] : null;
  };
"""
"""Describe the native form submission a control performs for the collector and
the recorder, so a person's input is described as an observation describes it.
"""

_PICKERS = """
  const containingForm = (el) => {
    for (let anchor = el; anchor; anchor = anchor.getRootNode().host) {
      const form = anchor.closest('form');
      if (form) { return form; }
    }
    return null;
  };
  const selectedState = (field, rule) => {
    if (!visible(field) || secretOf(field) || withinSecret(field)) { return false; }
    if (rule.source === 'native') {
      return tagOf(field) === 'select' && field.selectedOptions.length === 1 &&
        !field.selectedOptions[0].disabled && field.value !== '';
    }
    if (roleOf(field, tagOf(field), typeOf(field)) !== 'combobox') { return false; }
    if (rule.source === 'host') {
      const root = field.getRootNode(), host = root.host;
      if (!host || fieldsIn(root) !== 1) { return false; }
      try { return host[rule.property] === true; } catch (_) { return false; }
    }
    if (field.getAttribute('aria-expanded') === 'true') { return false; }
    const id = field.getAttribute('aria-controls');
    if (!id || squash(id).includes(' ')) { return false; }
    const root = field.getRootNode();
    const list = root.getElementById(id) || field.ownerDocument.getElementById(id);
    if (!list) { return false; }
    const selected = Array.from(list.querySelectorAll('[aria-selected="true"]'));
    if (selected.length !== 1 || secretOf(selected[0]) || withinSecret(selected[0])) {
      return false;
    }
    return nameOf(selected[0], tagOf(selected[0]), typeOf(selected[0])) ===
      squash(field.value || field.getAttribute('aria-valuetext') || '');
  };
  const selectionsFor = (el) => {
    const rules = options.pickerRules || [];
    if (!rules.length) { return []; }
    const form = containingForm(el), fields = [];
    if (!form) { return []; }
    eachCandidate(form, field => {
      if (FIELDS.includes(tagOf(field))) { fields.push(field); }
    });
    const groups = new Map();
    for (const rule of rules) {
      const found = fields.filter(field => slotOf(field) === rule.slot);
      const ready = found.length === 1 && selectedState(found[0], rule);
      groups.set(rule.operation, (groups.get(rule.operation) ?? true) && ready);
    }
    return Array.from(groups.entries());
  };
"""
"""Read only operator-declared selection state, never infer it from a query."""

SCRIPT = (
    """
(options) => {
  const MAX = options.maxNodes;
  const SKIP = options.skip || 0;
  const COUNT_LIMIT = options.countLimit;
  const WITHIN = options.scope || null;
  const KNOWN = options.knownElements || [];

  // Elements the adapter already holds are named by their index in KNOWN.
  // Anything else is returned beside the report, so the adapter can issue it
  // an id. Nothing is written into the page to tell elements apart.
  const fresh = [];
  const freshIndex = new Map();
  const refOf = (el) => {
    if (!el) { return null; }
    const held = KNOWN.indexOf(el);
    if (held >= 0) { return ['known', held]; }
    if (!freshIndex.has(el)) {
      freshIndex.set(el, fresh.length);
      fresh.push(el);
    }
    return ['fresh', freshIndex.get(el)];
  };
"""
    + _HELPERS
    + _FORMS
    + _PICKERS
    + """
  const refs = (pair) => pair ? [refOf(pair[0]), refOf(pair[1])] : null;
  const ATTRIBUTES = options.attributes;

  const interactionsOf = (el, tag, type, role) => {
    const found = [];
    const clickable = ['button', 'link', 'checkbox', 'radio', 'tab', 'menuitem',
                       'option', 'canvas', 'switch', 'treeitem'];
    if (clickable.includes(role) || tag === 'button' || tag === 'a' ||
        tag === 'summary' || el.hasAttribute('onclick')) {
      found.push('click');
    }
    const pressed = ['submit', 'button', 'reset', 'checkbox', 'radio', 'image'];
    const editable = (tag === 'input' && !pressed.includes(type)) ||
      tag === 'textarea' || (el.isContentEditable && !FIELDS.includes(tag));
    if (editable) { found.push('type'); }
    if (tag === 'select') { found.push('select'); }
    if (FIELDS.includes(tag) || editable) { found.push('press_key'); }
    found.push('read');
    return found;
  };

  // Boundaries use the same document identity as controls. Observation pages
  // can then be combined without connecting rows from different collections.
  const rowOf = (el) => {
    const row = el.closest(ROW);
    return refOf(row);
  };
  const ancestorsOf = (el) => {
    const found = [];
    const doc = el.ownerDocument;
    let at = el.parentElement;
    while (at && at !== doc.body && at !== doc.documentElement && found.length < 32) {
      found.push(refOf(at));
      at = at.parentElement;
    }
    return found;
  };
  const inView = (el) => {
    const box = el.getBoundingClientRect();
    return box.bottom > 0 && box.right > 0 &&
      box.top < window.innerHeight && box.left < window.innerWidth;
  };

  const nodes = [];
  let truncated = false;
  let closedRoots = 0;
  let shadowRoots = 0;
  let seen = 0;

  const describe = (el, root) => {
    const tag = tagOf(el);
    const type = typeOf(el);
    const role = roleOf(el, tag, type);
    const secret = secretOf(el);
    const attributes = [];
    for (const attribute of ATTRIBUTES) {
      if (secret && attribute === 'value') { continue; }
      const value = attributeOf(el, attribute);
      if (value) { attributes.push([attribute, value]); }
    }
    nodes.push({
      role: role,
      name: nameOf(el, tag, type),
      value: secret ? null : valueOf(el, tag),
      options: !secret && tag === 'select' ? Array.from(el.options).slice(0, 100)
        .map(option => ({label: cut(squash(option.label), NAME),
          value: cut(option.value, NAME),
          disabled: option.disabled || !!option.closest('optgroup[disabled]'),
          selected: option.selected})) : [],
      optionsComplete: secret || tag !== 'select' || el.options.length <= 100,
      tag: tag,
      attributes: attributes,
      classes: Array.from(el.classList).slice(0, 32),
      interactions: interactionsOf(el, tag, type, role),
      scope: scopeOf(el),
      context: contextOf(el),
      selections: selectionsFor(el),
      row: rowOf(el),
      ancestors: ancestorsOf(el),
      slot: slotOf(el),
      enabled: !el.disabled,
      visible: true,
      inView: inView(el),
      shadow: root !== el.ownerDocument,
      control: refOf(el),
      form: refOf(formOf(el)),
      submits: refs(submitsPair(el)),
      enter: refs(enterPair(el)),
      submitsAs: submissionAs(submitsPair(el)),
      enterAs: submissionAs(enterPair(el)),
      formAs: formAs(formOf(el)),
      secret: secret
    });
  };

  // A window over the frame: the first SKIP visible candidates are counted
  // and passed over, the next MAX are described, and the rest are counted up
  // to COUNT_LIMIT so the adapter can say how much was left out.
  eachCandidate(document, (el, root) => {
    if (!visible(el)) { return true; }
    if (WITHIN && !sameScope(scopeOf(el), WITHIN)) { return true; }
    seen += 1;
    if (seen <= SKIP) { return true; }
    if (nodes.length >= MAX) {
      truncated = true;
      return seen < COUNT_LIMIT;
    }
    describe(el, root);
    return true;
  }, (el, open) => {
    if (open) { shadowRoots += 1; }
    else if (typeof el.attachShadow === 'function' &&
             el.hasAttribute('data-closed-shadow')) { closedRoots += 1; }
  });
  const gone = [];
  KNOWN.forEach((el, index) => { if (!el.isConnected) { gone.push(index); } });
  return {
    report: {
      nodes: nodes,
      truncated: truncated,
      total: seen,
      counted: seen < COUNT_LIMIT,
      closedRoots: closedRoots,
      shadowRoots: shadowRoots,
      gone: gone,
      title: cut(squash(document.title || ''), TEXT),
      scrollX: Math.round(window.scrollX),
      scrollY: Math.round(window.scrollY),
      viewportWidth: Math.round(window.innerWidth),
      viewportHeight: Math.round(window.innerHeight)
    },
    fresh: fresh
  };
}
"""
)

RESOLVE = (
    """
(options) => {
"""
    + _HELPERS
    + """
  const found = [];
  eachCandidate(document, (el) => {
    if (!visible(el) || !namesElement(el, options.target)) { return true; }
    found.push(el);
    return found.length < options.limit;
  });
  return found;
}
"""
)
"""Every visible element a target names in this frame, in observation order.

The walk, the order, and every comparison are the observation's own, so the
index of a match here is the index an observation would give it. The caller
decides what one, none, or several matches mean.
"""

CONTEXT = (
    """
(el, options) => {
"""
    + _HELPERS
    + """
  return contextOf(el);
}
"""
)
"""The context path of one element, asked immediately before acting."""

CONTROL = (
    "(el, options) => {"
    + _HELPERS
    + "return matchesControl(el, options.observedControl); }"
)
"""Compare one element with its observed reusable metadata before screen input."""

SUBMISSION = (
    """
(el, options) => {
"""
    + _HELPERS
    + _FORMS
    + """
  return { submits: submissionAs(submitsPair(el)) || '',
           enter: submissionAs(enterPair(el)) || '' };
}
"""
)
"""The native submissions one element performs now, asked immediately before input."""

EVIDENCE = (
    """
(el, options) => {
"""
    + _HELPERS
    + """
  return evidenceOf(el, options.source, options.relation);
}
"""
)
"""Read record evidence again, and report whether it still bounds the target's record.

The value is read the way the collector reports it, the field value for a
form field and the accessible name for anything else, so what the model chose
from an observation and what is checked here are the same string. A secret
field is refused before its value is read. For a row or a container, the
boundary must hold exactly one control like the target and one value like the
evidence, which is what stops one member's number from vouching for another
member's button in a shared section.
"""

READ = (
    """
(el, options) => {
"""
    + _HELPERS
    + """
  if (concealed(el)) { return { secret: true, value: null }; }
  return { secret: false, value: shownOf(el) };
}
"""
)
"""Read what one element shows, or refuse when it holds a secret.

A field reports its value, a checkbox or radio its checked state, a select
its chosen options, and anything else its accessible name. The secret check
comes first and the value of a secret field is never read. An element inside
a secret editable region, or one that contains such a region, is refused too,
because its reading could include text the region holds.
"""

GUARD = (
    """
(el, options) => {
"""
    + _HELPERS
    + _FORMS
    + _PICKERS
    + """
  const win = el.ownerDocument.defaultView;
  const guard = { blocked: false, started: false, remove: () => {} };
  win[options.guardKey] = guard;
  const deadline = performance.now() + options.lifetime;
  let pixels = null;
  let checkedInput = false;
  if (options.paintedBinding) {
    if (secretOf(el) || withinSecret(el) || tagOf(el) !== 'canvas') {
      guard.blocked = 'operation';
    } else {
      try { pixels = el.getContext('2d').getImageData(0, 0, el.width, el.height); }
      catch (_) { guard.blocked = 'operation'; }
    }
  }
  const operationChanged = () => {
    if (options.observedControl && !matchesControl(el, options.observedControl)) {
      return true;
    }
    if (options.requiredSelections && options.requiredSelections.length) {
      const selected = new Map(selectionsFor(el));
      if (!options.requiredSelections.every(name => selected.get(name) === true)) {
        return true;
      }
    }
    if (options.operationRules) {
      const role = roleOf(el, tagOf(el), typeOf(el));
      const name = nameOf(el, tagOf(el), typeOf(el)), context = contextOf(el);
      const submits = Boolean(options.operationKind === 'click' ? submitsPair(el) :
        options.operationKind === 'press_key' && options.operationKey === 'Enter'
          ? enterPair(el) : null);
      const matches = options.operationRules.filter(rule =>
        (rule.role === '*' || rule.role === role) &&
        (rule.target === '*' || rule.target === name) &&
        rule.context.every(item => context.includes(item)) &&
        (rule.submission === 'any' || submits === (rule.submission === 'native')));
      const current = matches.length === 1 ? matches[0].name : '';
      if (current !== options.operationName) { return true; }
    }
    if (!pixels || (options.bindingOnce && checkedInput)) { return false; }
    if (el.width !== pixels.width || el.height !== pixels.height) { return true; }
    try {
      const current = el.getContext('2d').getImageData(0, 0, el.width, el.height).data;
      return current.some((value, index) => value !== pixels.data[index]);
    } catch (_) { return true; }
  };
  const stale = () => {
    if (!options.check) { return false; }
    const seen = evidenceOf(el, options.source, options.relation);
    return seen.secret || !seen.related || seen.value !== options.value;
  };
  const check = (event) => {
    // A script's own element.click() is not the driver's input. Only real
    // input is judged, so the page's own behaviour is left alone.
    if (!event.isTrusted) { return; }
    if (performance.now() > deadline) {
      if (options.operationRules || options.paintedBinding || options.observedControl) {
        guard.blocked = 'operation';
      } else { guard.remove(); return; }
    }
    const reaches = event.composedPath().includes(el);
    if (!guard.blocked && !reaches) { guard.blocked = 'target'; }
    if (!guard.blocked && stale()) { guard.blocked = 'record'; }
    if (!guard.blocked && operationChanged()) { guard.blocked = 'operation'; }
    if (guard.blocked) {
      if (guard.started &&
          (options.operationRules || options.paintedBinding ||
           options.observedControl)) {
        guard.blocked = 'uncertain_operation';
      }
      // Once anything is cancelled, everything after it is, including a
      // retry by the driver, until the adapter removes the guard.
      event.preventDefault();
      event.stopImmediatePropagation();
      return;
    }
    checkedInput = true;
    guard.started = true;
    if (event.type === options.last) { guard.remove(); }
  };
  guard.remove = () => {
    for (const type of options.events) { win.removeEventListener(type, check, true); }
  };
  for (const type of options.events) { win.addEventListener(type, check, true); }
  return true;
}
"""
)
"""Check the record again inside the page, as each input event reaches the target.

The listener sits on the target's window in the capture phase, so it runs
before any handler on the target or its ancestors. The check and the event
are one task in the page, with nothing between them. A record that changed
while the driver waited to click is caught here, and every remaining event of
that input is cancelled. So is input of that kind that reaches any other
element of the document during the action, which is what a replaced target or
a moved focus looks like. ``blocked`` says which it was. Only trusted events
are judged: an event a page script creates is not the driver's input. After a
cancellation the guard keeps cancelling until the adapter removes it, so a
retry by the driver cannot slip through. Otherwise it removes itself after
the last event of an input that passed, or ignores events once its lifetime
has passed.
"""

OPERATION = (
    "(el, options) => {"
    + _HELPERS
    + _FORMS
    + _PICKERS
    + """
      return { role: roleOf(el, tagOf(el), typeOf(el)),
               name: nameOf(el, tagOf(el), typeOf(el)), context: contextOf(el),
               submits: submissionAs(submitsPair(el)),
               enter: submissionAs(enterPair(el)),
               selections: selectionsFor(el),
               secret: secretOf(el) || withinSecret(el) };
    }
    """
)
"""Read the same operation metadata the collector used, immediately before input."""

FOCUSED = """
(el) => {
  let active = el.ownerDocument.activeElement;
  while (active && active.shadowRoot && active.shadowRoot.activeElement) {
    active = active.shadowRoot.activeElement;
  }
  return active === el;
}
"""
"""Report whether this element holds focus in its own document."""

KNOWN_AS = "(el, known) => known.indexOf(el)"
"""The index of this element among the elements the adapter already holds."""

PROTECT = "(el, mark) => { el.setAttribute(mark.name, mark.token); }"
"""Mark a field before a declared secret is typed into it."""

HELD = """
([el, mark]) => (el.isConnected ? 'connected' : 'removed') +
  (el.getAttribute(mark.name) === mark.token ? ' marked' : ' unmarked')
"""
"""Report whether a protected field remains in its document with its mark.

The answer is a string, which the driver hands back without asking the page
again, so the adapter can ask it through a call with a time limit.
"""

COVERS = "(els, held) => els.includes(held)"
"""Report whether the mask locator resolves to the protected element."""

PROMPTS = """
(selector) => {
  let found = 0;
  for (const el of document.querySelectorAll(selector)) {
    if (el.checkVisibility({ checkOpacity: true, checkVisibilityCSS: true })) {
      found += 1;
    }
  }
  return 'prompts ' + found;
}
"""
"""How many credential fields a frame shows, without reading any of them.

Only the selector and visibility decide. No value is read, not even whether
one is empty. The count comes back as a string for the same reason as
``HELD``'s answer.
"""

RECORDED = """
(request) => {
  const recorder = window[request.marker];
  return typeof recorder === 'function' ? recorder(request) : 'missing';
}
"""
"""Ask a frame's recorder for its document and last event, or to mark one field.

The recorder answers ``missing`` when the frame's document has none, and
``inert`` when the frame may not run script, so its listeners never run. Every
answer is either a marked element or a non-empty string, so a call that waits
for a true answer returns on its first evaluation.
"""

RECORDER = (
    """
(options) => {
  const send = window[options.binding];
  try { delete window[options.binding]; } catch (error) { /* already gone */ }
  if (typeof send !== 'function') { return; }
  const TOKEN = options.token;
  const now = Date.now.bind(Date);
  const later = window.setTimeout.bind(window);
  const listen = window.addEventListener.bind(window);
  const DOC = (() => {
    const bytes = new Uint32Array(4);
    crypto.getRandomValues(bytes);
    return Array.from(bytes, (part) => part.toString(16)).join('');
  })();
"""
    + _HELPERS
    + _FORMS
    + """
  const NOT_TEXT = ['checkbox', 'radio', 'submit', 'button', 'reset', 'image',
                    'file', 'range', 'color', 'hidden'];
  const CONTROL = 'a[href], button, input, select, textarea, summary, option, ' +
    'label, canvas, [role], [onclick], [tabindex], [contenteditable]';
  const MODIFIERS = ['Control', 'Meta', 'Shift', 'Alt', 'AltGraph'];

  // The element an editable region's text belongs to is its editing host,
  // the outermost editable element, which is what receives input events and
  // what a mark is placed on.
  const hostOf = (el) => {
    let at = el;
    while (at && at.isContentEditable && at.parentElement &&
           at.parentElement.isContentEditable) {
      at = at.parentElement;
    }
    return at;
  };
  const textEntry = (el) => {
    const tag = tagOf(el);
    if (tag === 'textarea') { return true; }
    if (tag === 'input') { return !NOT_TEXT.includes(typeOf(el)); }
    return el.isContentEditable;
  };
  const choice = (el) => tagOf(el) === 'select' ||
    (tagOf(el) === 'input' && ['checkbox', 'radio'].includes(typeOf(el)));
  const elementOf = (event) => {
    const first = event.composedPath()[0];
    return first && first.nodeType === 1 ? first : null;
  };
  // A click names the control it reached, not the text or icon inside it.
  const controlOf = (event) => {
    for (const node of event.composedPath()) {
      if (node.nodeType === 1 && node.matches(CONTROL)) { return node; }
    }
    return elementOf(event);
  };

  let seq = 0;
  let alive = false;
  let dirty = false;
  let beating = false;
  let keyed = false;
  let touched = -Infinity;
  let scrolled = -Infinity;
  let inputs = 0;
  let pressed = null;
  let claims = [];
  const edited = new Map();
  const targets = new Map();
  let reported = new WeakMap();
  let commanded = new WeakMap();
  let heard = false;
  listen(options.probe, () => { heard = true; }, true);

  // The browser's closing of a native dialog does not say which button
  // closed it. The page's own call does, when it returns, so each call is
  // noted with the dialog's kind, its message, and whether it was accepted.
  // A prompt's text is never kept. A dialog no page function opened, such
  // as the one before a page unloads, is not seen here.
  const answers = [];
  const answering = (name, accepted) => {
    const original = window[name];
    if (typeof original !== 'function') { return; }
    window[name] = function (...args) {
      const result = original.apply(this, args);
      answers.push({ kind: name, message: String(args[0] ?? ''),
                     accepted: accepted(result) });
      if (answers.length > 50) { answers.shift(); }
      return result;
    };
  };
  answering('alert', () => true);
  answering('confirm', (result) => result === true);
  answering('prompt', (result) => result !== null);

  const post = (payload) => {
    payload.token = TOKEN;
    payload.doc = DOC;
    try {
      const sent = send(payload);
      if (sent && typeof sent.catch === 'function') { sent.catch(() => {}); }
    } catch (error) { /* the document is going away */ }
  };
  const beat = () => {
    beating = false;
    if (!dirty) { return; }
    dirty = false;
    post({ kind: 'heartbeat', last: seq, at: now() });
  };
  // Secrecy and editability are decided before anything else is read. A
  // secret element is never named, and neither is an editable region, whose
  // name is its own text, which is what was typed into it. A field is named
  // by its label, which the collector never takes from its value.
  const describe = (el, named) => {
    const tag = tagOf(el);
    const type = typeOf(el);
    const secret = secretOf(el) || withinSecret(el);
    const editable = textEntry(el);
    const role = roleOf(el, tag, type);
    // A name made of an element's text includes any editable region inside
    // it, so an element that is or holds one is never named.
    const holdsEditable = el.isContentEditable ||
      !!el.querySelector('[contenteditable]:not([contenteditable="false"])');
    const said = named && !secret && !holdsEditable;
    const name = said ? cut(nameOf(el, tag, type), NAME) : '';
    return { tag: cut(tag, 64), role: cut(role, 64), name: name,
             type: cut(type, 32), secret: secret, editable: editable };
  };
  const BLANK = { tag: '', role: '', name: '', type: '', secret: false,
                  editable: false };
  // Keep the element that received a person's input so the adapter can report
  // which observed control it was, and which native submission it could
  // perform, the way an observation says it for the automation's own input.
  // It stays in the page until the adapter asks.
  const OPERATED = ['click', 'key', 'edit', 'select'];
  const remember = (number, el) => {
    const tag = tagOf(el);
    const type = typeOf(el);
    const holdsEditable = el.isContentEditable ||
      !!el.querySelector('[contenteditable]:not([contenteditable="false"])');
    const form = formOf(el);
    targets.set(number, {
      el: el, form: form, submits: submitsPair(el), enter: enterPair(el),
      submitsAs: submissionAs(submitsPair(el)), enterAs: submissionAs(enterPair(el)),
      formAs: formAs(form), tag: cut(tag, 64), role: cut(roleOf(el, tag, type), 64),
      name: holdsEditable ? '' : cut(nameOf(el, tag, type), NAME)
    });
    if (targets.size > 1000) { targets.delete(targets.keys().next().value); }
  };
  const emit = (kind, el, detail, named, markable, claim) => {
    if (operatorOf(el)) { return 0; }
    // The automation's own input is reported once per kind for each input
    // call, and only so the adapter can tell its own input from a person's.
    if (claim) {
      if (claim.sent.includes(kind)) { return 0; }
      claim.sent.push(kind);
    }
    seq += 1;
    if (!claim && el && OPERATED.includes(kind)) { remember(seq, el); }
    const described = el ? describe(el, named) : BLANK;
    if (markable) { described.editable = true; }
    post(Object.assign({ seq: seq, at: now(), kind: kind, detail: detail,
                         auto: !!claim, url: cut(location.href, 2000) },
                       described));
    dirty = true;
    if (!beating) { beating = true; later(beat, 250); }
    return seq;
  };
  // A field a person typed into is marked at once, in the same event, so no
  // reading or picture taken before the adapter hears of the edit can see it.
  const protect = (el) => {
    if (MARK) { el.setAttribute(MARK.name, MARK.token); }
  };

  // The adapter says, just before each input call, which kinds of input it
  // sends and to which element. An event is the automation's only when a
  // claim covers its kind and its element; anything else is a person's,
  // however close in time. A claim with no element covers the focused
  // element for keys, or anywhere in the document for a scroll or a drag.
  const coveredBy = (claim, el, path) => {
    if (claim.anywhere) { return true; }
    const own = claim.el;
    if (!own || !el) { return false; }
    if (own === document.body || own === document.documentElement) {
      return own === el;
    }
    if (own === el || own.contains(el) || path.includes(own)) { return true; }
    if (tagOf(own) === 'label' && own.control === el) { return true; }
    return tagOf(el) === 'form' && (own.form === el || el.contains(own));
  };
  const claimOf = (kind, el, event) => {
    const at = now();
    claims = claims.filter((claim) => claim.until >= at);
    const path = event ? event.composedPath() : [];
    for (const claim of claims) {
      if (claim.kinds.includes(kind) && coveredBy(claim, el, path)) {
        claim.counts[kind] = (claim.counts[kind] || 0) + 1;
        return claim;
      }
    }
    return null;
  };
  const focused = () => {
    let at = document.activeElement;
    while (at && at.shadowRoot && at.shadowRoot.activeElement) {
      at = at.shadowRoot.activeElement;
    }
    return at;
  };
  const fieldValue = (el) => {
    const tag = tagOf(el);
    return tag === 'input' || tag === 'textarea' ? el.value : null;
  };

  // A press without a click, such as a right button, a drag, or a press
  // the page answered on its own, is still something a person did there.
  listen('pointerdown', (event) => {
    if (!event.isTrusted) { return; }
    touched = now();
    const el = elementOf(event);
    if (el) {
      reported.delete(hostOf(el));
      commanded = new WeakMap();
    }
    const control = controlOf(event);
    pressed = { el: control, claim: claimOf('click', control, event), clicked: false };
  }, true);
  listen('pointerup', (event) => {
    if (!event.isTrusted) { return; }
    const held = pressed;
    later(() => {
      if (held && pressed === held && !held.clicked && held.el) {
        emit('click', held.el, '', true, false, held.claim);
      }
      if (pressed === held) { pressed = null; }
    }, 0);
  }, true);
  listen('click', (event) => {
    if (!event.isTrusted) { return; }
    touched = now();
    if (pressed) { pressed.clicked = true; }
    const el = controlOf(event);
    if (el) { emit('click', el, '', true, false, claimOf('click', el, event)); }
  }, true);

  // Every key a person presses is reported as a category. The category
  // is for the journal and says nothing about whether the key was harmless:
  // a page can act on any key, cancelled or not. A character or a deletion
  // in a text field is the field's edit, unless the field did not change,
  // when the page did something else with it. Any other key that is not a
  // movement key, a function key or a character outside a field, is a
  // command, reported once per focus so a count says nothing about length.
  const MOVES = ['Tab', 'ArrowUp', 'ArrowDown', 'ArrowLeft', 'ArrowRight',
                 'Home', 'End', 'PageUp', 'PageDown'];
  const keyAs = (el, detail, claim) => {
    if (detail === 'command' && !claim) {
      const host = el ? hostOf(el) : document;
      if (commanded.get(host)) { return; }
      commanded.set(host, true);
    }
    emit('key', el, detail, true, false, claim);
  };
  listen('keyup', () => { keyed = false; }, true);
  listen('keydown', (event) => {
    if (!event.isTrusted) { return; }
    keyed = true;
    touched = now();
    if (MODIFIERS.includes(event.key)) { return; }
    const el = elementOf(event);
    const claim = claimOf('key', el, event);
    const inField = !!el && (textEntry(hostOf(el)) || tagOf(el) === 'select');
    const shortcut = (event.ctrlKey || event.metaKey) &&
      !event.getModifierState('AltGraph');
    if (event.key === 'Enter') { keyAs(el, 'confirm', claim); return; }
    if (event.key === 'Escape') { keyAs(el, 'cancel', claim); return; }
    if (shortcut) { keyAs(el, 'shortcut', claim); return; }
    if (MOVES.includes(event.key)) { keyAs(el, 'move', claim); return; }
    if (!inField) { keyAs(el, 'command', claim); return; }
    const before = inputs;
    later(() => {
      if (inputs === before) { keyAs(el, 'command', claim); }
    }, 0);
  }, true);

  // One edit per element each time it takes focus, however much is typed,
  // so the number of edits says nothing about the length of a value. A
  // value set without a key press, the way a driver fills a field, is
  // followed by one more edit if keys are then typed into the same field.
  // An edit with no key or pointer input just before it may be the page's
  // own script, so it is reported as unprompted: the field is still marked,
  // but the edit does not count as a person changing the session. A text
  // field inside a closed shadow root reaches this listener as its host,
  // which is remembered and marked instead, so its box is masked. The
  // automation's own edits never stand in for a person's.
  listen('input', (event) => {
    if (!event.isTrusted) { return; }
    inputs += 1;
    const first = elementOf(event);
    const withKey = keyed;
    keyed = false;
    if (!first || choice(first)) { return; }
    const el = hostOf(first);
    const markable = textEntry(el) ||
      !el.matches('input, select, textarea, option, button');
    const claim = claimOf('edit', el, event);
    if (claim) {
      // Kept like any edit, so the adapter can still mark the field if the
      // session was not the automation's when this arrived.
      const number = emit('edit', el, '', true, markable, claim);
      if (number && markable) { edited.set(number, el); }
      return;
    }
    const before = reported.get(el);
    if (before !== 'keyed' && (before !== 'keyless' || withKey)) {
      reported.set(el, withKey ? 'keyed' : 'keyless');
      const prompted = withKey || now() - touched <= 1000;
      const number = emit('edit', el, prompted ? '' : 'unprompted', true,
                          markable, null);
      if (markable) { edited.set(number, el); }
    }
    // Marked after it is described, so the report still names the field.
    if (markable) { protect(el); }
  }, true);
  // A page's own script can toggle a box or submit a form, and the event
  // that causes is trusted too, so either counts only right after input.
  listen('change', (event) => {
    if (!event.isTrusted || now() - touched > 1000) { return; }
    const el = elementOf(event);
    if (el && choice(el)) {
      emit('select', el, '', true, false, claimOf('select', el, event));
    }
  }, true);
  listen('submit', (event) => {
    if (!event.isTrusted || now() - touched > 1000) { return; }
    const el = elementOf(event);
    if (el) { emit('submit', el, '', false, false, claimOf('submit', el, event)); }
  }, true);
  listen('focusout', (event) => {
    const el = elementOf(event);
    if (el) {
      reported.delete(hostOf(el));
      commanded.delete(hostOf(el));
    }
  }, true);
  // A wheel event comes only from a device. A scroll event also comes from
  // the page scrolling itself, so it would credit a person with the page's
  // own movement. The direction is for the journal; whether the page acted
  // on the wheel is not something this script can tell.
  listen('wheel', (event) => {
    if (!event.isTrusted) { return; }
    const dx = event.deltaX;
    const dy = event.deltaY;
    const at = now();
    if ((!dx && !dy) || at - scrolled < 500) { return; }
    scrolled = at;
    const direction = Math.abs(dy) >= Math.abs(dx)
      ? (dy > 0 ? 'down' : 'up') : (dx > 0 ? 'right' : 'left');
    emit('scroll', null, direction, false, false, claimOf('scroll', null, event));
  }, { capture: true, passive: true });
  listen('pagehide', () => {
    post({ kind: 'heartbeat', last: seq, at: now() });
  }, true);

  // An input call's claim ends when the call returns. Input that reached
  // the claimed element and that the call itself cannot account for, more
  // keys or edits than it sends or a field left with a value it did not
  // set, may be a person's. It is reported as uncertain, and a field is
  // marked, since nothing can say whose it is.
  const release = (claim) => {
    const over = Object.keys(claim.limits)
      .filter((kind) => (claim.counts[kind] || 0) > claim.limits[kind]);
    if (claim.value !== null && claim.el && fieldValue(claim.el) !== null &&
        fieldValue(claim.el) !== claim.value && !over.includes('edit')) {
      over.push('edit');
    }
    for (const kind of over) {
      const el = claim.el || focused();
      if (kind === 'edit' && el) {
        const host = hostOf(el);
        protect(host);
        const number = emit('edit', host, 'uncertain', true, true, null);
        edited.set(number, host);
      } else if (el) {
        emit(kind, el, 'uncertain', true, false, null);
      }
    }
    return over.length ? 'released uncertain' : 'released';
  };

  const recorder = (request) => {
    if (!request || typeof request !== 'object') { return 'unknown'; }
    // A frame that may not run script still answers the adapter, but none of
    // the listeners above ever run in it, so it says it cannot record.
    // A listener the page erased, with document.open, cannot record either,
    // so the recorder checks that its own listener still hears an event.
    if (request.op === 'report') {
      heard = false;
      window.dispatchEvent(new Event(options.probe));
      return alive && heard ? 'report ' + DOC + ' ' + seq : 'inert';
    }
    if (request.op === 'claim') {
      const el = request.target || (request.anywhere ? null : focused());
      claims.push({ id: request.id, el: el, anywhere: !!request.anywhere,
                    kinds: request.kinds, limits: request.limits || {},
                    value: typeof request.value === 'string' ? request.value : null,
                    counts: {}, sent: [], until: now() + request.lifetime });
      return 'claimed';
    }
    // Map each event to the index of the observed control it reached and
    // report which controls remain in the document. Descriptions reflect the
    // moment of the event.
    if (request.op === 'identify') {
      const known = request.known || [];
      // An element the adapter never observed is -1; no element is null.
      const index = (el) => el ? known.indexOf(el) : null;
      const pairOf = (pair) => pair ? [index(pair[0]), index(pair[1])] : null;
      const found = (request.seqs || []).map((number) => {
        const held = targets.get(number);
        if (!held) { return null; }
        return { control: index(held.el), form: index(held.form),
                 submits: pairOf(held.submits), enter: pairOf(held.enter),
                 submitsAs: held.submitsAs, enterAs: held.enterAs,
                 formAs: held.formAs, tag: held.tag, role: held.role,
                 name: held.name };
      });
      const present = [];
      known.forEach((el, at) => { if (el.isConnected) { present.push(at); } });
      return JSON.stringify({ found: found, present: present });
    }
    // A message identifies an answer only when exactly one call matches.
    // Repeated messages collected together are ambiguous, never guessed.
    if (request.op === 'answered') {
      const limit = request.limit;
      const matches = answers.filter((held) => held.kind === request.kind &&
        held.message.slice(0, limit) === request.message);
      if (matches.length > 1) {
        for (let at = answers.length - 1; at >= 0; at -= 1) {
          if (matches.includes(answers[at])) { answers.splice(at, 1); }
        }
        return 'unknown';
      }
      for (let at = answers.length - 1; at >= 0; at -= 1) {
        const held = answers[at];
        if (held.kind === request.kind &&
            held.message.slice(0, limit) === request.message) {
          const same = (item) => item.kind === held.kind &&
            item.message.slice(0, limit) === request.message;
          for (let before = at; before >= 0; before -= 1) {
            if (same(answers[before])) { answers.splice(before, 1); }
          }
          return held.accepted ? 'accepted' : 'dismissed';
        }
      }
      return 'unknown';
    }
    if (request.op === 'release') {
      const claim = claims.find((held) => held.id === request.id);
      claims = claims.filter((held) => held !== claim);
      return claim ? release(claim) : 'released';
    }
    // A person handed the session starts every count again, so their typing
    // into a field the automation typed into is still reported, and nothing
    // the automation claimed covers what they do.
    if (request.op === 'forget') {
      reported = new WeakMap();
      commanded = new WeakMap();
      claims = [];
      return 'forgot';
    }
    if (request.doc !== DOC) { return 'moved'; }
    const el = edited.get(request.seq);
    if (!el) { return 'unknown'; }
    el.setAttribute(request.mark.name, request.mark.token);
    if (!el.isConnected) { return 'gone'; }
    return el.getAttribute(request.mark.name) === request.mark.token ? el : 'failed';
  };
  Object.defineProperty(window, options.marker, {
    value: recorder, writable: false, configurable: false, enumerable: false
  });
  later(() => { alive = true; }, 0);
  post({ kind: 'hello', last: 0, at: now() });
}
"""
)
"""Report what a person does in one frame, without reading what they type.

The adapter installs this in every frame of a recorded session before the
first document loads. It sends each trusted input through a binding whose
name is random per session and which it removes from the page before any page
script runs, and every message carries a token that exists only inside this
function. The adapter drops any message without it.

That keeps an ordinary page from adding events, and no more. This runs in the
page's own world, so a page script can still stop it, replace the functions
it calls, call it, or delay its messages, and so drop, claim, or re-time a
person's events. Apart from that, nothing a recorded event says can loosen
anything. An event can only mark a field as holding a secret, void an
approval, stop the automation, or add a step a later capability recorder must
validate again. The adapter counts an event the page calls the automation's
as its own only while the automation holds the session.

The adapter tells it, just before each input call, which kinds of input the
call sends and to which element, and it attributes each event itself: an
event is the automation's only when such a claim covers its kind and its
element, never because it came soon after the automation's input. When the
call returns, input on the claimed element that the call cannot account for
is reported as uncertain, and a field there is marked.

What it sends is a category, never a value. A key is a confirm, cancel,
move, shortcut, or command, decided after it knows whether the key went into
a field, and a character is never sent. A category is for the journal only:
nothing here tells whether the page acted on an input, so nothing the adapter
decides about an approval rests on it. An
edit is one message per field per focus. A field a person typed into is
marked in the same event, and remembered by element so the adapter can hold
it, and the only thing left on ``window`` is the function the adapter calls.
"""

CHANGED = """
(mark) => {
  const NOT_TEXT = ['checkbox', 'radio', 'submit', 'button', 'reset', 'image',
                    'file', 'range', 'color', 'hidden'];
  const found = [];
  for (const el of document.querySelectorAll('input, textarea')) {
    const tag = el.tagName.toLowerCase();
    const type = (el.getAttribute('type') || '').toLowerCase();
    if (tag === 'input' && NOT_TEXT.includes(type)) { continue; }
    if (el.getAttribute(mark.name) === mark.token) { continue; }
    if (el.value === el.defaultValue) { continue; }
    el.setAttribute(mark.name, mark.token);
    found.push(el);
  }
  return found;
}
"""
"""Mark every text field whose value differs from the one its page set.

A field a person typed into can come back after its document went: the
browser restores form values when a person presses Back. The script compares
each value with its default inside the page and returns only the elements it
marked, never a value.
"""
