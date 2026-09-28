// Reproduce cases that simple pages omit. These include a record replaced at
// the same URL, a frame rewritten without a request, controls in shadow roots,
// a second painted pad, and a dialog. Every value here is invented.
(function () {
  var byId = function (id) { return document.getElementById(id); };

  // A record swapped in at the same URL, and the same record redrawn.
  var records = ['Record A', 'Record B', 'Record C'];
  var shown = 0;

  var approveButton = function () {
    var button = document.createElement('button');
    button.type = 'button';
    button.className = 'queue-approve';
    button.textContent = 'Approve';
    button.addEventListener('click', function () {
      byId('approved-record').textContent = byId('record-heading').textContent;
    });
    return button;
  };

  var slot = byId('approve-slot');
  slot.replaceChildren(approveButton());

  byId('move-approve').addEventListener('click', function () {
    // Draw the same record differently while keeping the button's heading.
    var moved = document.createElement('p');
    moved.style.marginLeft = '4rem';
    moved.appendChild(approveButton());
    slot.replaceChildren();
    slot.appendChild(moved);
  });

  byId('next-record').addEventListener('click', function () {
    shown = (shown + 1) % records.length;
    byId('record-heading').textContent = records[shown];
    slot.replaceChildren(approveButton());
  });

  // A frame whose contents are replaced in the page. No request is made, so
  // no route is consulted, and the src attribute still names the old route.
  byId('detach-ledger').addEventListener('click', function () {
    var ledger = document.querySelector('iframe[name="ledger"]');
    ledger.srcdoc = '<p>Ledger contents replaced in the page.</p>';
  });

  // A second painted pad, close in size to the first, and a swap that puts it
  // first in document order.
  var voidPad = byId('void-pad');
  var voidContext = voidPad.getContext('2d');
  var voidControl = { x: 40, y: 50, width: 150, height: 46 };

  var paintVoid = function () {
    voidContext.fillStyle = '#f4f4f4';
    voidContext.fillRect(0, 0, voidPad.width, voidPad.height);
    voidContext.fillStyle = '#7f2f2f';
    voidContext.fillRect(voidControl.x, voidControl.y, voidControl.width,
                         voidControl.height);
    voidContext.fillStyle = '#ffffff';
    voidContext.font = '16px Georgia';
    voidContext.fillText('Void entry', voidControl.x + 26, voidControl.y + 29);
  };

  voidPad.addEventListener('click', function (event) {
    var box = voidPad.getBoundingClientRect();
    var x = event.clientX - box.left;
    var y = event.clientY - box.top;
    if (x >= voidControl.x && x <= voidControl.x + voidControl.width &&
        y >= voidControl.y && y <= voidControl.y + voidControl.height) {
      byId('approval-state').textContent = 'voided';
    }
  });

  byId('swap-pads').addEventListener('click', function () {
    var pads = byId('pads');
    pads.insertBefore(byId('void-pad'), byId('approval-pad'));
  });

  paintVoid();

  // A canvas below the fold, painted so a scroll has something to reveal.
  var archive = byId('archive-pad');
  var archiveContext = archive.getContext('2d');
  archiveContext.fillStyle = '#f4f4f4';
  archiveContext.fillRect(0, 0, archive.width, archive.height);
  archiveContext.fillStyle = '#2f5f3f';
  archiveContext.fillRect(60, 40, 150, 46);
  archiveContext.fillStyle = '#ffffff';
  archiveContext.font = '16px Georgia';
  archiveContext.fillText('Archive', 86, 69);

  archive.addEventListener('click', function (event) {
    var box = archive.getBoundingClientRect();
    var x = event.clientX - box.left;
    var y = event.clientY - box.top;
    if (x >= 60 && x <= 210 && y >= 40 && y <= 86) {
      byId('approval-state').textContent = 'archived';
    }
  });

  // Draw an obstruction over the pads, as a modal or sticky banner can cover
  // a control that remains in the page.
  byId('cover-pads').addEventListener('click', function () {
    var pads = byId('pads').getBoundingClientRect();
    var cover = document.createElement('div');
    cover.id = 'pad-cover';
    cover.style.position = 'absolute';
    cover.style.left = (pads.left + window.scrollX) + 'px';
    cover.style.top = (pads.top + window.scrollY) + 'px';
    cover.style.width = pads.width + 'px';
    cover.style.height = pads.height + 'px';
    cover.style.background = 'transparent';
    document.body.appendChild(cover);
  });

  // An ordinary div hosting an open shadow root, with another open root
  // nested inside it. Neither host is a control itself.
  var host = byId('hold-host');
  var root = host.attachShadow({ mode: 'open' });
  var release = document.createElement('button');
  release.type = 'button';
  release.textContent = 'Release the hold';
  release.addEventListener('click', function () {
    byId('hold-state').textContent = 'released';
  });
  var innerHost = document.createElement('div');
  root.appendChild(release);
  root.appendChild(innerHost);

  var inner = innerHost.attachShadow({ mode: 'open' });
  var nested = document.createElement('button');
  nested.type = 'button';
  nested.textContent = 'Release the nested hold';
  nested.addEventListener('click', function () {
    byId('hold-state').textContent = 'nested release';
  });
  inner.appendChild(nested);

  // A custom element hosting its own open root.
  if (!customElements.get('ledger-card')) {
    customElements.define('ledger-card', class extends HTMLElement {
      connectedCallback() {
        var card = this.attachShadow({ mode: 'open' });
        var open = document.createElement('button');
        open.type = 'button';
        open.textContent = 'Open the card';
        open.addEventListener('click', function () {
          byId('hold-state').textContent = 'card opened';
        });
        card.appendChild(open);
      }
    });
  }

  // Two dialogs the page waits on, so one can replace the other.
  byId('post-entry').addEventListener('click', function () {
    var agreed = window.confirm('Post this entry to the ledger?');
    byId('dialog-state').textContent = agreed ? 'posted' : 'not posted';
  });

  byId('reverse-entry').addEventListener('click', function () {
    var agreed = window.confirm('Reverse this entry?');
    byId('reverse-state').textContent = agreed ? 'reversed' : 'not reversed';
  });

  // A page-rendered value attribute on a credential field, which is how a
  // legacy screen returns a saved passcode to the browser.
  byId('prefill-passcode').addEventListener('click', function () {
    byId('approver-pin').setAttribute('value', 'synthetic-prefilled-passcode');
  });

  // A plain text field that says nothing about holding a credential, renamed
  // or redrawn after something is typed into it. Redrawing copies the value
  // into a new element, which is what a page that re-renders a form does.
  var relabel = function (field) {
    var label = byId('branch-code-label');
    label.textContent = 'Code on file';
    label.htmlFor = field.id;
    field.name = 'code_on_file';
  };

  byId('rename-branch-code').addEventListener('click', function () {
    var field = document.querySelector('[name="branch_code"]');
    field.setAttribute('aria-label', 'Code on file');
    relabel(field);
  });

  byId('redraw-branch-code').addEventListener('click', function () {
    var old = document.querySelector('[name="branch_code"]');
    var fresh = document.createElement('input');
    fresh.type = 'text';
    fresh.id = 'code-on-file';
    fresh.value = old.value;
    old.replaceWith(fresh);
    relabel(fresh);
  });

  // A field that stops the page with a dialog while it is being filled.
  var checked = false;
  byId('supervisor-code').addEventListener('input', function () {
    if (checked) { return; }
    checked = true;
    window.confirm('Check the supervisor code with the branch?');
  });
})();
