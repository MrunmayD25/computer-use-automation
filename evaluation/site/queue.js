// Show one member at a time under a fixed heading. Loading the next member
// changes only the panel values, so only the member number identifies the
// record that the Approve button acts on. Every value is invented.
(function () {
  var byId = function (id) { return document.getElementById(id); };
  var members = [
    ['10001', 'Raise the daily card limit'],
    ['10002', 'Close the savings account'],
    ['10003', 'Waive the overdraft fee']
  ];
  var shown = 0;

  var approve = function () {
    var number = byId('member-number');
    byId('approved-member').textContent = number ? number.textContent : 'unknown';
  };

  var approveButton = function () {
    var button = document.createElement('button');
    button.type = 'button';
    button.textContent = 'Approve';
    button.addEventListener('click', approve);
    return button;
  };

  var slot = byId('approve-slot');
  slot.replaceChildren(approveButton());

  byId('next-member').addEventListener('click', function () {
    shown = (shown + 1) % members.length;
    byId('member-number').textContent = members[shown][0];
    byId('member-request').textContent = members[shown][1];
    slot.replaceChildren(approveButton());
  });

  byId('move-approve').addEventListener('click', function () {
    var moved = document.createElement('span');
    moved.style.marginLeft = '6rem';
    moved.appendChild(approveButton());
    slot.replaceChildren(moved);
  });

  byId('change-note').addEventListener('click', function () {
    byId('queue-note').textContent = 'Two requests are waiting.';
  });

  byId('repeat-number').addEventListener('click', function () {
    var copy = document.createElement('dd');
    copy.id = 'member-number';
    copy.textContent = byId('member-number').textContent;
    byId('member-request').after(copy);
  });

  byId('drop-number').addEventListener('click', function () {
    byId('member-number').remove();
  });

  var nextMember = function () { byId('next-member').click(); };

  // Cover the Approve button with an overlay. The member changes after 600 ms,
  // and the overlay disappears after 1,400 ms. A click that waits for the
  // overlay reaches the next member.
  byId('delayed-swap').addEventListener('click', function () {
    var box = slot.getBoundingClientRect();
    var cover = document.createElement('div');
    cover.id = 'approve-cover';
    cover.style.position = 'absolute';
    cover.style.left = (box.left + window.scrollX) + 'px';
    cover.style.top = (box.top + window.scrollY) + 'px';
    cover.style.width = box.width + 'px';
    cover.style.height = box.height + 'px';
    cover.style.background = 'rgba(0, 0, 0, 0.2)';
    document.body.appendChild(cover);
    setTimeout(nextMember, 600);
    setTimeout(function () { cover.remove(); }, 1400);
  });

  // Change the member when the next pointer press reaches the page, after any
  // pre-click check and before the click arrives.
  // The button stays the same element; only the member it acts on changes.
  var nextNumber = function () {
    shown = (shown + 1) % members.length;
    byId('member-number').textContent = members[shown][0];
    byId('member-request').textContent = members[shown][1];
  };
  byId('swap-on-press').addEventListener('click', function () {
    document.addEventListener('pointerdown', nextNumber, { capture: true, once: true });
  });

  // Paint the same Approve control on a screen with no button elements.
  var pad = byId('queue-pad');
  var context = pad.getContext('2d');
  context.fillStyle = '#f4f4f4';
  context.fillRect(0, 0, pad.width, pad.height);
  context.fillStyle = '#2f4f7f';
  context.fillRect(50, 22, 150, 46);
  context.fillStyle = '#ffffff';
  context.font = '16px Georgia';
  context.fillText('Approve', 92, 51);
  pad.addEventListener('click', function (event) {
    var box = pad.getBoundingClientRect();
    var x = event.clientX - box.left;
    var y = event.clientY - box.top;
    if (x >= 50 && x <= 200 && y >= 22 && y <= 68) { approve(); }
  });
})();
