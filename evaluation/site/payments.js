// Submit the payment form through its button or Enter in any field. Enter can
// also submit the adjacent search form. A simulated payment only writes the
// payee number into the page.
(function () {
  var byId = function (id) { return document.getElementById(id); };
  var payees = ['10001', '10002'];
  var shown = 0;

  // Listened for on the document, so a form drawn again still counts.
  document.addEventListener('submit', function (event) {
    if (!event.target.elements.namedItem('amount')) { return; }
    event.preventDefault();
    if (event.submitter && event.submitter.value === 'Save draft') {
      byId('drafts').textContent = String(Number(byId('drafts').textContent) + 1);
      return;
    }
    var sent = byId('payments-sent');
    var payee = byId('payee-number').textContent;
    sent.textContent = sent.textContent === 'none' ? payee : sent.textContent + ' ' + payee;
  });

  document.addEventListener('submit', function (event) {
    if (!event.target.elements.namedItem('lookup')) { return; }
    event.preventDefault();
    var searches = byId('searches');
    searches.textContent = String(Number(searches.textContent) + 1);
  });

  byId('next-payee').addEventListener('click', function () {
    shown = (shown + 1) % payees.length;
    byId('payee-number').textContent = payees[shown];
  });

  // The same submit button drawn again as a new element with a new name.
  byId('redraw-send').addEventListener('click', function () {
    var fresh = document.createElement('button');
    fresh.type = 'submit';
    fresh.id = 'pay-now';
    fresh.textContent = 'Pay now';
    byId('send-payment').replaceWith(fresh);
  });

  // The same button drawn again as a new element, name and all.
  byId('redraw-send-same').addEventListener('click', function () {
    var old = byId('send-payment');
    old.replaceWith(old.cloneNode(true));
  });

  // The whole payment form drawn again, exactly as it was.
  byId('rebuild-form').addEventListener('click', function () {
    var form = document.forms.payment;
    form.replaceWith(form.cloneNode(true));
  });

  // The payment form drawn again under another name, with its button
  // relabelled, so nothing observed says it is the same form.
  byId('rebuild-form-renamed').addEventListener('click', function () {
    var form = document.forms.payment;
    var copy = form.cloneNode(true);
    copy.name = 'payment-v2';
    copy.querySelector('#send-payment').textContent = 'Pay';
    form.replaceWith(copy);
  });

  // A transfer the page sends by script, from its button or from Enter.
  var transfer = function () {
    var sent = byId('transfers');
    sent.textContent = String(Number(sent.textContent) + 1);
  };
  byId('send-transfer').addEventListener('click', transfer);
  Array.prototype.forEach.call(document.forms.transfer.elements, function (field) {
    field.addEventListener('keydown', function (event) {
      if (event.key === 'Enter') { event.preventDefault(); transfer(); }
    });
  });

  // A page that moves focus somewhere else as soon as the amount gets it.
  var trapped = false;
  byId('trap-focus').addEventListener('click', function () { trapped = true; });
  byId('amount').addEventListener('focus', function () {
    if (trapped) { byId('lookup').focus(); }
  });
})();
