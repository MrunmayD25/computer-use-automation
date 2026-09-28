// Each Approve button acts on the member identified by its card or its
// description. Every value is invented.
(function () {
  var byId = function (id) { return document.getElementById(id); };
  var approved = byId('approved-member');

  var owner = function (button) {
    var described = button.getAttribute('aria-describedby');
    if (described) { return byId(described); }
    return button.closest('.member-card').querySelector('dd');
  };

  document.addEventListener('click', function (event) {
    var button = event.target.closest('button');
    if (!button || button.textContent !== 'Approve') { return; }
    approved.textContent = owner(button).textContent;
  });

  // Each card has a painted Approve control with no corresponding element.
  // Both cards use the same painting on purpose.
  Array.prototype.forEach.call(document.querySelectorAll('.card-pad'), function (pad) {
    var context = pad.getContext('2d');
    context.fillStyle = '#f4f4f4';
    context.fillRect(0, 0, pad.width, pad.height);
    context.fillStyle = '#2f4f7f';
    context.fillRect(20, 8, 140, 44);
    context.fillStyle = '#ffffff';
    context.font = '16px Georgia';
    context.fillText('Approve', 58, 36);
    pad.addEventListener('click', function () {
      approved.textContent = pad.closest('.member-card').querySelector('dd').textContent;
    });
  });

  // Reverse the two cards and move each button into its own paragraph.
  // Each button still belongs to the same card.
  byId('redraw-cards').addEventListener('click', function () {
    var cards = byId('cards');
    var listed = Array.prototype.slice.call(cards.children);
    listed.reverse().forEach(function (card) {
      var button = card.querySelector('button');
      var holder = document.createElement('p');
      card.appendChild(holder);
      holder.appendChild(button);
      cards.appendChild(card);
    });
  });
})();
