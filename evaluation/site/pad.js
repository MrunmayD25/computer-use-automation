// The approval control is painted instead of built from elements. It has only
// a position and size, so the browser must target it visually.
(function () {
  var pad = document.getElementById('approval-pad');
  var context = pad.getContext('2d');
  var control = { x: 30, y: 40, width: 150, height: 46 };

  function paint() {
    context.fillStyle = '#f4f4f4';
    context.fillRect(0, 0, pad.width, pad.height);
    context.fillStyle = '#2f4f7f';
    context.fillRect(control.x, control.y, control.width, control.height);
    context.fillStyle = '#ffffff';
    context.font = '16px Georgia';
    context.fillText('Post entry', control.x + 26, control.y + 29);
  }

  function inside(x, y) {
    return x >= control.x && x <= control.x + control.width &&
      y >= control.y && y <= control.y + control.height;
  }

  pad.addEventListener('click', function (event) {
    var box = pad.getBoundingClientRect();
    if (inside(event.clientX - box.left, event.clientY - box.top)) {
      document.getElementById('approval-state').textContent = 'posted';
    }
  });

  // Move the painted control to verify that a recorded target does not keep
  // its original position.
  window.movePaintedControl = function (x, y) {
    control.x = x;
    control.y = y;
    paint();
  };

  document.getElementById('move-control').addEventListener('click', function () {
    window.movePaintedControl(196, 74);
  });

  document.getElementById('clear-pad').addEventListener('click', function () {
    context.fillStyle = '#f4f4f4';
    context.fillRect(0, 0, pad.width, pad.height);
  });

  paint();
})();
