// Fullscreen toggle for the Tier List Maker
(function () {
  const btn = document.getElementById('bFull');
  if (!btn) return;

  btn.addEventListener('click', () => {
    if (document.fullscreenElement) {
      document.exitFullscreen();
    } else {
      document.documentElement.requestFullscreen().catch(() => {
        const t = document.getElementById('toast');
        if (t) { t.textContent = 'Fullscreen is not available here.'; t.classList.add('show'); setTimeout(() => t.classList.remove('show'), 2200); }
      });
    }
  });

  document.addEventListener('fullscreenchange', () => {
    btn.textContent = document.fullscreenElement ? 'Exit Fullscreen' : 'Fullscreen';
  });
})();
