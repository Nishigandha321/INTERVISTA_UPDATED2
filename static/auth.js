document.addEventListener('DOMContentLoaded', () => {
  document.querySelectorAll('.password-toggle').forEach((toggle) => {
    const input = document.getElementById(toggle.getAttribute('aria-controls'));
    if (!input) return;
    toggle.addEventListener('click', () => {
      const showing = input.type === 'password';
      input.type = showing ? 'text' : 'password';
      toggle.textContent = showing ? 'Hide' : 'Show';
      toggle.setAttribute('aria-pressed', String(showing));
      input.focus({ preventScroll: true });
    });
  });
});
