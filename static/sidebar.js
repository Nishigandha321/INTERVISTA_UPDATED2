(() => {
  const ready = () => {
    const app = document.body;
    const sidebar = document.getElementById('app-sidebar');
    const collapse = document.querySelector('.sidebar-toggle');
    const mobile = document.querySelector('.sidebar-mobile-trigger');
    const overlay = document.querySelector('.sidebar-overlay');
    if (!app || !sidebar || !collapse || !mobile || !overlay) return;
    const storageKey = 'intervista-sidebar-collapsed';
    const isMobile = () => window.matchMedia('(max-width: 700px)').matches;
    try { if (localStorage.getItem(storageKey) === 'true') app.classList.add('sidebar-collapsed'); } catch (_) {}
    const sync = () => {
      const open = isMobile() ? app.classList.contains('sidebar-open') : !app.classList.contains('sidebar-collapsed');
      collapse.setAttribute('aria-expanded', String(open));
      collapse.setAttribute('aria-label', app.classList.contains('sidebar-collapsed') ? 'Expand sidebar' : 'Collapse sidebar');
      collapse.title = app.classList.contains('sidebar-collapsed') ? 'Expand sidebar' : 'Collapse sidebar';
      mobile.setAttribute('aria-expanded', String(app.classList.contains('sidebar-open')));
      mobile.setAttribute('aria-label', app.classList.contains('sidebar-open') ? 'Close navigation' : 'Open navigation');
    };
    const toggle = () => {
      if (isMobile()) app.classList.toggle('sidebar-open');
      else {
        app.classList.toggle('sidebar-collapsed');
        try { localStorage.setItem(storageKey, String(app.classList.contains('sidebar-collapsed'))); } catch (_) {}
      }
      sync();
    };
    collapse.addEventListener('click', toggle);
    mobile.addEventListener('click', toggle);
    overlay.addEventListener('click', () => { app.classList.remove('sidebar-open'); sync(); });
    document.addEventListener('keydown', event => {
      if (event.key === 'Escape' && app.classList.contains('sidebar-open')) { app.classList.remove('sidebar-open'); sync(); mobile.focus(); }
    });
    window.addEventListener('resize', sync, { passive: true });
    sidebar.querySelectorAll('a').forEach(link => link.addEventListener('click', () => { if (isMobile()) { app.classList.remove('sidebar-open'); sync(); } }));
    sync();
  };
  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', ready, { once: true });
  else ready();
})();
