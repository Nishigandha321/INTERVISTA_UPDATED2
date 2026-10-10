document.addEventListener('DOMContentLoaded', () => {
  const phrases = [
    'Perform Technical Interviews.',
    'Practice Coding Rounds.',
    'Master Online Assessments.',
    'Practice HR Interviews.',
    'Excel at Group Discussions.',
    'Prepare for Your Dream Role.'
  ];
  const phrase = document.getElementById('rotatingText');
  const reduceMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
  if (phrase && !reduceMotion) {
    let current = 0;
    window.setInterval(() => {
      phrase.classList.add('is-leaving');
      window.setTimeout(() => {
        current = (current + 1) % phrases.length;
        phrase.textContent = phrases[current];
        phrase.classList.remove('is-leaving');
      }, 320);
    }, 2800);
  }

  const toggle = document.querySelector('.nav-toggle');
  const nav = document.getElementById('main-nav');
  toggle?.addEventListener('click', () => {
    const open = toggle.getAttribute('aria-expanded') !== 'true';
    toggle.setAttribute('aria-expanded', String(open));
    toggle.setAttribute('aria-label', open ? 'Close navigation' : 'Open navigation');
    nav?.classList.toggle('open', open);
  });

  const dropdowns = [...document.querySelectorAll('.nav-dropdown, .account-menu')];
  dropdowns.forEach((container) => {
    const button = container.querySelector('button');
    button?.addEventListener('click', () => {
      const open = !container.classList.contains('open');
      dropdowns.forEach((other) => {
        other.classList.remove('open');
        other.querySelector('button')?.setAttribute('aria-expanded', 'false');
      });
      container.classList.toggle('open', open);
      button.setAttribute('aria-expanded', String(open));
    });
  });
  document.addEventListener('click', (event) => {
    dropdowns.forEach((container) => {
      if (!container.contains(event.target)) {
        container.classList.remove('open');
        container.querySelector('button')?.setAttribute('aria-expanded', 'false');
      }
    });
  });
  document.addEventListener('keydown', (event) => {
    if (event.key === 'Escape') {
      dropdowns.forEach((container) => {
        container.classList.remove('open');
        container.querySelector('button')?.setAttribute('aria-expanded', 'false');
      });
      nav?.classList.remove('open');
      toggle?.setAttribute('aria-expanded', 'false');
      toggle?.setAttribute('aria-label', 'Open navigation');
      toggle?.focus();
    }
  });
  nav?.querySelectorAll('a').forEach((link) => link.addEventListener('click', () => {
    nav.classList.remove('open');
    toggle?.setAttribute('aria-expanded', 'false');
  }));
});
