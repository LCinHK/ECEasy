(function () {
  var prefersReducedMotion =
    typeof window.matchMedia === 'function' &&
    window.matchMedia('(prefers-reduced-motion: reduce)').matches;

  var year = new Date().getFullYear();
  var footer = document.querySelector('.site-footer p');
  if (footer && footer.textContent) {
    footer.textContent = 'ECEasy Demo ' + year + ' | Built for ECE advising exploration';
  }

  var header = document.querySelector('.site-header');
  var onScroll = function () {
    if (!header) {
      return;
    }
    if (window.scrollY > 12) {
      header.classList.add('is-scrolled');
    } else {
      header.classList.remove('is-scrolled');
    }
  };
  onScroll();
  window.addEventListener('scroll', onScroll, { passive: true });

  // Smooth scroll for in-page links.
  var anchors = document.querySelectorAll('a[href^="#"]');
  anchors.forEach(function (anchor) {
    anchor.addEventListener('click', function (event) {
      var href = anchor.getAttribute('href');
      if (!href || href === '#') {
        return;
      }
      var target = document.querySelector(href);
      if (!target) {
        return;
      }
      event.preventDefault();
      target.scrollIntoView({
        behavior: prefersReducedMotion ? 'auto' : 'smooth',
        block: 'start'
      });
    });
  });

  // Warn users if neither chat UI build appears available.
  var routeStatus = document.getElementById('route-status');
  var checkRoute = function (path) {
    return fetch(path, { method: 'GET', cache: 'no-store' }).then(function (res) {
      return res.ok;
    }).catch(function () {
      return false;
    });
  };

  if (routeStatus) {
    Promise.all([checkRoute('/newUI/index.html'), checkRoute('/ui/index.html')]).then(function (results) {
      var hasNewUi = results[0];
      var hasOldUi = results[1];
      routeStatus.classList.remove('success', 'info', 'warning');

      if (hasNewUi) {
        routeStatus.classList.add('success');
        routeStatus.textContent = 'Live route ready: /newUI/index.html.';
      } else if (hasOldUi) {
        routeStatus.classList.add('info');
        routeStatus.textContent = 'Fallback route ready: /ui/index.html (old UI).';
      } else {
        routeStatus.classList.add('warning');
        routeStatus.textContent = 'Chat UI build was not found at /newUI or /ui. Build and mount at least one UI before demoing.';
      }
    }).catch(function () {
      routeStatus.classList.remove('success', 'info');
      routeStatus.classList.add('warning');
      routeStatus.textContent = 'Unable to verify demo routes. Make sure the backend and static mounts are running.';
    });
  }
})();


