"""Burger menu that replaces Gradio's footers, small-screen layout fixes (header, tables, tabs, API and
Settings windows) and a fix for the Run history page.

Gradio shows "Runs / Use via API / Built with Gradio / Settings" under the app and "App / Built with Gradio" under
the Run history page (/gradio_api/runs). Both footers are hidden with CSS but stay in the page; each menu entry clicks
the matching footer control, so Gradio's own API docs / settings / run-history panels open as usual. Controls are found
by class, not text, because Gradio translates the footer into the browser language.

Everything is injected through `head`, because Gradio serves its own pages from the same HTML and `css=` would only
style the app page. On the Run history page the storage dropdown ("This browser") also closes on a click outside it
or on Escape. Pass to `launch(head=MENU_HEAD)`.
"""

MENU_CSS = """
footer[aria-label="Gradio footer navigation"] { display: none !important; }

/* ---- space above the page: the title no longer touches the top edge ---- */
.gradio-container .main { padding-top: 30px !important; }

/* ---- page header: a compact title, a short line of text and the six classes as small chips ---- */
#app-header, #app-header .html-container { padding: 0 !important; border: 0 !important;
                                          background: transparent !important; }
#app-header h1 { margin: 0 0 .35rem; padding-right: 56px; font-size: clamp(1.55rem, 6.4vw, 2.2rem); line-height: 1.15; }
#app-header .app-sub { margin: 0 0 .55rem; padding-right: 8px; font-size: .95rem; line-height: 1.45;
                       color: var(--body-text-color-subdued, #6b7280); }
#app-header .app-classes { margin: 0; display: flex; flex-wrap: wrap; gap: 6px; }
#app-header .app-classes span { padding: 2px 10px; font-size: .8rem; line-height: 1.5; border-radius: 999px;
                                color: var(--body-text-color, #1f2937);
                                border: 1px solid var(--border-color-primary, #d1d5db); }

/* ---- tables never squeeze their columns: they scroll sideways instead ---- */
.gradio-container .prose table { display: block; max-width: 100%; overflow-x: auto; -webkit-overflow-scrolling: touch;
                                 border-collapse: collapse; margin: .5rem 0 1rem; padding: 0; }
.gradio-container .prose th, .gradio-container .prose td { white-space: nowrap; word-break: normal;
                                                           overflow-wrap: normal; padding: 8px 14px; }
.gradio-container .prose th:first-child, .gradio-container .prose td:first-child {
    position: sticky; left: 0; background: var(--background-fill-primary, #ffffff); }

/* ---- tabs: labels stay on one line and never run into the neighbouring tab or the "..." button ---- */
.gradio-container [role="tab"] { white-space: nowrap; }
@media (max-width: 600px) {
    .gradio-container .tab-wrapper [role="tab"] { padding-left: 10px; padding-right: 10px; font-size: .92rem;
                                                  max-width: 44vw; overflow: hidden; text-overflow: ellipsis; }
}

/* ---- example photos: big enough to recognise (3 per row on a phone) ---- */
.gradio-container .gallery { gap: 10px; }
.gradio-container .gallery > .gallery-item { width: 150px; flex: 0 0 auto; }
.gradio-container .gallery > .gallery-item > span { display: block; width: 100%; }
.gradio-container .gallery > .gallery-item .container, .gradio-container .gallery > .gallery-item img {
    width: 100% !important; max-width: none !important; height: auto !important; aspect-ratio: 1;
    object-fit: cover; }
@media (max-width: 600px) {
    .gradio-container .gallery > .gallery-item { width: calc((100% - 20px) / 3); }
}

/* ---- upload in progress: spinner above the file name, both centred; a long name wraps in the middle ---- */
.gradio-container .upload-container .wrap.progress { text-align: center; padding: 0 16px; box-sizing: border-box; }
.gradio-container .upload-container .wrap.progress .file { flex-direction: column; align-items: center;
                                                           justify-content: center; gap: 8px; max-width: 100%; }
.gradio-container .upload-container .wrap.progress .file-name { display: block; max-width: 100%;
                                                                text-align: center; overflow-wrap: anywhere; }

/* ---- a gallery that has pictures is as tall as its pictures (Gradio keeps an empty 450 px box around them) ---- */
.gradio-container .grid-wrap.fixed-height:has(.grid-container) { min-height: 0 !important; }

/* ---- image galleries on a phone: two photos per row (three for the full set of 24), not one huge column ---- */
@media (max-width: 600px) {
    .gradio-container .grid-container { grid-template-columns: repeat(2, minmax(0, 1fr)) !important; }
    .gradio-container #all-synth .grid-container { grid-template-columns: repeat(3, minmax(0, 1fr)) !important; }
}

/* ---- API window: the lines of one entry keep their own height instead of running into each other ---- */
.api-docs .snippets { flex-wrap: wrap; gap: 6px; }
.api-docs .client-doc { flex-wrap: wrap; row-gap: 8px; }
.api-docs .endpoint-container h3 { line-height: 1.6; }
.api-docs .endpoint-container h3 .post { display: inline-block; margin: 2px 8px 2px 0; }
.api-docs .endpoint-container h3 .desc { display: block; margin-top: 6px; font-weight: 400; line-height: 1.55; }
.api-docs .endpoint-container h3 .analytics-inline, .api-docs .endpoint-container h3 .analytics-content {
    flex-wrap: wrap; row-gap: 4px; }
.api-docs .endpoint-container p { line-height: 1.85; overflow-wrap: anywhere; }
.api-docs .endpoint-container p .code { white-space: normal; overflow-wrap: anywhere; padding: 1px 6px;
                                         -webkit-box-decoration-break: clone; box-decoration-break: clone; }
.api-docs .endpoint-container p .code.highlight { display: inline; }

/* ---- the page behind the app follows the theme, so nothing white shows under a dark app ---- */
body { background-color: var(--body-background-fill, var(--background-fill-primary, #ffffff)); }

/* ---- small plain footer on every page ---- */
#site-footer { width: 100%; margin: 40px auto 0; padding: 16px 16px calc(36px + env(safe-area-inset-bottom, 0px));
               max-width: 1280px; box-sizing: border-box;
               border-top: 1px solid var(--border-color-primary, #d1d5db); text-align: center;
               font-size: 12px; line-height: 1.6; color: var(--body-text-color-subdued, #6b7280); }
#site-footer a { color: inherit; text-decoration: underline; }

/* ---- Gradio's own API / Settings windows: the close button sits in the corner, so the header leaves room for it
        and its parts (title, address, API Recorder) wrap onto a new line instead of running under the button ---- */
.api-docs .banner-wrap h2 { flex-wrap: wrap; row-gap: 6px; padding-right: 44px; }
.api-docs .banner-wrap h2 .title { flex-wrap: wrap; min-width: 0; row-gap: 2px; }
.api-docs .banner-wrap h2 .url { max-width: 100%; overflow-wrap: anywhere; font-size: .75em; }
.api-docs .banner-wrap h2 .counts { flex-wrap: wrap; min-width: 0; }

#app-menu-button {
    position: fixed; top: 22px; right: 18px; z-index: 1000;
    width: 42px; height: 42px; padding: 0; cursor: pointer;
    display: flex; align-items: center; justify-content: center;
    color: var(--body-text-color, #1f2937);
    background: var(--background-fill-primary, #ffffff);
    border: 1px solid var(--border-color-primary, #d1d5db); border-radius: 12px;
    box-shadow: 0 2px 8px rgba(0, 0, 0, .18);
    transition: transform .15s ease, box-shadow .15s ease, border-color .15s ease;
}
#app-menu-button:hover { transform: translateY(-1px); box-shadow: 0 4px 14px rgba(0, 0, 0, .25);
                         border-color: var(--color-accent, #f97316); }
#app-menu-button:focus-visible { outline: 2px solid var(--color-accent, #f97316); outline-offset: 2px; }
#app-menu-button svg { width: 22px; height: 22px; }
#app-menu-button line { stroke: currentColor; stroke-width: 2; stroke-linecap: round;
                        transform-box: fill-box; transform-origin: center;
                        transition: transform .2s ease, opacity .2s ease; }
#app-menu-button[aria-expanded="true"] .bar-top { transform: translateY(6px) rotate(45deg); }
#app-menu-button[aria-expanded="true"] .bar-mid { opacity: 0; }
#app-menu-button[aria-expanded="true"] .bar-bot { transform: translateY(-6px) rotate(-45deg); }

#app-menu-panel {
    position: fixed; top: 72px; right: 18px; z-index: 1000; min-width: 220px; padding: 6px;
    display: none; flex-direction: column; gap: 2px;
    background: var(--background-fill-primary, #ffffff);
    border: 1px solid var(--border-color-primary, #d1d5db); border-radius: 14px;
    box-shadow: 0 12px 32px rgba(0, 0, 0, .28);
}
#app-menu-panel.open { display: flex; animation: app-menu-in .14s ease-out; }
@keyframes app-menu-in { from { opacity: 0; transform: translateY(-6px); } to { opacity: 1; transform: none; } }

#app-menu-panel button {
    display: flex; align-items: center; gap: 12px; width: 100%; padding: 10px 12px;
    font: inherit; font-size: 14px; text-align: left; cursor: pointer;
    color: var(--body-text-color, #1f2937); background: transparent; border: 0; border-radius: 9px;
}
#app-menu-panel button:hover, #app-menu-panel button:focus-visible {
    background: var(--background-fill-secondary, #f3f4f6); outline: none; }
#app-menu-panel button svg { width: 18px; height: 18px; flex: none; color: var(--color-accent, #f97316); }
#app-menu-panel button:disabled { opacity: .5; cursor: default; background: transparent; }
#app-menu-panel button small { margin-left: auto; font-size: 11px; color: var(--body-text-color-subdued, #6b7280); }

/* ---- icons that replace the emoji and text symbols Gradio uses for some of its own controls ---- */
.gradio-container .ui-icon, .api-docs .ui-icon, .ui-icon {
           display: inline-block; width: 1.1em; height: 1.1em; vertical-align: -0.17em; flex: none; margin-right: .4em;
           fill: none; stroke: currentColor; stroke-width: 2; stroke-linecap: round; stroke-linejoin: round; }
.ui-icon.solo, .gradio-container .ui-icon.solo { margin-right: 0; }
.theme-button button { display: inline-flex; align-items: center; gap: 2px; white-space: nowrap; }
.ui-icon.ui-play path { fill: currentColor; stroke: none; }
"""

_MENU_JS = r"""
(function () {
  var path = location.pathname;
  var onRuns = /\/gradio_api\/runs\/?$/.test(path);
  if (/\/gradio_api\//.test(path) && !onRuns) return;  // other internal pages are left alone

  var ICONS = {
    app: '<path d="M3 11l9-8 9 8M5 10v10h5v-6h4v6h5V10"/>',
    api: '<path d="M8 7l-5 5 5 5M16 7l5 5-5 5M14 4l-4 16"/>',
    runs: '<path d="M3 12a9 9 0 1 0 3-6.7M3 4v5h5M12 7v5l3 2"/>',
    settings: '<circle cx="12" cy="12" r="3"/><path d="M19 12a7 7 0 0 0-.1-1.2l2-1.6-2-3.4-2.4 1a7 7 0 0 0-2-1.2L14 3h-4l-.4 2.6a7 7 0 0 0-2 1.2l-2.4-1-2 3.4 2 1.6A7 7 0 0 0 5 12c0 .4 0 .8.1 1.2l-2 1.6 2 3.4 2.4-1a7 7 0 0 0 2 1.2L10 21h4l.4-2.6a7 7 0 0 0 2-1.2l2.4 1 2-3.4-2-1.6c.1-.4.2-.8.2-1.2z"/>',
    gradio: '<path d="M12 3l9 5-9 5-9-5 9-5zM3 13l9 5 9-5M3 17l9 5 9-5"/>'
  };
  // Same order as Gradio's own footers.
  var ENTRIES = onRuns ? [
    { key: 'app', label: 'Back to app', selector: 'footer .app-link' },
    { key: 'gradio', label: 'Built with Gradio', selector: 'footer .built-with' }
  ] : [
    { key: 'runs', label: 'Runs', selector: 'footer .run-history', empty: 'no saved runs yet' },
    { key: 'api', label: 'Use via API', selector: 'footer .show-api' },
    { key: 'gradio', label: 'Built with Gradio', selector: 'footer .built-with' },
    { key: 'settings', label: 'Settings', selector: 'footer .settings' }
  ];

  function icon(key) {
    return '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" ' +
           'stroke-linejoin="round" aria-hidden="true">' + ICONS[key] + '</svg>';
  }

  // Gradio draws some of its own controls with emoji and text symbols (sun, moon, monitor, play, arrows ...).
  // Each one is swapped for an SVG icon of the same kind as the menu icons.
  var GLYPH_RE = /([↺▼⌄→▣▶☀⏾]|\u{1F5A5})[︎️]?/u;
  var GLYPH_NAME = { '↺': 'reset', '▼': 'chevron', '⌄': 'chevron', '→': 'arrow', '▣': 'browser',
                     '▶': 'play', '☀': 'sun', '⏾': 'moon', '\u{1F5A5}': 'monitor' };
  var GLYPH_SVG = {
    reset: '<path d="M3 12a9 9 0 1 0 3-6.7M3 4v5h5"/>',
    chevron: '<path d="M6 9l6 6 6-6"/>',
    arrow: '<path d="M5 12h14M13 6l6 6-6 6"/>',
    browser: '<rect x="3" y="4" width="18" height="16" rx="2"/><path d="M3 9h18"/>',
    play: '<path d="M8 5v14l11-7z"/>',
    sun: '<circle cx="12" cy="12" r="4"/><path d="M12 2v2M12 20v2M4.9 4.9l1.4 1.4M17.7 17.7l1.4 1.4M2 12h2M20 12h2M4.9 19.1l1.4-1.4M17.7 6.3l1.4-1.4"/>',
    moon: '<path d="M21 12.8A9 9 0 1 1 11.2 3a7 7 0 0 0 9.8 9.8z"/>',
    monitor: '<rect x="3" y="4" width="18" height="12" rx="2"/><path d="M8 20h8M12 16v4"/>'
  };

  function glyphIcon(name, solo) {
    var holder = document.createElement('span');
    holder.innerHTML = '<svg class="ui-icon ui-' + name + (solo ? ' solo' : '') + '" viewBox="0 0 24 24" ' +
                       'aria-hidden="true">' + GLYPH_SVG[name] + '</svg>';
    return holder.firstChild;
  }

  function iconifyText(node) {
    var m;
    while (node.parentNode && (m = GLYPH_RE.exec(node.data))) {
      var before = node.data.slice(0, m.index), after = node.data.slice(m.index + m[0].length);
      if (before) node.parentNode.insertBefore(document.createTextNode(before), node);
      node.parentNode.insertBefore(glyphIcon(GLYPH_NAME[m[1]], !before.trim() && !after.trim()), node);
      node.data = after.replace(/^\s+/, '');  // the same text node stays in place, so the page's own code still finds it
    }
  }

  function iconify(root) {
    if (root.nodeType === 3) { if (GLYPH_RE.test(root.data)) iconifyText(root); return; }
    if (root.nodeType !== 1 || /^(script|style|svg)$/i.test(root.tagName)) return;
    var walker = document.createTreeWalker(root, NodeFilter.SHOW_TEXT), found = [];
    while (walker.nextNode()) {
      var n = walker.currentNode;
      if (GLYPH_RE.test(n.data) && !(n.parentElement && n.parentElement.closest('script, style, #site-footer'))) found.push(n);
    }
    found.forEach(iconifyText);
  }

  function initIcons() {
    iconify(document.body);
    var queue = [], scheduled = false;
    new MutationObserver(function (records) {
      records.forEach(function (r) {
        r.addedNodes.forEach(function (n) { queue.push(n); });
        if (r.type === 'characterData') queue.push(r.target);
      });
      if (scheduled) return;
      scheduled = true;
      requestAnimationFrame(function () { scheduled = false; var q = queue; queue = []; q.forEach(iconify); });
    }).observe(document.body, { childList: true, subtree: true, characterData: true });
  }

  function initMenu() {
    if (document.getElementById('app-menu-button')) return;
    var button = document.createElement('button');
    button.id = 'app-menu-button';
    button.type = 'button';
    button.setAttribute('aria-label', 'Menu');
    button.setAttribute('aria-expanded', 'false');
    button.innerHTML = '<svg viewBox="0 0 24 24" aria-hidden="true">' +
      '<line class="bar-top" x1="4" y1="6" x2="20" y2="6"/>' +
      '<line class="bar-mid" x1="4" y1="12" x2="20" y2="12"/>' +
      '<line class="bar-bot" x1="4" y1="18" x2="20" y2="18"/></svg>';
    var panel = document.createElement('div');
    panel.id = 'app-menu-panel';
    panel.setAttribute('role', 'menu');
    document.body.appendChild(panel);
    document.body.appendChild(button);

    function close() {
      panel.classList.remove('open');
      button.setAttribute('aria-expanded', 'false');
    }

    function render() {
      panel.innerHTML = '';
      ENTRIES.forEach(function (entry) {
        var target = document.querySelector(entry.selector);
        var item = document.createElement('button');
        item.type = 'button';
        item.setAttribute('role', 'menuitem');
        item.innerHTML = icon(entry.key) + '<span>' + entry.label + '</span>' +
          (!target && entry.empty ? '<small>' + entry.empty + '</small>' : '');
        if (target) item.addEventListener('click', function () { close(); target.click(); });
        else item.disabled = true;
        panel.appendChild(item);
      });
    }

    button.addEventListener('click', function (e) {
      e.stopPropagation();
      if (panel.classList.contains('open')) { close(); return; }
      render();
      panel.classList.add('open');
      button.setAttribute('aria-expanded', 'true');
    });
    document.addEventListener('click', function (e) {
      if (!panel.contains(e.target)) close();
    });
    document.addEventListener('keydown', function (e) {
      if (e.key === 'Escape') close();
    });
  }

  // Run history page: close the "History storage" dropdown by clicking outside it, not only on its button.
  function initStorageDropdown() {
    function closeIfOpen(inside) {
      var trigger = document.querySelector('.storage-trigger[aria-expanded="true"]');
      if (trigger && !inside) trigger.click();
    }
    document.addEventListener('click', function (e) {
      closeIfOpen(e.target.closest && e.target.closest('.storage-picker'));
    });
    document.addEventListener('keydown', function (e) {
      if (e.key === 'Escape') closeIfOpen(false);
    });
  }

  function initFooter() {
    if (document.getElementById('site-footer')) return;
    var f = document.createElement('footer');
    f.id = 'site-footer';
    f.innerHTML = '&copy; 2026 Scenery Recognition System &middot; Computer Vision, Assignment 2 &middot; ' +
      'Astana IT University &middot; <a href="https://github.com/Xurmayo/Scenery_Recognition_System" ' +
      'target="_blank" rel="noopener">Source code</a>';
    document.body.appendChild(f);
  }

  function init() {
    initMenu();
    initFooter();
    initIcons();
    if (onRuns) initStorageDropdown();
  }
  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', init);
  else init();
})();
"""

MENU_HEAD = f"<style>{MENU_CSS}</style>\n<script>{_MENU_JS}</script>"
