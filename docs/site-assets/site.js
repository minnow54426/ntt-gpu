"use strict";

(function () {
  const menu = document.querySelector(".site-menu-button");
  const sidebar = document.querySelector(".site-sidebar");
  function closeMenu() {
    menu.setAttribute("aria-expanded", "false");
    sidebar.classList.remove("site-open");
  }
  menu.addEventListener("click", function () {
    const expanded = menu.getAttribute("aria-expanded") !== "true";
    menu.setAttribute("aria-expanded", String(expanded));
    sidebar.classList.toggle("site-open", expanded);
  });
  document.addEventListener("click", function (event) {
    if (!sidebar.contains(event.target) && !menu.contains(event.target)) closeMenu();
  });
  document.addEventListener("keydown", function (event) {
    if (event.key === "Escape") closeMenu();
  });
  sidebar.addEventListener("click", function (event) {
    if (event.target.closest("a")) closeMenu();
  });

  document.querySelectorAll(".site-article table").forEach(function (table) {
    const wrapper = document.createElement("div");
    wrapper.className = "site-table-scroll";
    wrapper.tabIndex = 0;
    wrapper.setAttribute("role", "region");
    const heading = table.querySelector("th");
    wrapper.setAttribute("aria-label", heading ? heading.textContent + " table" : "Scrollable table");
    table.parentNode.insertBefore(wrapper, table);
    wrapper.appendChild(table);
  });

  const dialog = document.querySelector(".site-search-dialog");
  const input = document.querySelector("#site-search-input");
  const results = document.querySelector(".site-search-results");
  const status = document.querySelector(".site-search-status");
  const indexUrl = new URL(document.body.dataset.searchIndex, window.location.href);
  let searchIndex;
  let pendingIndex;
  let requestSequence = 0;

  function loadIndex() {
    if (!pendingIndex) {
      const entries = Array.isArray(window.nttNotebookSearchIndex)
        ? Promise.resolve(window.nttNotebookSearchIndex)
        : fetch(indexUrl).then(function (response) {
        if (!response.ok) throw new Error("Search index unavailable");
        return response.json();
      });
      pendingIndex = entries.then(function (entries) {
        searchIndex = entries.map(function (entry) {
          return Object.assign({}, entry, {
            haystack: (entry.title + " " + entry.summary + " " + entry.text).toLowerCase()
          });
        });
      }).catch(function (error) {
        pendingIndex = null;
        throw error;
      });
    }
    return pendingIndex;
  }

  async function search() {
    const sequence = ++requestSequence;
    const query = input.value.trim().toLowerCase();
    results.replaceChildren();
    if (!query) {
      status.textContent = "Search articles, plans, and daily entries.";
      return;
    }
    status.textContent = "Searching…";
    try {
      await loadIndex();
      if (sequence !== requestSequence) return;
      const terms = query.split(/\s+/);
      const matches = searchIndex.filter(function (entry) {
        return terms.every(function (term) { return entry.haystack.includes(term); });
      }).sort(function (left, right) {
        return Number(right.title.toLowerCase().includes(query)) - Number(left.title.toLowerCase().includes(query));
      });
      status.textContent = matches.length ? matches.length + " matching page" + (matches.length === 1 ? "" : "s") : "No matching pages. Try another term.";
      matches.slice(0, 20).forEach(function (entry) {
        const link = document.createElement("a");
        link.href = new URL(entry.url, indexUrl).href;
        const category = document.createElement("small");
        category.textContent = entry.group;
        const title = document.createElement("strong");
        title.textContent = entry.title;
        const description = document.createElement("p");
        description.textContent = entry.summary;
        link.append(category, title, description);
        results.appendChild(link);
      });
    } catch (_) {
      if (sequence !== requestSequence) return;
      status.textContent = "Search could not load. Serve the generated site over HTTP, then retry.";
    }
  }
  function openSearch() {
    closeMenu();
    if (!dialog.open) dialog.showModal();
    input.focus();
  }
  document.querySelector(".site-search-button").addEventListener("click", openSearch);
  document.querySelector(".site-search-close").addEventListener("click", function () { dialog.close(); });
  input.addEventListener("input", search);
  document.addEventListener("keydown", function (event) {
    if ((event.metaKey || event.ctrlKey) && event.key.toLowerCase() === "k") {
      event.preventDefault();
      openSearch();
    }
  });
  dialog.addEventListener("click", function (event) {
    if (event.target === dialog) {
      const bounds = dialog.getBoundingClientRect();
      if (event.clientX < bounds.left || event.clientX > bounds.right ||
          event.clientY < bounds.top || event.clientY > bounds.bottom) dialog.close();
    }
  });
})();
