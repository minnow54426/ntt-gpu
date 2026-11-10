"use strict";

(function () {
  const menu = document.querySelector(".site-menu-button");
  const sidebar = document.querySelector(".site-sidebar");
  const contentsButton = document.querySelector(".site-contents-button");
  const contents = document.querySelector(".site-on-this-page");
  function closeContents() {
    if (!contents) return;
    contentsButton.setAttribute("aria-expanded", "false");
    contents.classList.remove("site-open");
  }
  function closeMenu() {
    menu.setAttribute("aria-expanded", "false");
    sidebar.classList.remove("site-open");
  }
  menu.addEventListener("click", function () {
    closeContents();
    const expanded = menu.getAttribute("aria-expanded") !== "true";
    menu.setAttribute("aria-expanded", String(expanded));
    sidebar.classList.toggle("site-open", expanded);
  });
  document.addEventListener("click", function (event) {
    if (!sidebar.contains(event.target) && !menu.contains(event.target)) closeMenu();
    if (contents && !contents.contains(event.target) && !contentsButton.contains(event.target)) closeContents();
  });
  document.addEventListener("keydown", function (event) {
    if (event.key === "Escape") closeMenu();
    if (event.key === "Escape" && contents && contents.classList.contains("site-open")) {
      closeContents();
      contentsButton.focus();
    }
  });
  sidebar.addEventListener("click", function (event) {
    if (event.target.closest("a")) closeMenu();
  });

  if (contents) {
    contentsButton.addEventListener("click", function () {
      closeMenu();
      const expanded = contentsButton.getAttribute("aria-expanded") !== "true";
      contentsButton.setAttribute("aria-expanded", String(expanded));
      contents.classList.toggle("site-open", expanded);
      if (expanded) contents.querySelector("a").focus();
    });
    contents.querySelector(".site-contents-close").addEventListener("click", function () {
      closeContents();
      contentsButton.focus();
    });
    const links = [...contents.querySelectorAll("a")];
    const sections = links.map(link => document.getElementById(decodeURIComponent(link.hash.slice(1))));
    contents.addEventListener("click", function (event) {
      const link = event.target.closest("a");
      if (!link) return;
      closeContents();
      const section = document.getElementById(decodeURIComponent(link.hash.slice(1)));
      section.tabIndex = -1;
      section.focus({preventScroll:true});
    });
    let framePending = false;
    function updateSection() {
      let selected = 0;
      sections.forEach((section, index) => {
        if (section.getBoundingClientRect().top <= 140) selected = index;
      });
      if (window.scrollY > 0 && window.innerHeight + window.scrollY >= document.documentElement.scrollHeight - 2) {
        selected = sections.length - 1;
      }
      links.forEach((link, index) => {
        if (index === selected) link.setAttribute("aria-current", "location");
        else link.removeAttribute("aria-current");
      });
      framePending = false;
    }
    window.addEventListener("scroll", function () {
      if (!framePending) {
        framePending = true;
        requestAnimationFrame(updateSection);
      }
    }, {passive:true});
    window.addEventListener("resize", updateSection);
    updateSection();
  }

  document.querySelectorAll(".site-article table, .site-legacy table").forEach(function (table) {
    if (table.closest(".site-table-scroll, .matrix-scroll")) return;
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
    closeContents();
    if (!dialog.open) dialog.showModal();
    input.focus();
  }
  document.querySelector(".site-search-button").addEventListener("click", openSearch);
  document.querySelector(".site-search-close").addEventListener("click", function () { dialog.close(); });
  input.addEventListener("input", search);
  document.addEventListener("keydown", function (event) {
    if (event.key === "Escape" && dialog.open) {
      event.preventDefault();
      dialog.close();
    }
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
