/* KaTeX is loaded locally before this deferred script, including for file:// pages. */
"use strict";

document.addEventListener("DOMContentLoaded", function () {
  renderMathInElement(document.body, {
    delimiters: [
      { left: "\\[", right: "\\]", display: true },
      { left: "\\(", right: "\\)", display: false }
    ],
    throwOnError: true,
    trust: false
  });
});
