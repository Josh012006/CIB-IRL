// MathJax 3 for the formulas written by pymdownx.arithmatex (generic mode): \( ... \) inline, \[ ... \] displayed.
window.MathJax = {
  tex: {
    inlineMath: [["\\(", "\\)"]],
    displayMath: [["\\[", "\\]"]],
    processEscapes: true,
    processEnvironments: true,
    tags: "none"
  },
  options: {
    ignoreHtmlClass: ".*|",
    processHtmlClass: "arithmatex|md-ellipsis"      // the formulas of the headings also show in the sidebar
  }
};

// Material loads pages without a full reload (instant navigation): typeset again after each page.
document$.subscribe(() => {
  MathJax.startup.output.clearCache();
  MathJax.typesetClear();
  MathJax.texReset();
  MathJax.typesetPromise();
});
