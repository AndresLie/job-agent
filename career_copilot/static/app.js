(function () {
  const form = document.querySelector("[data-review-form]");
  initializeTabs();

  if (form) {
    form.addEventListener("submit", function (event) {
      document.querySelectorAll("[data-alert='error']").forEach(function (alert) {
        alert.remove();
      });

      const submitter = event.submitter || form.querySelector("[data-submit-button]");
      const label = submitter ? submitter.querySelector("[data-submit-label]") : null;
      const status = form.querySelector("[data-submit-status]");
      const statusLabel = form.querySelector("[data-submit-status-label]");

      form.querySelectorAll("[data-submit-button]").forEach(function (button) {
        button.disabled = true;
        button.setAttribute("aria-busy", "true");
      });
      if (submitter) {
        submitter.classList.add("is-loading");
      }
      if (label) {
        label.textContent = submitter.getAttribute("data-loading-label") || "Running...";
      }
      if (status) {
        status.hidden = false;
      }
      if (statusLabel && submitter) {
        statusLabel.textContent = submitter.getAttribute("data-status-label") || "Running...";
      }
    });
  }

  document.querySelectorAll("[data-copy-target]").forEach(function (button) {
    button.addEventListener("click", function () {
      const target = document.getElementById(button.getAttribute("data-copy-target"));
      const text = target ? target.textContent.trim() : "";
      if (!text || !navigator.clipboard) {
        return;
      }
      navigator.clipboard.writeText(text).then(function () {
        const original = button.textContent;
        button.textContent = "Copied";
        window.setTimeout(function () {
          button.textContent = original;
        }, 1400);
      });
    });
  });

  function initializeTabs() {
    document.addEventListener("click", function (event) {
      const button = event.target.closest("[data-tab-target]");
      if (!button) {
        return;
      }

      const tabGroup = button.closest("[data-tabs]");
      const target = button.getAttribute("data-tab-target");
      if (!tabGroup || !target) {
        return;
      }

      const panelRoot = document.querySelector("[data-tab-panels]");
      const panels = panelRoot ? panelRoot.querySelectorAll("[data-tab-panel]") : [];
      const buttons = tabGroup.querySelectorAll("[data-tab-target]");

      buttons.forEach(function (item) {
        const selected = item === button;
        item.classList.toggle("is-active", selected);
        item.setAttribute("aria-selected", selected ? "true" : "false");
      });

      panels.forEach(function (panel) {
        const selected = panel.getAttribute("data-tab-panel") === target;
        panel.hidden = !selected;
        panel.classList.toggle("is-active", selected);
      });
    });
  }
})();
