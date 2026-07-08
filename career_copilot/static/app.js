(function () {
  const form = document.querySelector("[data-review-form]");
  if (!form) {
    return;
  }

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
})();
