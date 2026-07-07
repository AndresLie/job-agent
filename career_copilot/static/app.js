(function () {
  const form = document.querySelector("[data-review-form]");
  if (!form) {
    return;
  }

  form.addEventListener("submit", function () {
    document.querySelectorAll("[data-alert='error']").forEach(function (alert) {
      alert.remove();
    });

    const button = form.querySelector("[data-submit-button]");
    const label = form.querySelector("[data-submit-label]");
    const status = form.querySelector("[data-submit-status]");

    if (button) {
      button.disabled = true;
      button.classList.add("is-loading");
      button.setAttribute("aria-busy", "true");
    }
    if (label) {
      label.textContent = "Running...";
    }
    if (status) {
      status.hidden = false;
    }
  });
})();
