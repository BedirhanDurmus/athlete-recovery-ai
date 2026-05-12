document.addEventListener("DOMContentLoaded", () => {
  const slides = document.querySelectorAll(".slide");
  const progressFill = document.getElementById("progress-fill");
  const progressWrap = document.getElementById("progress-wrap");
  const loadLines = document.getElementById("load-lines");
  const startBtn = document.getElementById("start-btn");

  let slideIdx = 0;
  setInterval(() => {
    slides[slideIdx].classList.remove("active");
    slideIdx = (slideIdx + 1) % slides.length;
    slides[slideIdx].classList.add("active");
  }, 3200);

  const steps = [
    { pct: 22, line: "scikit-learn pipeline yükleniyor", final: false },
    { pct: 42, line: "GradientBoosting modeli hazırlanıyor", final: false },
    { pct: 62, line: "Özellik mühendisliği aktifleştiriliyor", final: false },
    { pct: 85, line: "Biyometri ve antrenman verisi senkronize ediliyor", final: false },
    { pct: 100, line: "Sistem çevrimiçi — başlamaya hazır!", final: true },
  ];

  let currentStep = 0;
  let currentPct = 0;
  let targetPct = 0;

  function setAriaProgress(v) {
    if (progressWrap) {
      progressWrap.setAttribute("aria-valuenow", String(Math.round(v)));
    }
  }

  function tickCounter() {
    if (currentPct < targetPct) {
      currentPct = Math.min(targetPct, currentPct + 1);
      progressFill.style.width = currentPct + "%";
      setAriaProgress(currentPct);
      requestAnimationFrame(tickCounter);
    }
  }

  function dimPreviousLines() {
    loadLines.querySelectorAll(".load-line:not(.load-line--final)").forEach((el) => {
      el.classList.add("load-line--muted");
    });
  }

  function addLine(text, isFinal) {
    dimPreviousLines();
    const p = document.createElement("p");
    p.className = "load-line" + (isFinal ? " load-line--final" : "");
    p.textContent = text;
    loadLines.appendChild(p);
  }

  function advanceStep() {
    if (currentStep >= steps.length) {
      finishLoading();
      return;
    }
    const s = steps[currentStep];
    addLine(s.line, Boolean(s.final));
    targetPct = s.pct;
    tickCounter();

    currentStep++;
    const delay = currentStep === steps.length ? 900 : 950 + Math.random() * 450;
    setTimeout(advanceStep, delay);
  }

  function finishLoading() {
    startBtn.disabled = false;
    startBtn.classList.add("ready");
  }

  setTimeout(advanceStep, 500);

  startBtn.addEventListener("click", () => {
    const root = document.querySelector(".splash");
    root.style.transition = "opacity .55s ease";
    root.style.opacity = "0";
    setTimeout(() => {
      window.location.href = "/app";
    }, 550);
  });
});
