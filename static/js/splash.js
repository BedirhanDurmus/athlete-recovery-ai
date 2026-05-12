document.addEventListener("DOMContentLoaded", () => {
  const slides = document.querySelectorAll(".slide");
  const progressFill = document.getElementById("progress-fill");
  const progressPercent = document.getElementById("progress-percent");
  const statusLine = document.getElementById("status-line");
  const startBtn = document.getElementById("start-btn");

  let slideIdx = 0;
  setInterval(() => {
    slides[slideIdx].classList.remove("active");
    slideIdx = (slideIdx + 1) % slides.length;
    slides[slideIdx].classList.add("active");
  }, 3200);

  const steps = [
    { pct: 20, msg: "scikit-learn pipeline yükleniyor" },
    { pct: 40, msg: "GradientBoosting modeli hazırlanıyor" },
    { pct: 60, msg: "Özellik mühendisliği aktifleştiriliyor" },
    { pct: 85, msg: "Biyometri & antrenman verisi senkronize ediliyor" },
    { pct: 100, msg: "Sistem çevrimiçi, başlamaya hazır" },
  ];

  let currentStep = 0;
  let currentPct = 0;
  let targetPct = 0;

  function setStatus(msg) {
    statusLine.classList.add("is-switching");
    setTimeout(() => {
      statusLine.textContent = msg;
      statusLine.classList.remove("is-switching");
    }, 160);
  }

  function tickCounter() {
    if (currentPct < targetPct) {
      currentPct = Math.min(targetPct, currentPct + 1);
      progressPercent.textContent = currentPct + "%";
      progressFill.style.width = currentPct + "%";
      requestAnimationFrame(tickCounter);
    }
  }

  function advanceStep() {
    if (currentStep >= steps.length) {
      startBtn.disabled = false;
      startBtn.classList.add("ready");
      return;
    }
    const s = steps[currentStep];
    setStatus(s.msg);
    targetPct = s.pct;
    tickCounter();
    currentStep++;
    const delay = currentStep === steps.length ? 900 : 1000 + Math.random() * 500;
    setTimeout(advanceStep, delay);
  }

  setTimeout(advanceStep, 500);

  startBtn.addEventListener("click", () => {
    document.querySelector(".splash").style.transition = "opacity .55s ease";
    document.querySelector(".splash").style.opacity = "0";
    setTimeout(() => {
      window.location.href = "/app";
    }, 550);
  });
});
