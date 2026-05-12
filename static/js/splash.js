document.addEventListener("DOMContentLoaded", () => {
  const slides = document.querySelectorAll(".slide");
  const progressFill = document.getElementById("progress-fill");
  const progressPercent = document.getElementById("progress-percent");
  const statusText = document.getElementById("status-text");
  const checklistItems = document.querySelectorAll("#checklist li");
  const startBtn = document.getElementById("start-btn");

  // ── Slide carousel ─────────────────────────────────
  let slideIdx = 0;
  setInterval(() => {
    slides[slideIdx].classList.remove("active");
    slideIdx = (slideIdx + 1) % slides.length;
    slides[slideIdx].classList.add("active");
  }, 3200);

  // ── Loading sequence ───────────────────────────────
  const steps = [
    { pct: 18, msg: "scikit-learn pipeline yükleniyor...",          stepIdx: 0 },
    { pct: 38, msg: "GradientBoosting modeli hazırlanıyor...",     stepIdx: 1 },
    { pct: 58, msg: "Özellik mühendisliği aktifleştiriliyor...",   stepIdx: 2 },
    { pct: 82, msg: "Biyometri & antrenman verisi senkronize ediliyor...", stepIdx: 3 },
    { pct: 100, msg: "Sistem çevrimiçi — başlamaya hazır!",        stepIdx: 4 },
  ];

  let currentStep = 0;
  let currentPct = 0;
  let targetPct = 0;

  // Smooth percent counter
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
      finishLoading();
      return;
    }
    const s = steps[currentStep];
    statusText.textContent = s.msg;
    targetPct = s.pct;
    tickCounter();

    // Mark previous as done, current as active
    checklistItems.forEach((li, i) => {
      li.classList.remove("active");
      if (i < s.stepIdx) li.classList.add("done");
      else if (i === s.stepIdx) li.classList.add("active");
    });

    currentStep++;
    const delay = currentStep === steps.length ? 900 : 1100 + Math.random() * 600;
    setTimeout(advanceStep, delay);
  }

  function finishLoading() {
    // mark all done
    checklistItems.forEach((li) => {
      li.classList.remove("active");
      li.classList.add("done");
    });
    startBtn.disabled = false;
    startBtn.classList.add("ready");
  }

  // Kick off after a short delay
  setTimeout(advanceStep, 600);

  // ── Start button → go to main app ──────────────────
  startBtn.addEventListener("click", () => {
    document.querySelector(".splash").style.transition = "opacity .6s ease";
    document.querySelector(".splash").style.opacity = "0";
    setTimeout(() => {
      window.location.href = "/app";
    }, 600);
  });
});
