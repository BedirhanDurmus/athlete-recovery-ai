document.addEventListener("DOMContentLoaded", function () {
  var slides = document.querySelectorAll(".slide");
  var bar = document.getElementById("progress-fill");
  var statusEl = document.getElementById("status-line");
  var startBtn = document.getElementById("start-btn");

  /* ── Slide carousel ──────────────────────────────── */
  var idx = 0;
  setInterval(function () {
    slides[idx].classList.remove("active");
    idx = (idx + 1) % slides.length;
    slides[idx].classList.add("active");
  }, 3200);

  /* ── Loading steps ───────────────────────────────── */
  var steps = [
    { pct: 20,  text: "scikit-learn pipeline yükleniyor..." },
    { pct: 40,  text: "GradientBoosting modeli hazırlanıyor..." },
    { pct: 62,  text: "Özellik mühendisliği aktifleştiriliyor..." },
    { pct: 84,  text: "Biyometri ve antrenman verisi senkronize ediliyor..." },
    { pct: 100, text: "Sistem çevrimiçi — başlamaya hazır!" },
  ];

  var step = 0;

  function run() {
    if (step >= steps.length) {
      statusEl.classList.add("status--done");
      startBtn.disabled = false;
      startBtn.classList.add("ready");
      return;
    }
    var s = steps[step];
    statusEl.textContent = s.text;
    bar.style.width = s.pct + "%";

    step++;
    var delay = (step >= steps.length) ? 800 : 900 + Math.random() * 500;
    setTimeout(run, delay);
  }

  setTimeout(run, 400);

  /* ── Start → ana sayfa ───────────────────────────── */
  startBtn.addEventListener("click", function () {
    var root = document.querySelector(".splash");
    root.style.transition = "opacity .5s ease";
    root.style.opacity = "0";
    setTimeout(function () { window.location.href = "/app"; }, 500);
  });
});
