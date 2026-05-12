document.addEventListener("DOMContentLoaded", () => {
  const slidesRoot = document.getElementById("slides-root");
  const slides = document.querySelectorAll(".slide");
  const progressFill = document.getElementById("progress-fill");
  const progressPercent = document.getElementById("progress-percent");
  const statusText = document.getElementById("status-text");
  const typeCursor = document.getElementById("type-cursor");
  const startBtn = document.getElementById("start-btn");

  const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

  /** Tüm splash görsellerini tarayıcı önbelleğine çeker. */
  function preloadImages(urls) {
    return Promise.all(
      urls.map(
        (src) =>
          new Promise((resolve) => {
            const img = new Image();
            img.onload = () => resolve();
            img.onerror = () => resolve();
            img.src = src;
          })
      )
    );
  }

  function collectSlideUrls() {
    return Array.from(slides).map((el) => el.getAttribute("src")).filter(Boolean);
  }

  let slideIdx = 0;
  let carouselTimer = null;

  function startCarousel() {
    if (carouselTimer) return;
    carouselTimer = setInterval(() => {
      slides[slideIdx].classList.remove("active");
      slideIdx = (slideIdx + 1) % slides.length;
      slides[slideIdx].classList.add("active");
    }, 3400);
  }

  let currentPct = 0;
  let targetPct = 0;
  let rafId = null;

  function tickCounter() {
    rafId = null;
    if (currentPct < targetPct) {
      currentPct = Math.min(targetPct, currentPct + 2);
      progressPercent.textContent = currentPct + "%";
      progressFill.style.width = currentPct + "%";
      rafId = requestAnimationFrame(tickCounter);
    } else if (currentPct > targetPct) {
      currentPct = targetPct;
      progressPercent.textContent = currentPct + "%";
      progressFill.style.width = currentPct + "%";
    }
  }

  function setTargetPct(n) {
    targetPct = Math.min(100, Math.max(0, n));
    if (rafId) cancelAnimationFrame(rafId);
    rafId = requestAnimationFrame(tickCounter);
  }

  async function typeLine(text, msPerChar = 16) {
    typeCursor.classList.remove("is-hidden");
    statusText.textContent = "";
    for (let i = 0; i < text.length; i++) {
      statusText.textContent += text[i];
      await sleep(msPerChar);
    }
  }

  async function runBootSequence() {
    const lines = [
      "scikit-learn pipeline yükleniyor…",
      "GradientBoosting modeli hazırlanıyor…",
      "Özellik mühendisliği aktifleştiriliyor…",
      "Biyometri ve antrenman verisi senkronize ediliyor…",
    ];
    const pctsAfterLine = [22, 45, 68, 88];

    for (let i = 0; i < lines.length; i++) {
      await typeLine(lines[i], 14);
      setTargetPct(pctsAfterLine[i]);
      await sleep(280);
    }

    setTargetPct(100);
    await sleep(550);

    await typeLine("Çevrimiçi — hazır.", 12);
    typeCursor.classList.add("is-hidden");

    startBtn.disabled = false;
    startBtn.classList.add("ready");
  }

  startBtn.addEventListener("click", () => {
    document.querySelector(".splash").style.transition = "opacity .55s ease";
    document.querySelector(".splash").style.opacity = "0";
    setTimeout(() => {
      window.location.href = "/app";
    }, 550);
  });

  (async () => {
    const urls = collectSlideUrls();
    await preloadImages(urls);
    slidesRoot.classList.remove("is-pending");
    slidesRoot.classList.add("is-ready");
    startCarousel();

    await sleep(120);
    await runBootSequence();
  })();
});
