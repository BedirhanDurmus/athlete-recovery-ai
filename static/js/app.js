document.addEventListener("DOMContentLoaded", () => {
  const form = document.getElementById("predict-form");
  const btn = document.getElementById("btn-predict");
  const resultsWrap = document.getElementById("results-wrap");
  const placeholder = document.getElementById("placeholder");

  form.addEventListener("submit", async (e) => {
    e.preventDefault();
    btn.classList.add("loading");
    btn.innerHTML = '<span class="spinner">⏳</span> Hesaplanıyor...';

    const fd = new FormData(form);
    const body = {};
    fd.forEach((v, k) => { body[k] = v; });

    try {
      const res = await fetch("/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      });
      const data = await res.json();
      renderResults(data);
    } catch (err) {
      alert("Tahmin sırasında hata: " + err.message);
    } finally {
      btn.classList.remove("loading");
      btn.innerHTML = 'Tahmin Yap <span class="arrow">→</span>';
    }
  });

  function renderResults(d) {
    placeholder.style.display = "none";
    resultsWrap.style.display = "flex";
    resultsWrap.classList.add("animate-in");

    drawGauge(d.score, d.level.color);
    document.getElementById("score-num").textContent = d.score;
    document.getElementById("score-label").textContent = d.level.label;
    document.getElementById("score-label").style.color = d.level.color;
    document.getElementById("range-marker").style.left = d.score + "%";

    const factorsEl = document.getElementById("factors-list");
    factorsEl.innerHTML = d.factors.map((f) => {
      const pct = f.level === "Yüksek" ? 90 : f.level === "Orta" ? 55 : 25;
      return `
        <div class="factor-row">
          <span class="factor-icon">${f.icon}</span>
          <span class="factor-name">${f.name}</span>
          <div class="factor-bar-bg">
            <div class="factor-bar-fill" style="width:${pct}%;background:${f.color}"></div>
          </div>
          <span class="factor-level" style="color:${f.color}">${f.level}</span>
        </div>`;
    }).join("");

    const recsEl = document.getElementById("recs-list");
    recsEl.innerHTML = d.recommendations.map((r) => `
      <div class="rec-item">
        <span class="rec-icon">${r.icon}</span>
        <div class="rec-text">
          <div class="rec-title">${r.title}</div>
          <div class="rec-desc">${r.desc}</div>
        </div>
      </div>`).join("");

    document.getElementById("recovery-hours").textContent = d.recovery_hours;
  }

  function drawGauge(score, color) {
    const circle = document.getElementById("gauge-fill");
    const r = 90;
    const circ = 2 * Math.PI * r;
    circle.style.strokeDasharray = circ;
    circle.style.strokeDashoffset = circ;
    circle.style.stroke = color;
    requestAnimationFrame(() => {
      circle.style.strokeDashoffset = circ - (score / 100) * circ;
    });
  }
});
