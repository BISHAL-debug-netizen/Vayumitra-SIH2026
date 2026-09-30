const form = document.getElementById("form");
const out = document.getElementById("out");
const bars = document.getElementById("bars");
const liveBtn = document.getElementById("live-btn");
const cityInput = document.getElementById("city-input");
const statusEl = document.getElementById("status");
const advisorForm = document.getElementById("advisor-form");
let lastWeather = null;

form.addEventListener("submit", async (e) => {
  e.preventDefault();
  const data = Object.fromEntries(new FormData(form).entries());
  const res = await fetch("/api/predict", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(data),
  });
  const json = await res.json();
  renderManual(json);
});

liveBtn.addEventListener("click", loadLive);
cityInput.addEventListener("keydown", (e) => {
  if (e.key === "Enter") {
    e.preventDefault();
    loadLive();
  }
});

window.addEventListener("load", loadLive);

async function loadLive() {
  const city = (cityInput.value || "Guwahati").trim();
  statusEl.textContent = `Loading live weather and AQI for ${city}…`;
  statusEl.classList.remove("error");
  liveBtn.disabled = true;
  try {
    const res = await fetch("/api/live?city=" + encodeURIComponent(city));
    const json = await res.json();
    if (!res.ok || json.ok === false) {
      throw new Error(json.error || "Live request failed");
    }
    lastWeather = json;
    renderLive(json);
    renderAdvisor(json.advisor);
    statusEl.textContent = `Updated for ${json.place.name} (${json.now.observed_at || "now"})`;
  } catch (err) {
    statusEl.textContent = String(err.message || err);
    statusEl.classList.add("error");
  } finally {
    liveBtn.disabled = false;
  }
}

function applySky(w) {
  const sky = w.sky || {};
  const theme = sky.theme || "day-clear";
  document.body.className = "theme-" + theme;
  const rain = document.getElementById("rain-layer");
  const snow = document.getElementById("snow-layer");
  rain.innerHTML = "";
  snow.innerHTML = "";
  if (sky.mood === "rain" || sky.mood === "storm") {
    const n = sky.mood === "storm" ? 70 : 45;
    for (let i = 0; i < n; i += 1) {
      const d = document.createElement("span");
      d.className = "drop";
      d.style.left = Math.random() * 100 + "%";
      d.style.animationDuration = 0.5 + Math.random() * 0.7 + "s";
      d.style.animationDelay = "-" + Math.random() * 2 + "s";
      d.style.opacity = String(0.35 + Math.random() * 0.5);
      rain.appendChild(d);
    }
  }
  if (sky.mood === "snow") {
    for (let i = 0; i < 40; i += 1) {
      const f = document.createElement("span");
      f.className = "flake";
      f.style.left = Math.random() * 100 + "%";
      f.style.animationDuration = 4 + Math.random() * 5 + "s";
      f.style.animationDelay = "-" + Math.random() * 5 + "s";
      snow.appendChild(f);
    }
  }
}

function renderLive(j) {
  const w = j.now || {};
  const a = w.indian_aqi || {};
  const sky = w.sky || {};
  applySky(w);
  document.getElementById("now-grid").hidden = false;
  document.getElementById("outlook-card").hidden = false;
  document.getElementById("daily-card").hidden = false;
  document.getElementById("hourly-wrap").hidden = false;

  document.getElementById("weather-now").innerHTML = `
    <p class="hint">${j.place.name}</p>
    <p class="temp-num">${dash(w.temp_c)}°</p>
    <p class="cat">${w.weather || ""} · feels ${dash(w.feels_like)}°</p>
    <div class="sky-chip">
      <span>${sky.period_label || "Daytime"}</span>
      <span>${sky.mood_label || "Clear"}</span>
      <span>Sunrise ${fmtClock(w.sunrise || sky.sunrise)}</span>
      <span>Sunset ${fmtClock(w.sunset || sky.sunset)}</span>
    </div>
    <div class="meta">
      <span>Humidity ${dash(w.humidity)}%</span>
      <span>Wind ${dash(w.wind_ms)} m/s</span>
      <span>Rain ${dash(w.rain_mm)} mm</span>
      <span>Pressure ${dash(w.pressure)} hPa</span>
    </div>
  `;

  document.getElementById("aqi-now").innerHTML = `
    <p class="hint">Indian AQI (CPCB scale)</p>
    <p class="aqi-num" style="color:${a.color || "#9aa8b6"}">${dash(a.aqi)}</p>
    <p class="cat" style="color:${a.color || "#9aa8b6"}">${a.category || "Unknown"} · ${a.prominent_pollutant || "n/a"}</p>
    <p>${a.advice || ""}</p>
    <ul>${(a.actions || []).map((x) => `<li>${x}</li>`).join("")}</ul>
  `;

  document.getElementById("outlook").textContent = j.outlook || "";
  const src = j.source || {};
  document.getElementById("source-note").textContent =
    `${src.weather || ""}. ${src.note || ""}`;

  // --- ML BADGE INJECTION ---
  document.getElementById("days").innerHTML = (j.daily || [])
    .map((d) => {
      // Create the badge if the data came from the ML model
      const mlBadge = d.is_ml ? `<span style="font-size: 0.7rem; margin-left: 4px;" title="Estimated by ML Model">✨ ML</span>` : "";
      
      const pill = d.aqi_peak
        ? `<span class="pill" style="background:${d.aqi_color || "#ccc"}">AQI ${d.aqi_peak} ${mlBadge}</span>`
        : `<span class="hint">AQI n/a</span>`;
        
      return `<div class="day">
        <b>${d.date.slice(5)}</b>
        <div>${d.weather}</div>
        <div>${dash(d.tmin)}° / ${dash(d.tmax)}°</div>
        <div>Rain ${dash(d.rain_chance)}%</div>
        ${pill}
      </div>`;
    })
    .join("");

  if (j.daily && j.daily.length > 0) {
      renderAQIChart(j.daily);
  }

  document.getElementById("hourly-weather").innerHTML = table(
    ["Time", "Sky", "°C", "Rain %", "Wind"],
    (j.hourly_weather || []).map((h) => [
      hour(h.time),
      h.weather,
      dash(h.temp_c),
      dash(h.rain_chance),
      dash(h.wind_ms),
    ])
  );

  document.getElementById("hourly-aqi").innerHTML = table(
    ["Time", "AQI", "Category", "PM2.5"],
    (j.hourly_aqi || []).map((h) => [
      hour(h.time),
      dash(h.aqi),
      h.category || "—",
      dash(h.pm25),
    ])
  );

  if (a.pollutants) {
    const p = a.pollutants;
    const manual = form.elements;
    if (manual.pm25) manual.pm25.value = p.pm25 ?? "";
    if (manual.pm10) manual.pm10.value = p.pm10 ?? "";
    if (manual.no2) manual.no2.value = p.no2 ?? "";
    if (manual.so2) manual.so2.value = p.so2 ?? "";
    if (manual.co) manual.co.value = p.co ?? "";
    if (manual.o3) manual.o3.value = p.o3 ?? "";
    if (manual.temp_c) manual.temp_c.value = w.temp_c ?? "";
    if (manual.humidity) manual.humidity.value = w.humidity ?? "";
    if (manual.wind_ms) manual.wind_ms.value = w.wind_ms ?? "";
  }
}

function renderManual(j) {
  const color = j.color || "#9aa8b6";
  const ml = j.ml ? `<p>ML estimate: <strong>${j.ml.ml_aqi}</strong></p>` : "";
  out.innerHTML = `
    <p class="aqi-num" style="color:${color}">${j.aqi ?? "—"}</p>
    <p class="cat" style="color:${color}">${j.category || "Unknown"} · ${j.city || ""}</p>
    <p>${j.advice || ""}</p>
    <p>Prominent pollutant: <strong>${j.prominent_pollutant || "n/a"}</strong></p>
    ${ml}
    <ul>${(j.actions || []).map((a) => `<li>${a}</li>`).join("")}</ul>
  `;
  const subs = j.sub_indices || {};
  bars.innerHTML = Object.entries(subs)
    .map(([k, v]) => {
      const pct = Math.min(100, (Number(v) / 500) * 100);
      return `<div class="bar-row"><span>${k}</span><div class="track"><div class="fill" style="width:${pct}%"></div></div><span>${v}</span></div>`;
    })
    .join("");
}

function table(headers, rows) {
  const head = headers.map((h) => `<th>${h}</th>`).join("");
  const body = rows
    .map((r) => `<tr>${r.map((c) => `<td>${c}</td>`).join("")}</tr>`)
    .join("");
  return `<table><thead><tr>${head}</tr></thead><tbody>${body}</tbody></table>`;
}

function hour(iso) {
  if (!iso) return "—";
  return iso.slice(11, 16);
}

function dash(v) {
  return v === null || v === undefined || v === "" ? "—" : v;
}

advisorForm.addEventListener("submit", async (e) => {
  e.preventDefault();
  if (!lastWeather || !lastWeather.now) {
    statusEl.textContent = "Load live weather first.";
    return;
  }
  const prefs = [...advisorForm.querySelectorAll('input[name="pref"]:checked')].map((el) => el.value);
  const duration = advisorForm.elements.duration.value;
  const res = await fetch("/api/advise", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ weather: lastWeather, profile: { prefs, duration } }),
  });
  const json = await res.json();
  if (!res.ok || json.ok === false) {
    statusEl.textContent = json.error || "Could not update advice.";
    statusEl.classList.add("error");
    return;
  }
  renderAdvisor(json);
});

function renderAdvisor(a) {
  const card = document.getElementById("advisor-card");
  const body = document.getElementById("advisor-body");
  if (!a || !a.ok) {
    card.hidden = true;
    return;
  }
  card.hidden = false;
  const who = (a.who || []).map((x) => `<li>${x}</li>`).join("");
  const prec = (a.precautions || []).map((x) => `<li>${x}</li>`).join("");
  const pack = (a.pack || []).map((x) => `<li>${x}</li>`).join("");
  const why = (a.why || []).map((x) => `<div>${x}</div>`).join("");
  const drivers = (a.drivers || [])
    .map((d) => {
      const sign = d.delta > 0 ? "+" : "";
      const cls = d.positive ? "factor-pos" : "factor-neg";
      return `<li class="${cls}">${d.label} (${sign}${d.delta})</li>`;
    })
    .join("");
  const personal = (a.personal_notes || []).map((x) => `<li>${x}</li>`).join("");
  const missing = (a.missing_notes || []).map((x) => `<li>${x}</li>`).join("");
  body.innerHTML = `
    <div class="score-row">
      <p class="score"><span class="sr-only">Weather suitability score</span>${a.score}/100</p>
      <p class="level">${a.title} — ${a.general_label}</p>
    </div>
    <p>${a.summary || ""}</p>
    ${a.best_time ? `<p>${a.best_time}</p>` : ""}
    <h3>Why</h3>
    <div class="why">${why}</div>
    ${a.why_blurb ? `<p>${a.why_blurb}</p>` : ""}
    <details>
      <summary>Score factors</summary>
      <ul>${drivers}</ul>
    </details>
    ${who ? `<h3>People who may find this challenging</h3><ul>${who}</ul>` : ""}
    <h3>Recommended precautions</h3>
    <ul>${prec}</ul>
    ${pack ? `<h3>What to pack</h3><ul>${pack}</ul>` : ""}
    ${personal ? `<h3>Based on your selections</h3><ul>${personal}</ul>` : ""}
    ${missing ? `<h3>Missing data</h3><ul>${missing}</ul>` : ""}
    <p class="hint">${a.disclaimer}</p>
  `;
}

function fmtClock(value) {
  if (!value) return "—";
  const text = String(value);
  if (text.includes("T")) return text.slice(11, 16);
  if (text.includes(" ")) return text;
  return text.slice(0, 5);
}

// --- CHART.JS WITH ML FIX ---
function renderAQIChart(dailyData) {
  const canvas = document.getElementById('aqiChart');
  if (!canvas) return; 
  
  const ctx = canvas.getContext('2d');

  const labels = dailyData.map(d => d.date ? d.date.slice(5) : "—");
  // Maps exactly to aqi_peak so the graph smoothly connects ML estimates without dropping to zero
  const aqiValues = dailyData.map(d => d.aqi_peak); 

  if (window.myLineChart) {
      window.myLineChart.destroy();
  }

  window.myLineChart = new Chart(ctx, {
      type: 'line',
      data: {
          labels: labels,
          datasets: [{
              label: 'Predicted AQI Peak',
              data: aqiValues,
              borderColor: '#3B82F6',
              backgroundColor: 'rgba(59, 130, 246, 0.2)',
              borderWidth: 3,
              tension: 0.4,
              fill: true,
              pointBackgroundColor: '#3B82F6',
              pointBorderColor: '#ffffff',
              pointBorderWidth: 2,
              pointRadius: 4
          }]
      },
      options: {
          responsive: true,
          maintainAspectRatio: false, 
          scales: { 
              x: {
                  ticks: { color: 'rgba(255, 255, 255, 0.8)' },
                  grid: { color: 'rgba(255, 255, 255, 0.1)' }
              },
              y: { 
                  beginAtZero: true,
                  title: { 
                      display: true, 
                      text: 'AQI Level',
                      color: 'rgba(255, 255, 255, 0.9)', 
                      font: { weight: 'bold' }
                  },
                  ticks: { color: 'rgba(255, 255, 255, 0.8)' },
                  grid: { color: 'rgba(255, 255, 255, 0.1)' }
              } 
          },
          plugins: { 
              legend: { display: false } 
          }
      }
  });
}

// --- TAB SWITCHING LOGIC ---
function openTab(tabId, btn) {
  document.querySelectorAll('.tab-pane').forEach(p => p.classList.remove('active'));
  document.querySelectorAll('.dash-tab').forEach(b => b.classList.remove('active'));

  const selectedPane = document.getElementById(tabId);
  if (selectedPane) selectedPane.classList.add('active');
  
  if (btn) btn.classList.add('active');

  if (tabId === 'tab-live' && window.myLineChart) {
    window.myLineChart.resize();
  }
}

// --- SIDEBAR EXPAND/COLLAPSE LOGIC ---
function toggleSidebar() {
  const sidebar = document.getElementById('app-sidebar');
  if (sidebar) {
    sidebar.classList.toggle('expanded');
  }
}