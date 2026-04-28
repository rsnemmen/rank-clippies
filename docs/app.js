"use strict";

let currentData = null;

async function loadCategory(cat) {
  const resp = await fetch(`data/${cat}.json`);
  if (!resp.ok) throw new Error(`Failed to load data/${cat}.json: ${resp.status}`);
  return resp.json();
}

function formatGeneratedAt(generatedAt) {
  if (!generatedAt) return "unknown";
  const timestamp = Date.parse(generatedAt);
  if (Number.isNaN(timestamp)) return generatedAt;
  return new Intl.DateTimeFormat("en-US", {
    dateStyle: "medium",
    timeStyle: "short",
    timeZone: "UTC",
  }).format(timestamp) + " UTC";
}

function updateDataTimestamp(data) {
  const el = document.getElementById("data-updated-at");
  el.textContent = `Benchmark data last updated: ${formatGeneratedAt(data.generated_at)}`;
}

function getFilters() {
  return {
    hideOpen: document.getElementById("hideOpen").checked,
    hideSparse: document.getElementById("hideSparse").checked,
    showQuadrants: document.getElementById("showQuadrants").checked,
  };
}

function filterModels(models, { hideOpen, hideSparse }) {
  return models.filter(m => {
    if (hideOpen && m.is_open) return false;
    if (hideSparse && m.n_bench <= 2) return false;
    return true;
  });
}

// Deterministic pseudo-random for reproducible jitter
function pseudoRand(seed) {
  const x = Math.sin(seed + 1) * 10000;
  return x - Math.floor(x);
}

// ── Scatter chart ────────────────────────────────────────────────────────────

function buildScatterTraces(models, palette) {
  // One trace per (tier, is_open) pair
  const groups = new Map();
  for (const m of models) {
    if (m.rel_cost == null) continue;
    const key = `${m.tier}-${m.is_open ? 1 : 0}`;
    if (!groups.has(key)) groups.set(key, { tier: m.tier, is_open: m.is_open, models: [] });
    groups.get(key).models.push(m);
  }

  const traces = [];
  const tiersSeen = new Set();

  const sortedGroups = [...groups.entries()].sort(([a], [b]) => a.localeCompare(b));
  for (const [, grp] of sortedGroups) {
    const { tier, is_open, models: gm } = grp;
    const color = palette[(tier - 1) % palette.length];
    const lgKey = `tier-${tier}`;
    const showLegend = !tiersSeen.has(lgKey);
    if (showLegend) tiersSeen.add(lgKey);

    traces.push({
      type: "scatter",
      x: gm.map(m => m.rel_cost),
      y: gm.map(m => m.avg_pct * 100),
      error_y: {
        type: "data",
        array: gm.map(m => (m.semi_iqr ?? 0) * 100),
        visible: true,
        color,
        thickness: 1.5,
        width: 4,
      },
      mode: "markers",
      marker: {
        symbol: is_open ? "diamond" : "circle",
        size: 10,
        color,
        line: { color: "white", width: 1 },
        opacity: 0.88,
      },
      name: `Tier ${tier}`,
      legendgroup: lgKey,
      showlegend: showLegend,
      customdata: gm.map(m => ({
        rank: m.rank,
        name: m.name,
        n_bench: m.n_bench,
        cost: m.cost != null ? m.cost.toFixed(0) : "N/A",
        open: m.is_open,
        raw: m.raw_scores.map(s => (s * 100).toFixed(1)).join(", ") || "—",
      })),
      hovertemplate:
        "<b>%{customdata.name}</b> (Rank %{customdata.rank})<br>" +
        "Score: %{y:.1f}%ile<br>" +
        "Cost: %{x:.2f}× best<br>" +
        "Benchmarks (n=%{customdata.n_bench}): %{customdata.raw}<br>" +
        "%{customdata.open}<extra></extra>",
    });
  }

  return traces;
}

function buildScatterLayout(models, palette, showQuadrants, title) {
  const plottable = models.filter(m => m.rel_cost != null);
  const scores = plottable.map(m => m.avg_pct * 100);
  const costs = plottable.map(m => m.rel_cost);
  const logCosts = costs.map(c => Math.log10(c));
  const maxScore = scores.length ? Math.max(...scores) : 100;
  const upper = Math.min(Math.ceil(maxScore / 10) * 10, 100);
  const xPadding = 0.4;
  const logXMin = logCosts.length ? Math.min(...logCosts) - xPadding : -xPadding;
  const logXMax = logCosts.length ? Math.max(...logCosts) + xPadding : xPadding;

  const layout = {
    title: { text: `LLM Performance vs. Cost<br><sub>${title}</sub>`, x: 0.5, xanchor: "center" },
    xaxis: {
      title: "Cost relative to best model (log scale)",
      type: "log",
      range: [logXMin, logXMax],
      showgrid: true,
      gridcolor: "#d0d0d0",
      gridwidth: 0.6,
      zeroline: false,
      showline: true,
      linewidth: 0.8,
      mirror: false,
      tickfont: { size: 10 },
    },
    yaxis: {
      title: "Percentile rank (lower = better)",
      range: [upper, 0],
      showgrid: true,
      gridcolor: "#d0d0d0",
      gridwidth: 0.6,
      zeroline: false,
      showline: true,
      linewidth: 0.8,
      tickfont: { size: 10 },
    },
    legend: {
      title: { text: "Performance tier" },
      x: 1.02,
      y: 0.98,
      yanchor: "top",
      bgcolor: "rgba(255,255,255,0.9)",
      bordercolor: "#cccccc",
      borderwidth: 1,
    },
    plot_bgcolor: "white",
    paper_bgcolor: "white",
    margin: { t: 80, r: 160, b: 60, l: 70 },
    height: 560,
    font: { family: 'Arial, "Helvetica Neue", Helvetica, sans-serif', size: 11 },
    shapes: [],
    annotations: [],
  };

  if (showQuadrants && costs.length >= 2) {
    const logXMid = logCosts.reduce((a, b) => a + b, 0) / logCosts.length;
    const xMin = 10 ** logXMin;
    const yMid = [...scores].sort((a, b) => a - b)[Math.floor(scores.length / 2)];
    const xMid = 10 ** logXMid;
    const xMax = 10 ** logXMax;
    const labelOffset = 0.15;
    const leftLabelX = 10 ** (logXMin + labelOffset);
    const rightLabelX = 10 ** (logXMax - labelOffset);

    layout.shapes = [
      { type: "rect", x0: xMin, x1: xMid, y0: 0,    y1: yMid,  fillcolor: "#a8d8a8", opacity: 0.25, line: { width: 0 }, layer: "below" },
      { type: "rect", x0: xMid, x1: xMax, y0: 0,    y1: yMid,  fillcolor: "#a8c8e8", opacity: 0.25, line: { width: 0 }, layer: "below" },
      { type: "rect", x0: xMin, x1: xMid, y0: yMid, y1: upper, fillcolor: "#f8e8a0", opacity: 0.25, line: { width: 0 }, layer: "below" },
      { type: "rect", x0: xMid, x1: xMax, y0: yMid, y1: upper, fillcolor: "#f8b8b8", opacity: 0.25, line: { width: 0 }, layer: "below" },
      { type: "line", x0: xMid, x1: xMid, y0: 0,    y1: upper, line: { color: "#aaaaaa", width: 1, dash: "dash" }, layer: "below" },
      { type: "line", x0: xMin, x1: xMax, y0: yMid, y1: yMid,  line: { color: "#aaaaaa", width: 1, dash: "dash" }, layer: "below" },
    ];
    layout.annotations = [
      { x: leftLabelX, y: yMid * 0.08 + 1, xref: "x", yref: "y", text: "Best value", showarrow: false, font: { size: 12, color: "#888888" }, fontstyle: "italic" },
      { x: rightLabelX, y: yMid * 0.08 + 1, xref: "x", yref: "y", text: "Premium", showarrow: false, font: { size: 12, color: "#888888" }, xanchor: "right" },
      { x: leftLabelX, y: upper - 2, xref: "x", yref: "y", text: "Budget", showarrow: false, font: { size: 12, color: "#888888" }, yanchor: "bottom" },
      { x: rightLabelX, y: upper - 2, xref: "x", yref: "y", text: "Avoid", showarrow: false, font: { size: 12, color: "#888888" }, xanchor: "right", yanchor: "bottom" },
    ];
  }

  return layout;
}

// ── Ranking chart ────────────────────────────────────────────────────────────

function buildRankingTraces(models, palette) {
  const sorted = [...models].sort((a, b) => a.avg_pct - b.avg_pct);
  const n = sorted.length;

  const costs = sorted.map(m => m.cost).filter(c => c != null);
  const logMin = costs.length ? Math.log10(Math.min(...costs)) : 0;
  const logMax = costs.length ? Math.log10(Math.max(...costs)) : 1;

  const traces = [];
  const tiers = [...new Set(sorted.map(m => m.tier))].sort((a, b) => a - b);

  for (const tier of tiers) {
    const color = palette[(tier - 1) % palette.length];

    for (const isOpen of [false, true]) {
      const gm = sorted
        .map((m, i) => ({ ...m, pos: i }))
        .filter(m => m.tier === tier && m.is_open === isOpen);
      if (!gm.length) continue;

      const markerSizes = gm.map(m => {
        if (m.cost == null || logMax === logMin) return 8;
        return 5 + ((Math.log10(m.cost) - logMin) / (logMax - logMin)) * 12;
      });

      traces.push({
        type: "scatter",
        x: gm.map(m => m.avg_pct * 100),
        y: gm.map(m => m.pos),
        error_x: {
          type: "data",
          array: gm.map(m => (m.semi_iqr ?? 0) * 100),
          visible: true,
          color,
          thickness: 1.2,
          width: 3,
        },
        mode: "markers",
        marker: {
          symbol: isOpen ? "diamond" : "circle",
          size: markerSizes,
          color,
          line: { color: "white", width: 0.8 },
          opacity: 0.88,
        },
        name: `Tier ${tier}`,
        legendgroup: `tier-${tier}`,
        showlegend: !isOpen,
        customdata: gm.map(m => ({
          rank: m.rank,
          name: m.name,
          n_bench: m.n_bench,
          raw: m.raw_scores.map(s => (s * 100).toFixed(1)).join(", ") || "—",
          open: m.is_open,
        })),
        hovertemplate:
          "<b>%{customdata.name}</b> (Rank %{customdata.rank})<br>" +
          "Score: %{x:.1f}%ile<br>" +
          "Benchmarks (n=%{customdata.n_bench}): %{customdata.raw}<br>" +
          "%{customdata.open}<extra></extra>",
      });
    }
  }

  // Jittered raw-score overlay dots
  const rawX = [], rawY = [], rawColors = [];
  for (const [i, m] of sorted.entries()) {
    const color = palette[(m.tier - 1) % palette.length];
    for (const [j, score] of m.raw_scores.entries()) {
      rawX.push(score * 100);
      rawY.push(i + (pseudoRand(i * 100 + j) - 0.5) * 0.28);
      rawColors.push(color);
    }
  }
  if (rawX.length) {
    traces.push({
      type: "scatter",
      x: rawX,
      y: rawY,
      mode: "markers",
      marker: { size: 4, color: rawColors, opacity: 0.45, line: { color: "white", width: 0.4 } },
      name: "Individual benchmarks",
      showlegend: true,
      hoverinfo: "skip",
    });
  }

  return { traces, sorted, n };
}

function buildRankingLayout(sorted, n, title) {
  const scores = sorted.map(m => m.avg_pct * 100);
  const maxScore = scores.length ? Math.max(...scores) : 100;
  const upper = Math.min(Math.ceil(maxScore / 10) * 10, 100);

  const tickvals = sorted.map((_, i) => i);
  const ticktext = sorted.map((m, i) => {
    const suffix = m.n_bench === 1 ? " ‡" : m.n_bench === 2 ? " †" : "";
    return `${i + 1}. ${m.name}${suffix}`;
  });

  // Alternating row background shapes
  const rowShapes = sorted.map((_, i) => ({
    type: "rect",
    x0: 0, x1: upper,
    y0: i - 0.5, y1: i + 0.5,
    fillcolor: i % 2 === 0 ? "#f5f5f5" : "white",
    opacity: 1,
    line: { width: 0 },
    layer: "below",
  }));

  return {
    title: { text: `Model Ranking — ${title}`, x: 0.5, xanchor: "center" },
    xaxis: {
      title: "Percentile rank (lower = better)",
      range: [0, upper],
      showgrid: true,
      gridcolor: "#d0d0d0",
      zeroline: false,
      showline: true,
      linewidth: 0.8,
      tickfont: { size: 10 },
    },
    yaxis: {
      tickvals,
      ticktext,
      tickfont: { size: 9.5 },
      range: [n - 0.5, -0.5],
      showline: false,
      showgrid: false,
      zeroline: false,
      ticklen: 0,
    },
    legend: {
      title: { text: "Tier" },
      x: 1.02,
      y: 1,
      yanchor: "top",
      bgcolor: "rgba(255,255,255,0.9)",
      bordercolor: "#cccccc",
      borderwidth: 1,
    },
    plot_bgcolor: "white",
    paper_bgcolor: "white",
    margin: { t: 60, r: 160, b: 50, l: 185 },
    height: Math.max(320, n * 25 + 90),
    font: { family: 'Arial, "Helvetica Neue", Helvetica, sans-serif', size: 11 },
    shapes: rowShapes,
  };
}

// ── Render ───────────────────────────────────────────────────────────────────

const PLOTLY_CONFIG = { responsive: true, displayModeBar: true, displaylogo: false };

function renderCharts(data) {
  const filters = getFilters();
  const models = filterModels(data.models, filters);
  const { palette, title } = data;

  const scatterTraces = buildScatterTraces(models, palette);
  const scatterLayout = buildScatterLayout(models, palette, filters.showQuadrants, title);
  Plotly.react("scatter-chart", scatterTraces, scatterLayout, PLOTLY_CONFIG);

  const { traces: rankTraces, sorted, n } = buildRankingTraces(models, palette);
  const rankLayout = buildRankingLayout(sorted, n, title);
  Plotly.react("ranking-chart", rankTraces, rankLayout, PLOTLY_CONFIG);
}

// ── Bootstrap ────────────────────────────────────────────────────────────────

async function switchCategory(cat) {
  document.querySelectorAll(".tab").forEach(t => t.classList.toggle("active", t.dataset.cat === cat));
  try {
    currentData = await loadCategory(cat);
    updateDataTimestamp(currentData);
    renderCharts(currentData);
  } catch (err) {
    console.error(err);
    updateDataTimestamp({ generated_at: null });
    document.getElementById("scatter-chart").innerHTML =
      `<p style="color:#c00;padding:1rem">Error loading data: ${err.message}</p>`;
  }
}

document.addEventListener("DOMContentLoaded", () => {
  document.querySelectorAll(".tab").forEach(tab => {
    tab.addEventListener("click", () => switchCategory(tab.dataset.cat));
  });
  ["hideOpen", "hideSparse", "showQuadrants"].forEach(id => {
    document.getElementById(id).addEventListener("change", () => {
      if (currentData) renderCharts(currentData);
    });
  });
  switchCategory("general");
});
