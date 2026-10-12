/* Shared Chart.js helpers: theme-aware colours, INR formatting, defaults. */
(function () {
  const css = (name) => getComputedStyle(document.documentElement).getPropertyValue(name).trim();

  window.T = function () {
    return {
      s1: css("--series-1"), s2: css("--series-2"), s3: css("--series-3"),
      text: css("--text"), text2: css("--text-2"), muted: css("--muted"),
      grid: css("--grid"), surface: css("--surface"), neutral: css("--neutral-mark"),
      good: css("--good"), bad: css("--bad"),
    };
  };

  window.alpha = (hex, a) => {
    const h = hex.replace("#", "");
    const n = parseInt(h.length === 3 ? h.split("").map((c) => c + c).join("") : h, 16);
    return `rgba(${(n >> 16) & 255}, ${(n >> 8) & 255}, ${n & 255}, ${a})`;
  };

  // ₹ formatting (Indian grouping) and lakhs-per-annum
  window.inr = (v) => "₹" + Math.round(v).toLocaleString("en-IN");
  window.lpa = (v) => (v / 1e5).toFixed(2) + " LPA";
  window.lpaShort = (v) => (v / 1e5).toFixed(0) + "L";

  window.applyChartDefaults = function () {
    const t = T();
    Chart.defaults.font.family = getComputedStyle(document.body).fontFamily;
    Chart.defaults.font.size = 12;
    Chart.defaults.color = t.text2;
    Chart.defaults.borderColor = t.grid;
    Chart.defaults.maintainAspectRatio = false;
    Chart.defaults.animation.duration = 500;
    Chart.defaults.plugins.legend.display = false;
    Chart.defaults.plugins.tooltip.backgroundColor = t.text;
    Chart.defaults.plugins.tooltip.titleColor = t.surface;
    Chart.defaults.plugins.tooltip.bodyColor = t.surface;
    Chart.defaults.plugins.tooltip.padding = 10;
    Chart.defaults.plugins.tooltip.cornerRadius = 8;
    Chart.defaults.plugins.tooltip.displayColors = false;
    Chart.defaults.elements.bar.borderRadius = 4;
    Chart.defaults.elements.bar.borderSkipped = "start";
    Chart.defaults.elements.line.borderWidth = 2;
    Chart.defaults.elements.point.radius = 0;
    Chart.defaults.elements.point.hoverRadius = 5;
    Chart.defaults.scale.grid.drawTicks = false;
    Chart.defaults.scale.border = { display: false };
    Chart.defaults.scale.ticks.padding = 8;
  };

  // registry so charts can be rebuilt when the theme changes
  const builders = [];
  window.registerChart = function (build) { builders.push({ build, chart: null }); };
  window.renderCharts = function () {
    applyChartDefaults();
    builders.forEach((b) => { if (b.chart) b.chart.destroy(); b.chart = b.build(T()); });
  };

  // y-axis helpers
  window.lpaAxis = (extra = {}) => ({ beginAtZero: true, ticks: { callback: (v) => lpaShort(v) }, ...extra });
  window.noGridAxis = (extra = {}) => ({ grid: { display: false }, ...extra });
})();
