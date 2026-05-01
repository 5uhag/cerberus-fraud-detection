const uploadForm = document.getElementById("uploadForm");
const fileInput = document.getElementById("transactionsFile");
const fileLabel = document.getElementById("fileLabel");
const uploadBox = document.getElementById("uploadBox");
const errorPanel = document.getElementById("errorPanel");
const summarySection = document.getElementById("summarySection");
const tableSection = document.getElementById("tableSection");
const downloadBtn = document.getElementById("downloadBtn");
const resultTable = document.getElementById("resultTable");
const thresholdInput = document.getElementById("threshold");
const analyticsSection = document.getElementById("analyticsSection");
const metricsSection = document.getElementById("metricsSection");
const metricPrecision = document.getElementById("metricPrecision");
const metricRecall = document.getElementById("metricRecall");
const metricF1 = document.getElementById("metricF1");
const metricConfusion = document.getElementById("metricConfusion");

const totalTxn = document.getElementById("totalTxn");
const fraudTxn = document.getElementById("fraudTxn");
const legitTxn = document.getElementById("legitTxn");
const fraudRate = document.getElementById("fraudRate");

function toggleError(message) {
  if (!message) {
    errorPanel.classList.add("hidden");
    errorPanel.textContent = "";
    return;
  }
  errorPanel.classList.remove("hidden");
  errorPanel.textContent = message;
}

function renderTable(rows) {
  if (!rows || rows.length === 0) {
    tableSection.classList.add("hidden");
    return;
  }

  const headers = Object.keys(rows[0]);
  let html = "<thead><tr>";
  headers.forEach((header) => {
    html += `<th>${header}</th>`;
  });
  html += "</tr></thead><tbody>";

  rows.forEach((row) => {
    const isFraud = row["prediction"] === "Fraud";
    html += `<tr class="${isFraud ? "row-fraud" : "row-legit"}">`;
    headers.forEach((header) => {
      let cellClass = "";
      if (header === "prediction") {
        cellClass = isFraud ? "cell-fraud" : "cell-legit";
      }
      html += `<td${cellClass ? ` class="${cellClass}"` : ""}>${row[header]}</td>`;
    });
    html += "</tr>";
  });

  html += "</tbody>";
  resultTable.innerHTML = html;
  tableSection.classList.remove("hidden");
}

function updateSummary(summary) {
  totalTxn.textContent = summary.total_transactions;
  fraudTxn.textContent = summary.fraud_transactions;
  legitTxn.textContent = summary.legit_transactions;
  fraudRate.textContent = `${summary.fraud_rate}%`;
  summarySection.classList.remove("hidden");
}

function formatConfusionMatrix(matrix) {
  if (!Array.isArray(matrix) || matrix.length !== 2 || !Array.isArray(matrix[0]) || !Array.isArray(matrix[1])) {
    return "Unavailable";
  }

  return `TN ${matrix[0][0]} | FP ${matrix[0][1]} | FN ${matrix[1][0]} | TP ${matrix[1][1]}`;
}

async function loadMetrics() {
  if (!metricsSection) {
    return;
  }

  try {
    const response = await fetch("/metrics");
    if (!response.ok) {
      metricsSection.classList.add("hidden");
      return;
    }

    const payload = await response.json();
    if (!payload.available) {
      metricsSection.classList.add("hidden");
      return;
    }

    metricPrecision.textContent = Number(payload.precision || 0).toFixed(3);
    metricRecall.textContent = Number(payload.recall || 0).toFixed(3);
    metricF1.textContent = Number(payload.f1_score || 0).toFixed(3);
    metricConfusion.textContent = formatConfusionMatrix(payload.confusion_matrix);
    metricsSection.classList.remove("hidden");
  } catch {
    metricsSection.classList.add("hidden");
  }
}

fileInput.addEventListener("change", () => {
  if (fileInput.files.length > 0) {
    fileLabel.textContent = `Selected file: ${fileInput.files[0].name}`;
  } else {
    fileLabel.textContent = "Required columns: Time, Amount, V1 to V28";
  }
});

["dragenter", "dragover"].forEach((eventName) => {
  uploadBox.addEventListener(eventName, (event) => {
    event.preventDefault();
    uploadBox.classList.add("dragging");
  });
});

["dragleave", "drop"].forEach((eventName) => {
  uploadBox.addEventListener(eventName, (event) => {
    event.preventDefault();
    uploadBox.classList.remove("dragging");
  });
});

uploadBox.addEventListener("drop", (event) => {
  const files = event.dataTransfer.files;
  if (files && files.length > 0) {
    fileInput.files = files;
    fileLabel.textContent = `Selected file: ${files[0].name}`;
  }
});

uploadForm.addEventListener("submit", async (event) => {
  event.preventDefault();
  toggleError("");

  if (!fileInput.files.length) {
    toggleError("Upload a CSV file before analysis.");
    return;
  }

  const submitBtn = uploadForm.querySelector('button[type="submit"]');
  const originalLabel = submitBtn.textContent;
  submitBtn.disabled = true;
  submitBtn.textContent = "Analyzing…";

  const formData = new FormData(uploadForm);

  try {
    const response = await fetch("/predict", {
      method: "POST",
      body: formData,
    });

    const payload = await response.json();
    if (!response.ok) {
      toggleError(payload.error || "Prediction failed.");
      return;
    }

    updateSummary(payload.summary);
    renderAnalytics(payload.analytics, payload.summary);
    renderTable(payload.preview);
    downloadBtn.disabled = false;
  } catch {
    toggleError("Unexpected server error. Please try again.");
  } finally {
    submitBtn.disabled = false;
    submitBtn.textContent = originalLabel;
  }
});

downloadBtn.addEventListener("click", async () => {
  toggleError("");

  if (!fileInput.files.length) {
    toggleError("Upload a CSV file before downloading results.");
    return;
  }

  const formData = new FormData(uploadForm);

  try {
    const response = await fetch("/download", {
      method: "POST",
      body: formData,
    });

    if (!response.ok) {
      const payload = await response.json();
      toggleError(payload.error || "Download failed.");
      return;
    }

    const blob = await response.blob();
    const url = window.URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = "fraud_predictions.csv";
    document.body.appendChild(a);
    a.click();
    a.remove();
    window.URL.revokeObjectURL(url);
  } catch {
    toggleError("Could not download results right now.");
  }
});

// Generate sample CSV and trigger download
const generateBtn = document.getElementById("generateBtn");
if (generateBtn) {
  generateBtn.addEventListener("click", async () => {
    toggleError("");
    const rows = document.getElementById("genRows").value || 10;
    try {
      const response = await fetch(`/generate_sample?rows=${encodeURIComponent(rows)}`, { method: "GET" });
      if (!response.ok) {
        const payload = await response.json();
        toggleError(payload.error || "Could not generate sample.");
        return;
      }
      const blob = await response.blob();
      const url = window.URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = `generated_sample_${rows}.csv`;
      document.body.appendChild(a);
      a.click();
      a.remove();
      window.URL.revokeObjectURL(url);
    } catch (e) {
      toggleError("Could not generate sample right now.");
    }
  });
}

const chartInstances = {};

function destroyChart(id) {
  if (chartInstances[id]) {
    chartInstances[id].destroy();
    delete chartInstances[id];
  }
}

function renderAnalytics(analytics, summary) {
  if (!analyticsSection || !analytics) return;
  analyticsSection.classList.remove("hidden");

  // 1. Donut — Fraud vs Legit
  destroyChart("donut");
  chartInstances["donut"] = new Chart(document.getElementById("chartDonut"), {
    type: "doughnut",
    data: {
      labels: ["Legit", "Fraud"],
      datasets: [{
        data: [summary.legit_transactions, summary.fraud_transactions],
        backgroundColor: ["#1b8f5a", "#cc3d2f"],
        borderWidth: 2,
        borderColor: "#fff",
      }],
    },
    options: { plugins: { legend: { position: "bottom" } }, cutout: "62%" },
  });

  // 2. Bar — Amount distribution
  destroyChart("amount");
  chartInstances["amount"] = new Chart(document.getElementById("chartAmount"), {
    type: "bar",
    data: {
      labels: analytics.amount_distribution.map((b) => b.label),
      datasets: [
        {
          label: "Legit",
          data: analytics.amount_distribution.map((b) => b.total - b.fraud),
          backgroundColor: "rgba(27,143,90,0.7)",
        },
        {
          label: "Fraud",
          data: analytics.amount_distribution.map((b) => b.fraud),
          backgroundColor: "rgba(204,61,47,0.7)",
        },
      ],
    },
    options: {
      plugins: { legend: { position: "bottom" } },
      scales: { x: { stacked: true }, y: { stacked: true, beginAtZero: true } },
    },
  });

  // 3. Bar — Probability distribution
  destroyChart("prob");
  chartInstances["prob"] = new Chart(document.getElementById("chartProb"), {
    type: "bar",
    data: {
      labels: analytics.probability_distribution.map((b) => b.label),
      datasets: [{
        label: "Transactions",
        data: analytics.probability_distribution.map((b) => b.count),
        backgroundColor: analytics.probability_distribution.map((_, i) =>
          i >= 5 ? "rgba(204,61,47,0.7)" : "rgba(15,123,140,0.6)"
        ),
      }],
    },
    options: {
      plugins: { legend: { display: false } },
      scales: { y: { beginAtZero: true } },
    },
  });

  // 4. Scatter — Amount over Time
  destroyChart("scatter");
  const legitPts = analytics.time_series.filter((p) => p.fraud === 0).map((p) => ({ x: p.time, y: p.amount }));
  const fraudPts = analytics.time_series.filter((p) => p.fraud === 1).map((p) => ({ x: p.time, y: p.amount }));
  chartInstances["scatter"] = new Chart(document.getElementById("chartScatter"), {
    type: "scatter",
    data: {
      datasets: [
        { label: "Legit", data: legitPts, backgroundColor: "rgba(27,143,90,0.5)", pointRadius: 3 },
        { label: "Fraud", data: fraudPts, backgroundColor: "rgba(204,61,47,0.8)", pointRadius: 4 },
      ],
    },
    options: {
      plugins: { legend: { position: "bottom" } },
      scales: {
        x: { title: { display: true, text: "Time (s)" } },
        y: { title: { display: true, text: "Amount ($)" }, beginAtZero: true },
      },
    },
  });

  // 5. Top risks table
  const topTable = document.getElementById("topRisksTable");
  let html = "<thead><tr><th>Row</th><th>Amount</th><th>Fraud Probability</th></tr></thead><tbody>";
  analytics.top_risks.forEach((r) => {
    html += `<tr><td>#${r.row + 1}</td><td>$${r.amount.toFixed(2)}</td><td class="cell-fraud">${r.probability}%</td></tr>`;
  });
  html += "</tbody>";
  topTable.innerHTML = html;
}

loadMetrics();
