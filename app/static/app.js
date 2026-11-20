// app.js - Client interaction logic for Surface Defect Inspection Platform

let currentFile = null;
let currentThreshold = 0.60;

// DOM Elements
const thresholdSlider = document.getElementById("thresholdSlider");
const thresholdValue = document.getElementById("thresholdValue");
const sampleGallery = document.getElementById("sampleGallery");

const dropZone = document.getElementById("dropZone");
const fileInput = document.getElementById("fileInput");
const dropPrompt = document.getElementById("dropPrompt");
const previewContainer = document.getElementById("previewContainer");
const imagePreview = document.getElementById("imagePreview");
const btnClear = document.getElementById("btnClear");
const btnAnalyze = document.getElementById("btnAnalyze");
const analyzeSpinner = document.getElementById("analyzeSpinner");

const resultsCard = document.getElementById("resultsCard");
const visualsCard = document.getElementById("visualsCard");
const verdictCard = document.getElementById("verdictCard");
const verdictIcon = document.getElementById("verdictIcon");
const verdictTitle = document.getElementById("verdictTitle");
const verdictDesc = document.getElementById("verdictDesc");
const metricClass = document.getElementById("metricClass");
const metricConfidence = document.getElementById("metricConfidence");
const metricSeverity = document.getElementById("metricSeverity");
const metricLatency = document.getElementById("metricLatency");
const latencyBadge = document.getElementById("latencyBadge");
const inspectionReportText = document.getElementById("inspectionReportText");
const probBarsContainer = document.getElementById("probBarsContainer");

const viewOriginal = document.getElementById("viewOriginal");
const viewBox = document.getElementById("viewBox");
const viewGradcam = document.getElementById("viewGradcam");
const viewSegmentation = document.getElementById("viewSegmentation");

// 1. Slider Event
thresholdSlider.addEventListener("input", (e) => {
  const val = e.target.value;
  currentThreshold = val / 100.0;
  thresholdValue.textContent = `${val}%`;
});

// 2. Fetch Sample Images Gallery
async function loadSampleGallery() {
  try {
    const res = await fetch("/samples");
    if (!res.ok) return;
    const samples = await res.json();
    sampleGallery.innerHTML = "";
    
    samples.forEach(sample => {
      const card = document.createElement("div");
      card.className = "sample-card";
      card.innerHTML = `
        <img src="${sample.url}" alt="${sample.class_name}">
        <span>${sample.class_name.replace('_', ' ')}</span>
      `;
      card.addEventListener("click", () => selectSampleImage(sample));
      sampleGallery.appendChild(card);
    });
  } catch (err) {
    console.error("Error loading sample gallery:", err);
  }
}

async function selectSampleImage(sample) {
  try {
    const response = await fetch(sample.url);
    const blob = await response.blob();
    const file = new File([blob], sample.filename, { type: blob.type || "image/jpeg" });
    handleSelectedFile(file);
  } catch (e) {
    console.error("Could not load sample blob:", e);
  }
}

// 3. Drop Zone & File Selection
dropZone.addEventListener("click", () => fileInput.click());

fileInput.addEventListener("change", (e) => {
  if (e.target.files && e.target.files[0]) {
    handleSelectedFile(e.target.files[0]);
  }
});

["dragenter", "dragover"].forEach(eventName => {
  dropZone.addEventListener(eventName, (e) => {
    e.preventDefault();
    e.stopPropagation();
    dropZone.classList.add("dragover");
  });
});

["dragleave", "drop"].forEach(eventName => {
  dropZone.addEventListener(eventName, (e) => {
    e.preventDefault();
    e.stopPropagation();
    dropZone.classList.remove("dragover");
  });
});

dropZone.addEventListener("drop", (e) => {
  if (e.dataTransfer.files && e.dataTransfer.files[0]) {
    handleSelectedFile(e.dataTransfer.files[0]);
  }
});

btnClear.addEventListener("click", (e) => {
  e.stopPropagation();
  clearSelectedFile();
});

function handleSelectedFile(file) {
  currentFile = file;
  const reader = new FileReader();
  reader.onload = (e) => {
    imagePreview.src = e.target.result;
    dropPrompt.style.display = "none";
    previewContainer.style.display = "flex";
    btnAnalyze.disabled = false;
  };
  reader.readAsDataURL(file);
}

function clearSelectedFile() {
  currentFile = null;
  fileInput.value = "";
  imagePreview.src = "";
  previewContainer.style.display = "none";
  dropPrompt.style.display = "block";
  btnAnalyze.disabled = true;
  resultsCard.style.display = "none";
  visualsCard.style.display = "none";
}

// 4. Analyze Action
btnAnalyze.addEventListener("click", async () => {
  if (!currentFile) return;

  btnAnalyze.disabled = true;
  analyzeSpinner.style.display = "inline-block";

  const formData = new FormData();
  formData.append("file", currentFile);

  try {
    const url = `/predict?threshold=${currentThreshold}`;
    const response = await fetch(url, {
      method: "POST",
      body: formData
    });

    const data = await response.json();
    if (!response.ok) {
      alert(`Inspection Error: ${data.detail || "Upload failed"}`);
      return;
    }

    renderInspectionResults(data);
  } catch (err) {
    alert(`Server Error: ${err.message}`);
  } finally {
    btnAnalyze.disabled = false;
    analyzeSpinner.style.display = "none";
  }
});

// 5. Render Results Function
function renderInspectionResults(data) {
  resultsCard.style.display = "block";
  visualsCard.style.display = "block";

  // Verdict
  verdictCard.className = `verdict-card ${data.verdict_color}`;
  verdictIcon.textContent = data.is_uncertain ? "!" : (data.verdict_status === "PASS" ? "✓" : "✗");
  verdictTitle.textContent = data.verdict;
  verdictDesc.textContent = data.verdict_message;

  // Key Metrics
  metricClass.textContent = data.predicted_class.replace("_", " ");
  metricConfidence.textContent = `${data.confidence_percentage}%`;
  metricLatency.textContent = `${data.inference_latency_ms} ms`;
  latencyBadge.textContent = `Latency: ${data.inference_latency_ms} ms`;

  // Estimate severity
  const bb = data.bounding_box;
  let severity = "Moderate";
  if (bb) {
    const areaPct = Math.round(((bb[2] - bb[0]) * (bb[3] - bb[1])) / (200 * 200) * 100);
    severity = areaPct > 35 ? "High (Severe)" : (areaPct < 12 ? "Minor" : "Moderate");
  }
  metricSeverity.textContent = severity;

  // Inspection Report Text
  inspectionReportText.textContent = data.inspection_report;

  // Probability Bars
  probBarsContainer.innerHTML = "";
  const dist = data.probability_distribution;
  for (const [cls, pct] of Object.entries(dist)) {
    const isTop = cls === data.predicted_class;
    const row = document.createElement("div");
    row.className = "prob-bar-row";
    row.innerHTML = `
      <span class="bar-class-name">${cls.replace("_", " ")}</span>
      <div class="bar-track">
        <div class="bar-fill ${isTop ? 'highlight' : ''}" style="width: ${pct}%"></div>
      </div>
      <span class="bar-percent">${pct}%</span>
    `;
    probBarsContainer.appendChild(row);
  }

  // Visualizations Matrix
  const v = data.visualizations;
  viewOriginal.src = v.original_image;
  viewBox.src = v.bounding_box_overlay || v.original_image;
  viewGradcam.src = v.gradcam_heatmap_overlay || v.original_image;
  viewSegmentation.src = v.segmentation_mask_overlay || v.original_image;

  // Smooth scroll down to results
  resultsCard.scrollIntoView({ behavior: "smooth", block: "start" });
}

// Initial Call
loadSampleGallery();
