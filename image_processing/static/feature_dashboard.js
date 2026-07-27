const form = document.querySelector("#upload-form");
const input = document.querySelector("#image-input");
const dropZone = document.querySelector("#drop-zone");
const dropEmpty = document.querySelector("#drop-empty");
const previewWrap = document.querySelector("#preview-wrap");
const previewImage = document.querySelector("#image-preview");
const previewName = document.querySelector("#preview-name");
const previewSize = document.querySelector("#preview-size");
const analyzeButton = document.querySelector("#analyze-button");
const changeImageButton = document.querySelector("#change-image");
const formMessage = document.querySelector("#form-message");
const emptyResult = document.querySelector("#empty-result");
const loadingResult = document.querySelector("#loading-result");
const summaryResult = document.querySelector("#summary-result");
const featureSection = document.querySelector("#feature-section");
const tableSection = document.querySelector("#table-section");
const spectrumChart = document.querySelector("#spectrum-chart");
const tableBody = document.querySelector("#feature-table-body");
const downloadButton = document.querySelector("#download-json");
const filterButtons = [...document.querySelectorAll(".filter-button")];

const groupColors = {
  intensity: "var(--ochre)",
  texture: "var(--blue)",
  frequency: "var(--green)",
};

let selectedFile = null;
let latestResult = null;
let previewUrl = null;

function formatBytes(bytes) {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(2)} MB`;
}

function formatValue(value) {
  if (value === 0) return "0";
  const absolute = Math.abs(value);
  if (absolute < 0.0001 || absolute >= 10000) return value.toExponential(6);
  return Number(value).toFixed(9).replace(/0+$/, "").replace(/\.$/, "");
}

function escapeHtml(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

function setSelectedFile(file) {
  if (!file) return;
  if (file.type && !file.type.startsWith("image/")) {
    showError("这个文件不是可识别的图片，请选择 PNG、JPG、BMP 或 TIFF 文件。");
    return;
  }
  if (file.size > 16 * 1024 * 1024) {
    showError("图片超过 16 MB，请压缩后再试。");
    return;
  }

  selectedFile = file;
  formMessage.textContent = "";
  if (previewUrl) URL.revokeObjectURL(previewUrl);
  previewUrl = URL.createObjectURL(file);
  previewImage.src = previewUrl;
  previewName.textContent = file.name;
  previewSize.textContent = formatBytes(file.size);
  dropEmpty.hidden = true;
  previewWrap.hidden = false;
  analyzeButton.disabled = false;
  changeImageButton.disabled = false;
}

function resetSelection() {
  selectedFile = null;
  input.value = "";
  if (previewUrl) URL.revokeObjectURL(previewUrl);
  previewUrl = null;
  previewImage.removeAttribute("src");
  previewWrap.hidden = true;
  dropEmpty.hidden = false;
  analyzeButton.disabled = true;
  changeImageButton.disabled = true;
  formMessage.textContent = "";
  input.click();
}

function showError(message) {
  formMessage.textContent = message;
  emptyResult.hidden = false;
  loadingResult.hidden = true;
  summaryResult.hidden = true;
  dropZone.classList.remove("is-analyzing");
  analyzeButton.disabled = !selectedFile;
  analyzeButton.querySelector(".button-label").textContent = "重新分析";
}

function setLoading(isLoading) {
  dropZone.classList.toggle("is-analyzing", isLoading);
  analyzeButton.disabled = isLoading || !selectedFile;
  changeImageButton.disabled = isLoading || !selectedFile;
  analyzeButton.querySelector(".button-label").textContent = isLoading ? "正在解析图像…" : "开始特征解析";
  if (isLoading) {
    emptyResult.hidden = true;
    loadingResult.hidden = false;
    summaryResult.hidden = true;
  } else {
    loadingResult.hidden = true;
  }
}

function renderSummary(result) {
  document.querySelector("#scene-label").textContent = result.scene_label;
  document.querySelector("#scene-code").textContent = result.scene;
  document.querySelector("#confidence-value").textContent = `${(result.confidence * 100).toFixed(2)}%`;
  document.querySelector("#confidence-bar").style.width = `${result.confidence * 100}%`;
  document.querySelector("#fact-dimensions").textContent = `${result.image.width} × ${result.image.height} px`;
  document.querySelector("#fact-format").textContent = `${result.image.format} / ${result.image.mode}`;
  document.querySelector("#fact-time").textContent = `${result.processing_ms} ms`;
  document.querySelector("#fact-count").textContent = `${result.selected_feature_count} 项`;

  const probabilityList = document.querySelector("#probability-list");
  probabilityList.innerHTML = result.probabilities.map((item) => `
    <div class="probability-row ${item.scene === result.scene ? "is-predicted" : ""}">
      <span>${escapeHtml(item.label)}</span>
      <span class="probability-bar"><i style="--value: ${(item.value * 100).toFixed(3)}%"></i></span>
      <strong>${(item.value * 100).toFixed(1)}%</strong>
    </div>
  `).join("");

  emptyResult.hidden = true;
  loadingResult.hidden = true;
  summaryResult.hidden = false;
}

function renderSpectrum(features) {
  const axis = `
    <div class="spectrum-axis" aria-hidden="true">
      <span>特征名称</span>
      <span class="axis-labels"><i>−3σ</i><i>0</i><i>+3σ</i></span>
      <span>z-score</span>
    </div>
  `;
  const rows = features.map((feature, index) => {
    const clamped = Math.max(-3, Math.min(3, feature.z_score));
    const position = ((clamped + 3) / 6) * 100;
    const sign = feature.z_score > 0 ? "+" : "";
    return `
      <div class="spectrum-row" title="${escapeHtml(feature.description)}">
        <span class="spectrum-name">${escapeHtml(feature.label)}</span>
        <span class="feature-scale">
          <i class="feature-marker ${feature.group}" style="--position: ${position}%; --i: ${index}"></i>
        </span>
        <strong class="spectrum-value">${sign}${feature.z_score.toFixed(2)}</strong>
      </div>
    `;
  }).join("");
  spectrumChart.innerHTML = axis + rows;
}

function renderTable(features, group = "all") {
  const visible = group === "all" ? features : features.filter((feature) => feature.group === group);
  const maxImportance = Math.max(...features.map((feature) => feature.importance), 0.000001);
  tableBody.innerHTML = visible.map((feature) => {
    const scoreClass = feature.z_score >= 0 ? "positive" : "negative";
    const sign = feature.z_score > 0 ? "+" : "";
    const importanceWidth = (feature.importance / maxImportance) * 100;
    return `
      <tr title="${escapeHtml(feature.description)}">
        <td class="feature-index">${String(feature.index).padStart(2, "0")}</td>
        <td>
          <span class="feature-name-cell">
            <i class="feature-dot" style="--group-color: ${groupColors[feature.group]}"></i>
            ${escapeHtml(feature.label)}
          </span>
        </td>
        <td><code class="feature-code">${escapeHtml(feature.name)}</code></td>
        <td class="raw-value">${formatValue(feature.value)}</td>
        <td class="z-score ${scoreClass}">${sign}${feature.z_score.toFixed(4)}</td>
        <td>
          <span class="importance-cell">
            ${feature.importance.toFixed(4)}
            <i class="importance-track"><i style="--importance: ${importanceWidth}%"></i></i>
          </span>
        </td>
      </tr>
    `;
  }).join("");
}

function renderResult(result) {
  latestResult = result;
  renderSummary(result);
  renderSpectrum(result.features);
  renderTable(result.features);
  filterButtons.forEach((button) => button.classList.toggle("active", button.dataset.group === "all"));
  featureSection.hidden = false;
  tableSection.hidden = false;
  featureSection.scrollIntoView({ behavior: "smooth", block: "start" });
}

async function analyzeImage(event) {
  event.preventDefault();
  if (!selectedFile) {
    showError("请先选择一张图片。");
    return;
  }

  formMessage.textContent = "";
  setLoading(true);
  const payload = new FormData();
  payload.append("image", selectedFile, selectedFile.name);

  try {
    const response = await fetch("/api/analyze", { method: "POST", body: payload });
    const result = await response.json();
    if (!response.ok) throw new Error(result.message || "图像分析失败，请稍后重试。");
    renderResult(result);
  } catch (error) {
    showError(error.message || "无法连接分析服务，请确认程序仍在运行。");
  } finally {
    setLoading(false);
  }
}

function downloadResult() {
  if (!latestResult) return;
  const exportData = {
    image: latestResult.image,
    scene: latestResult.scene,
    scene_label: latestResult.scene_label,
    confidence: latestResult.confidence,
    probabilities: Object.fromEntries(latestResult.probabilities.map((item) => [item.scene, item.value])),
    selected_feature_count: latestResult.selected_feature_count,
    selected_feature_values: Object.fromEntries(latestResult.features.map((feature) => [feature.name, feature.value])),
  };
  const blob = new Blob([JSON.stringify(exportData, null, 2)], { type: "application/json;charset=utf-8" });
  const url = URL.createObjectURL(blob);
  const anchor = document.createElement("a");
  const baseName = latestResult.image.name.replace(/\.[^.]+$/, "");
  anchor.href = url;
  anchor.download = `${baseName}_特征值.json`;
  document.body.append(anchor);
  anchor.click();
  anchor.remove();
  URL.revokeObjectURL(url);
}

input.addEventListener("change", () => setSelectedFile(input.files[0]));
form.addEventListener("submit", analyzeImage);
changeImageButton.addEventListener("click", resetSelection);
downloadButton.addEventListener("click", downloadResult);

["dragenter", "dragover"].forEach((eventName) => {
  dropZone.addEventListener(eventName, (event) => {
    event.preventDefault();
    dropZone.classList.add("is-dragging");
  });
});

["dragleave", "drop"].forEach((eventName) => {
  dropZone.addEventListener(eventName, (event) => {
    event.preventDefault();
    dropZone.classList.remove("is-dragging");
  });
});

dropZone.addEventListener("drop", (event) => setSelectedFile(event.dataTransfer.files[0]));

filterButtons.forEach((button) => {
  button.addEventListener("click", () => {
    if (!latestResult) return;
    filterButtons.forEach((item) => item.classList.remove("active"));
    button.classList.add("active");
    renderTable(latestResult.features, button.dataset.group);
  });
});
