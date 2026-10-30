/**
 * ==============================================================================
 * SafeSite AI - Frontend Controller Script
 * Handles Image Upload, API Communication (Railway Backend),
 * and Reactive UI Rendering (Zero Emojis, Clean Professional Metrics)
 * ==============================================================================
 */

// 1. Application State & Elements
const state = {
    selectedFile: null,
    apiBaseUrl: localStorage.getItem("safesite_api_url") || "http://localhost:8000",
    isAnalyzing: false,
};

// DOM Selectors
const elements = {
    // API Status & Settings
    apiStatusBadge: document.getElementById("apiStatusBadge"),
    statusDot: document.getElementById("statusDot"),
    statusText: document.getElementById("statusText"),
    btnSettings: document.getElementById("btnSettings"),
    settingsDrawer: document.getElementById("settingsDrawer"),
    apiUrlInput: document.getElementById("apiUrlInput"),
    btnSaveApiUrl: document.getElementById("btnSaveApiUrl"),

    // File Upload
    dropZone: document.getElementById("dropZone"),
    fileInput: document.getElementById("fileInput"),
    uploadPrompt: document.getElementById("uploadPrompt"),
    selectedFileInfo: document.getElementById("selectedFileInfo"),
    selectedFileName: document.getElementById("selectedFileName"),
    selectedFileSize: document.getElementById("selectedFileSize"),
    btnChangeFile: document.getElementById("btnChangeFile"),

    // Controls
    confSlider: document.getElementById("confSlider"),
    confValue: document.getElementById("confValue"),
    chkBanner: document.getElementById("chkBanner"),
    btnAnalyze: document.getElementById("btnAnalyze"),
    btnReset: document.getElementById("btnReset"),

    // Loading & Results
    loadingState: document.getElementById("loadingState"),
    resultsSection: document.getElementById("resultsSection"),

    // Compliance Card
    complianceCard: document.getElementById("complianceCard"),
    complianceIcon: document.getElementById("complianceIcon"),
    complianceStatusTag: document.getElementById("complianceStatusTag"),
    complianceMessage: document.getElementById("complianceMessage"),
    inspectionTime: document.getElementById("inspectionTime"),

    // Metrics
    metricComplianceRate: document.getElementById("metricComplianceRate"),
    metricHelmets: document.getElementById("metricHelmets"),
    metricViolations: document.getElementById("metricViolations"),
    metricTotalEvaluated: document.getElementById("metricTotalEvaluated"),

    // Imagery
    imgOriginal: document.getElementById("imgOriginal"),
    imgAnnotated: document.getElementById("imgAnnotated"),
    btnDownload: document.getElementById("btnDownload"),

    // Audit Table
    detectionsTableBody: document.getElementById("detectionsTableBody"),
    detectionCountLabel: document.getElementById("detectionCountLabel"),
};

// ==============================================================================
// 2. Initialization & API Health Check
// ==============================================================================
document.addEventListener("DOMContentLoaded", () => {
    // Restore saved API URL to input
    elements.apiUrlInput.value = state.apiBaseUrl;

    // Check backend connection
    checkBackendHealth();

    // Attach all event listeners
    bindEvents();
});

async function checkBackendHealth() {
    setApiStatus("checking", "Connecting to API...");
    try {
        const response = await fetch(`${state.apiBaseUrl}/health`, {
            method: "GET",
            headers: { "Accept": "application/json" }
        });

        if (response.ok) {
            setApiStatus("online", "Backend: Operational");
        } else {
            setApiStatus("offline", "Backend: Unreachable (Error)");
        }
    } catch (err) {
        setApiStatus("offline", "Backend: Offline (Check API URL)");
    }
}

function setApiStatus(status, text) {
    elements.statusText.textContent = text;
    elements.statusDot.className = "status-indicator";

    if (status === "online") {
        elements.statusDot.classList.add("online");
    } else if (status === "offline") {
        elements.statusDot.classList.add("offline");
    }
}

// ==============================================================================
// 3. Event Listeners
// ==============================================================================
function bindEvents() {
    // Settings Drawer Toggle
    elements.btnSettings.addEventListener("click", () => {
        elements.settingsDrawer.classList.toggle("hidden");
    });

    // Save API URL
    elements.btnSaveApiUrl.addEventListener("click", () => {
        const inputUrl = elements.apiUrlInput.value.trim().replace(/\/$/, "");
        if (inputUrl) {
            state.apiBaseUrl = inputUrl;
            localStorage.setItem("safesite_api_url", inputUrl);
            checkBackendHealth();
            elements.settingsDrawer.classList.add("hidden");
        }
    });

    // Confidence Slider Input
    elements.confSlider.addEventListener("input", (e) => {
        elements.confValue.textContent = `${e.target.value}%`;
    });

    // File Input Change
    elements.fileInput.addEventListener("change", (e) => {
        const file = e.target.files[0];
        if (file) handleFileSelected(file);
    });

    // Drag and Drop Events
    ["dragenter", "dragover"].forEach(event => {
        elements.dropZone.addEventListener(event, (e) => {
            e.preventDefault();
            e.stopPropagation();
            elements.dropZone.classList.add("drag-over");
        });
    });

    ["dragleave", "drop"].forEach(event => {
        elements.dropZone.addEventListener(event, (e) => {
            e.preventDefault();
            e.stopPropagation();
            elements.dropZone.classList.remove("drag-over");
        });
    });

    elements.dropZone.addEventListener("drop", (e) => {
        const files = e.dataTransfer.files;
        if (files && files.length > 0) {
            handleFileSelected(files[0]);
        }
    });

    // Change File Button
    elements.btnChangeFile.addEventListener("click", (e) => {
        e.stopPropagation();
        elements.fileInput.value = "";
        elements.fileInput.click();
    });

    // Run Safety Inspection Button
    elements.btnAnalyze.addEventListener("click", runInspection);

    // Reset Button
    elements.btnReset.addEventListener("click", resetInspection);
}

// ==============================================================================
// 4. File Handling & Preview
// ==============================================================================
function handleFileSelected(file) {
    if (!file.type.startsWith("image/")) {
        alert("Please upload a valid image file (PNG, JPG, JPEG, WEBP).");
        return;
    }

    state.selectedFile = file;

    // Update UI File Info
    elements.selectedFileName.textContent = file.name;
    elements.selectedFileSize.textContent = `(${(file.size / (1024 * 1024)).toFixed(2)} MB)`;

    elements.uploadPrompt.classList.add("hidden");
    elements.selectedFileInfo.classList.remove("hidden");

    // Enable Buttons
    elements.btnAnalyze.disabled = false;
    elements.btnReset.disabled = false;

    // Hide old results if analyzing a new image
    elements.resultsSection.classList.add("hidden");
}

function resetInspection() {
    state.selectedFile = null;
    elements.fileInput.value = "";

    elements.uploadPrompt.classList.remove("hidden");
    elements.selectedFileInfo.classList.add("hidden");

    elements.btnAnalyze.disabled = true;
    elements.btnReset.disabled = true;

    elements.resultsSection.classList.add("hidden");
    elements.loadingState.classList.add("hidden");
}

// ==============================================================================
// 5. API Request & Execution
// ==============================================================================
async function runInspection() {
    if (!state.selectedFile || state.isAnalyzing) return;

    state.isAnalyzing = true;
    elements.btnAnalyze.disabled = true;
    elements.btnReset.disabled = true;

    // Display Loading State
    elements.resultsSection.classList.add("hidden");
    elements.loadingState.classList.remove("hidden");

    try {
        const formData = new FormData();
        formData.append("file", state.selectedFile);

        const confidenceThreshold = (parseFloat(elements.confSlider.value) / 100).toFixed(2);
        const includeBanner = elements.chkBanner.checked;

        const requestUrl = `${state.apiBaseUrl}/detect?confidence_threshold=${confidenceThreshold}&include_banner=${includeBanner}`;

        const response = await fetch(requestUrl, {
            method: "POST",
            body: formData,
        });

        if (!response.ok) {
            const errorData = await response.json().catch(() => ({}));
            throw new Error(errorData.detail || `Server error: HTTP ${response.status}`);
        }

        const data = await response.json();
        renderResults(data);

    } catch (err) {
        console.error("Detection Error:", err);
        alert(`Inspection Failed: ${err.message}\n\nPlease ensure your Railway/Local API is running at: ${state.apiBaseUrl}`);
    } finally {
        state.isAnalyzing = false;
        elements.loadingState.classList.add("hidden");
        elements.btnAnalyze.disabled = false;
        elements.btnReset.disabled = false;
    }
}

// ==============================================================================
// 6. Results Rendering (Zero Emojis, Pure Professional Visuals)
// ==============================================================================
function renderResults(data) {
    const { compliance, detections, annotated_image } = data;

    // 1. Render Compliance Banner Card
    renderComplianceCard(compliance);

    // 2. Render Metrics Cards
    renderMetrics(compliance);

    // 3. Render Images (Original vs Annotated)
    elements.imgOriginal.src = URL.createObjectURL(state.selectedFile);
    elements.imgAnnotated.src = annotated_image;
    elements.btnDownload.href = annotated_image;

    // 4. Populate Audit Log Table
    renderAuditTable(detections);

    // 5. Show Results Section with smooth entrance
    elements.resultsSection.classList.remove("hidden");
    elements.resultsSection.scrollIntoView({ behavior: "smooth", block: "start" });
}

function renderComplianceCard(compliance) {
    const { status, status_level, message } = compliance;

    // Reset card classes
    elements.complianceCard.className = "compliance-card";
    elements.complianceCard.classList.add(status_level);

    // Set Status Tag
    elements.complianceStatusTag.textContent = status;
    elements.complianceMessage.textContent = message;
    elements.inspectionTime.textContent = `Audited at ${new Date().toLocaleTimeString()}`;

    // SVG Icons (Clean stroke vector icons, no emojis)
    let iconSvg = "";
    if (status_level === "safe") {
        iconSvg = `
            <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round">
                <path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/>
                <path d="m9 12 2 2 4-4"/>
            </svg>
        `;
    } else if (status_level === "danger") {
        iconSvg = `
            <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round">
                <path d="m21.73 18-8-14a2 2 0 0 0-3.48 0l-8 14A2 2 0 0 0 4 21h16a2 2 0 0 0 1.73-3Z"/>
                <line x1="12" y1="9" x2="12" y2="13"/>
                <line x1="12" y1="17" x2="12.01" y2="17"/>
            </svg>
        `;
    } else {
        iconSvg = `
            <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round">
                <circle cx="12" cy="12" r="10"/>
                <line x1="12" y1="8" x2="12" y2="12"/>
                <line x1="12" y1="16" x2="12.01" y2="16"/>
            </svg>
        `;
    }

    elements.complianceIcon.innerHTML = iconSvg;
}

function renderMetrics(compliance) {
    const { compliance_rate, counts } = compliance;

    elements.metricComplianceRate.textContent = `${compliance_rate}%`;
    elements.metricHelmets.textContent = counts.helmet || 0;
    elements.metricViolations.textContent = counts.head || 0;
    elements.metricTotalEvaluated.textContent = counts.total_heads_evaluated || (counts.helmet + counts.head);
}

function renderAuditTable(detections) {
    elements.detectionsTableBody.innerHTML = "";
    elements.detectionCountLabel.textContent = `${detections.length} objects detected`;

    if (detections.length === 0) {
        elements.detectionsTableBody.innerHTML = `
            <tr>
                <td colspan="5" style="text-align: center; color: var(--color-text-muted); padding: 2rem;">
                    No objects detected in the current imagery at this confidence threshold.
                </td>
            </tr>
        `;
        return;
    }

    detections.forEach((det, index) => {
        const { class_name, confidence, box } = det;

        let statusPill = "";
        if (class_name === "helmet") {
            statusPill = `<span class="pill pill-safe">Compliant</span>`;
        } else if (class_name === "head") {
            statusPill = `<span class="pill pill-danger">Violation</span>`;
        } else {
            statusPill = `<span class="pill pill-info">Personnel</span>`;
        }

        const coordinatesFormatted = `[${box.join(", ")}]`;

        const row = document.createElement("tr");
        row.innerHTML = `
            <td>${index + 1}</td>
            <td style="font-weight: 600;">${class_name.toUpperCase()}</td>
            <td>${statusPill}</td>
            <td style="font-family: var(--font-mono);">${(confidence * 100).toFixed(1)}%</td>
            <td style="font-family: var(--font-mono); font-size: 0.78rem; color: var(--color-text-muted);">${coordinatesFormatted}</td>
        `;
        elements.detectionsTableBody.appendChild(row);
    });
}
