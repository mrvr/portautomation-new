const tabs = document.querySelectorAll(".tab");
const panels = document.querySelectorAll(".panel");

tabs.forEach((tab) => {
  tab.addEventListener("click", () => {
    tabs.forEach((item) => item.classList.remove("active"));
    panels.forEach((panel) => panel.classList.remove("active"));
    tab.classList.add("active");
    document.getElementById(`${tab.dataset.tab}-tab`).classList.add("active");
  });
});

const startBtn = document.getElementById("start-btn");
const stopBtn = document.getElementById("stop-btn");
const modelSelect = document.getElementById("model-select");
const statusCard = document.getElementById("status-card");
const globalStatus = document.getElementById("global-status");
const appLogs = document.getElementById("app-logs");
const testsOutput = document.getElementById("tests-output");
const actionBanner = document.getElementById("action-banner");

const runTestsBtn = document.getElementById("run-tests-btn");
const generateTestsBtn = document.getElementById("generate-tests-btn");
const watchToggle = document.getElementById("watch-toggle");

let fastPollTimer = null;

async function fetchJson(url, options = {}) {
  const response = await fetch(url, {
    headers: { "Content-Type": "application/json" },
    ...options,
  });
  let payload;
  try {
    payload = await response.json();
  } catch (error) {
    throw new Error(response.statusText || "Invalid server response");
  }
  if (!response.ok) {
    throw new Error(payload.detail || payload.message || response.statusText);
  }
  return payload;
}

function unwrap(payload) {
  return payload.data !== undefined ? payload.data : payload;
}

function showBanner(message, type = "info") {
  if (!actionBanner) return;
  actionBanner.textContent = message;
  actionBanner.className = `action-banner ${type}`;
  actionBanner.hidden = false;
}

function formatElapsed(seconds) {
  if (seconds == null) return "-";
  const mins = Math.floor(seconds / 60);
  const secs = Math.round(seconds % 60);
  return mins > 0 ? `${mins}m ${secs}s` : `${secs}s`;
}

function formatTime(unixSeconds) {
  if (!unixSeconds) return "-";
  return new Date(unixSeconds * 1000).toLocaleString();
}

function scrollToBottom(element) {
  element.scrollTop = element.scrollHeight;
}

function setButtonLoading(button, loading, loadingText) {
  if (!button) return;
  if (loading) {
    button.dataset.originalText = button.textContent;
    button.textContent = loadingText;
    button.disabled = true;
  } else if (button.dataset.originalText) {
    button.textContent = button.dataset.originalText;
    button.disabled = false;
  }
}

function updateAppStatus(data) {
  document.getElementById("app-state").textContent = data.state;
  document.getElementById("app-model").textContent = data.model || "-";
  document.getElementById("app-pid").textContent = data.pid ?? "-";
  document.getElementById("app-elapsed").textContent = formatElapsed(data.elapsed_seconds);
  document.getElementById("app-device").textContent = data.device
    ? `${data.device.device_type} (${data.device.device_name})`
    : "-";
  document.getElementById("app-message").textContent = data.message || "-";
  appLogs.textContent = (data.logs || []).join("\n") || "No logs yet.";
  scrollToBottom(appLogs);

  statusCard.className = `card status-card ${data.state}`;
  startBtn.disabled = data.is_running;
  stopBtn.disabled = !data.is_running;

  if (data.is_running) {
    globalStatus.textContent = "Training running";
    globalStatus.style.borderColor = "#2563eb";
  } else if (data.state === "failed") {
    globalStatus.textContent = "Training failed";
    globalStatus.style.borderColor = "#dc2626";
  } else if (data.state === "completed") {
    globalStatus.textContent = "Training completed";
    globalStatus.style.borderColor = "#16a34a";
  } else if (data.state === "stopped") {
    globalStatus.textContent = "Training stopped";
    globalStatus.style.borderColor = "#d97706";
  } else {
    globalStatus.textContent = "Idle";
    globalStatus.style.borderColor = "#374151";
  }
}

function updateTestsStatus(data) {
  document.getElementById("stat-passed").textContent = data.passed ?? 0;
  document.getElementById("stat-failed").textContent = (data.failed ?? 0) + (data.errors ?? 0);
  document.getElementById("stat-skipped").textContent = data.skipped ?? 0;
  document.getElementById("stat-total").textContent = data.total ?? 0;

  const meta = [
    `Status: ${data.status || "idle"}`,
    `Trigger: ${data.trigger || "-"}`,
    `Duration: ${data.duration_seconds ?? 0}s`,
    `Last run: ${formatTime(data.finished_at)}`,
    data.message || "",
  ].filter(Boolean).join(" | ");
  document.getElementById("tests-meta").textContent = meta;

  const tbody = document.getElementById("tests-table-body");
  const tests = data.tests || [];
  if (!tests.length) {
    tbody.innerHTML = `<tr><td colspan="5">${data.is_running ? "Running tests..." : "No test results yet."}</td></tr>`;
  } else {
    tbody.innerHTML = tests
      .map(
        (test) => `
        <tr>
          <td><span class="badge ${test.status}">${test.status}</span></td>
          <td>${test.name}</td>
          <td>${test.classname}</td>
          <td>${Number(test.time || 0).toFixed(3)}</td>
          <td>${test.message || ""}</td>
        </tr>`
      )
      .join("");
  }

  runTestsBtn.disabled = !!data.is_running;
  generateTestsBtn.disabled = !!data.is_running;
}

function updateTestsOutput(output) {
  testsOutput.textContent = output || "No test output yet.";
  scrollToBottom(testsOutput);
}

async function refreshAppStatus() {
  try {
    const payload = await fetchJson("/api/app/status");
    updateAppStatus(unwrap(payload));
  } catch (error) {
    globalStatus.textContent = "Backend unavailable";
  }
}

async function refreshAppLogs() {
  try {
    const payload = await fetchJson("/api/app/logs");
    const data = unwrap(payload);
    appLogs.textContent = (data.logs || []).join("\n") || "No logs yet.";
    scrollToBottom(appLogs);
  } catch (error) {
    // Ignore log refresh errors; status endpoint still provides logs.
  }
}

async function refreshTestsStatus() {
  try {
    const payload = await fetchJson("/api/tests/status");
    updateTestsStatus(unwrap(payload));
  } catch (error) {
    document.getElementById("tests-meta").textContent = "Unable to load test status.";
  }
}

async function refreshTestsOutput() {
  try {
    const payload = await fetchJson("/api/tests/output");
    const data = unwrap(payload);
    updateTestsOutput(data.output || "");
  } catch (error) {
    testsOutput.textContent = "Unable to load test output.";
  }
}

function startFastPolling() {
  if (fastPollTimer) clearInterval(fastPollTimer);
  fastPollTimer = setInterval(async () => {
    await refreshAppStatus();
    await refreshAppLogs();
    await refreshTestsStatus();
    await refreshTestsOutput();
  }, 1000);
  setTimeout(() => {
    if (fastPollTimer) {
      clearInterval(fastPollTimer);
      fastPollTimer = null;
    }
  }, 30000);
}

startBtn.addEventListener("click", async () => {
  setButtonLoading(startBtn, true, "Starting...");
  try {
    const payload = await fetchJson("/api/app/start", {
      method: "POST",
      body: JSON.stringify({ model: modelSelect.value }),
    });
    const data = unwrap(payload);
    updateAppStatus(data);
    showBanner(payload.message, payload.success ? "success" : "error");
    startFastPolling();
  } catch (error) {
    showBanner(`Failed to start training: ${error.message}`, "error");
  } finally {
    setButtonLoading(startBtn, false);
    await refreshAppStatus();
    await refreshAppLogs();
  }
});

stopBtn.addEventListener("click", async () => {
  setButtonLoading(stopBtn, true, "Stopping...");
  try {
    const payload = await fetchJson("/api/app/stop", { method: "POST" });
    updateAppStatus(unwrap(payload));
    showBanner(payload.message, payload.success ? "success" : "info");
    startFastPolling();
  } catch (error) {
    showBanner(`Failed to stop training: ${error.message}`, "error");
  } finally {
    setButtonLoading(stopBtn, false);
    await refreshAppStatus();
  }
});

runTestsBtn.addEventListener("click", async () => {
  setButtonLoading(runTestsBtn, true, "Running...");
  try {
    const payload = await fetchJson("/api/tests/run", { method: "POST" });
    updateTestsStatus(unwrap(payload));
    showBanner(payload.message, "success");
    startFastPolling();
  } catch (error) {
    showBanner(`Failed to run tests: ${error.message}`, "error");
  } finally {
    setButtonLoading(runTestsBtn, false);
    await refreshTestsStatus();
    await refreshTestsOutput();
  }
});

generateTestsBtn.addEventListener("click", async () => {
  setButtonLoading(generateTestsBtn, true, "Regenerating...");
  try {
    const payload = await fetchJson("/api/tests/generate", {
      method: "POST",
      body: JSON.stringify({ run_after: true }),
    });
    const data = unwrap(payload);
    if (data.tests) updateTestsStatus(data.tests);
    showBanner(payload.message, "success");
    startFastPolling();
  } catch (error) {
    showBanner(`Failed to regenerate tests: ${error.message}`, "error");
  } finally {
    setButtonLoading(generateTestsBtn, false);
    await refreshTestsStatus();
    await refreshTestsOutput();
  }
});

watchToggle.addEventListener("change", async () => {
  try {
    const payload = await fetchJson("/api/tests/watch", {
      method: "POST",
      body: JSON.stringify({ enabled: watchToggle.checked }),
    });
    showBanner(payload.message, "info");
  } catch (error) {
    showBanner(`Failed to update watcher: ${error.message}`, "error");
  }
});

async function initWatcherToggle() {
  try {
    const payload = await fetchJson("/api/tests/watch");
    const data = unwrap(payload);
    watchToggle.checked = !!data.enabled;
    if (!data.available) {
      watchToggle.checked = false;
      watchToggle.disabled = true;
    }
  } catch (error) {
    watchToggle.disabled = true;
  }
}

setInterval(async () => {
  await refreshAppStatus();
  await refreshTestsStatus();
}, 2000);

setInterval(async () => {
  await refreshAppLogs();
  await refreshTestsOutput();
}, 3000);

initWatcherToggle();
refreshAppStatus();
refreshAppLogs();
refreshTestsStatus();
refreshTestsOutput();
