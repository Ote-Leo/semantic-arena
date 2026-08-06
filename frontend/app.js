const state = {
  datasets: [],
  engines: [],
  selectedDataset: "",
  activeJob: null,
  deletingEngine: null,
};

const elements = {
  dataset: document.querySelector("#dataset"),
  datasetDescription: document.querySelector("#dataset-description"),
  engines: document.querySelector("#engine-list"),
  indexStatus: document.querySelector("#index-status"),
  indexButton: document.querySelector("#index-button"),
  queryForm: document.querySelector("#query-form"),
  query: document.querySelector("#query"),
  askButton: document.querySelector("#ask-button"),
  notice: document.querySelector("#notice"),
  results: document.querySelector("#results"),
  template: document.querySelector("#result-template"),
};

async function request(path, options = {}) {
  const response = await fetch(path, {
    ...options,
    headers: { "Content-Type": "application/json", ...(options.headers || {}) },
  });
  if (!response.ok) {
    const body = await response.json().catch(() => ({}));
    throw new Error(body.detail || `Request failed with ${response.status}.`);
  }
  if (response.status === 204) return null;
  return response.json();
}

function selectedEngineIds() {
  return [...elements.engines.querySelectorAll("input:checked")].map((input) => input.value);
}

function selectedDataset() {
  return state.datasets.find((dataset) => dataset.id === state.selectedDataset);
}

function showNotice(message, type = "") {
  elements.notice.hidden = !message;
  elements.notice.textContent = message;
  elements.notice.className = `notice ${type}`;
}

function renderEngines() {
  elements.engines.replaceChildren();
  for (const engine of state.engines) {
    const label = document.createElement("label");
    label.className = "engine-option";
    label.innerHTML = `
      <input type="checkbox" value="${engine.id}" ${engine.available ? "checked" : "disabled"}>
      <span>${engine.label}<small>${engine.available ? "service ready" : "service unavailable"}</small></span>
    `;
    elements.engines.append(label);
  }
}

function renderDatasets() {
  const previousSelection = state.selectedDataset;
  elements.dataset.replaceChildren();
  for (const dataset of state.datasets) {
    const option = new Option(dataset.name, dataset.id);
    elements.dataset.add(option);
  }
  state.selectedDataset = state.datasets.some((dataset) => dataset.id === previousSelection)
    ? previousSelection
    : state.datasets[0]?.id || "";
  elements.dataset.value = state.selectedDataset;
  elements.dataset.disabled = !state.selectedDataset;
  renderDatasetState();
}

function renderDatasetState() {
  const dataset = selectedDataset();
  if (!dataset) {
    elements.datasetDescription.textContent = "No local dataset manifests were found.";
    elements.indexStatus.textContent = "Add sample-data/<id>/manifest.yaml to begin.";
    setActionAvailability();
    return;
  }
  elements.datasetDescription.textContent = dataset.description || "Local indexed source material.";
  renderIndexState(dataset);
  setActionAvailability();
}

function renderIndexState(dataset) {
  elements.indexStatus.replaceChildren();
  for (const engine of state.engines) {
    const indexed = Boolean(dataset.engine_indexes[engine.id]);
    const deleting = state.deletingEngine === engine.id;
    const row = document.createElement("div");
    row.className = "index-row";

    const identity = document.createElement("div");
    identity.className = "index-identity";
    const engineLabel = document.createElement("strong");
    engineLabel.textContent = engine.label;
    const name = document.createElement("code");
    name.textContent = dataset.index_name;
    identity.append(engineLabel, name);

    const badge = document.createElement("span");
    badge.className = "index-badge";
    if (!engine.available) {
      badge.textContent = "unavailable";
      badge.classList.add("unavailable");
    } else if (deleting) {
      badge.textContent = "deleting";
      badge.classList.add("working");
    } else if (indexed) {
      badge.textContent = "ready";
      badge.classList.add("ready");
    } else {
      badge.textContent = "not indexed";
    }

    const deleteButton = document.createElement("button");
    deleteButton.type = "button";
    deleteButton.className = "index-delete";
    deleteButton.textContent = "Delete";
    deleteButton.hidden = !indexed;
    deleteButton.dataset.available = String(engine.available);
    deleteButton.setAttribute(
      "aria-label",
      `Delete ${dataset.index_name} from ${engine.label}`,
    );
    deleteButton.addEventListener("click", () => deleteIndex(engine.id));

    row.append(identity, badge, deleteButton);
    elements.indexStatus.append(row);
  }
}

function setActionAvailability() {
  const hasDataset = Boolean(selectedDataset());
  const hasEngines = selectedEngineIds().length > 0;
  const busy = Boolean(state.activeJob || state.deletingEngine);
  elements.indexButton.disabled = !hasDataset || !hasEngines || busy;
  elements.askButton.disabled = !hasDataset || !hasEngines || busy;
  for (const button of elements.indexStatus.querySelectorAll(".index-delete")) {
    button.disabled = busy || button.dataset.available !== "true";
  }
}

async function refreshData() {
  const [engines, datasets, health] = await Promise.all([
    request("/api/engines"),
    request("/api/datasets"),
    request("/api/health"),
  ]);
  state.engines = engines;
  state.datasets = datasets;
  renderEngines();
  renderDatasets();
  if (!health.embedding_configured) {
    showNotice("An embedding endpoint is not configured. Set the embedding values in .env before indexing or searching.", "error");
  } else if (engines.some((engine) => !engine.available)) {
    showNotice("One or more search services are unavailable. You can still compare the available engines.", "error");
  }
}

async function startIndex() {
  const engineIds = selectedEngineIds();
  state.activeJob = "starting";
  setActionAvailability();
  showNotice("Preparing shared chunks and embeddings...", "");
  try {
    const job = await request("/api/indexes", {
      method: "POST",
      body: JSON.stringify({ dataset_id: state.selectedDataset, engine_ids: engineIds }),
    });
    state.activeJob = job.id;
    await pollJob(job.id);
    await refreshData();
    showNotice("Index build completed. The selected engines received identical chunks and vectors.", "");
  } catch (error) {
    showNotice(error.message, "error");
  } finally {
    state.activeJob = null;
    renderDatasetState();
  }
}

async function pollJob(jobId) {
  for (;;) {
    const job = await request(`/api/jobs/${jobId}`);
    elements.indexStatus.textContent = `Index build: ${job.status}; ${job.processed_chunks.toLocaleString()} chunks processed.`;
    if (job.status === "succeeded") return;
    if (job.status === "failed") throw new Error(job.error || "Index build failed.");
    await new Promise((resolve) => setTimeout(resolve, 500));
  }
}

async function deleteIndex(engineId) {
  const dataset = selectedDataset();
  const engine = state.engines.find((candidate) => candidate.id === engineId);
  if (!dataset || !engine) return;
  const confirmed = window.confirm(
    `Delete ${dataset.index_name} from ${engine.label}? This cannot be undone.`,
  );
  if (!confirmed) return;

  state.deletingEngine = engineId;
  renderDatasetState();
  showNotice(`Deleting ${dataset.index_name} from ${engine.label}...`, "");
  try {
    await request(
      `/api/indexes/${encodeURIComponent(dataset.id)}/${encodeURIComponent(engineId)}`,
      { method: "DELETE" },
    );
    await refreshData();
    showNotice(`Deleted ${dataset.index_name} from ${engine.label}.`, "");
  } catch (error) {
    showNotice(error.message, "error");
  } finally {
    state.deletingEngine = null;
    renderDatasetState();
  }
}

async function submitQuery(event) {
  event.preventDefault();
  elements.askButton.disabled = true;
  elements.results.replaceChildren();
  showNotice("Running lexical and vector retrieval...", "");
  try {
    const result = await request("/api/chat", {
      method: "POST",
      body: JSON.stringify({
        dataset_id: state.selectedDataset,
        query: elements.query.value,
        engine_ids: selectedEngineIds(),
      }),
    });
    showNotice(`Shared query embedding: ${result.embedding_ms.toFixed(2)} ms.`, "");
    for (const engineResult of result.results) renderResult(engineResult);
  } catch (error) {
    showNotice(error.message, "error");
  } finally {
    setActionAvailability();
  }
}

function renderResult(result) {
  const card = elements.template.content.firstElementChild.cloneNode(true);
  card.querySelector(".engine-kicker").textContent = result.engine_id;
  card.querySelector("h3").textContent = result.engine_label;
  const badge = card.querySelector(".result-state");
  badge.textContent = result.status;
  if (result.status === "error") badge.classList.add("error");
  const answer = card.querySelector(".answer");
  answer.textContent = result.error || result.message || "No response.";
  if (result.no_data) answer.classList.add("no-data");
  const citations = card.querySelector(".citations");
  const citationList = citations.querySelector("ol");
  if (!result.citations.length) {
    citations.hidden = true;
  } else {
    result.citations.forEach((citation) => {
      const item = document.createElement("li");
      const button = document.createElement("button");
      button.textContent = `${citation.title} / ${citation.source_path} / chunk ${citation.chunk_index}`;
      button.addEventListener("click", () => openCitation(citation));
      item.append(button);
      citationList.append(item);
    });
  }
  const diagnostics = card.querySelector(".diagnostics");
  if (!result.timings) {
    diagnostics.hidden = true;
  } else {
    diagnostics.querySelector(".timings").innerHTML = [
      ["lex", result.timings.lexical_ms],
      ["vec", result.timings.vector_ms],
      ["fuse", result.timings.fusion_ms],
      ["select", result.timings.selection_ms],
      ["total", result.timings.total_ms],
    ].map(([label, value]) => `<span>${label}: ${Number(value).toFixed(2)}ms</span>`).join("");
    const candidateList = diagnostics.querySelector(".candidates");
    result.candidates.forEach((candidate) => {
      const item = document.createElement("li");
      item.innerHTML = `<span class="candidate-source">${escapeHtml(candidate.title)} / RRF ${candidate.fused_score.toFixed(4)}</span>${escapeHtml(candidate.excerpt)}`;
      candidateList.append(item);
    });
  }
  elements.results.append(card);
}

async function openCitation(citation) {
  const modal = document.createElement("div");
  modal.className = "modal";
  modal.innerHTML = `
      <div class="modal-dialog" role="dialog" aria-modal="true" aria-label="Citation source">
        <header><div><p class="eyebrow">${escapeHtml(citation.source_path)} / CHUNK ${citation.chunk_index}</p><h3>${escapeHtml(citation.title)}</h3></div><button type="button">Close</button></header>
        <pre></pre>
      </div>`;
  modal.querySelector("pre").textContent = citation.text;
  modal.querySelector("button").addEventListener("click", () => modal.remove());
  modal.addEventListener("click", (event) => { if (event.target === modal) modal.remove(); });
  document.body.append(modal);
}

function escapeHtml(value) {
  const element = document.createElement("span");
  element.textContent = value;
  return element.innerHTML;
}

elements.dataset.addEventListener("change", () => {
  state.selectedDataset = elements.dataset.value;
  elements.results.replaceChildren();
  renderDatasetState();
});
elements.engines.addEventListener("change", renderDatasetState);
elements.indexButton.addEventListener("click", startIndex);
elements.queryForm.addEventListener("submit", submitQuery);

refreshData().catch((error) => showNotice(`Could not load arena status: ${error.message}`, "error"));
