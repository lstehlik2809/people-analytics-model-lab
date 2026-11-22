import "./style.css";
import { AnalysisRunner, type Setup } from "./analysis.ts";
import { fields, groups, parseData, type Field, type Row } from "./data.ts";
import type { Task } from "./kumo/preprocess.ts";
import { count, emptyResults, escape, renderResults } from "./results.ts";

function get<T extends HTMLElement = HTMLElement>(id: string) {
  return document.getElementById(id) as T;
}

const form = get<HTMLFormElement>("analysis-form");
const inputs = get<HTMLFieldSetElement>("inputs");
const outcomeInput = get<HTMLSelectElement>("outcome");
const picker = get<HTMLDetailsElement>("predictor-picker");
const options = get("predictor-options");
const selected = get("selected-predictors");
const runButton = get<HTMLButtonElement>("run");
const resetButton = get<HTMLButtonElement>("reset");
const cancelButton = get<HTMLButtonElement>("cancel");
const dataMessage = get("data-message");
const status = get("run-status");
const content = get("results-content");
const runner = new AnalysisRunner();

const defaults = {
  classification: {
    outcome: "Attrition",
    predictors: ["Age", "OverTime", "JobSatisfaction", "MonthlyIncome", "JobRole", "YearsAtCompany", "DistanceFromHome"],
  },
  regression: {
    outcome: "MonthlyIncome",
    predictors: ["JobLevel", "JobRole", "TotalWorkingYears", "Age", "Department", "YearsAtCompany", "Education"],
  },
};

function defaultSetup() {
  return {
    classification: { ...defaults.classification, predictors: new Set(defaults.classification.predictors) },
    regression: { ...defaults.regression, predictors: new Set(defaults.regression.predictors) },
  };
}

let task: Task = "classification";
let selections = defaultSetup();
let dataset: Row[] | null = null;
let active: AbortController | null = null;
let stopCharts: (() => void) | undefined;
let shownResult = false;

function available(): Field[] {
  return fields.filter(({ key }) => key !== selections[task].outcome);
}

function setup(): Setup {
  return {
    task,
    outcome: fields.find(({ key }) => key === selections[task].outcome)!,
    predictors: available().filter(({ key }) => selections[task].predictors.has(key)),
  };
}

function updateSelection() {
  const { predictors } = setup();
  get("predictor-count").textContent = String(predictors.length);
  const suffix = predictors.length > 4 ? ` + ${predictors.length - 4} more` : "";
  selected.textContent = predictors.length === 0
    ? "Choose at least one predictor."
    : predictors.slice(0, 4).map(({ name }) => name).join(", ") + suffix;
  runButton.disabled = predictors.length === 0;
}

function renderControls() {
  const { outcome } = setup();
  const kind = task === "classification" ? "categorical" : "numerical";
  outcomeInput.replaceChildren(...fields.filter((field) => field.kind === kind).map((field) => new Option(field.name, field.key)));
  outcomeInput.value = outcome.key;
  get("outcome-help").textContent = task === "classification" ? "Choose a category to predict." : "Choose a numeric value to predict.";
  const current = selections[task].predictors;
  const choices = available();
  options.innerHTML = groups.map((group) => `<fieldset class="option-group">
    <legend>${group}</legend>
    ${choices.filter((field) => field.group === group).map(({ key, name }) => `
      <label class="predictor"><input type="checkbox" value="${key}"${current.has(key) ? " checked" : ""}>
      <span>${escape(name)}</span></label>`).join("")}
  </fieldset>`).join("");
  updateSelection();
}

function setEmpty() {
  stopCharts?.();
  stopCharts = undefined;
  emptyResults(content, setup(), shownResult);
}

function setupChanged() {
  status.replaceChildren();
  setEmpty();
}

function setBusy(busy: boolean) {
  inputs.disabled = busy || !dataset;
  resetButton.disabled = busy || !dataset;
  cancelButton.hidden = !busy;
  content.setAttribute("aria-busy", String(busy));
}

form.addEventListener("change", (event) => {
  const target = event.target as HTMLInputElement | HTMLSelectElement;
  if (target instanceof HTMLInputElement && target.name === "task") {
    task = target.value as Task;
    renderControls();
  } else if (target === outcomeInput) {
    selections[task].outcome = target.value;
    selections[task].predictors.delete(target.value);
    renderControls();
  } else if (target instanceof HTMLInputElement && target.type === "checkbox") {
    const predictors = selections[task].predictors;
    if (target.checked) predictors.add(target.value);
    else predictors.delete(target.value);
    updateSelection();
  } else {
    return;
  }
  setupChanged();
});

for (const [id, select] of [["select-all", true], ["clear", false]] as const) {
  get(id).addEventListener("click", () => {
    selections[task].predictors = new Set(select ? available().map(({ key }) => key) : []);
    options.querySelectorAll("input").forEach((input) => { input.checked = select; });
    updateSelection();
    setupChanged();
  });
}

resetButton.addEventListener("click", () => {
  selections = defaultSetup();
  task = "classification";
  form.querySelector<HTMLInputElement>('input[value="classification"]')!.checked = true;
  picker.open = false;
  shownResult = false;
  renderControls();
  setupChanged();
});

cancelButton.addEventListener("click", () => active?.abort());

function announce(message: string, detail = "", progress?: number) {
  status.innerHTML = message ? `<div class="status-message">
    <strong>${escape(message)}</strong>${detail ? `<p>${escape(detail)}</p>` : ""}
    ${progress === undefined ? "" : `<progress value="${progress}" max="100" aria-label="Kumo model download"></progress>`}</div>` : "";
}

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  if (!dataset || active || runButton.disabled) return;
  active = new AbortController();
  const { signal } = active;
  const current = setup();
  picker.open = false;
  stopCharts?.();
  content.replaceChildren();
  setBusy(true);
  cancelButton.focus({ preventScroll: true });
  let downloadBucket = -1;
  let usingFallback = false;
  let completed = false;
  try {
    const result = await runner.run(dataset, current, (update) => {
      if (update.phase === "prepare") announce("Preparing the analysis", "Creating the shared training and test sets.");
      if (update.phase === "fallback") {
        usingFallback = true;
        downloadBucket = -1;
      }
      if (update.phase === "load") {
        const progress = update.total ? Math.min(100, Math.floor(100 * (update.loaded ?? 0) / update.total)) : undefined;
        const bucket = progress === undefined ? -1 : Math.floor(progress / 5);
        if (progress === undefined || bucket > downloadBucket) {
          downloadBucket = bucket;
          announce(usingFallback ? "Switching to CPU" : "Loading Kumo Tabular Small",
            progress === undefined ? "The first download may take a minute." : `${progress}% downloaded`, progress);
        }
      }
      if (update.phase === "predict") {
        announce("Running both models", update.device === "webgpu" ? "Comparing held-out predictions on WebGPU." : "Comparing held-out predictions on your CPU. This may take a minute.");
      }
    }, signal);
    stopCharts = renderResults(content, result);
    shownResult = true;
    completed = true;
    status.innerHTML = '<span class="sr-only">Comparison complete.</span>';
  } catch (error) {
    setEmpty();
    if (signal.aborted) {
      announce("Run canceled.");
    } else {
      console.error(error);
      announce("The comparison could not finish.", "Check that the model can download, then try again. A recent browser and more available memory may help.");
    }
  } finally {
    const restoreFocus = document.activeElement === cancelButton;
    active = null;
    setBusy(false);
    if (restoreFocus) {
      if (completed) get("results-title").focus();
      else runButton.focus({ preventScroll: true });
    }
  }
});

renderControls();
setBusy(false);
setEmpty();
fetch(`${import.meta.env.BASE_URL}data/ibm-hr-attrition.csv`)
  .then((response) => {
    if (!response.ok) throw new Error("Dataset request failed.");
    return response.text();
  })
  .then((csv) => {
    dataset = parseData(csv);
    dataMessage.textContent = `${count(dataset.length)} fictional employees · IBM HR dataset`;
    setBusy(false);
  })
  .catch((error) => {
    console.error(error);
    dataMessage.textContent = "The dataset could not load. Reload the page to try again.";
  });
