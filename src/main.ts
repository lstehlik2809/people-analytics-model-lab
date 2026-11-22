import "./style.css";
import { AnalysisRunner, type Setup } from "./analysis.ts";
import { fields, groups, parseData, type Field, type Row } from "./data.ts";
import type { Task } from "./kumo/preprocess.ts";
import { count, emptyResults, escape, plannedSplits, renderResults, type ResultsView } from "./results.ts";

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
const activity = get("run-activity");
const dataMessage = get("data-message");
const status = get("run-status");
const resultsPanel = get("results-panel");
const resultFreshness = get("result-freshness");
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
let view: ResultsView | undefined;
let completedSetup: Setup | null = null;

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

function showResultFreshness(stale: boolean) {
  if (resultsPanel.classList.contains("is-stale") === stale) return;
  resultsPanel.classList.toggle("is-stale", stale);
  resultFreshness.innerHTML = stale
    ? "<strong>Previous results</strong> Run the comparison again to use the current setup."
    : "";
}

function clearResults() {
  completedSetup = null;
  view?.stop();
  view = undefined;
  content.replaceChildren();
  showResultFreshness(false);
}

function setupChanged() {
  status.replaceChildren();
  const current = setup();
  if (!completedSetup) {
    emptyResults(content, current);
    return;
  }
  const previous = completedSetup;
  if (current.task !== previous.task) return;
  // Both predictor lists follow the same field order from setup().
  showResultFreshness(current.outcome.key !== previous.outcome.key ||
    current.predictors.length !== previous.predictors.length ||
    current.predictors.some(({ key }, index) => key !== previous.predictors[index].key));
}

function setBusy(busy: boolean) {
  inputs.disabled = busy || !dataset;
  resetButton.disabled = busy || !dataset;
  cancelButton.hidden = !busy;
  activity.hidden = !busy;
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
  clearResults();
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
  clearResults();
  setBusy(true);
  cancelButton.focus({ preventScroll: true });
  let downloadBucket = -1;
  try {
    const result = await runner.run(dataset, current, (update) => {
      if (update.phase === "prepare") announce("Preparing the analysis", "Creating the shared training and test sets.");
      if (update.phase === "load") {
        const progress = update.total ? Math.min(100, Math.floor(100 * (update.loaded ?? 0) / update.total)) : undefined;
        const bucket = progress === undefined ? -1 : Math.floor(progress / 5);
        if (progress === undefined || bucket > downloadBucket) {
          downloadBucket = bucket;
          announce("Loading Kumo Tabular Small",
            progress === undefined ? "The first download may take a minute." : `${progress}% downloaded`, progress);
        }
      }
      if (update.phase === "predict") {
        announce("Running three models", "Comparing held-out predictions on your CPU. This may take a minute.");
      }
    }, signal);
    view = renderResults(content, result);
    completedSetup = current;
    status.innerHTML = '<span class="sr-only">Comparison complete.</span>';
  } catch (error) {
    emptyResults(content, current);
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
      if (completedSetup) {
        const heading = get("results-title");
        const needsScroll = heading.getBoundingClientRect().top > window.innerHeight / 2;
        heading.focus({ preventScroll: needsScroll });
        if (needsScroll) heading.scrollIntoView({ block: "start" });
      } else {
        runButton.focus({ preventScroll: true });
      }
    }
  }
});

// Repeats use the completed setup, so the table stays tied to the displayed results.
async function repeatSplits() {
  if (!dataset || active || !view || !completedSetup) return;
  active = new AbortController();
  const { signal } = active;
  const current = view;
  setBusy(true);
  current.setRunning(true);
  cancelButton.focus({ preventScroll: true });
  try {
    while (current.splitCount() < plannedSplits) {
      const repeat = current.splitCount();
      status.innerHTML = `<span class="sr-only">Running split ${repeat + 1} of ${plannedSplits}.</span>`;
      current.addSplit(await runner.run(dataset, completedSetup, () => {}, signal, repeat));
    }
    status.innerHTML = '<span class="sr-only">Repeated splits complete.</span>';
  } catch (error) {
    const kept = current.splitCount() > 1 ? "Completed splits stay in the table." : "";
    if (signal.aborted) {
      announce("Repeats canceled.", kept);
    } else {
      console.error(error);
      announce("The repeated splits could not finish.", `${kept} Try again to continue.`.trim());
    }
  } finally {
    const restoreFocus = document.activeElement === cancelButton;
    active = null;
    setBusy(false);
    current.setRunning(false);
    if (restoreFocus) (get("repeat-splits") ?? get("results-title")).focus({ preventScroll: true });
  }
}

content.addEventListener("click", (event) => {
  if ((event.target as HTMLElement).id === "repeat-splits") void repeatSplits();
});

renderControls();
setBusy(false);
emptyResults(content, setup());
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
