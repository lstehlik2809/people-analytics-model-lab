import type { Result, Setup } from "./analysis.ts";
import { drawPrecisionRecall, drawScatter, type ModelKey } from "./charts.ts";
import { classScores, classificationMetrics, precisionRecall, regressionMetrics } from "./metrics.ts";

const integer = new Intl.NumberFormat("en");
const decimal = new Intl.NumberFormat("en", { maximumFractionDigits: 2 });
const percent = new Intl.NumberFormat("en", { style: "percent", maximumFractionDigits: 1 });
export const count = (value: number) => integer.format(value);
const measure = (value: number | null) => value === null ? "—" : decimal.format(value);
export const escape = (value: string) => value.replace(/&/g, "&amp;").replace(/</g, "&lt;")
  .replace(/>/g, "&gt;").replace(/"/g, "&quot;").replace(/'/g, "&#39;");

const kumoName = "Kumo Tabular Small";

type Model = { key: ModelKey; name: string };

function modelList(baseline: string): Model[] {
  return [
    { key: "kumo", name: kumoName },
    { key: "baseline", name: baseline },
    { key: "forest", name: "Random Forest" },
  ];
}

function modelLabel({ key, name }: Model) {
  return `<i class="model-marker ${key}" aria-hidden="true"></i>${name}`;
}

export function emptyResults(container: HTMLElement, setup: Setup, changed: boolean) {
  const models = modelList(setup.task === "classification" ? "Logistic regression" : "Linear regression");
  container.innerHTML = `
    <div class="empty">
      <div class="empty-diagram" aria-hidden="true">
        <div class="diagram-source"><span></span><span></span><span></span><span></span><span></span></div>
        <div class="diagram-connector"></div>
        <div class="diagram-models"><span></span><span></span><span></span></div>
      </div>
      <h3>${changed ? "Ready for a new comparison" : "Compare three models on the same test rows"}</h3>
      <p>Run the comparison to see how the models predict <strong>${escape(setup.outcome.name.toLowerCase())}</strong>.</p>
    </div>
    <div class="empty-footer">
      ${models.map((model) => `<span>${modelLabel(model)}</span>`).join("")}
    </div>`;
}

type Metric = { name: string; direction: "higher" | "lower"; values: string[]; scope?: string };

function metricTable(metrics: Metric[], models: Model[]) {
  return `<div class="table-wrap">
    <table class="metrics">
      <caption class="sr-only">Model metrics on the held-out rows</caption>
      <thead><tr><th scope="col">Metric</th>
      ${models.map((model) => `<th scope="col">${modelLabel(model)}</th>`).join("")}</tr></thead>
      <tbody>${metrics.map(({ name, direction, values, scope }) => `<tr>
        <th scope="row">${name} <small>${scope ? `${escape(scope)} · ` : ""}${direction === "higher" ? "↑ Higher" : "↓ Lower"} is better</small></th>
        ${values.map((value) => `<td>${value}</td>`).join("")}</tr>`).join("")}</tbody>
    </table>
  </div>`;
}

function resultScope(result: Result) {
  const { setup, trainCount, testCount } = result;
  return `<div class="result-scope">
    <div><h3>${escape(setup.outcome.name)}</h3>
      <p>${count(trainCount)} training · ${count(testCount)} test · ${setup.predictors.length} predictors</p></div>
    <span class="compute">CPU only</span>
  </div>`;
}

function confusionTable(matrix: number[][], classes: string[], model: string) {
  const labels = classes.map((value) => escape(value));
  return `<div class="table-wrap" tabindex="0" role="region" aria-label="${model} confusion table">
    <table class="detail-table">
      <caption>${model} · predicted class by column</caption>
      <thead><tr><th scope="col">Actual class</th>${labels.map((value) => `<th scope="col">${value}</th>`).join("")}</tr></thead>
      <tbody>${matrix.map((row, i) => `<tr><th scope="row">${labels[i]}</th>${row.map((n) => `<td>${count(n)}</td>`).join("")}</tr>`).join("")}</tbody>
    </table>
  </div>`;
}

function classification(container: HTMLElement, result: Extract<Result, { task: "classification" }>) {
  const { actual, classes, setup } = result;
  const n = classes.length;
  const labels = classes.map((value) => setup.outcome.labels?.[value] ?? value);
  const models = modelList("Logistic regression");
  const modelMetrics = models.map(({ key, name }) => ({
    name: key === "baseline" && n > 2 ? "One-vs-rest logistic regression" : name,
    ...classificationMetrics(actual, result[key], n),
  }));
  const positives = classes.map((_, index) => actual.filter((code) => code === index).length);
  const choices = classes.map((_, code) => code).sort((a, b) => labels[a].localeCompare(labels[b], undefined, { numeric: true }));
  container.innerHTML = `${resultScope(result)}
    <div id="metric-table"></div>
    <div class="chart-block">
      <div class="chart-heading">
        <div><h3>Precision–recall</h3><p id="prevalence"></p></div>
        <label class="chart-control">Class of interest <select id="positive-class">${choices.map((code) =>
    `<option value="${code}">${escape(labels[code])}</option>`).join("")}</select></label>
      </div>
      <div class="legend">
        ${models.map(({ key, name }) => `<span><i class="line-key ${key}" aria-hidden="true"></i>${name}</span>`).join("")}
        <span><i class="line-key dotted" aria-hidden="true"></i>Class prevalence</span>
      </div>
      <canvas class="pr-chart" role="img"></canvas>
      <p class="chart-note">Each point uses a different score threshold. A higher curve gives more precision at the same recall.</p>
    </div>
    <details class="result-detail">
      <summary>Confusion tables and metric definitions</summary>
      <p>Counts use the highest-scoring class for each row. Average precision and the curve use the selected class. Accuracy and balanced accuracy cover all classes.</p>
      ${modelMetrics.map(({ confusion, name }) => confusionTable(confusion, labels, name)).join("")}
      <dl class="definitions">
        <dt>Average precision</dt><dd>Precision averaged across thresholds, weighted by gains in recall. The test-set prevalence gives the no-skill reference.</dd>
        <dt>Balanced accuracy</dt><dd>Mean recall across the outcome classes. Each class gets the same weight.</dd>
        <dt>Accuracy</dt><dd>Share of test rows assigned to the correct class.</dd>
      </dl>
    </details>`;
  const picker = container.querySelector<HTMLSelectElement>("#positive-class")!;
  const canvas = container.querySelector("canvas")!;
  picker.value = String(positives.indexOf(Math.min(...positives)));
  let disconnect: (() => void) | undefined;
  function updateClass() {
    const code = Number(picker.value);
    const curves = models.map(({ key, name }) => ({ key, name, ...precisionRecall(actual, classScores(result[key], n, code), code) }));
    const prevalence = curves[0].prevalence;
    container.querySelector("#prevalence")!.textContent =
      `${labels[code]} · ${count(positives[code])} of ${count(actual.length)} rows (${percent.format(prevalence)})`;
    container.querySelector("#metric-table")!.innerHTML = metricTable([
      { name: "Average precision", scope: labels[code], direction: "higher", values: curves.map(({ ap }) => percent.format(ap)) },
      { name: "Balanced accuracy", direction: "higher", values: modelMetrics.map(({ balanced }) => percent.format(balanced)) },
      { name: "Accuracy", direction: "higher", values: modelMetrics.map(({ accuracy }) => percent.format(accuracy)) },
    ], models);
    const summaries = curves.map(({ name, ap }) => `${name} ${percent.format(ap)}`).join(", ");
    canvas.setAttribute("aria-label", `Precision–recall for ${labels[code]}. Average precision ${summaries}. Reference prevalence ${percent.format(prevalence)}.`);
    disconnect?.();
    disconnect = drawPrecisionRecall(canvas, curves, prevalence);
  }
  picker.addEventListener("change", updateClass);
  updateClass();
  return () => disconnect?.();
}

function regression(container: HTMLElement, result: Extract<Result, { task: "regression" }>) {
  const { actual, setup } = result;
  const models = modelList("Linear regression");
  const modelMetrics = models.map(({ key }) => regressionMetrics(actual, result[key]));
  const unit = setup.outcome.unit ?? "dataset units";
  const predictions = models.flatMap(({ key }) => Array.from(result[key]));
  const min = Math.min(...actual, ...predictions);
  const max = Math.max(...actual, ...predictions);
  const step = 10 ** Math.floor(Math.log10(Math.max(1, max - min) / 4));
  const bounds: [number, number] = [Math.floor(min / step) * step, (Math.floor(max / step) + 1) * step];
  const rows = actual.map((value, row) => ({ value, row, maxError: Math.max(...models.map(({ key }) => Math.abs(result[key][row] - value))) }))
    .sort((a, b) => b.maxError - a.maxError).slice(0, 10);
  container.innerHTML = `${resultScope(result)}
    ${metricTable([
    { name: "Mean absolute error", direction: "lower", values: modelMetrics.map(({ mae }) => measure(mae)) },
    { name: "Root mean squared error", direction: "lower", values: modelMetrics.map(({ rmse }) => measure(rmse)) },
    { name: "R²", direction: "higher", values: modelMetrics.map(({ r2 }) => measure(r2)) },
  ], models)}
    <div class="chart-block">
      <div class="chart-heading"><div><h3>Actual vs. predicted</h3><p>All plots use the same scale · ${escape(unit)}</p></div></div>
      <div class="scatter-grid">
        ${models.map((model) => `<figure><figcaption>${modelLabel(model)}</figcaption><canvas class="scatter-chart" role="img"></canvas></figure>`).join("")}
      </div>
      <p class="chart-note">Each mark is one test row. Predictions above the dashed line are too high. Predictions below it are too low.</p>
    </div>
    <details class="result-detail">
      <summary>Largest errors and metric definitions</summary>
      <p>Ten test rows with the largest absolute errors across the models. Values are in ${escape(unit)}.</p>
      <div class="table-wrap" tabindex="0" role="region" aria-label="Largest errors table">
        <table class="detail-table">
          <caption class="sr-only">Observed and predicted values for rows with the largest errors</caption>
          <thead><tr><th scope="col">Actual</th>${models.map(({ name }) => `<th scope="col">${name} prediction</th>`).join("")}</tr></thead>
          <tbody>${rows.map(({ value, row }) => `<tr><td>${measure(value)}</td>${models.map(({ key }) =>
      `<td>${measure(result[key][row])}</td>`).join("")}</tr>`).join("")}</tbody>
        </table>
      </div>
      <dl class="definitions">
        <dt>Mean absolute error</dt><dd>Average size of a prediction error in ${escape(unit)}.</dd>
        <dt>Root mean squared error</dt><dd>Square root of the average squared error, in ${escape(unit)}. Large misses count more.</dd>
        <dt>R²</dt><dd>One minus the ratio of squared prediction error to squared variation around the observed test mean. It can be negative and is undefined when all test outcomes are equal.</dd>
      </dl>
    </details>`;
  const canvases = container.querySelectorAll("canvas");
  const disconnect = models.map(({ key, name }, index) => {
    const canvas = canvases[index];
    canvas.setAttribute("aria-label", `${name} predicted versus actual. Mean absolute error ${measure(modelMetrics[index].mae)} ${unit}.`);
    return drawScatter(canvas, actual, result[key], bounds, key);
  });
  return () => disconnect.forEach((stop) => stop());
}

export function renderResults(container: HTMLElement, result: Result) {
  return result.task === "classification" ? classification(container, result) : regression(container, result);
}
