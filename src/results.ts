import type { Result, Setup } from "./analysis.ts";
import { drawPrecisionRecall, drawScatter } from "./charts.ts";
import { classScores, classificationMetrics, precisionRecall, regressionMetrics } from "./metrics.ts";

const integer = new Intl.NumberFormat("en");
const decimal = new Intl.NumberFormat("en", { maximumFractionDigits: 2 });
const percent = new Intl.NumberFormat("en", { style: "percent", maximumFractionDigits: 1 });
export const count = (value: number) => integer.format(value);
const measure = (value: number | null) => value === null ? "—" : decimal.format(value);
export const escape = (value: string) => value.replace(/&/g, "&amp;").replace(/</g, "&lt;")
  .replace(/>/g, "&gt;").replace(/"/g, "&quot;").replace(/'/g, "&#39;");

const kumoName = "Kumo Tabular Small";

export function emptyResults(container: HTMLElement, setup: Setup, changed: boolean) {
  const baseline = setup.task === "classification" ? "Logistic regression" : "Linear regression";
  container.innerHTML = `
    <div class="empty">
      <div class="empty-diagram" aria-hidden="true">
        <div class="diagram-source"><span></span><span></span><span></span><span></span><span></span></div>
        <div class="diagram-connector"></div>
        <div class="diagram-models"><span></span><span></span></div>
      </div>
      <h3>${changed ? "Ready for a new comparison" : "Compare two models on the same test rows"}</h3>
      <p>Run the comparison to see how ${kumoName} and ${baseline.toLowerCase()} predict <strong>${escape(setup.outcome.name.toLowerCase())}</strong>.</p>
    </div>
    <div class="empty-footer">
      <span><i class="model-marker kumo" aria-hidden="true"></i>${kumoName}</span>
      <span><i class="model-marker baseline" aria-hidden="true"></i>${baseline}</span>
    </div>`;
}

type Metric = { name: string; direction: "higher" | "lower"; values: [string, string]; scope?: string };

function metricTable(metrics: Metric[], baselineName: string) {
  return `<div class="table-wrap">
    <table class="metrics">
      <caption class="sr-only">Model metrics on the held-out rows</caption>
      <thead><tr><th scope="col">Metric</th>
      <th scope="col"><i class="model-marker kumo" aria-hidden="true"></i>${kumoName}</th>
      <th scope="col"><i class="model-marker baseline" aria-hidden="true"></i>${baselineName}</th></tr></thead>
      <tbody>${metrics.map(({ name, direction, values, scope }) => `<tr>
        <th scope="row">${name} <small>${scope ? `${escape(scope)} · ` : ""}${direction === "higher" ? "↑ Higher" : "↓ Lower"} is better</small></th>
        <td>${values[0]}</td><td>${values[1]}</td></tr>`).join("")}</tbody>
    </table>
  </div>`;
}

function resultScope(result: Result) {
  const { setup, trainCount, testCount, device, fallback } = result;
  return `<div class="result-scope">
    <div><h3>${escape(setup.outcome.name)}</h3>
      <p>${count(trainCount)} training · ${count(testCount)} test · ${setup.predictors.length} predictors</p></div>
    <span class="compute">${device === "webgpu" ? "WebGPU" : fallback ? "CPU (WebAssembly fallback)" : "CPU (WebAssembly)"}</span>
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
  const { actual, classes, kumo, baseline, setup } = result;
  const n = classes.length;
  const labels = classes.map((value) => setup.outcome.labels?.[value] ?? value);
  const modelMetrics = [classificationMetrics(actual, kumo, n), classificationMetrics(actual, baseline, n)] as const;
  const baselineName = n === 2 ? "Logistic regression" : "One-vs-rest logistic regression";
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
      <div class="legend"><span><i class="line-key solid"></i>Kumo</span><span><i class="line-key dashed"></i>Logistic regression</span><span><i class="line-key dotted"></i>Class prevalence</span></div>
      <canvas class="pr-chart" role="img"></canvas>
      <p class="chart-note">Each point uses a different score threshold. A higher curve gives more precision at the same recall.</p>
    </div>
    <details class="result-detail">
      <summary>Confusion tables and metric definitions</summary>
      <p>Counts use the most probable class for each row. Average precision and the curve use the selected class. Accuracy and balanced accuracy cover all classes.</p>
      ${modelMetrics.map(({ confusion }, index) => confusionTable(confusion, labels, index === 0 ? kumoName : baselineName)).join("")}
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
    const model = precisionRecall(actual, classScores(kumo, n, code), code);
    const base = precisionRecall(actual, classScores(baseline, n, code), code);
    container.querySelector("#prevalence")!.textContent =
      `${labels[code]} · ${count(positives[code])} of ${count(actual.length)} rows (${percent.format(model.prevalence)})`;
    container.querySelector("#metric-table")!.innerHTML = metricTable([
      { name: "Average precision", scope: labels[code], direction: "higher", values: [percent.format(model.ap), percent.format(base.ap)] },
      { name: "Balanced accuracy", direction: "higher", values: [percent.format(modelMetrics[0].balanced), percent.format(modelMetrics[1].balanced)] },
      { name: "Accuracy", direction: "higher", values: [percent.format(modelMetrics[0].accuracy), percent.format(modelMetrics[1].accuracy)] },
    ], baselineName);
    canvas.setAttribute("aria-label", `Precision–recall for ${labels[code]}. Average precision ${percent.format(model.ap)} for Kumo and ${percent.format(base.ap)} for logistic regression. Reference prevalence ${percent.format(model.prevalence)}.`);
    disconnect?.();
    disconnect = drawPrecisionRecall(canvas, model.points, base.points, model.prevalence);
  }
  picker.addEventListener("change", updateClass);
  updateClass();
  return () => disconnect?.();
}

function regression(container: HTMLElement, result: Extract<Result, { task: "regression" }>) {
  const { actual, kumo, baseline, setup } = result;
  const model = regressionMetrics(actual, kumo);
  const base = regressionMetrics(actual, baseline);
  const unit = setup.outcome.unit ?? "dataset units";
  const min = Math.min(...actual, ...kumo, ...baseline);
  const max = Math.max(...actual, ...kumo, ...baseline);
  const step = 10 ** Math.floor(Math.log10(Math.max(1, max - min) / 4));
  const bounds: [number, number] = [Math.floor(min / step) * step, (Math.floor(max / step) + 1) * step];
  const rows = actual.map((value, row) => ({ value, row, maxError: Math.max(Math.abs(kumo[row] - value), Math.abs(baseline[row] - value)) }))
    .sort((a, b) => b.maxError - a.maxError).slice(0, 10);
  container.innerHTML = `${resultScope(result)}
    ${metricTable([
    { name: "Mean absolute error", direction: "lower", values: [measure(model.mae), measure(base.mae)] },
    { name: "Root mean squared error", direction: "lower", values: [measure(model.rmse), measure(base.rmse)] },
    { name: "R²", direction: "higher", values: [measure(model.r2), measure(base.r2)] },
  ], "Linear regression")}
    <div class="chart-block">
      <div class="chart-heading"><div><h3>Actual vs. predicted</h3><p>Both plots use the same scale · ${escape(unit)}</p></div></div>
      <div class="scatter-grid">
        <figure><figcaption><i class="model-marker kumo" aria-hidden="true"></i>Kumo Tabular Small</figcaption><canvas class="scatter-chart" role="img"></canvas></figure>
        <figure><figcaption><i class="model-marker baseline" aria-hidden="true"></i>Linear regression</figcaption><canvas class="scatter-chart" role="img"></canvas></figure>
      </div>
      <p class="chart-note">Each mark is one test row. Predictions above the dashed line are too high. Predictions below it are too low.</p>
    </div>
    <details class="result-detail">
      <summary>Largest errors and metric definitions</summary>
      <p>Ten test rows with the largest absolute errors from either model. Values are in ${escape(unit)}.</p>
      <div class="table-wrap" tabindex="0" role="region" aria-label="Largest errors table">
        <table class="detail-table">
          <caption class="sr-only">Observed and predicted values for rows with the largest errors</caption>
          <thead><tr><th scope="col">Actual</th><th scope="col">Kumo prediction</th><th scope="col">Linear prediction</th></tr></thead>
          <tbody>${rows.map(({ value, row }) => `<tr><td>${measure(value)}</td><td>${measure(kumo[row])}</td><td>${measure(baseline[row])}</td></tr>`).join("")}</tbody>
        </table>
      </div>
      <dl class="definitions">
        <dt>Mean absolute error</dt><dd>Average size of a prediction error in ${escape(unit)}.</dd>
        <dt>Root mean squared error</dt><dd>Square root of the average squared error, in ${escape(unit)}. Large misses count more.</dd>
        <dt>R²</dt><dd>One minus the ratio of squared prediction error to squared variation around the observed test mean. It can be negative and is undefined when all test outcomes are equal.</dd>
      </dl>
    </details>`;
  const [left, right] = container.querySelectorAll("canvas");
  left.setAttribute("aria-label", `Kumo predicted versus actual. Mean absolute error ${measure(model.mae)} ${unit}.`);
  right.setAttribute("aria-label", `Linear regression predicted versus actual. Mean absolute error ${measure(base.mae)} ${unit}.`);
  const stopLeft = drawScatter(left, actual, kumo, bounds, true);
  const stopRight = drawScatter(right, actual, baseline, bounds, false);
  return () => { stopLeft(); stopRight(); };
}

export function renderResults(container: HTMLElement, result: Result) {
  return result.task === "classification" ? classification(container, result) : regression(container, result);
}
