import { getFeatures, splitRows, type Field, type Row } from "./data.ts";
import { classProbabilities, prepare, regressionOutputs, type Task } from "./kumo/preprocess.ts";
import type { Head, WorkerRequest, WorkerResponse } from "./kumo/worker.ts";

export type Setup = { task: Task; outcome: Field; predictors: Field[] };
type ResultBase = { setup: Setup; trainCount: number; testCount: number; baseline: Float64Array; forest: Float64Array };
export type Result =
  | (ResultBase & { task: "classification"; classes: string[]; actual: number[]; kumo: Float32Array })
  | (ResultBase & { task: "regression"; actual: number[]; kumo: Float64Array });

type Status =
  | { phase: "prepare" }
  | { phase: "load"; loaded?: number; total?: number }
  | { phase: "predict" };

export function prepareAnalysis(rows: Row[], setup: Setup, repeat = 0) {
  const { task, outcome, predictors } = setup;
  const eligible = predictors.filter((field) => field.key !== outcome.key);
  if (eligible.length !== predictors.length || !eligible.length) throw new Error("Choose at least one predictor other than the outcome.");
  const { train, test } = splitRows(rows, outcome, task, repeat);
  const features = getFeatures(rows, [...train, ...test], eligible);
  const labels = train.map((index) => rows[index][outcome.key]);
  const targets = task === "regression" ? Float64Array.from(labels, Number) : null;
  const prepared = prepare(features, targets ?? labels, task, train.length);
  if (!prepared.columns) throw new Error("The selected predictors have no variation in the training rows.");
  if (prepared.classes && (prepared.classes.length < 2 || prepared.classes.length > 10)) {
    throw new Error("Kumo’s classifier needs between 2 and 10 outcome classes.");
  }
  return { train, test, features, prepared, targets };
}

function checkedOutput(result: Extract<WorkerResponse, { type: "result" }>, rows: number, activeColumns: number, head: Head) {
  const width = head === "classifier" ? 10 : 999;
  const predictionCount = rows * (head === "classifier" ? activeColumns : 1);
  if (result.dims.length !== 2 || result.dims[0] !== rows || result.dims[1] !== width ||
    result.logits.length !== rows * width || result.baseline.length !== predictionCount || result.forest.length !== predictionCount) {
    throw new Error("A model returned an unexpected number of scores or rows.");
  }
  for (let row = 0; row < rows; row++) {
    for (let column = 0; column < activeColumns; column++) {
      if (!Number.isFinite(result.logits[row * width + column])) throw new Error("Kumo produced non-finite predictions.");
    }
  }
  if (!result.baseline.every(Number.isFinite)) throw new Error("The traditional model produced non-finite predictions.");
  if (!result.forest.every(Number.isFinite)) throw new Error("Random Forest produced non-finite predictions.");
  return result;
}

export class AnalysisRunner {
  private worker: Worker | null = null;

  private request(request: WorkerRequest, status: (status: Status) => void, signal: AbortSignal) {
    signal.throwIfAborted();
    const worker = this.worker ??= new Worker(new URL("./kumo/worker.ts", import.meta.url), { type: "module" });
    return new Promise<Exclude<WorkerResponse, { type: "progress" | "error" }>>((resolve, reject) => {
      const clean = () => {
        signal.removeEventListener("abort", cancel);
        worker.onmessage = null;
        worker.onerror = null;
        worker.onmessageerror = null;
      };
      const cancel = () => {
        clean();
        this.reset();
        reject(signal.reason);
      };
      const fail = (message: string) => {
        clean();
        reject(new Error(message));
      };
      signal.addEventListener("abort", cancel, { once: true });
      worker.onerror = (error) => fail(error.message || "The browser could not start the model worker.");
      worker.onmessageerror = () => fail("The browser could not read the model response.");
      worker.onmessage = ({ data }: MessageEvent<WorkerResponse>) => {
        if (data.type === "progress") {
          status({ phase: "load", loaded: data.loaded, total: data.total });
          return;
        }
        if (data.type === "error") return fail(data.message);
        clean();
        resolve(data);
      };
      worker.postMessage(request);
    });
  }

  private reset() {
    this.worker?.terminate();
    this.worker = null;
  }

  async run(rows: Row[], setup: Setup, status: (status: Status) => void, signal: AbortSignal, repeat = 0): Promise<Result> {
    status({ phase: "prepare" });
    const { train, test, features, prepared, targets } = prepareAnalysis(rows, setup, repeat);
    const head = setup.task === "classification" ? "classifier" : "regressor";
    const classCount = prepared.classes?.length ?? 0;
    let output: Extract<WorkerResponse, { type: "result" }>;
    try {
      status({ phase: "load" });
      const loaded = await this.request({ type: "load", head }, status, signal);
      if (loaded.type !== "loaded") throw new Error("The model could not start.");
      status({ phase: "predict" });
      const result = await this.request({
        type: "run", head, features, classCount, regressionTargets: targets,
        x: prepared.x, rows: prepared.rows, columns: prepared.columns,
        yContext: prepared.yContext, categoricalMask: prepared.categoricalMask,
      }, status, signal);
      if (result.type !== "result") throw new Error("The model did not return predictions.");
      output = checkedOutput(result, test.length, classCount || 999, head);
    } catch (error) {
      this.reset();
      throw error;
    }
    const common = { setup, trainCount: train.length, testCount: test.length, baseline: output.baseline, forest: output.forest };
    if (prepared.classes) {
      const classes = prepared.classes;
      return {
        ...common,
        task: "classification",
        classes,
        actual: test.map((row) => classes.indexOf(rows[row][setup.outcome.key])),
        kumo: classProbabilities(output.logits, test.length, classCount),
      };
    }
    return {
      ...common,
      task: "regression",
      actual: test.map((row) => Number(rows[row][setup.outcome.key])),
      kumo: regressionOutputs(output.logits, test.length, prepared.target!).point,
    };
  }
}
