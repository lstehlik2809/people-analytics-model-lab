import { getFeatures, splitRows, type Field, type Row } from "./data.ts";
import { classProbabilities, prepare, regressionOutputs, type Task } from "./kumo/preprocess.ts";
import type { Device, Head, WorkerRequest, WorkerResponse } from "./kumo/worker.ts";

export type Setup = { task: Task; outcome: Field; predictors: Field[] };
type ResultBase = { setup: Setup; trainCount: number; testCount: number; device: Device; fallback: boolean };
export type Result =
  | (ResultBase & { task: "classification"; classes: string[]; actual: number[]; kumo: Float32Array; baseline: Float64Array })
  | (ResultBase & { task: "regression"; actual: number[]; kumo: Float64Array; baseline: Float64Array });

type Status =
  | { phase: "prepare" }
  | { phase: "load"; device: Device; loaded?: number; total?: number }
  | { phase: "predict"; device: Device }
  | { phase: "fallback" };

export function prepareAnalysis(rows: Row[], setup: Setup) {
  const { task, outcome, predictors } = setup;
  const eligible = predictors.filter((field) => field.key !== outcome.key);
  if (eligible.length !== predictors.length || !eligible.length) throw new Error("Choose at least one predictor other than the outcome.");
  const { train, test } = splitRows(rows, outcome, task);
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

function availableDevice(): Device {
  return "gpu" in navigator ? "webgpu" : "wasm";
}

function checkedOutput(result: Extract<WorkerResponse, { type: "result" }>, rows: number, activeColumns: number, head: Head) {
  const width = head === "classifier" ? 10 : 999;
  if (result.dims.length !== 2 || result.dims[0] !== rows || result.dims[1] !== width ||
    result.logits.length !== rows * width || result.baseline.length !== rows * (head === "classifier" ? activeColumns : 1)) {
    throw new Error("A model returned an unexpected number of scores or rows.");
  }
  for (let row = 0; row < rows; row++) {
    for (let column = 0; column < activeColumns; column++) {
      if (!Number.isFinite(result.logits[row * width + column])) throw new Error("Kumo produced non-finite predictions.");
    }
  }
  if (!result.baseline.every(Number.isFinite)) throw new Error("The traditional model produced non-finite predictions.");
  return result;
}

export class AnalysisRunner {
  private worker: Worker | null = null;
  private device = availableDevice();

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
          status({ phase: "load", device: this.device, loaded: data.loaded, total: data.total });
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

  async run(rows: Row[], setup: Setup, status: (status: Status) => void, signal: AbortSignal): Promise<Result> {
    status({ phase: "prepare" });
    const { train, test, features, prepared, targets } = prepareAnalysis(rows, setup);
    const head = setup.task === "classification" ? "classifier" : "regressor";
    const classCount = prepared.classes?.length ?? 0;
    const runModel = async () => {
      status({ phase: "load", device: this.device });
      const loaded = await this.request({ type: "load", head, device: this.device }, status, signal);
      if (loaded.type !== "loaded") throw new Error("The model could not start.");
      status({ phase: "predict", device: this.device });
      const result = await this.request({
        type: "run", head, features, classCount, baselineTargets: targets,
        x: prepared.x, rows: prepared.rows, columns: prepared.columns,
        yContext: prepared.yContext, categoricalMask: prepared.categoricalMask,
      }, status, signal);
      if (result.type !== "result") throw new Error("The model did not return predictions.");
      return checkedOutput(result, test.length, classCount || 999, head);
    };
    let output: Awaited<ReturnType<typeof runModel>>;
    let fallback = false;
    try {
      output = await runModel();
    } catch (error) {
      this.reset();
      signal.throwIfAborted();
      if (this.device !== "webgpu") throw error;
      this.device = "wasm";
      fallback = true;
      status({ phase: "fallback" });
      try {
        output = await runModel();
      } catch (fallbackError) {
        this.reset();
        throw fallbackError;
      }
    }
    const common = { setup, trainCount: train.length, testCount: test.length, baseline: output.baseline, device: this.device, fallback };
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
