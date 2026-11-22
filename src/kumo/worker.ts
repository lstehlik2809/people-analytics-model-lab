/// <reference lib="webworker" />
// Adapted from Shreyas Karnik's kumo-tabular-web (Apache-2.0).
// Source and license are listed in public/licenses/kumo-browser-notice.txt.
// Runs all three models off the main thread.
import { AutoModel, Tensor, env } from "@huggingface/transformers";
import { designMatrix, linearRegression, logisticRegression } from "../baseline.ts";
import { forestClassification, forestRegression } from "../forest.ts";
import type { FeatureColumn } from "./preprocess.ts";

export type Head = "classifier" | "regressor";
export type WorkerRequest =
  | { type: "load"; head: Head }
  | {
      type: "run";
      head: Head;
      x: Float32Array;
      rows: number;
      columns: number;
      yContext: Int32Array | Float32Array;
      categoricalMask: Uint8Array;
      features: FeatureColumn[];
      classCount: number;
      regressionTargets: Float64Array | null;
    };
export type WorkerResponse =
  | { type: "progress"; head: Head; loaded: number; total: number }
  | { type: "loaded"; head: Head }
  | { type: "result"; head: Head; logits: Float32Array; dims: number[]; baseline: Float64Array; forest: Float64Array }
  | { type: "error"; message: string };

const post = (message: WorkerResponse, transfer: Transferable[] = []) =>
  (self as unknown as DedicatedWorkerGlobalScope).postMessage(message, transfer);

type Model = ((inputs: Record<string, Tensor>) => Promise<{ logits: Tensor }>) & { dispose(): Promise<void> };
const revisions = {
  classifier: "db7fec7ac021759105caaaccc7865e329c1520a0",
  regressor: "71ad7c7f8aa4325b1925ea2866820c4d8b5ac235",
};
let current: { head: Head; model: Model } | null = null;
env.allowLocalModels = false;
if (env.backends.onnx.wasm) env.backends.onnx.wasm.numThreads = self.crossOriginIsolated ? 4 : 1;

async function load(head: Head) {
  const model = (await AutoModel.from_pretrained(`onnx-community/kumo-tabular-small-${head}-ONNX`, {
    revision: revisions[head],
    device: "wasm",
    dtype: "fp32",
    progress_callback: (info: { status: string; file?: string; loaded?: number; total?: number }) => {
      if (info.status === "progress" && info.file?.endsWith(".onnx_data")) {
        post({ type: "progress", head, loaded: info.loaded ?? 0, total: info.total ?? 0 });
      }
    },
  } as never)) as unknown as Model;
  return { head, model };
}

self.onmessage = async (event: MessageEvent<WorkerRequest>) => {
  const msg = event.data;
  try {
    if (msg.type === "load") {
      if (current?.head !== msg.head) {
        await current?.model.dispose();
        current = null;
        current = await load(msg.head);
      }
      post({ type: "loaded", head: msg.head });
      return;
    }
    if (current?.head !== msg.head) throw new Error(`Model ${msg.head} is not loaded.`);
    const matrix = designMatrix(msg.features, msg.yContext.length, msg.rows);
    const baseline = msg.yContext instanceof Int32Array
      ? logisticRegression(matrix, msg.yContext, msg.classCount)
      : linearRegression(matrix, msg.regressionTargets!);
    const forest = msg.yContext instanceof Int32Array
      ? forestClassification(matrix, msg.yContext, msg.classCount)
      : forestRegression(matrix, msg.regressionTargets!);
    const inputs = {
      x: new Tensor("float32", msg.x, [msg.rows, msg.columns]),
      y_context: msg.yContext instanceof Int32Array
        ? new Tensor("int32", msg.yContext, [msg.yContext.length])
        : new Tensor("float32", msg.yContext, [msg.yContext.length]),
      categorical_mask: new Tensor("bool", msg.categoricalMask, [msg.columns]),
    };
    try {
      const { logits: output } = await current.model(inputs);
      try {
        const logits = Float32Array.from(output.data as Float32Array);
        post({ type: "result", head: msg.head, logits, dims: output.dims, baseline, forest },
          [logits.buffer, baseline.buffer, forest.buffer]);
      } finally {
        output.dispose();
      }
    } finally {
      Object.values(inputs).forEach((tensor) => tensor.dispose());
    }
  } catch (err) {
    post({ type: "error", message: (err as Error)?.message ?? String(err) });
  }
};
