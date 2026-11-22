import type { Point } from "./metrics.ts";

type Surface = { context: CanvasRenderingContext2D; width: number; height: number };
export type ModelKey = "kumo" | "baseline" | "forest";
const palette = {
  kumo: "#136658",
  baseline: "#815032",
  forest: "#65498a",
  reference: "#626e68",
  axis: "#53655c",
  grid: "#dce6e0",
};
const linePattern: Record<ModelKey, number[]> = {
  kumo: [],
  baseline: [7, 4],
  forest: [8, 3, 2, 3],
};

function connect(canvas: HTMLCanvasElement, draw: (surface: Surface) => void) {
  const repaint = () => {
    const { width, height } = canvas.getBoundingClientRect();
    if (!width || !height) return;
    const ratio = window.devicePixelRatio || 1;
    canvas.width = Math.round(width * ratio);
    canvas.height = Math.round(height * ratio);
    const context = canvas.getContext("2d");
    if (!context) return;
    context.scale(ratio, ratio);
    draw({ context, width, height });
  };
  const observer = new ResizeObserver(repaint);
  observer.observe(canvas);
  return () => observer.disconnect();
}

function axes(surface: Surface, min: number, max: number, horizontal: string, vertical: string, format: (value: number) => string) {
  const { context: ctx, width, height } = surface;
  ctx.font = "12px system-ui, sans-serif";
  ctx.textBaseline = "middle";
  const intervals = width < 350 ? 4 : 5;
  const ticks = Array.from({ length: intervals + 1 }, (_, index) => {
    const value = min + (max - min) * index / intervals;
    const label = format(value);
    return { value, label, width: ctx.measureText(label).width };
  });
  const edgeGap = 5;
  const tickGap = 8;
  const titleMetrics = ctx.measureText(vertical);
  const titleWidth = titleMetrics.actualBoundingBoxAscent + titleMetrics.actualBoundingBoxDescent;
  const titleX = edgeGap + titleWidth / 2;
  const left = Math.ceil(edgeGap + titleWidth + 2 * tickGap + Math.max(...ticks.map((tick) => tick.width)));
  const right = width - Math.ceil(edgeGap + ticks[ticks.length - 1].width / 2);
  const top = 14;
  const bottom = height - 46;
  const x = (n: number) => left + (n - min) / (max - min) * (right - left);
  const y = (n: number) => bottom - (n - min) / (max - min) * (bottom - top);
  ctx.lineWidth = 1;
  for (const { value, label } of ticks) {
    ctx.strokeStyle = palette.grid;
    ctx.beginPath();
    ctx.moveTo(left, y(value));
    ctx.lineTo(right, y(value));
    ctx.moveTo(x(value), top);
    ctx.lineTo(x(value), bottom);
    ctx.stroke();
    ctx.fillStyle = palette.axis;
    ctx.textAlign = "right";
    ctx.fillText(label, left - tickGap, y(value));
    ctx.textAlign = "center";
    ctx.fillText(label, x(value), bottom + 13);
  }
  ctx.fillText(horizontal, (left + right) / 2, height - 8);
  ctx.save();
  ctx.translate(titleX, (bottom + top) / 2);
  ctx.rotate(-Math.PI / 2);
  ctx.fillText(vertical, 0, 0);
  ctx.restore();
  return { ctx, x, y };
}

export function drawPrecisionRecall(
  canvas: HTMLCanvasElement,
  series: { key: ModelKey; points: Point[] }[],
  prevalence: number,
) {
  return connect(canvas, (surface) => {
    const { ctx, x, y } = axes(surface, 0, 1, "Recall", "Precision", (value) => `${Math.round(value * 100)}%`);
    ctx.strokeStyle = palette.reference;
    ctx.lineWidth = 1.4;
    ctx.setLineDash([2, 4]);
    ctx.beginPath();
    ctx.moveTo(x(0), y(prevalence));
    ctx.lineTo(x(1), y(prevalence));
    ctx.stroke();
    for (const { key, points } of series) {
      ctx.strokeStyle = palette[key];
      ctx.lineWidth = 2;
      ctx.setLineDash(linePattern[key]);
      ctx.beginPath();
      ctx.moveTo(x(points[0].x), y(points[0].y));
      for (let i = 1; i < points.length; i++) {
        ctx.lineTo(x(points[i - 1].x), y(points[i].y));
        ctx.lineTo(x(points[i].x), y(points[i].y));
      }
      ctx.stroke();
    }
  });
}

export function drawScatter(canvas: HTMLCanvasElement, actual: number[], predicted: Float64Array, bounds: [number, number], model: ModelKey) {
  const formatter = new Intl.NumberFormat("en", { notation: "compact", maximumFractionDigits: 1 });
  return connect(canvas, (surface) => {
    const [min, max] = bounds;
    const { ctx, x, y } = axes(surface, min, max, "Actual", "Predicted", (value) => formatter.format(value));
    ctx.strokeStyle = palette.reference;
    ctx.lineWidth = 1.4;
    ctx.setLineDash([3, 5]);
    ctx.beginPath();
    ctx.moveTo(x(min), y(min));
    ctx.lineTo(x(max), y(max));
    ctx.stroke();
    ctx.fillStyle = palette[model];
    ctx.globalAlpha = 0.66;
    actual.forEach((value, row) => {
      const px = x(value);
      const py = y(predicted[row]);
      if (model === "kumo") {
        ctx.beginPath();
        ctx.arc(px, py, 2.45, 0, Math.PI * 2);
        ctx.fill();
      } else if (model === "forest") {
        ctx.beginPath();
        ctx.moveTo(px, py - 3);
        ctx.lineTo(px + 3, py);
        ctx.lineTo(px, py + 3);
        ctx.lineTo(px - 3, py);
        ctx.closePath();
        ctx.fill();
      } else {
        ctx.fillRect(px - 2, py - 2, 4, 4);
      }
    });
  });
}
