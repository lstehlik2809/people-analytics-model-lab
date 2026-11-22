# People Analytics Model Lab

A standalone browser app for exploring how NVIDIA Kumo Tabular Small compares with logistic and linear regression on IBM’s fictional employee data. Change the outcome and predictors, run both models, then check their scores and where their predictions go wrong.

## Run locally

```sh
npm ci
npm run dev
npm test
npm run build
npm run preview
```

Use Node.js 24. The production build in `dist/` uses relative paths and can run on GitHub Pages as a static site. First use requires internet access to download the selected model head (about 115–120 MB) and the ONNX Runtime. The model runs in a worker on the device and the browser cache can reuse downloads. The app attempts WebGPU and retries on WebAssembly if session creation, inference, or output validation fails. It keeps one model session in memory at a time.

## Evaluation

Each analysis uses all 1,470 rows, with a reproducible approximately 80/20 split. Classification allocates approximately one fifth of every outcome class to the test set. Regression uses a fixed random split. Both models receive the same labeled training rows and unlabeled test features. Evaluation uses the test labels after inference. Changing predictors preserves the split for the selected outcome. Repeated exploration can guide hypotheses that need independent validation.

All 31 nonconstant, nonidentifier fields can be predictors. The outcome is excluded from that run’s predictors. Outcome choices follow field types. For instance, job level and satisfaction codes are categories. Monthly income uses unspecified dataset currency units.

- Kumo uses the Small ONNX classifier or regressor, one estimator and upstream context-only preprocessing. Predicted class probabilities use a softmax over the active class outputs. Regression averages the model’s 999 quantiles after reversing the training-label standardization.
- The traditional models use an intercept, training-standardized numerical fields, and one-hot encoding with a reference level. Numeric constants in training contribute no features. Linear regression uses a ridge penalty of `1e-8` times the training count on slopes and logistic regression uses `1e-4` times the training count on slopes. Multiclass logistic regression fits one-vs-rest classifiers and normalizes their probabilities across classes.

| What the reader needs | Display and definition |
| --- | --- |
| Compare how well the models rank rows from one outcome class | Average precision weights precision at score thresholds by recall gains. The precision–recall graph compares both models on common axes and marks the selected class’s test prevalence. A lower-prevalence class can be more challenging to rank. The selector defaults to the rarest test class. |
| Compare predictions across all classes | Accuracy is the share of test rows assigned to the correct class. Balanced accuracy averages recall equally across classes. Confusion tables use test counts with observed classes in rows and highest-probability classes in columns. |
| Compare prediction errors for a numeric outcome | MAE averages absolute test errors. RMSE is the square root of the mean squared test error and gives large errors greater weight. Both use the outcome’s units. R² compares total squared test error with variance around the test mean and can be negative. |
| Inspect numeric predictions | Side-by-side scatterplots use identical axes and an equality line. The detail table shows the ten test rows with the largest absolute errors from either model. |

The app is an exploratory tool for people analytics practitioners. The reading order is setup, shared-split scope, side-by-side metrics, then error diagnostics. Line patterns, point shapes, labels, and tables accompany model color. Narrow layouts reflow, and native controls work with keyboard and touch.

## Sources and licenses

- Fictional IBM HR Attrition & Performance dataset. [Dataset source and license](public/data/SOURCE.md)
- NVIDIA Kumo Tabular Small ONNX weights, OpenMDW 1.1. [Classifier export](https://huggingface.co/onnx-community/kumo-tabular-small-classifier-ONNX/tree/db7fec7ac021759105caaaccc7865e329c1520a0), [regressor export](https://huggingface.co/onnx-community/kumo-tabular-small-regressor-ONNX/tree/71ad7c7f8aa4325b1925ea2866820c4d8b5ac235)
- Kumo browser preprocessing and worker adapted from Shreyas Karnik’s Apache 2.0 implementation. [Source notice](public/licenses/kumo-browser-notice.txt), [Apache 2.0 license](public/licenses/kumo-browser-apache-2.0.txt)
