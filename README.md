# People Analytics Model Lab

A browser lab for exploring tabular foundation models on IBM’s fictional employee data. It uses NVIDIA Kumo Tabular Small as an example of a pretrained model that predicts from labeled rows in context. Linear or logistic regression and Random Forest fit to those same training rows. Change the outcome and predictors to compare all three on the same held-out rows. [NVIDIA Kumo Tabular documentation](https://docs.nvidia.com/sdgm/structured-data-models/models/kumo-tabular)

[Open People Analytics Model Lab](https://lstehlik2809.github.io/people-analytics-model-lab/)

## Run locally

```sh
npm ci
npm run dev
npm test
npm run build
npm run preview
```

Use Node.js 24. The production build in `dist/` uses relative paths and can run on GitHub Pages as a static site. All three models run on the device’s CPU in a web worker. Kumo uses WebAssembly and keeps one model session in memory at a time. First use requires internet access to download the selected model head (about 115–120 MB per mode) and the ONNX Runtime as needed. The browser cache can reuse downloads.

## Evaluation

Each analysis uses all 1,470 rows, with a reproducible approximately 80/20 split. Classification allocates approximately one fifth of every outcome class to the test set. Regression uses a fixed random split. The three models receive the same labeled training rows and unlabeled test features. Evaluation uses the test labels after inference. Changing predictors preserves the split for the selected outcome. Repeated exploration can guide hypotheses that need independent validation.

All 31 nonconstant, nonidentifier fields can be predictors. The outcome is excluded from that run’s predictors. Outcome choices follow field types. For instance, job level and satisfaction codes are categories. Monthly income uses unspecified dataset currency units.

- Kumo uses the Small ONNX classifier or regressor, one estimator and upstream context-only preprocessing. Predicted class probabilities use a softmax over the active class outputs. Regression averages the model’s 999 quantiles after reversing the training-label standardization.
- The traditional models use an intercept, training-standardized numerical fields, and one-hot encoding with a reference level. Numeric constants in training contribute no features. Linear regression uses a ridge penalty of `1e-8` times the training count on slopes and logistic regression uses `1e-4` times the training count on slopes. Multiclass logistic regression fits one-vs-rest classifiers and normalizes their probabilities across classes.
- Random Forest uses 200 CART trees and seed 20817. It uses the same training-only predictor encoding as traditional regression. The trees omit the intercept column, and each uses a bootstrap sample the size of the training set. Both tasks use a maximum tree depth of 12 and a minimum of three bootstrap draws per child.
  - Classification draws a fresh subset of distinct encoded predictors at each split. The initial subset size is the square root of the encoded predictor count, rounded down. If no valid split on those predictors reduces Gini impurity, the tree checks additional predictors. Class probabilities average the class proportions from the leaf reached in each tree. [Scikit-learn RandomForestClassifier reference](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html)
  - Regression considers all encoded predictors at each split and selects the split with the largest reduction in squared error. The forest averages tree predictions.

| What the reader needs | Display and definition |
| --- | --- |
| Compare how well the models rank rows from one outcome class | Average precision weights precision at score thresholds by recall gains. The precision–recall graph compares all three models on common axes and marks the selected class’s test prevalence. A lower-prevalence class can be more challenging to rank. The selector defaults to the rarest test class. |
| Compare predictions across all classes | Accuracy is the share of test rows assigned to the correct class. Balanced accuracy averages recall equally across classes. Confusion tables use test counts with observed classes in rows and highest-scoring classes in columns. |
| Compare prediction errors for a numeric outcome | MAE averages absolute test errors. RMSE is the square root of the mean squared test error and gives large errors greater weight. Both use the outcome’s units. R² compares total squared test error with variance around the test mean and can be negative. |
| Inspect numeric predictions | The scatterplots use identical axes and an equality line. The detail table shows the ten test rows with the largest absolute errors across the models. |

The app is an exploratory tool for people analytics practitioners. The reading order is setup, shared-split scope, side-by-side metrics, then error diagnostics. Line patterns, point shapes, labels, and tables accompany model color. Narrow layouts reflow, and native controls work with keyboard and touch.

## Sources and licenses

- Fictional IBM HR Attrition & Performance dataset. [Dataset source and license](public/data/SOURCE.md)
- NVIDIA Kumo Tabular Small ONNX weights, OpenMDW 1.1. [Classifier export](https://huggingface.co/onnx-community/kumo-tabular-small-classifier-ONNX/tree/db7fec7ac021759105caaaccc7865e329c1520a0), [regressor export](https://huggingface.co/onnx-community/kumo-tabular-small-regressor-ONNX/tree/71ad7c7f8aa4325b1925ea2866820c4d8b5ac235)
- Kumo browser preprocessing and worker adapted from Shreyas Karnik’s Apache 2.0 implementation. [Source notice](public/licenses/kumo-browser-notice.txt), [Apache 2.0 license](public/licenses/kumo-browser-apache-2.0.txt)
