# C1-Tabular: Insurance Premium Prediction (Tabular ML)

This project predicts individual medical insurance charges using tabular features and multiple regression approaches, with a final focus on tuned XGBoost models.

## Project Goals
- Build reliable regression models for `charges`.
- Compare baseline vs tuned performance.
- Improve generalization and reduce overfitting.
- Prepare a deployable model export (`model.onnx`).

## Dataset and Features
The cleaned dataset includes core columns such as:
- `age`
- `bmi`
- `children`
- `gender_encoded`
- `smoker_encoded`
- region dummies (`region_northwest`, `region_southeast`, `region_southwest`)
- target: `charges`

Additional engineered features used in experiments:
- `age_squared`
- `smoker_bmi` (interaction between smoking status and BMI)

## Main Notebooks
- `eda.ipynb`  
  Exploratory data analysis and early baseline comparisons.
- `xgboost_reg.ipynb`  
  XGBoost regression directly on original `charges` scale.
- `xgboost_log_reg.ipynb`  
  XGBoost regression trained on `log1p(charges)` and evaluated on original scale via `expm1`.

## Modeling Workflow (xgboost_log_reg)
1. Load cleaned data.
2. Feature engineering (`age_squared`, `smoker_bmi`).
3. Train-test split.
4. Transform target for training: `y_train_log = log1p(y_train)`.
5. Train baseline XGBoost.
6. Tune hyperparameters with `RandomizedSearchCV`.
7. Use early stopping for training monitoring.
8. Train final tuned model.
9. Convert predictions back to original scale with `expm1`.
10. Evaluate with `R2`, `RMSE`, and `MAE`.
11. Run 5-fold cross-validation.
12. Analyze feature importance and residuals.

## Key Result Snapshot (xgboost_log_reg tuned)
On original `charges` scale:
- Test `R2`: `0.8555`
- Test `RMSE`: `4481.0091`
- Test `MAE`: `1980.3446`

Generalization signal:
- No strong overfitting indication from train-vs-validation RMSE gap in log space.

## Model Export
The final notebook includes ONNX export code:
- Output file: `model.onnx`
- Input order must follow `feature_cols` exactly.
- Model output is still in log space (`log1p` target), so convert back with:

```python
charges = np.expm1(pred_log_charges)
```

## How to Run
1. Open the project in Jupyter.
2. Run cells in order inside `xgboost_log_reg.ipynb`.
3. Re-run the final export cell to generate `model.onnx`.

## Notes
- If ONNX conversion raises feature-name errors, use the latest export cell version that remaps booster feature names to `f0, f1, ...`.
- Keep package versions consistent across training and export environments (`xgboost`, `onnxmltools`, `onnx`).



Input yang diperlukan
Dari model kamu, urutan feature_cols yang dipakai adalah:

age
age_squared
bmi
children
gender_encoded
smoker_encoded
region_northwest
region_southeast
region_southwest
smoker_bmi
Catatan:

age_squared = age * age
smoker_bmi = smoker_encoded * bmi
Region pakai one-hot (3 kolom), jadi hanya salah satu biasanya 1, sisanya 0 (atau semua 0 kalau baseline region yang tidak di-encode).
Contoh Swift (ONNX Runtime)
Berikut contoh alur inferensi (konsepnya):

import Foundation
import onnxruntime_objc
struct ChargesInput {
    let age: Float
    let bmi: Float
    let children: Float
    let genderEncoded: Float      // 0/1
    let smokerEncoded: Float      // 0/1
    let regionNorthwest: Float    // 0/1
    let regionSoutheast: Float    // 0/1
    let regionSouthwest: Float    // 0/1
}
final class ChargesPredictor {
    private let env: ORTEnv
    private let session: ORTSession
    init(modelPath: String) throws {
        env = try ORTEnv(loggingLevel: ORTLoggingLevel.warning)
        let options = try ORTSessionOptions()
        session = try ORTSession(env: env, modelPath: modelPath, sessionOptions: options)
    }
    func predict(_ x: ChargesInput) throws -> Double {
        let ageSquared = x.age * x.age
        let smokerBMI = x.smokerEncoded * x.bmi
        // HARUS sama persis dengan urutan feature_cols saat training/export
        var features: [Float] = [
            x.age,
            ageSquared,
            x.bmi,
            x.children,
            x.genderEncoded,
            x.smokerEncoded,
            x.regionNorthwest,
            x.regionSoutheast,
            x.regionSouthwest,
            smokerBMI
        ]
        // shape: [1, 10]
        let shape: [NSNumber] = [1, 10]
        let inputTensor = try ORTValue(
            tensorData: Data(bytes: &features, count: features.count * MemoryLayout<Float>.size),
            elementType: ORTTensorElementDataType.float,
            shape: shape
        )
        // "input" sesuai export cell ONNX kamu
        let outputs = try session.run(
            withInputs: ["input": inputTensor],
            outputNames: nil,
            runOptions: nil
        )
        // Ambil output pertama (prediksi log-space)
        guard let firstOutput = outputs.first?.value else {
            throw NSError(domain: "PredictError", code: -1, userInfo: [NSLocalizedDescriptionKey: "No model output"])
        }
        let outputData = try firstOutput.tensorData()
        let predLog = outputData.withUnsafeBytes { ptr -> Float in
            ptr.bindMemory(to: Float.self)[0]
        }
        // Model output masih log1p(charges) -> balik ke charges
        let charges = Foundation.expm1(Double(predLog))
        return charges
    }
}
Hal penting biar prediksi benar
Urutan input wajib identik dengan feature_cols.
Tipe data gunakan Float32.
Model output kamu masih log-space, jadi wajib expm1.
Nama input ONNX dari export kamu adalah "input".
Kalau mau, aku bisa kasih versi helper function yang langsung menerima data mentah user (mis. "male", "yes", "southeast") lalu otomatis encode ke 10 fitur di atas.