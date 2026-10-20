Dementia 3D-face Mapping

### Interpretable facial micro-dynamics analysis for dementia-screening research

An interactive research application that converts talking-head video into facial movement time series, baseline-relative behavioural descriptors, exploratory visualisations, and supervised classification outputs.

The project investigates a central question:

> Can temporal patterns in facial movement provide useful evidence for dementia-screening research without relying on spoken content or transcripts?

**Research lead and first author:** Filippo Cenacchi, Macquarie University.

**Research prototype — not a clinically validated diagnostic tool.**

[Paper](https://arxiv.org/abs/2511.13802) ·
[Project overview](https://cenacchi2000.github.io/filippocenacchi.github.io/passive-dementia-screening.html) ·
[Academic profile](https://cenacchi2000.github.io/filippocenacchi.github.io/)

---

## Research motivation

Speech-based cognitive-health models can depend on language, transcription quality, interview structure, and recording conditions. Facial movement offers a complementary source of temporal evidence that can be examined without interpreting what a person says.

This project studies observable movement patterns such as:

- Blink activity and eyelid dynamics.
- Mouth opening and jaw variability.
- Gaze variability.
- Brow asymmetry.
- Small head movements.

The goal is to make these signals inspectable: a researcher should be able to examine the video, extracted movements, temporal summaries, and model outputs together.

These measurements are behavioural descriptors. Their presence or magnitude does not establish a disease-specific biomarker or clinical diagnosis.

## Main contributions

### 1. Video-to-behaviour analysis pipeline

The application connects video processing, facial landmark tracking, head-pose estimation, signal extraction, temporal aggregation, and interactive analysis in one workflow.

The implementation includes landmark and pose smoothing to reduce visual instability. It exposes processing controls so researchers can inspect the effects of frame skipping, image downscaling, baseline duration, and analysis-window length.

This provides a practical bridge between raw recordings and structured behavioural features that can be examined or used in downstream models.

### 2. Interpretable temporal facial descriptors

Rather than relying exclusively on opaque image embeddings, the application extracts named facial measurements.

The supervised feature vector can include:

| Feature family | Example measurements |
|---|---|
| Eyelids and blinks | Eye-aspect ratios, blink activity, blink rate, eyelid variability |
| Mouth and jaw | Mouth opening and jaw variability |
| Gaze | Gaze dispersion and variability |
| Brows | Brow asymmetry and asymmetry amplitude |
| Head movement | Head jitter |

For each available feature stream, the current `Vectorizer` computes:

- Mean.
- Standard deviation.
- Median.
- 10th percentile.
- 90th percentile.

This creates a compact representation whose dimensions remain linked to identifiable movement measurements.

### 3. Baseline-relative behavioural exploration

The application includes an initial baseline period and baseline-relative signal analysis, with an optional Isolation Forest component for behavioural deviation exploration.

The default analysis settings are:

| Setting | Default |
|---|---:|
| Baseline duration | 15 seconds |
| Window length | 6 seconds |
| Window step | 2 seconds |

Baseline-relative deviations describe changes relative to the selected recording baseline. They should not be interpreted as deviations from a clinically established healthy population.

### 4. Interactive visual inspection

The Streamlit interface supports:

- Video previews with facial landmark and pose overlays.
- Three-dimensional facial landmark visualisations.
- Temporal behavioural traces and live readouts.
- PCA-based visualisation using a fixed live projection basis.
- Optional UMAP and HDBSCAN exploration.
- Feature-contribution and heuristic reasoning displays.
- Batch-analysis dashboards.

These views support exploratory inspection and debugging. Projection clusters and heuristic explanations are not, by themselves, evidence of disease separation or causal mechanisms.

### 5. Supervised modelling and local experiment state

Labelled recordings can be added to a local feature database.

The current training implementation combines:

1. Feature standardisation.
2. PCA.
3. A stacked classifier using SVC, Random Forest, and Logistic Regression.
4. Randomised hyperparameter search with stratified cross-validation.
5. Sigmoid probability calibration.
6. A locally selected decision threshold.

The application also includes an online SGD-based update path.

**Implementation detail:** although the interface exposes classifier names, the current `train()` method constructs the stacked pipeline described above. The selected interface label should not be treated as proof that an independent single-model baseline was trained.

---

## Associated paper and reported results

**Passive Dementia Screening via Facial Temporal Micro-Dynamics Analysis of In-the-Wild Talking-Head Video**

Filippo Cenacchi, Longbing Cao, Mitchell McEwan, and Deborah Richards.

[Read the paper](https://arxiv.org/abs/2511.13802)

The paper introduces **YT-DemTalk**, comprising 300 talking-head clips:

- 150 clips associated with self-reported dementia.
- 150 control clips.

The paper reports:

| Metric | Reported value |
|---|---:|
| AUROC | 0.953 |
| Average precision | 0.961 |
| F1-score | 0.851 |
| Accuracy | 0.857 |

**These are paper-reported benchmark results, not results automatically reproduced by launching this application.**

The repository currently contains an interactive application and supporting dependency files. It does not include a complete benchmark reproduction package with the dataset, frozen subject-level splits, trained paper model, and experiment outputs.

The application's statistical-feature and stacked-classifier workflow should also be distinguished from the paper's motion-composition representation and associated experimental pipeline.

---

## Installation

### Recommended environment

Use **Python 3.10 or 3.11**.

The dependency file installs MediaPipe only for Python versions below 3.12. Using a newer interpreter may therefore leave the facial-landmark dependency unavailable.

```bash
git clone https://github.com/cenacchi2000/dementia_3d_face_mapping.git
cd dementia_3d_face_mapping

python -m venv .venv
```

Activate the environment:

```bash
# macOS / Linux
source .venv/bin/activate

# Windows PowerShell
.venv\Scripts\Activate.ps1
```

Install dependencies:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### Optional capabilities

Additional packages may be needed for specific features:

- `yt-dlp` and a working FFmpeg installation for supported online-video workflows.
- `openpyxl` for `.xlsx` batch tables.
- `xlrd` for legacy `.xls` tables.
- PyTorch for the optional landmark-autoencoder functionality.
- InsightFace and its runtime dependencies for the optional detector.

Start with a local video to check the core analysis workflow before enabling optional features.

### Dependency compatibility

The current training code uses the older
`CalibratedClassifierCV(base_estimator=...)` constructor argument.
Recent scikit-learn versions use `estimator=...`.

If training raises an unexpected-keyword error, update that constructor call for your installed version. The published dependency range and current training code require compatibility review before benchmark use.

## Launch

```bash
streamlit run dementia_3d_face_mapping.py
```

Open the local address printed by Streamlit.

## Analyse a single video

1. Upload a supported video or enter a local file path.
2. Configure the baseline, window length, and processing controls.
3. Select **Visualize & Analyze**.
4. Inspect the tracked face, movement traces, and exploratory outputs.

Supported upload extensions include:

```text
.mp4 .mov .avi .mkv .mpg .mpeg .m4v
```

For labelled development data, enable **Add this run to DB (and auto-train)** when you intentionally want to update the local training database.

Keep this option disabled for held-out evaluation recordings.

## Batch processing

Upload a CSV containing video paths and explicit labels:

```csv
url,label
/absolute/path/control_01.mp4,Normotypical
/absolute/path/dementia_01.mp4,Dementia
```

The loader also recognises several alternative source and label column names.

An optional `severity` column is accepted by the loader, but this does not establish validated severity estimation or guarantee that every processing path uses that field.

Available batch actions include:

- **Run batch (auto-train & log)**.
- **Quick Analyze batch (live 6 graphs)**.

Batch processing randomises recording order. Randomisation is not a substitute for a subject-disjoint evaluation split.

**Always provide explicit labels:** the current loader can default missing labels to `Normotypical`.

## Local files and saved state

The application uses a local `cache/` directory. Depending on the workflow, it may create:

| File | Purpose |
|---|---|
| `features_db.json` | Labelled feature vectors |
| `research_model.joblib` | Saved supervised model |
| `model_meta.json` | Model metadata and recorded score |
| `decision_threshold.json` | Selected classification threshold |
| `online_model.joblib` | Online-update model |
| `perf_log.json` | Application performance records |
| `uploads/` | Uploaded video files |

These files are development state, not a frozen benchmark release.

Reset controls can remove saved models, logs, or cached videos. Preserve any experiment state you need before using them.

---

## Evaluation boundaries and limitations

### Subject independence

The current training method uses stratified cross-validation over stored feature rows. It does not enforce grouping by person.

If several clips belong to the same person, construct subject-disjoint training, validation, and test sets outside this application.

### Model selection and threshold selection

The hyperparameter-search score is a model-selection result, not an unbiased final test estimate.

The current decision threshold is selected using predictions on the stored training data. A defensible benchmark should select thresholds on separate validation data and report results on an untouched test set.

### Interactive retraining

Adding recordings, retraining, and updating models changes the predictor. Dashboard metrics collected across these changes should not be interpreted as evaluation of one fixed model.

### Output interpretation

Supervised probabilities, anomaly scores, heuristic severity readouts, and exploratory clusters have different meanings.

When no saved supervised model is available, the classifier's current fallback is `0.5`; this is a placeholder, not an evidence-based risk estimate.

### Clinical generalisation

The associated dataset uses self-reported dementia labels and publicly available videos. Recording conditions, demographics, identity, editing, medication, and other factors may confound observed patterns.

Prospective, independently recruited clinical evaluation is needed before drawing conclusions about diagnostic performance.

### Reproducibility status

This release supports code inspection and interactive experimentation. It should not currently be described as a complete reproduction of the paper's headline results.

---

## Repository structure

```text
dementia_3d_face_mapping/
├── dementia_3d_face_mapping.py
├── requirements.txt
├── packages.txt
├── runtime.txt
└── README.md
```

## Citation

```bibtex
@misc{cenacchi2025passivedementia,
  title = {Passive Dementia Screening via Facial Temporal
           Micro-Dynamics Analysis of In-the-Wild
           Talking-Head Video},
  author = {Cenacchi, Filippo and Cao, Longbing
            and McEwan, Mitchell and Richards, Deborah},
  year = {2025},
  eprint = {2511.13802},
  archivePrefix = {arXiv},
  primaryClass = {cs.CV},
  url = {https://arxiv.org/abs/2511.13802}
}
```

## Contact

**Filippo Cenacchi**  
PhD Researcher, Computer Science, Macquarie University  
Email: filippo.cenacchi@hdr.mq.edu.au  
[Research portfolio](https://cenacchi2000.github.io/filippocenacchi.github.io/)
