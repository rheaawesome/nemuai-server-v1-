# NemuAI Server

FastAPI wrapper around the NemuAI inference pipeline (ported from `NemuAI.ipynb`).
Takes an audio recording, runs it through YAMNet + the trained logistic
regression classifier, and returns the same report JSON shape the original
Colab notebook produces (`schema_version: "nemuai-final-colab-1"`).

**Not a diagnostic tool.** Per the bundle's own README and config
(`medical_validation: false`), this flags "possible obstructive/mixed
apnea-like" audio patterns for an experimental prototype. It does not
diagnose sleep apnea or calculate clinical AHI. Keep that disclaimer visible
in the app UI wherever these results are shown -- it's already included in
every report under the `"disclaimer"` field.

## 1. Add the model bundle

Copy these 6 files from `nemuai_final_colab_bundle.zip` into `model_bundle/`:

```
model_bundle/
  nemuai_final_model.joblib
  nemuai_final_scaler.joblib
  nemuai_final_threshold.json
  nemuai_final_config.json
  nemuai_final_selection.json
  README.txt
```

## 2. Local run

Requires Python 3.11 and system `ffmpeg`/`ffprobe` on PATH.

```bash
# macOS
brew install ffmpeg

pip install -r requirements.txt
uvicorn main:app --reload --port 8000
```

Test it:

```bash
curl -F "audio=@america.m4a" http://localhost:8000/analyze | python3 -m json.tool
```

Compare against the known-good output:

```bash
python3 -c "
import json
a = json.load(open('server_output.json'))
b = json.load(open('america_nemuai_final_report.json'))
# quick smoke check -- see 'Validating output' below for a real diff
print(a['overall_experimental_result'], b['overall_experimental_result'])
"
```

## 3. Deploy to Render

Push this folder (with `model_bundle/` populated) to a repo, then on Render:

- New → Web Service → connect the repo
- Environment: **Docker** (the Dockerfile installs ffmpeg, which Render's
  native Python runtime won't let you do)
- Render sets `$PORT` automatically -- the Dockerfile already respects it
- First deploy will be slow (~1-2 min extra) while TensorFlow downloads
  YAMNet from tfhub.dev on startup; consider Render's persistent disk or a
  larger instance if cold starts are too slow

## Validating output matches the original notebook

Since this is a hand-ported copy of the notebook logic, don't trust it blind.
Run `america.m4a` through this server and diff the result against
`america_nemuai_final_report.json` (the notebook's own output for that file):

```bash
python3 -c "
import json
mine = json.load(open('server_output.json'))
theirs = json.load(open('america_nemuai_final_report.json'))
for key in ['overall_experimental_result']:
    print(key, '-> match:', mine[key] == theirs[key])
for i, (a, b) in enumerate(zip(mine['windows'], theirs['windows'])):
    if a['predicted_label'] != b['predicted_label']:
        print(f'window {i} label mismatch:', a['predicted_label'], 'vs', b['predicted_label'])
    if a['smoothed_probability'] is not None and b['smoothed_probability'] is not None:
        if abs(a['smoothed_probability'] - b['smoothed_probability']) > 1e-4:
            print(f'window {i} probability mismatch:', a['smoothed_probability'], 'vs', b['smoothed_probability'])
print('done')
"
```

`generated_at` will always differ (timestamp) -- ignore that field. Everything
else should match exactly or very closely (floating point noise aside),
since the pipeline code was copied line-for-line from the notebook rather
than reimplemented from a description.
