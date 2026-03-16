# Local Tailor — User Guide

This guide walks you through using Local Tailor from first install to daily use. No programming experience required.

---

## 1. Installation (One-Time Setup)

### Step 1: Install Python

Download Python 3.10 or newer from [python.org](https://www.python.org/downloads/).

During installation, **check "Add Python to PATH"**.

### Step 2: Open a Terminal

- **Windows**: Press `Win + R`, type `cmd`, press Enter
- **Mac**: Open Terminal from Applications > Utilities

### Step 3: Navigate to the Project Folder

```bash
cd "C:\python\Local Tailor"
```

Replace with your actual folder path.

### Step 4: Create a Virtual Environment

```bash
python -m venv venv
```

### Step 5: Activate the Virtual Environment

**Windows:**
```bash
venv\Scripts\activate
```

**Mac/Linux:**
```bash
source venv/bin/activate
```

You should see `(venv)` at the start of your terminal line.

### Step 6: Install Dependencies

```bash
pip install -r requirements.txt
```

This takes a few minutes.

---

## 2. Running for the First Time

```bash
python run_pipeline.py first-time
```

**What happens:**
1. A demo dataset of 150 pillow shop comments is generated
2. AI models are downloaded (~580MB, only on the first run)
3. Models are trained on your dimension examples (~30 seconds each)
4. All comments are analyzed (span extraction, classification)
5. Accuracy is evaluated against ground truth
6. The dashboard opens in your browser at **http://localhost:8501**

The first run takes 5-10 minutes. After that, models are cached and it's much faster.

---

## 3. Using the Dashboard

### Dimension Board

Shows comments organized by what customers are talking about.

- **Select a dimension** from the dropdown (e.g., Comfort, Shape, Price)
- Comments are grouped by value (e.g., "too firm", "just right", "too soft")
- **Yellow highlights** show the specific part of the comment that was detected
- Confidence scores show how sure the model is
### Intent Queue

Shows comments that need your attention, sorted by priority:

1. **Needs reply** — customers asking questions (highest priority)
2. **Negative review** — complaints to address
3. **Comparison** — customers comparing to competitors
4. **Monitor** — general discussion worth watching
5. **Positive review** — happy customers
6. **Spam** — off-topic content

Use the filter to focus on specific intents.

### Analytics

Visual charts and statistics:

- **Coverage** — what percentage of comments mention each dimension
- **Value Distribution** — breakdown of values per dimension
- **Intent Summary** — counts by intent type
- **Accuracy** — how well the model performs (if evaluation data exists)
- **Cross-Dimension** — how dimensions relate to each other

### Export

Generate shareable reports:

1. Choose what to include (accuracy, intent queue, top comments)
2. Choose format (HTML, PDF, or both)
3. Click **Generate Report**
4. Download the file using the download button

Previously generated reports are listed at the bottom.

### Config (Editing Dimensions)

Edit your analysis setup without touching any files:

1. **Add a dimension**: Expand "Add new dimension", enter a name and comma-separated values
2. **Edit values**: Change descriptions, add/remove values within a dimension
3. **Manage examples**: Add training examples for each value (8+ recommended per value)
4. **Enable/disable**: Toggle dimensions on/off
5. **Save**: Click "Save changes" to write to disk

After saving, you must retrain for changes to take effect:
```bash
python run_pipeline.py retrain
```

---

## 4. Opening the Dashboard Again (After First Run)

If you've already run the pipeline and just want to view results:

```bash
streamlit run localtailor/app.py
```

Make sure your virtual environment is activated first (`venv\Scripts\activate`).

To stop the dashboard, press `Ctrl+C` in the terminal.

---

## 5. Retraining After Editing Examples

When you add, remove, or change training examples:

```bash
python run_pipeline.py retrain
```

This retrains all models and re-analyzes all comments. Takes about 3-5 minutes.

**Tip**: You need at least 2 examples per value for training to work, but 8+ examples gives much better accuracy.

---

## 6. Common Tasks

### "I want to add a new dimension to track"

Example: You want to track "material" with values "cotton", "polyester", "silk".

**From the dashboard:**
1. Go to **Config**
2. Expand "Add new dimension"
3. Name: `material`
4. Values: `cotton, polyester, silk`
5. Click **Add dimension**
6. Add 8+ example sentences for each value
7. Click **Save changes**
8. Run `python run_pipeline.py retrain` in your terminal

### "I want to remove a dimension"

1. Go to **Config**
2. Select the dimension
3. Click **Delete dimension**
4. Click **Save changes**
5. Run `python run_pipeline.py retrain`

### "I want to improve accuracy for a specific value"

1. Go to **Config**
2. Select the dimension
3. Find the value with low accuracy
4. Add more training examples (aim for 8-12 diverse examples)
5. Click **Save changes**
6. Run `python run_pipeline.py retrain`

### "I want to run on a different port"

```bash
streamlit run localtailor/app.py --server.port 8502
```

### "I want to just open the dashboard without reprocessing"

```bash
python run_pipeline.py load-data
```

---

## 7. Understanding Confidence Scores

Each prediction has a confidence score between 0 and 1:

| Score | Meaning | Flag |
|-------|---------|------|
| 0.80 - 1.00 | Very confident | Classified |
| 0.50 - 0.79 | Reasonably confident | Classified |
| 0.00 - 0.49 | Not confident | Unclear |

Comments marked **Unclear** appear at the top of the Dimension Board so you can review them. Adding similar examples to your training data and retraining usually resolves unclear predictions.

---

## 8. Tips for Better Results

1. **Write diverse examples** — don't just rephrase the same sentence 8 times. Use different words, lengths, and styles.
2. **Match your real data** — examples should sound like your actual customer comments (casual, typos, slang are OK).
3. **Check the Unclear group** — comments the model isn't sure about. If you see a pattern, add similar examples.
4. **Retrain after changes** — run `python run_pipeline.py retrain` after updating examples to see the improvement.
5. **Keep descriptions clear** — the description in each dimension value helps the span extractor find relevant text.
