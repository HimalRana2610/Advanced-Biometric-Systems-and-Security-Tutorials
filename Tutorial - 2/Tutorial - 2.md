# Tutorial 2: Age-Gap Face Recognition

## Problem statement

Design and evaluate a face-recognition system that decides whether two face
images belong to the same person when the images were captured at
significantly different ages.

For example, the system should compare a childhood photograph with an adult
photograph of the same person and distinguish that genuine pair from images
belonging to two different people. Since aging changes facial appearance,
simple pixel or handcrafted-feature comparisons may not be reliable.

In this tutorial, use a pretrained FaceNet model to create identity
embeddings, compare the embeddings, and measure recognition performance.

## Learning objectives

By completing this tutorial, you should be able to:

1. Load and inspect an age-gap face dataset.
2. Preprocess face images for a deep-learning model.
3. Generate 512-dimensional embeddings with FaceNet.
4. Compare embeddings using cosine similarity.
5. Construct genuine and impostor comparison pairs.
6. Apply similarity thresholds to accept or reject pairs.
7. Calculate FMR, FNMR, TAR, ROC, AUC, and EER.
8. Explain the effect of aging on biometric recognition.

## Dataset recommendation

The most appropriate dataset for this problem is the **FG-NET Aging
Database**, because it contains multiple photographs of the same people at
different ages. Use the [FG-NET dataset
page](http://www.fgnet.rsunit.com/) to request/download it and follow its
license and access conditions. Do not redistribute the images.

If FG-NET is unavailable, use **AgeDB-30** as an alternative. It was created
specifically for face recognition across age variation; obtain it from the
[AgeDB resource page](https://ibug.doc.ic.ac.uk/resources/agedb/). A smaller
student project may also use a legally obtained **CACD** subset, but the
subset and subject identities must be documented because CACD access and
licensing vary.

**Recommended choice for this tutorial:** FG-NET, with one fixed pair of
different-age images per subject and at least 30 subjects if possible. Keep
the same dataset and pair list for every method being compared.

The repository's supplied `age_gap_dataset` is already arranged for this
tutorial and can be used when available. Each subject has two face images
captured at different ages:

```text
age_gap_dataset/
├── metadata.csv
├── subject_001/
│   ├── age_00.jpg
│   └── age_54.jpg
├── subject_002/
│   ├── age_03.jpg
│   └── age_54.jpg
└── ...
```

The metadata columns are:

```text
subject_id,image_1,age_1,image_2,age_2,age_gap_years
```

Each metadata row describes one genuine pair. The current dataset contains 82
subjects and 82 genuine pairs.

If you download another dataset, create an equivalent `metadata.csv` before
starting:

```csv
subject_id,image_1,age_1,image_2,age_2,age_gap_years
subject_001,subject_001/age_04.jpg,4,subject_001/age_22.jpg,22,18
```

Use relative paths, record ages in years, and calculate
`age_gap_years = abs(age_2 - age_1)`. Do not put the same image in both a
genuine pair and an impostor pair.

## Software and hardware

Use Python 3.9 or newer. Install the required packages in a virtual
environment:

```powershell
python -m venv .venv
.venv\Scripts\activate
python -m pip install numpy opencv-python matplotlib scikit-learn
python -m pip install torch torchvision facenet-pytorch
```

A GPU is helpful but not required. The first model run may download FaceNet
weights, so ensure that the environment has internet access or place the
weights in the local cache first.

## Required procedure

### 1. Preprocess each image

For every face image:

1. Load the image with OpenCV.
2. Convert it from BGR to RGB.
3. Resize it to `160 x 160` pixels.
4. Convert pixel values to floating point.
5. Normalize values from `[0, 1]` to `[-1, 1]`.

The images are already face crops, so a separate face detector is not
required.

### 2. Generate face embeddings

Use the pretrained FaceNet implementation:

```python
InceptionResnetV1(pretrained="vggface2")
```

The model converts each face into a 512-dimensional identity embedding.
L2-normalize the embeddings before comparison.

### 3. Calculate similarity

For two normalized embeddings, calculate cosine similarity:

```text
similarity = embedding_1 dot embedding_2
```

A larger value indicates a stronger identity match.

### 4. Create comparison pairs

Create the following two types of comparisons:

- **Genuine pair:** the two images belonging to the same subject.
- **Impostor pair:** images belonging to two different subjects.

The default implementation uses 5,000 randomly selected impostor trials.
Use a fixed random seed and save the selected impostor pairs. This makes the
results reproducible. Sample only pairs with different subject IDs and report
the number of genuine and impostor comparisons.

### 5. Classify pairs using a threshold

For a similarity threshold `t`:

```text
If similarity >= t:
    Accept the pair as the same person
Otherwise:
    Reject the pair
```

Evaluate several thresholds over the observed similarity-score range.

## Implementation algorithm

Implement the following sequence, either in a Python script or notebook:

1. Read `metadata.csv`.
2. Load every image only once and store its normalized FaceNet embedding.
3. For every metadata row, compare `image_1` and `image_2` and label the
   result `1` (genuine).
4. Randomly choose image pairs from different subjects and label them `0`
   (impostor).
5. Store a table with `label`, `subject_id` (when applicable), `age_gap_years`
   (when applicable), and `similarity`.
6. For each threshold, compute the three rates below.
7. Use all labels and similarities to calculate ROC, AUC, and EER.
8. Save the table, numerical results, and figures.

Never tune a threshold using the same results that are reported as an
independent test set. If you create a development/test split, choose the
threshold on the development split and report final performance on the test
split.

## Performance measures

For each threshold, calculate:

### False Match Rate (FMR)

The fraction of impostor pairs incorrectly accepted:

```text
FMR = accepted impostor pairs / total impostor pairs
```

### False Non-Match Rate (FNMR)

The fraction of genuine pairs incorrectly rejected:

```text
FNMR = rejected genuine pairs / total genuine pairs
```

### True Accept Rate (TAR)

The fraction of genuine pairs correctly accepted:

```text
TAR = accepted genuine pairs / total genuine pairs
```

Also calculate:

- **ROC curve:** true-positive rate versus false-positive rate.
- **AUC:** overall quality of genuine/impostor score separation.
- **EER:** the operating point where FMR and FNMR are approximately equal.

An AUC near 0.50 indicates random separation; an AUC near 1.00 indicates
strong separation. Lower EER is better.

Report percentages to at least two decimal places. Include the threshold at
which EER was obtained and state whether your score convention treats larger
values as stronger matches.

## Expected deliverables

Your completed tutorial should include:

1. Genuine and impostor similarity scores.
2. Score-distribution plots.
3. FMR, FNMR, and TAR values for multiple thresholds.
4. An ROC curve.
5. AUC and EER values.
6. A threshold-performance table or graph.
7. A short discussion of the results.

Your discussion should answer:

- How well does FaceNet recognize people across the available age gaps?
- Which threshold provides a reasonable balance between FMR and FNMR?
- Why is age-gap recognition difficult?
- How does FaceNet compare with raw pixels, image moments, color histograms,
  and SIFT?

Also report results grouped by age-gap ranges, for example `0-10`, `11-20`,
and `21+` years, if the selected dataset contains enough pairs. This shows
whether performance decreases as the age gap increases.

## Expected result

The validated implementation gives approximately:

```text
AUC: 0.7473
EER: 32.34%
```

This is better than random matching, but the result also shows that very
large age gaps remain challenging.

## Running and checking your work

Run your script or notebook after activating the environment:

```powershell
python analysis.py
```

Before submitting, verify that:

- Every metadata image exists and loads successfully.
- Genuine and impostor counts are reported.
- The random seed and FaceNet model/pretrained weights are recorded.
- Genuine/impostor score distributions and an ROC curve are included.
- FMR, FNMR, TAR, AUC, and EER are calculated from the same documented
  protocol.
- The threshold table, figures, and numerical results can be regenerated.

Do not report the supplied reference result as your own result unless you
actually reproduce it with the same data and protocol. The supplied result
(AUC `0.7473`, EER `32.34%`) is only a reference; different datasets,
preprocessing, pair selection, and random seeds will produce different
values.