# api/db_ops.py
# Tests all three tables by inserting and querying real data
# This is exactly what FastAPI will do on Day 7 — same functions, same DB calls

import sys, os
sys.path.append(os.getcwd())

import pickle
from datetime import datetime
from sqlalchemy.orm import Session
from api.db import SessionLocal, ModelVersion, Prediction, Experiment


def log_model_version(db: Session):
    """Register your trained pipeline in the database."""

    # Load metrics from your experiment log
    import json
    with open("model/artifacts/experiment_log.json", "r") as f:
        metrics = json.load(f)

    model = ModelVersion(
        name          = "tfidf_lr_v1",
        version       = "1.0.0",
        algorithm     = "TF-IDF + LogisticRegression",
        artifact_path = "model/artifacts/best_pipeline.pkl",
        parameters    = metrics.get("best_params", {}),
        metrics       = {
            "test_f1_macro" : metrics.get("test_f1_macro"),
            "cv_f1_macro"   : metrics.get("cv_f1_macro"),
        },
        is_active     = True,
        created_at    = datetime.utcnow()
    )

    db.add(model)
    db.commit()
    db.refresh(model)

    print(f"Registered model version: {model}")
    print(f"  ID            : {model.id}")
    print(f"  Parameters    : {model.parameters}")
    print(f"  Metrics       : {model.metrics}")
    print(f"  Active        : {model.is_active}")
    return model


def log_prediction(db: Session, text: str, model_version: str = "1.0.0"):
    """Run a prediction and store the full result in the database."""

    import time

    # Load pipeline and predict
    with open("model/artifacts/best_pipeline.pkl", "rb") as f:
        pipeline = pickle.load(f)

    LABEL_MAP = {0: "World", 1: "Sports", 2: "Business", 3: "Sci/Tech"}

    start      = time.time()
    pred_idx   = pipeline.predict([text])[0]
    proba      = pipeline.predict_proba([text])[0]
    latency_ms = (time.time() - start) * 1000

    predicted_label = LABEL_MAP[pred_idx]
    confidence      = float(proba.max())
    all_probs       = {LABEL_MAP[i]: round(float(p), 4) for i, p in enumerate(proba)}

    prediction = Prediction(
        input_text      = text,
        predicted_label = predicted_label,
        confidence      = confidence,
        all_probs       = all_probs,
        model_version   = model_version,
        latency_ms      = round(latency_ms, 3),
        metadata        = {"source": "manual_test", "day": 5},
        created_at      = datetime.utcnow()
    )

    db.add(prediction)
    db.commit()
    db.refresh(prediction)

    print(f"\nPrediction stored:")
    print(f"  ID         : {prediction.id}")
    print(f"  Text       : {text[:60]}...")
    print(f"  Label      : {predicted_label}")
    print(f"  Confidence : {confidence:.4f}")
    print(f"  All probs  : {all_probs}")
    print(f"  Latency    : {latency_ms:.2f}ms")
    return prediction


def log_experiment(db: Session):
    """Store your Day 4 GridSearchCV run as an experiment record."""

    import json
    with open("model/artifacts/experiment_log.json", "r") as f:
        metrics = json.load(f)

    experiment = Experiment(
        experiment_name = "tfidf_lr_gridsearch_v1",
        model_type      = "LogisticRegression",
        dataset         = "ag_news_120k",
        parameters      = {
            "param_grid"  : {
                "tfidf__max_features" : [5000, 10000, 20000],
                "classifier__C"       : [0.1, 1.0, 10.0]
            },
            "best_params" : metrics.get("best_params", {}),
            "cv_folds"    : 5,
            "scoring"     : "f1_macro"
        },
        metrics         = metrics,
        notes           = "Baseline TF-IDF + LR pipeline. Best params: max_features=20000, C=1.0",
        created_at      = datetime.utcnow()
    )

    db.add(experiment)
    db.commit()
    db.refresh(experiment)

    print(f"\nExperiment logged:")
    print(f"  ID      : {experiment.id}")
    print(f"  Name    : {experiment.experiment_name}")
    print(f"  Metrics : {experiment.metrics}")
    return experiment


def query_and_display(db: Session):
    """
    Run analytical queries — the kind you'll use in production
    to monitor model performance over time.
    """
    print("\n" + "="*55)
    print("Database analytics")
    print("="*55)

    # Query 1 — all active models
    active_models = db.query(ModelVersion).filter(
        ModelVersion.is_active == True
    ).all()
    print(f"\nActive models: {len(active_models)}")
    for m in active_models:
        print(f"  {m.name} v{m.version} — F1: {m.metrics.get('test_f1_macro')}")

    # Query 2 — all predictions with confidence above threshold
    high_conf = db.query(Prediction).filter(
        Prediction.confidence >= 0.90
    ).all()
    print(f"\nHigh confidence predictions (>=0.90): {len(high_conf)}")
    for p in high_conf:
        print(f"  [{p.predicted_label}] conf={p.confidence:.4f} — {p.input_text[:50]}...")

    # Query 3 — low confidence predictions worth reviewing
    low_conf = db.query(Prediction).filter(
        Prediction.confidence < 0.70
    ).all()
    print(f"\nLow confidence predictions (<0.70): {len(low_conf)}")
    for p in low_conf:
        print(f"  [{p.predicted_label}] conf={p.confidence:.4f} — {p.input_text[:50]}...")

    # Query 4 — prediction count per label
    from sqlalchemy import func
    label_counts = db.query(
        Prediction.predicted_label,
        func.count(Prediction.id).label("count"),
        func.avg(Prediction.confidence).label("avg_confidence")
    ).group_by(Prediction.predicted_label).all()

    print(f"\nPredictions per label:")
    for row in label_counts:
        print(f"  {row.predicted_label:<12} count={row.count}  avg_conf={row.avg_confidence:.4f}")



if __name__ == "__main__":
    db = SessionLocal()

    try:
        print("Step 1 — Register model version")
        print("-"*40)
        model = log_model_version(db)

        print("\nStep 2 — Log sample predictions")
        print("-"*40)
        test_texts = [
            "Goldman Sachs reported record quarterly profits driven by investment banking fees",
            "Virat Kohli smashed a brilliant century in the third Test match against Australia",
            "The UN Security Council convened an emergency session over the border dispute",
            "OpenAI released GPT-5 with significantly improved reasoning capabilities",
            "This article has very ambiguous content that could belong to multiple categories"
        ]
        for text in test_texts:
            log_prediction(db, text)

        print("\nStep 3 — Log experiment")
        print("-"*40)
        log_experiment(db)

        print("\nStep 4 — Query and analyse")
        print("-"*40)
        query_and_display(db)

    finally:
        db.close()
        print("\nDone. Database session closed.")