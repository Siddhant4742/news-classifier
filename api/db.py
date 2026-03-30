# api/db.py
# Defines the database tables as Python classes
# SQLAlchemy maps these classes to PostgreSQL tables automatically

import os
from datetime import datetime
from sqlalchemy import (
    create_engine, Column, Integer, String,
    Float, DateTime, Boolean, Text, JSON
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import declarative_base, sessionmaker,Session
from dotenv import load_dotenv

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")
engine       = create_engine(DATABASE_URL, echo=False)
SessionLocal = sessionmaker(bind=engine, autocommit=False, autoflush=False)
Base         = declarative_base()


class ModelVersion(Base):
    """
    Tracks every trained model artifact.
    One row per trained model — not per prediction.
    """
    __tablename__ = "model_versions"

    id            = Column(Integer, primary_key=True, index=True)
    name          = Column(String(100), nullable=False)        # "tfidf_lr_v1"
    version       = Column(String(20),  nullable=False)        # "1.0.0"
    algorithm     = Column(String(100), nullable=False)        # "TF-IDF + LogisticRegression"
    artifact_path = Column(String(255), nullable=False)        # "model/artifacts/best_pipeline.pkl"
    parameters    = Column(JSONB, default={})                  # {"C": 1.0, "max_features": 20000}
    metrics       = Column(JSONB, default={})                  # {"test_f1": 0.9175, "cv_f1": 0.9046}
    is_active     = Column(Boolean, default=False)             # only one model active at a time
    created_at    = Column(DateTime, default=datetime.utcnow)

    def __repr__(self):
        return f"<ModelVersion {self.name} v{self.version} active={self.is_active}>"


class Prediction(Base):
    """
    Stores every prediction made by the API.
    One row per request — this is your audit trail.
    """
    __tablename__ = "predictions"

    id              = Column(Integer, primary_key=True, index=True)
    input_text      = Column(Text,        nullable=False)
    predicted_label = Column(String(50),  nullable=False)      # "Business"
    true_label      = Column(String(50),  nullable=True)       # null until human reviews
    confidence      = Column(Float,       nullable=False)      # max probability score
    all_probs       = Column(JSONB,       default={})          # {"World":0.02, "Sports":0.01, ...}
    model_version   = Column(String(20),  nullable=False)      # "1.0.0"
    is_correct      = Column(Boolean,     nullable=True)       # null until true_label known
    latency_ms      = Column(Float,       nullable=True)       # inference time in milliseconds
    request_metadata        = Column(JSONB,       default={})          # request context, source, etc
    created_at      = Column(DateTime,    default=datetime.utcnow)

    def __repr__(self):
        return f"<Prediction {self.predicted_label} conf={self.confidence:.3f}>"


class Experiment(Base):
    """
    Logs every training run with full hyperparameters and results.
    Enables comparing model versions over time.
    """
    __tablename__ = "experiments"

    id              = Column(Integer, primary_key=True, index=True)
    experiment_name = Column(String(100), nullable=False)      # "tfidf_lr_gridsearch_v1"
    model_type      = Column(String(100), nullable=False)      # "LogisticRegression"
    dataset         = Column(String(100), nullable=False)      # "ag_news_120k"
    parameters      = Column(JSONB, default={})                # full param grid + best params
    metrics         = Column(JSONB, default={})                # all metrics from training
    notes           = Column(Text,  nullable=True)             # anything you want to record
    created_at      = Column(DateTime, default=datetime.utcnow)

    def __repr__(self):
        return f"<Experiment {self.experiment_name}>"


def get_db():
    """
    Dependency function for FastAPI.
    Yields a database session and closes it after the request.
    On Day 7 FastAPI will import and use this directly.
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# Add to bottom of api/db.py

def get_active_model_version(db: Session) -> ModelVersion:
    """
    Returns the currently active model version from the database.
    FastAPI will call this on startup to know which model to load.
    If no active model exists, raises a clear error rather than
    serving with no model — fail loud, not silent.
    """
    from sqlalchemy.orm import Session as SessionType

    model = db.query(ModelVersion).filter(
        ModelVersion.is_active == True
    ).first()

    if not model:
        raise RuntimeError(
            "No active model version found in database. "
            "Run db_ops.py to register a model before starting the API."
        )

    return model


def deactivate_all_models(db: Session):
    """
    Sets is_active=False on all model versions.
    Call this before activating a new model version
    to ensure only one model is active at a time.
    """
    db.query(ModelVersion).update({"is_active": False})
    db.commit()
    print("All model versions deactivated.")


def activate_model(db: Session, model_id: int):
    """
    Activates a specific model version by ID.
    Deactivates all others first — enforces single active model.
    Use this when deploying a new model version (e.g. DistilBERT on Day 11).
    """
    deactivate_all_models(db)

    model = db.query(ModelVersion).filter(
        ModelVersion.id == model_id
    ).first()

    if not model:
        raise ValueError(f"No model version found with id={model_id}")

    model.is_active = True
    db.commit()
    db.refresh(model)
    print(f"Activated: {model.name} v{model.version}")
    return model



if __name__ == "__main__":
    # Quick connection test — run this block directly to verify DB is reachable
    try:
        with engine.connect() as conn:
            print("Database connection successful.")
            print(f"Connected to: {DATABASE_URL}")
    except Exception as e:
        print(f"Connection failed: {e}")
        print("Is your Docker container running? Try: docker ps")

# from dotenv import load_dotenv
# import os
# load_dotenv()
# print(os.getenv('DATABASE_URL'))