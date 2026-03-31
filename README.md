# News Classifier - Day 6

ML-powered news classification API built with FastAPI and PostgreSQL.

## 📋 Progress

### Day 5: Database Setup ✅
- ✅ PostgreSQL database with 3 SQLAlchemy models
- ✅ Docker + docker-compose configuration
- ✅ Database connection setup & validation
- ✅ .gitignore updated with sensitive files

## Tech Stack

- **Database**: PostgreSQL 15 (Docker)
- **ORM**: SQLAlchemy
- **Containerization**: Docker & docker-compose
- **Python**: 3.10+

## Quick Start

### Prerequisites
```bash
docker-compose up -d              # Start database
pip install -r requirements.txt   # Install dependencies
```

### Verify Setup
```bash
python api/db.py                  # Test connection
```

## Database Architecture (Day 5)

### 3 Core Models

**ModelVersion** - Tracks trained models
```
Fields: id, name, version, algorithm, artifact_path, 
        parameters, metrics, is_active, created_at
Purpose: Store model versions and metadata
```

**Prediction** - Audit trail of predictions
```
Fields: id, input_text, predicted_label, true_label,
        confidence, all_probs, model_version, is_correct,
        latency_ms, request_metadata, created_at
Purpose: Log every prediction + human feedback
```

**Experiment** - Training run logs
```
Fields: id, experiment_name, model_type, dataset,
        parameters, metrics, notes, created_at
Purpose: Track training experiments
```

## Environment Setup (Day 5)

### .env (not in git)
```
DATABASE_URL=postgresql://user:password@127.0.0.1:5433/news_classifier
```

### docker-compose.yml
```yaml
services:
  db:
    image: postgres:15
    container_name: news-classifier-db
    ports:
      - "5433:5432"
```

**Usage**:
```bash
docker-compose up -d      # Start
docker-compose down       # Stop
docker-compose logs -f    # View logs
```

## Project Structure (Day 5)

```
├── api/
│   ├── db.py              # Database models & connection
│   ├── db_ops.py          # Database operations
├── docker-compose.yml     # Database config
├── .env                   # Credentials (not in git)
├── .gitignore             # Security config
└── requirements.txt
```

