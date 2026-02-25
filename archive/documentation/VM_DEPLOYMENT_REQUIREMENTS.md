# Patient Feedback System - VM Deployment Requirements

**Document Version:** 1.0  
**Date:** February 18, 2026  
**Prepared For:** IT Infrastructure Team  

---

## 1. Executive Summary

This document outlines the hardware requirements for deploying the Patient Feedback System in an offline/air-gapped VM environment. The application includes AI-powered Speech-to-Text (STT) and text classification capabilities that require specific compute resources.

---

## 2. Application Overview

| Component | Technology |
|-----------|------------|
| **Backend API** | Python 3.x + FastAPI + Uvicorn |
| **Frontend** | Streamlit |
| **Database** | Microsoft SQL Server (via pyodbc) |
| **AI/ML Framework** | PyTorch (CPU-only), Transformers, Faster-Whisper |

### Key Features Requiring Compute Resources
- **Speech-to-Text**: Arabic audio transcription using Whisper Medium model
- **Text Classification**: Multi-label classification (domain, category, severity, stage)
- **Named Entity Recognition**: Staff and doctor name extraction using GLiNER
- **Text Embeddings**: Multilingual MPNet for semantic analysis

---

## 3. User Load Profile

| User Type | Count | Activity Description | Resource Impact |
|-----------|-------|----------------------|-----------------|
| **Data Entry Workers** | 2 | Voice recording, transcription, complaint submission | High (STT usage) |
| **Complaint Supervisor** | 1 | Voice recording, classification review, quality control | High (STT usage) |
| **General Users** | 130 | Dashboard viewing, report reading, status updates | Low (DB reads only) |
| **Total** | **133** | | |

### Usage Pattern Analysis
- Heavy users (3 people) work sequentially - unlikely to have simultaneous STT requests
- Light users (130 people) access system once daily for read operations
- Peak concurrent API requests estimated at 10-20 (light operations)
- STT concurrency: 1-2 simultaneous requests maximum

---

## 4. AI/ML Models Inventory

| Model | Purpose | Disk Size | RAM (Loaded) | Source |
|-------|---------|-----------|--------------|--------|
| **Faster-Whisper Medium** | Speech-to-Text (Arabic) | ~1.5 GB | 2-4 GB | Local (offline) |
| **MPNet Base v2** | Text embeddings | ~1 GB | 1-2 GB | Local (offline) |
| **GLiNER Arabic v2.1** | Named Entity Recognition | ~500 MB | 0.5-1 GB | Local (offline) |
| **scikit-learn classifiers** | Severity, Stage, Harm prediction | ~100 MB | <100 MB | Local (offline) |
| **XGBoost models** | Additional classification | ~50 MB | <100 MB | Local (offline) |

**Note:** All models are pre-downloaded and stored locally. No internet connectivity required during operation.

---

## 5. Hardware Requirements

### 5.1 Minimum Specification (Budget Option)

| Resource | Specification | Notes |
|----------|---------------|-------|
| **CPU** | 4 vCPUs (x86-64) | Modern Intel/AMD processor |
| **RAM** | 8 GB | Tight but functional for described usage |
| **Storage** | 30 GB SSD | Models (3 GB) + App + DB space |
| **Network** | Internal LAN only | Air-gapped environment supported |

**Limitations at Minimum Spec:**
- STT processing: 20-40 seconds per minute of audio
- Slight delays if 2 users trigger STT simultaneously
- Limited headroom for future growth

### 5.2 Recommended Specification

| Resource | Specification | Notes |
|----------|---------------|-------|
| **CPU** | 6 vCPUs (x86-64) | Comfortable headroom |
| **RAM** | 12 GB | Smooth operation with buffer |
| **Storage** | 50 GB SSD | Room for audio files and logs |
| **Network** | Internal LAN only | |

### 5.3 Comfortable Specification (Future-Proof)

| Resource | Specification | Notes |
|----------|---------------|-------|
| **CPU** | 8 vCPUs (x86-64) | Fast STT, handles concurrency |
| **RAM** | 16 GB | All models loaded comfortably |
| **Storage** | 100 GB SSD | Long-term audio/log retention |
| **Network** | Internal LAN only | |

---

## 6. Software Requirements

### Operating System
- **Windows Server 2019/2022** (recommended) or
- **Windows 10/11 Pro** or
- **Ubuntu 22.04 LTS** (Linux alternative)

### Runtime Dependencies
| Software | Version | Purpose |
|----------|---------|---------|
| Python | 3.10+ | Application runtime |
| ODBC Driver | 17+ | SQL Server connectivity |
| Visual C++ Redistributable | 2019+ | Native extensions support |

### Python Package Summary
- Total packages: ~100 (see `requirements.txt`)
- Key frameworks: FastAPI, PyTorch (CPU), Transformers, Faster-Whisper
- Size after installation: ~4-5 GB (including venv)

---

## 7. Performance Expectations

| Operation | Minimum Spec (4 CPU/8 GB) | Recommended Spec (6 CPU/12 GB) |
|-----------|---------------------------|--------------------------------|
| **STT (1 min audio)** | 25-40 seconds | 15-25 seconds |
| **Text Classification** | 1-3 seconds | <1 second |
| **Dashboard Load** | <1 second | <500 ms |
| **Report Generation** | 2-4 seconds | 1-2 seconds |
| **Concurrent Light Users** | 50+ supported | 100+ supported |

---

## 8. Deployment Checklist

### Pre-Deployment
- [ ] Provision VM with specifications above
- [ ] Install Windows/Python/ODBC drivers
- [ ] Configure network access to SQL Server
- [ ] Copy application folder to VM
- [ ] Create Python virtual environment
- [ ] Install requirements: `pip install -r requirements.txt`

### Model Files Required (Copy to VM)
```
models_directory/
├── STT_Models/
│   └── (Whisper model files - auto-downloaded on first run or pre-copy)
├── Classification_Models/
│   └── model_storage/
│       └── mpnet_embeddings/  (~1 GB)
└── NER_Model/
    └── (GLiNER files - auto-downloaded or pre-copy)
```

### Post-Deployment Verification
- [ ] Start backend: `uvicorn main:app --host 0.0.0.0 --port 8000`
- [ ] Start frontend: `streamlit run app.py`
- [ ] Test STT endpoint with sample audio
- [ ] Test classification with sample text
- [ ] Verify database connectivity
- [ ] Test with actual user login

---

## 9. Network Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Internal Network (LAN)                    │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│   ┌──────────────┐         ┌──────────────────────────────┐ │
│   │   Users      │         │     Patient Feedback VM      │ │
│   │  (Browsers)  │◄───────►│  ┌────────────────────────┐  │ │
│   │   133 total  │  HTTP   │  │  Streamlit (Frontend)  │  │ │
│   └──────────────┘  :8501  │  │        Port 8501       │  │ │
│                            │  └───────────┬────────────┘  │ │
│                            │              │               │ │
│                            │              ▼               │ │
│                            │  ┌────────────────────────┐  │ │
│                            │  │  FastAPI (Backend)     │  │ │
│                            │  │      Port 8000         │  │ │
│                            │  │  ┌──────────────────┐  │  │ │
│                            │  │  │ AI/ML Models     │  │  │ │
│                            │  │  │ (Whisper, MPNet) │  │  │ │
│                            │  │  └──────────────────┘  │  │ │
│                            │  └───────────┬────────────┘  │ │
│                            └──────────────┼───────────────┘ │
│                                           │                 │
│   ┌──────────────────────────────────────┐│                 │
│   │         SQL Server Database          ││                 │
│   │         (Existing Server)            │◄┘                │
│   └──────────────────────────────────────┘                  │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

---

## 10. Scaling Considerations

### If User Count Increases
| Scenario | Recommendation |
|----------|----------------|
| 200+ light users | No change needed - current spec handles it |
| 5+ heavy STT users | Upgrade to 8 CPU / 16 GB RAM |
| 10+ concurrent STT | Consider dedicated ML worker process or second VM |

### If Response Time Requirements Change
| Requirement | Recommendation |
|-------------|----------------|
| STT < 10 seconds | Add GPU (NVIDIA with CUDA support) |
| Classification < 500ms | Current spec is sufficient |

---

## 11. Backup & Recovery

### Critical Data Locations
| Data | Location | Backup Frequency |
|------|----------|------------------|
| Application Code | `/Patient_Feedback/` | Weekly |
| ML Models | `/models_directory/` | Monthly (static) |
| Configuration | `/backend/config/` | Weekly |
| Database | SQL Server | Per IT policy |

### Recovery Time Objective
- **Full VM failure**: 2-4 hours (restore from backup + reinstall)
- **Model corruption**: 1 hour (re-copy from source)

---

## 12. Contact & Support

| Role | Contact |
|------|---------|
| Application Developer | [Your Name/Contact] |
| IT Infrastructure | [IT Team Contact] |

---

## Appendix A: Key Python Dependencies

```
fastapi==0.127.0
uvicorn==0.40.0
torch==2.9.1+cpu
transformers==4.57.3
faster-whisper==1.2.1
gliner==0.2.24
scikit-learn==1.7.2
xgboost==3.1.1
pyodbc==5.3.0
streamlit==1.51.0
pandas==2.3.3
numpy==2.3.4
```

Full dependency list available in `requirements.txt` (106 packages total).

---

## Appendix B: Quick Start Commands

```powershell
# Activate virtual environment
.\venv\Scripts\Activate.ps1

# Start Backend API
cd backend
uvicorn main:app --host 0.0.0.0 --port 8000

# Start Frontend (separate terminal)
streamlit run app.py --server.port 8501
```

---

**Document End**
