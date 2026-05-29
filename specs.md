# Moralization Detection Platform — Project Outline

## 1. Project Overview

### Goal

Build a GDPR-compliant web platform for detecting and analyzing moralization in text using:

* Fine-tuned RoBERTa models
* Optional LLM providers (OpenAI / Claude)
* Batch CSV/JSON processing
* Coupon-based API access system

The platform should support:

* research workflows
* demos
* explainability
* scalable future extensions

---

# 2. Core Functionalities

## 2.1 Single Text Inference

### Input

* User enters text manually

### Processing

* preprocessing pipeline
* provider selection
* model inference

### Output

* moralization prediction
* confidence scores
* optional explanations
* optional structured JSON output

---

## 2.2 Batch Processing

### Supported Formats

* CSV
* JSON

### Features

* upload files
* asynchronous processing
* downloadable output
* optional queue system

### Output Formats

* CSV
* JSON

---

## 2.3 Provider Selection

The system should support multiple inference providers through a unified interface.

### Initial Providers

#### Local Provider

* Fine-tuned RoBERTa
* Hosted on Hetzner VPS
* Primary inference backend

#### Optional Providers

* OpenAI API
* Claude API
* Hugging Face API (fallback/demo use)

---

# 3. System Architecture

## 3.1 High-Level Architecture

```text
Frontend
    ↓
FastAPI Backend
    ↓
Provider Router
 ├── Local RoBERTa
 ├── OpenAI
 ├── Claude
 └── Hugging Face
```

---

## 3.2 Frontend

### Recommended Stack

* Next.js
* React
* TailwindCSS

### Responsibilities

* text input UI
* batch upload UI
* provider selection
* coupon input
* result visualization
* download interface

---

## 3.3 Backend

### Recommended Stack

* FastAPI
* Python
* Pydantic

### Responsibilities

* inference orchestration
* coupon validation
* rate limiting
* preprocessing
* batching
* provider abstraction
* logging
* GDPR compliance logic

---

## 3.4 Model Layer

### Initial Model

* Fine-tuned RoBERTa
* CPU inference
* ONNX optimized (recommended)

### Runtime

* ONNX Runtime preferred
* PyTorch acceptable initially

### Important Constraints

* load model once at startup
* single worker on 4GB RAM machine
* avoid per-request model loading

---

# 4. Hosting & Infrastructure

## 4.1 VPS Hosting

### Recommended Provider

Hetzner Cloud

### Recommended Machine

* 2 vCPU
* 4 GB RAM
* x86 Intel/AMD architecture
* 40 GB SSD

### Why

* low cost
* GDPR-friendly EU hosting
* stable CPU inference
* enough RAM for RoBERTa

---

## 4.2 Deployment Architecture

```text
Nginx
 ├── Frontend
 ├── FastAPI Backend
 └── RoBERTa Inference
```

---

## 4.3 Reverse Proxy

### Recommended

* Nginx

### Responsibilities

* HTTPS termination
* static frontend serving
* routing
* compression
* request size limits

---

# 5. Database Design

## 5.1 Recommended Database

* PostgreSQL
* Supabase optional

---

## 5.2 Coupon Table

```sql
CREATE TABLE coupons (
    id UUID PRIMARY KEY,
    code TEXT UNIQUE NOT NULL,
    credits_remaining INTEGER NOT NULL,
    max_credits INTEGER NOT NULL,
    allowed_providers TEXT[],
    is_active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP,
    expires_at TIMESTAMP
);
```

---

## 5.3 Usage Logs

```sql
CREATE TABLE api_usage (
    id UUID PRIMARY KEY,
    coupon_code TEXT,
    provider TEXT,
    credits_used INTEGER,
    created_at TIMESTAMP
);
```

---

# 6. Coupon System

## Goals

* avoid full account system
* minimize GDPR burden
* support research/demo access

---

## Features

* coupon-based access
* limited credits
* provider restrictions
* expiration dates
* usage tracking
* revocation support

---

## Example Coupon Types

| Coupon    | Purpose             |
| --------- | ------------------- |
| DEMO100   | conference demo     |
| REVIEW500 | reviewer access     |
| CHAI1000  | collaborator access |

---

# 7. Rate Limiting

## Goals

* prevent abuse
* protect API costs
* prevent provider throttling

---

## Recommended Limits

| User Type      | Limit  |
| -------------- | ------ |
| Anonymous      | low    |
| Coupon Users   | medium |
| Internal/Admin | high   |

---

## Recommended Technology

* Redis
* Upstash Redis optional

---

# 8. GDPR Compliance Strategy

## Core Principle

```text
Process minimally
Store minimally
Delete aggressively
```

---

## Recommendations

### Avoid storing:

* uploaded raw text
* personal information
* user accounts if unnecessary

### Prefer:

* temporary processing
* ephemeral uploads
* short retention logs

---

## Important Compliance Steps

### Privacy Policy

Must explain:

* what data is processed
* external providers used
* retention policies
* user rights

---

## External Providers

Document:

* OpenAI
* Anthropic

including:

* potential non-EU processing

---

# 9. Security

## Required Measures

### Secrets

* environment variables only
* never expose API keys to frontend

---

## HTTPS

* mandatory in production

---

## Validation

* file upload limits
* input sanitization
* request size limits

---

## Logging

Avoid logging:

* sensitive text content
* personal information

---

# 10. Model Optimization

## Strong Recommendation

Convert RoBERTa to ONNX.

### Benefits

* lower RAM usage
* faster CPU inference
* lower latency
* improved stability

---

## Recommended Tools

* Hugging Face Optimum
* ONNX Runtime

---

# 11. Development Workflow

## Recommended Workflow

```text
Local Development
    ↓
Local Production Simulation
    ↓
Dockerization
    ↓
Hetzner Deployment
```

---

## Local Development

Run:

* frontend locally
* backend locally
* model locally

before deployment.

---

# 12. Future Extensions

## Possible Features

### Explainability

* token importance visualization
* rationale generation
* attention heatmaps

---

## Research Tooling

* annotation interfaces
* comparison mode
* batch evaluation

---

## Analytics

* provider usage statistics
* coupon usage tracking
* inference benchmarking

---

# 13. Suggested Roadmap

## Phase 1 — MVP

* local RoBERTa inference
* single text UI
* FastAPI backend
* Hetzner deployment

---

## Phase 2 — Infrastructure

* coupon system
* rate limiting
* logging
* HTTPS

---

## Phase 3 — Batch Processing

* CSV upload
* async jobs
* downloadable outputs

---

## Phase 4 — Multi-Provider

* OpenAI integration
* Claude integration
* provider routing abstraction

---

## Phase 5 — Research Features

* explainability
* evaluation dashboards
* annotation tooling

---

# 14. Recommended Tech Stack Summary

| Component     | Recommendation     |
| ------------- | ------------------ |
| Frontend      | Next.js            |
| Backend       | FastAPI            |
| Model Runtime | ONNX Runtime       |
| Hosting       | Hetzner            |
| Reverse Proxy | Nginx              |
| Database      | PostgreSQL         |
| Rate Limiting | Redis              |
| Model         | Fine-tuned RoBERTa |

---

# 15. Final Architecture Recommendation

```text
Frontend (Next.js)
        ↓
Nginx
        ↓
FastAPI Backend
        ↓
Provider Router
 ├── Local RoBERTa (primary)
 ├── OpenAI
 ├── Claude
```

The local RoBERTa backend should remain the primary inference engine for:

* cost stability
* GDPR control
* predictable performance
* scalable research usage.
