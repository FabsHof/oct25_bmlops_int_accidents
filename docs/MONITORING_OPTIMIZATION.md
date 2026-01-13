# Prometheus, Grafana, and Alertmanager Optimization

## Summary of Changes

This document outlines the fixes and optimizations made to the monitoring stack to ensure dashboards show data and alerts fire correctly.

## Issues Fixed

### 1. Prometheus Configuration
**Problem:** Alert rules file (`alert_rules.yml`) was not mounted in the Prometheus container.
**Fix:** Added volume mount in `docker-compose.yml`:
```yaml
- ./configs/prometheus/alert_rules.yml:/etc/prometheus/alert_rules.yml:ro
```

### 2. Alert Rule Queries
**Problem:** Histogram-based alert queries were incorrect and would never fire.

**Fixes:**
- **HighPredictionLatency**: Changed from bucket comparison to proper histogram quantile:
  ```promql
  # Old (incorrect):
  prediction_latency_seconds_bucket{le="1.0"} < 0.5 * prediction_latency_seconds_count
  
  # New (correct):
  histogram_quantile(0.95, rate(prediction_latency_seconds_bucket[5m])) > 1.0
  ```

- **HighPredictionErrorRate**: Added proper aggregation with `sum()` by error_type:
  ```promql
  # Old:
  rate(prediction_errors_total[5m]) / (rate(predictions_total[5m]) + 0.001) > 0.05
  
  # New:
  sum(rate(prediction_errors_total[5m])) by (error_type) / (sum(rate(predictions_total[5m])) + 0.001) > 0.05
  ```

### 3. Alertmanager Environment Variables
**Problem:** Alertmanager configuration used environment variable placeholders (`${VAR}`), but the container doesn't substitute them by default.

**Fix:** Created custom entrypoint script (`scripts/alertmanager-entrypoint.sh`) that:
- Installs `envsubst` if not available
- Substitutes environment variables in the config
- Starts Alertmanager with the processed config

Updated `docker-compose.yml` to use the custom entrypoint:
```yaml
entrypoint: ["/scripts/alertmanager-entrypoint.sh"]
volumes:
  - ./scripts/alertmanager-entrypoint.sh:/scripts/alertmanager-entrypoint.sh:ro
```

### 4. Grafana Dashboard Improvements
**Problem:** Some panels had incorrect or missing data.

**Fixes:**
- Verified histogram quantile queries use `rate()` function
- Added two new panels to ML Monitoring dashboard:
  - **Prediction Errors by Type**: Shows error rates over time by error type
  - **API Requests by Endpoint**: Shows request rates by endpoint and status

## Metrics Exposed by FastAPI

The API ([src/api/main.py](../../src/api/main.py)) exposes these Prometheus metrics at `/metrics`:

| Metric | Type | Labels | Description |
|--------|------|--------|-------------|
| `predictions_total` | Counter | - | Total number of predictions served |
| `prediction_latency_seconds` | Histogram | - | Prediction latency distribution |
| `predictions_by_severity` | Counter | `severity` | Predictions grouped by severity level |
| `prediction_errors_total` | Counter | `error_type` | Total prediction errors by type |
| `api_requests_total` | Counter | `endpoint`, `method`, `status` | API requests by endpoint and status |
| `data_drift_score` | Gauge | - | Overall data drift score (0-1) |
| `feature_drift` | Gauge | `feature` | Per-feature drift scores |

## Alert Rules Configured

The following alerts are configured in [configs/prometheus/alert_rules.yml](../../configs/prometheus/alert_rules.yml):

### Critical Alerts
1. **APIServiceDown**: API is unreachable for 2+ minutes
2. **MLflowServerDown**: MLflow is unreachable for 2+ minutes
3. **CriticalDataDrift**: Drift score > 0.5 for 5+ minutes

### Warning Alerts
1. **HighPredictionLatency**: P95 latency > 1s for 5+ minutes
2. **HighPredictionErrorRate**: Error rate > 5% for 5+ minutes
3. **NoPredictionsServed**: No predictions in last 30 minutes
4. **HighDataDrift**: Drift score > 0.3 for 5+ minutes
5. **HighFeatureDrift**: Feature drift > 0.4 for 10+ minutes
6. **ModelPerformanceDegraded**: Too many severity=4 predictions

### Info Alerts
1. **ModelRetrainingNeeded**: Drift > 0.2 with 100+ predictions in 24h

## Grafana Dashboards

### 1. ML Model Monitoring (`ml-model-monitoring`)
**Panels:**
- Total Predictions (stat)
- Average Prediction Latency (stat)
- Prediction Request Rate (time series)
- Predictions by Severity (pie chart)
- Data Drift Score (stat with thresholds)
- Feature Drift Over Time (time series)
- **NEW:** Prediction Errors by Type (time series)
- **NEW:** API Requests by Endpoint (time series)

### 2. Alerts Dashboard (`mlops_alerts`)
**Panels:**
- Active Alerts (alert list)
- API Health Summary (stat)
- Data Drift Score (gauge with thresholds)
- Prediction Latency P95 (stat)
- Recent Alert Summary (table)

### 3. System Monitoring (`system-monitoring`)
**Panels:**
- CPU Usage (from node-exporter)
- Memory Usage
- Disk Usage
- Network I/O
- Container metrics

## Testing the Setup

### 1. Restart Monitoring Services
```bash
make down
make up
```

### 2. Verify Prometheus Targets
1. Open http://localhost:9090
2. Go to **Status → Targets**
3. Verify all targets are **UP**:
   - `prometheus` (self-monitoring)
   - `fastapi` (API at port 8000)
   - `mlflow` (may be down if not exposing metrics)
   - `node-exporter` (system metrics)

### 3. Verify Alert Rules
1. In Prometheus UI, go to **Status → Rules**
2. Verify all alert rules are loaded from `alert_rules.yml`
3. Check which alerts are currently firing (if any)

### 4. View Grafana Dashboards
1. Open http://localhost:3000
2. Login with credentials from `.env`:
   - Username: `${GRAFANA_ADMIN_USER}`
   - Password: `${GRAFANA_ADMIN_PASSWORD}`
3. Navigate to **Dashboards** and open:
   - ML Model Monitoring
   - MLOps Alerts & Incidents
   - System Monitoring
4. Verify all panels show data (not "No data")

### 5. Test Alertmanager
1. Open http://localhost:9093
2. Check if any alerts are active
3. Verify alert routing configuration
4. To test email alerts (optional):
   - Configure SMTP settings in `.env`
   - Trigger an alert (e.g., stop the API container)
   - Check email inbox after 2+ minutes

### 6. Trigger Test Alerts

**Stop API to trigger APIServiceDown:**
```bash
docker stop api_container
```
Wait 2 minutes, then check Prometheus Alerts and Alertmanager.

**Restart API:**
```bash
docker start api_container
```

**Submit high drift score to trigger HighDataDrift:**
```bash
# Get access token first
TOKEN=$(curl -X POST "http://localhost:8000/token" \
  -H "Content-Type: application/x-www-form-urlencoded" \
  -d "username=admin&password=admin" | jq -r '.access_token')

# Submit drift metrics
curl -X POST "http://localhost:8000/metrics/drift" \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "overall_drift_score": 0.35,
    "feature_drift_scores": {
      "year": 0.05,
      "month": 0.45,
      "hour": 0.30
    },
    "timestamp": "2026-01-13T10:00:00Z"
  }'
```

## Environment Variables Required

Ensure these are set in your `.env` file:

```bash
# Prometheus
PROMETHEUS_PORT=9090

# Grafana
GRAFANA_PORT=3000
GRAFANA_ADMIN_USER=admin
GRAFANA_ADMIN_PASSWORD=admin

# Alertmanager
ALERTMANAGER_PORT=9093

# SMTP Configuration (for email alerts)
ALERTMANAGER_SMTP_HOST=smtp.gmail.com
ALERTMANAGER_SMTP_PORT=587
ALERTMANAGER_SMTP_USER=your-email@gmail.com
ALERTMANAGER_SMTP_PASSWORD=your-app-password

# Email Recipients
ALERTMANAGER_EMAIL_FROM=mlops-alerts@your-domain.com
ALERTMANAGER_EMAIL_TO=team@your-domain.com
ALERTMANAGER_EMAIL_CRITICAL=oncall@your-domain.com
ALERTMANAGER_EMAIL_WARNING=team@your-domain.com
ALERTMANAGER_EMAIL_DATA_TEAM=data-team@your-domain.com
ALERTMANAGER_EMAIL_ML_TEAM=ml-team@your-domain.com
ALERTMANAGER_EMAIL_OPS_TEAM=ops-team@your-domain.com
```

## Common Issues & Solutions

### Panels Show "No Data"
**Cause:** Metrics not being scraped or counter hasn't incremented yet.
**Solution:**
1. Make a prediction request to generate metrics
2. Check Prometheus targets are UP
3. Verify metric names in Prometheus → Graph → Metrics explorer

### Alerts Not Firing
**Cause:** Alert rules not loaded or conditions not met.
**Solution:**
1. Check Prometheus → Status → Rules
2. Verify alert_rules.yml is mounted in container
3. Check alert conditions are being met in Prometheus → Graph

### Alertmanager Not Sending Emails
**Cause:** SMTP not configured or environment variables not substituted.
**Solution:**
1. Verify `.env` has correct SMTP settings
2. Check Alertmanager logs: `docker logs alertmanager_container`
3. Test SMTP credentials externally

### Dashboard Queries Return Empty
**Cause:** Incorrect PromQL query syntax.
**Solution:**
1. Test query in Prometheus → Graph first
2. Use `rate()` for counters: `rate(metric_total[5m])`
3. Use `histogram_quantile()` with `rate()` for histograms

## Maintenance

### Adding New Alerts
1. Edit [configs/prometheus/alert_rules.yml](../../configs/prometheus/alert_rules.yml)
2. Add new alert rule under appropriate group
3. Restart Prometheus: `docker restart prometheus_container`

### Adding Dashboard Panels
1. Edit JSON files in [configs/grafana/provisioning/dashboards/json/](../../configs/grafana/provisioning/dashboards/json/)
2. Or create in Grafana UI, then export JSON
3. Restart Grafana: `docker restart grafana_container`

### Modifying Email Templates
Edit files in [configs/alertmanager/templates/](../../configs/alertmanager/templates/):
- `email.default.tmpl`
- `email.critical.tmpl`
- `email.warning.tmpl`

## Architecture Diagram

```
┌─────────────┐
│   FastAPI   │ exposes /metrics
│   (port     │────────────────┐
│    8000)    │                │
└─────────────┘                │
                               │ scrapes every 15s
┌─────────────┐                │
│ Node        │                │
│ Exporter    │────────────────┤
│ (port 9100) │                │
└─────────────┘                │
                               ▼
                        ┌──────────────┐
                        │  Prometheus  │
                        │  (port 9090) │
                        └──────┬───────┘
                               │ evaluates rules
                               │ every 15s
                               ▼
                        ┌──────────────┐
                        │ Alertmanager │
                        │  (port 9093) │
                        └──────┬───────┘
                               │ sends alerts
                               ▼
                        ┌──────────────┐
                        │  Email/Slack │
                        └──────────────┘

┌─────────────┐
│   Grafana   │ queries Prometheus
│  (port      │────────────────────▶ Dashboards
│   3000)     │
└─────────────┘
```

## References

- [Prometheus Querying Basics](https://prometheus.io/docs/prometheus/latest/querying/basics/)
- [Alerting Rules](https://prometheus.io/docs/prometheus/latest/configuration/alerting_rules/)
- [Grafana Provisioning](https://grafana.com/docs/grafana/latest/administration/provisioning/)
- [Alertmanager Configuration](https://prometheus.io/docs/alerting/latest/configuration/)
