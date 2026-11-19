"""
DataSentry Web Dashboard - Embedded Single-Page Application
"""

DASHBOARD_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>DataSentry | Enterprise ML Data-Reliability Engine</title>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
  <style>
    :root {
      --bg: #0b0f19;
      --card-bg: #111827;
      --border: #1f2937;
      --accent: #0f62fe;
      --accent-hover: #0353e9;
      --success: #198038;
      --success-bg: #0e2a18;
      --warning: #f1c21b;
      --danger: #da1e28;
      --text: #f3f4f6;
      --text-muted: #9ca3af;
    }
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body {
      font-family: 'Inter', -apple-system, sans-serif;
      background: var(--bg);
      color: var(--text);
      line-height: 1.5;
      padding: 2rem 1rem;
    }
    .container { max-width: 1100px; margin: 0 auto; }
    header {
      display: flex;
      justify-content: space-between;
      align-items: center;
      padding-bottom: 1.5rem;
      border-bottom: 1px solid var(--border);
      margin-bottom: 2rem;
    }
    .logo-area { display: flex; align-items: center; gap: 0.75rem; }
    .badge-icon {
      width: 36px; height: 36px; background: var(--accent);
      border-radius: 8px; display: flex; align-items: center;
      justify-content: center; font-weight: 700; font-size: 1.1rem;
    }
    .title h1 { font-size: 1.35rem; font-weight: 700; letter-spacing: -0.02em; }
    .title p { font-size: 0.85rem; color: var(--text-muted); }
    .header-links a {
      color: var(--accent);
      text-decoration: none;
      font-size: 0.9rem;
      margin-left: 1.25rem;
      font-weight: 500;
    }
    .header-links a:hover { text-decoration: underline; }
    
    .stats-grid {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
      gap: 1rem;
      margin-bottom: 2rem;
    }
    .stat-card {
      background: var(--card-bg);
      border: 1px solid var(--border);
      border-radius: 8px;
      padding: 1rem 1.25rem;
    }
    .stat-label { font-size: 0.75rem; text-transform: uppercase; color: var(--text-muted); letter-spacing: 0.05em; }
    .stat-value { font-size: 1.25rem; font-weight: 700; margin-top: 0.25rem; }
    .pill {
      display: inline-block; padding: 0.15rem 0.5rem; border-radius: 9999px;
      font-size: 0.75rem; font-weight: 600;
    }
    .pill-green { background: #0e2a18; color: #42be65; border: 1px solid #198038; }
    .pill-blue { background: #0f2744; color: #4589ff; border: 1px solid #0f62fe; }

    .grid-2 { display: grid; grid-template-columns: 1fr 1fr; gap: 1.5rem; margin-bottom: 2rem; }
    @media (max-width: 768px) { .grid-2 { grid-template-columns: 1fr; } }
    
    .card {
      background: var(--card-bg);
      border: 1px solid var(--border);
      border-radius: 8px;
      padding: 1.5rem;
    }
    .card h2 { font-size: 1.1rem; font-weight: 600; margin-bottom: 0.5rem; }
    .card p.desc { font-size: 0.85rem; color: var(--text-muted); margin-bottom: 1.25rem; }
    
    .btn {
      background: var(--accent);
      color: white;
      border: none;
      padding: 0.6rem 1.1rem;
      border-radius: 6px;
      font-size: 0.875rem;
      font-weight: 600;
      cursor: pointer;
      transition: background 0.15s;
    }
    .btn:hover { background: var(--accent-hover); }
    .btn-secondary { background: #374151; }
    .btn-secondary:hover { background: #4b5563; }
    .btn:disabled { opacity: 0.5; cursor: not-allowed; }
    
    .console-out {
      background: #000;
      color: #10b981;
      font-family: 'JetBrains Mono', monospace;
      font-size: 0.8rem;
      padding: 1rem;
      border-radius: 6px;
      margin-top: 1rem;
      max-height: 260px;
      overflow-y: auto;
      white-space: pre-wrap;
      border: 1px solid var(--border);
    }
    .report-box {
      margin-top: 1rem;
      padding: 1rem;
      border-radius: 6px;
      background: #1e293b;
      font-size: 0.85rem;
      border-left: 4px solid var(--accent);
    }
    .report-box h3 { font-size: 0.95rem; margin-bottom: 0.5rem; color: #93c5fd; }
  </style>
</head>
<body>
  <div class="container">
    <header>
      <div class="logo-area">
        <div class="badge-icon">DS</div>
        <div class="title">
          <h1>DataSentry Engine</h1>
          <p>Safety-Gated Data-Reliability Microservice for ML Pipelines</p>
        </div>
      </div>
      <div class="header-links">
        <a href="/docs" target="_blank">Interactive API Docs (Swagger) &rarr;</a>
        <a href="/redoc" target="_blank">ReDoc</a>
        <a href="/health" target="_blank">Health JSON</a>
      </div>
    </header>

    <div class="stats-grid">
      <div class="stat-card">
        <div class="stat-label">Microservice Status</div>
        <div class="stat-value" id="status-val"><span class="pill pill-green">RUNNING (200 OK)</span></div>
      </div>
      <div class="stat-card">
        <div class="stat-label">Safety Mode</div>
        <div class="stat-value"><span class="pill pill-blue">DRY-RUN GATED</span></div>
      </div>
      <div class="stat-card">
        <div class="stat-label">Triad Axes Active</div>
        <div class="stat-value">Quality &bull; Anomaly &bull; Drift</div>
      </div>
      <div class="stat-card">
        <div class="stat-label">Granite Verifier</div>
        <div class="stat-value"><span class="pill pill-green">0.0% Hallucination Gate</span></div>
      </div>
    </div>

    <div class="grid-2">
      <!-- Card 1: Batch Reliability Scan -->
      <div class="card">
        <h2>1. Tabular Reliability Scan</h2>
        <p class="desc">Ingest current telemetry records, run Great Expectations quality checks, PyTorch Autoencoder anomaly inference, and KS/PSI drift detection.</p>
        <button class="btn" id="btn-scan" onclick="runScan()">Scan Sample Batch</button>
        <div id="scan-out" class="console-out">// Ready to scan. Click above to test sample sensor telemetry.</div>
      </div>

      <!-- Card 2: watsonx.ai Granite Diagnosis -->
      <div class="card">
        <h2>2. Root-Cause Diagnosis & Audit</h2>
        <p class="desc">Synthesizes the reliability report using IBM watsonx.ai Granite and runs the AST verifier requiring cited finding IDs and numeric tolerances.</p>
        <button class="btn btn-secondary" id="btn-explain" onclick="runExplain()" disabled>Generate Diagnosis</button>
        <div id="explain-out" class="console-out">// Requires completing Step 1 scan first.</div>
      </div>
    </div>

    <!-- Card 3: Autonomous Remediation Agent -->
    <div class="card" style="margin-bottom: 2rem;">
      <h2>3. Safety-Gated Remediation Agent</h2>
      <p class="desc">Invokes the autonomous remediation agent under strict role-based access control (RBAC), loop breaker constraints, and human-in-the-loop gates.</p>
      <div style="display: flex; gap: 0.5rem; align-items: center; margin-bottom: 1rem;">
        <select id="agent-role" style="background:#1f2937; color:#fff; padding:0.5rem; border-radius:6px; border:1px solid #374151;">
          <option value="remediation_agent">Role: remediation_agent (Default)</option>
          <option value="pipeline_operator">Role: pipeline_operator</option>
          <option value="security_auditor">Role: security_auditor (Read-Only)</option>
        </select>
        <button class="btn" id="btn-remediate" onclick="runRemediate()" disabled>Execute Remediation</button>
      </div>
      <div id="remediate-out" class="console-out">// Requires completing Step 1 scan first.</div>
    </div>

    <div class="report-box">
      <h3>About this Dashboard & Healthcheck</h3>
      <p style="color: #cbd5e1; font-size: 0.85rem;">
        &bull; <strong>Why this server exists:</strong> DataSentry runs as a high-performance REST microservice in production data platforms (e.g. Kubernetes, Airflow, Kubeflow).<br>
        &bull; <strong>What the <code>/health</code> endpoint is for:</strong> Production orchestrators (load balancers, Kubernetes probes, Airflow sensors) ping <code>/health</code> automatically to verify uptime, active security policies, and rate limits before routing mission-critical batches.<br>
        &bull; <strong>Interactive API Playground:</strong> Visit <a href="/docs" style="color:#60a5fa;">/docs</a> to test every endpoint with custom JSON payloads.
      </p>
    </div>
  </div>

  <script>
    let latestReport = null;

    const sampleTelemetry = [
      {"temperature": 68.4, "vibration_x": 0.052, "vibration_y": 0.051, "rotational_speed": 1745.0, "oil_pressure": 44.8, "status": "NORMAL"},
      {"temperature": 69.1, "vibration_x": 0.054, "vibration_y": 0.049, "rotational_speed": 1752.0, "oil_pressure": 45.2, "status": "NORMAL"},
      {"temperature": 94.2, "vibration_x": 0.098, "vibration_y": 0.112, "rotational_speed": 1890.0, "oil_pressure": 31.0, "status": "WARNING"},
      {"temperature": 67.8, "vibration_x": 0.048, "vibration_y": 0.050, "rotational_speed": 1748.0, "oil_pressure": 44.5, "status": "NORMAL"},
      {"temperature": 66.5, "vibration_x": 0.050, "vibration_y": 0.052, "rotational_speed": 1740.0, "oil_pressure": 46.1, "status": "NORMAL"}
    ];

    const sampleReference = [
      {"temperature": 65.0, "vibration_x": 0.050, "vibration_y": 0.050, "rotational_speed": 1750.0, "oil_pressure": 45.0, "status": "NORMAL"},
      {"temperature": 65.2, "vibration_x": 0.049, "vibration_y": 0.051, "rotational_speed": 1751.0, "oil_pressure": 44.9, "status": "NORMAL"},
      {"temperature": 64.8, "vibration_x": 0.051, "vibration_y": 0.050, "rotational_speed": 1749.0, "oil_pressure": 45.1, "status": "NORMAL"}
    ];

    async function runScan() {
      const out = document.getElementById("scan-out");
      out.textContent = "[*] Dispatching POST /scan with 5 telemetry records...";
      try {
        const res = await fetch("/scan", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            dataset_name: "iot_sensor_feed",
            current_records: sampleTelemetry,
            reference_records: sampleReference
          })
        });
        const report = await res.json();
        latestReport = report;
        out.textContent = "[+] Scan Complete! Status: " + report.overall_status.toUpperCase() + "\\n" +
          "    - Dataset: " + report.dataset_name + "\\n" +
          "    - Quality Issues: " + (report.quality ? report.quality.failed_validations : 0) + "\\n" +
          "    - Anomaly Outliers: " + (report.anomaly ? report.anomaly.anomalies_detected : 0) + "\\n" +
          "    - Drifted Features: " + (report.drift ? report.drift.drifted_features.length : 0) + "\\n\\n" +
          JSON.stringify(report, null, 2);

        document.getElementById("btn-explain").disabled = false;
        document.getElementById("btn-remediate").disabled = false;
      } catch (err) {
        out.textContent = "[-] Error calling /scan: " + err;
      }
    }

    async function runExplain() {
      if (!latestReport) return;
      const out = document.getElementById("explain-out");
      out.textContent = "[*] Dispatching POST /explain with report and verifier enabled...";
      try {
        const res = await fetch("/explain", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            report: latestReport,
            verify_faithfulness: true
          })
        });
        const data = await res.json();
        out.textContent = "[+] Granite Diagnosis Received!\\n" +
          "    - Faithfulness Score: " + (data.faithfulness_score * 100).toFixed(1) + "%\\n" +
          "    - Hallucination Detected: " + data.has_hallucinations + "\\n" +
          "    - Citation Precision: " + (data.citation_precision * 100).toFixed(1) + "%\\n\\n" +
          data.explanation.explanation_markdown;
      } catch (err) {
        out.textContent = "[-] Error calling /explain: " + err;
      }
    }

    async function runRemediate() {
      if (!latestReport) return;
      const out = document.getElementById("remediate-out");
      const role = document.getElementById("agent-role").value;
      out.textContent = "[*] Dispatching POST /remediate with role: " + role + "...";
      try {
        const res = await fetch("/remediate", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            report: latestReport,
            role: role,
            human_approved: false,
            dry_run: true
          })
        });
        const data = await res.json();
        out.textContent = "[+] Remediation Agent Decision: " + data.action.toUpperCase() + "\\n" +
          "    - Tool Called: " + data.tool_called + "\\n" +
          "    - Guardrail Allowed: " + data.allowed + "\\n" +
          "    - Executed: " + data.executed + "\\n" +
          "    - Risk Tier: " + data.risk_tier.toUpperCase() + "\\n" +
          "    - Policy Reason: " + data.reason;
      } catch (err) {
        out.textContent = "[-] Error calling /remediate: " + err;
      }
    }
  </script>
</body>
</html>
"""
