"use client";

import { useEffect, useMemo, useState } from "react";

import styles from "./page.module.css";

type PredictionResponse = {
  label: string;
  confidence: number;
};

type BatchStatus = "idle" | "uploading" | "processing" | "error" | "done";
type ViewMode = "single" | "batch";


type BatchMetrics = {
  accuracy: string;
  precision: string;
  recall: string;
  f1: string;
  tp: string;
  fp: string;
  tn: string;
  fn: string;
};

type BatchStatusResponse = {
  job_id: string;
  status: "queued" | "running" | "completed" | "failed";
  processed: number;
  total: number;
  progress: number;
  metrics?: BatchMetrics;
  error?: string;
};

const MAX_PREVIEW_ROWS = 5;
const MAX_PREVIEW_COLS = 5;

const DEFAULT_TEXT =
  "Wer so handelt, liegt moralisch daneben, und die Gesellschaft darf das nicht akzeptieren.";

const INITIAL_TEXT = DEFAULT_TEXT;

const parseCsvLine = (line: string): string[] => {
  const cells: string[] = [];
  let current = "";
  let inQuotes = false;

  for (let i = 0; i < line.length; i += 1) {
    const char = line[i];

    if (char === '"') {
      if (inQuotes && line[i + 1] === '"') {
        current += '"';
        i += 1;
      } else {
        inQuotes = !inQuotes;
      }
      continue;
    }

    if (char === "," && !inQuotes) {
      cells.push(current.trim());
      current = "";
      continue;
    }

    current += char;
  }

  cells.push(current.trim());
  return cells;
};

const parseCsvPreview = (text: string): string[][] => {
  const rows = text.split(/\r?\n/).filter((line) => line.trim().length > 0);
  return rows.slice(0, MAX_PREVIEW_ROWS).map((row) => {
    const cells = parseCsvLine(row);
    return cells.slice(0, MAX_PREVIEW_COLS);
  });
};

const parseCsvAll = (text: string): string[][] => {
  const rows = text.split(/\r?\n/).filter((line) => line.trim().length > 0);
  return rows.map((row) => parseCsvLine(row));
};

export default function Home() {
  const currentYear = new Date().getFullYear();
  const [text, setText] = useState(DEFAULT_TEXT);
  const [viewMode, setViewMode] = useState<ViewMode>(() => {
    if (typeof window === "undefined") return "single";
    return (sessionStorage.getItem("viewMode") as ViewMode) || "single";
  });
  const [status, setStatus] = useState<"idle" | "loading" | "error">("idle");
  const [errorMessage, setErrorMessage] = useState<string | null>(null);
  const [result, setResult] = useState<PredictionResponse | null>(null);
  const [batchFile, setBatchFile] = useState<File | null>(null);
  const [batchStatus, setBatchStatus] = useState<BatchStatus>("idle");
  const [batchError, setBatchError] = useState<string | null>(null);
  const [batchDownloadUrl, setBatchDownloadUrl] = useState<string | null>(null);
  const [batchOutputFormat, setBatchOutputFormat] = useState<
    "csv" | "json" | null
  >(null);
  const [batchFilename, setBatchFilename] = useState<string | null>(null);
  const [batchMetricsFilename, setBatchMetricsFilename] = useState<
    string | null
  >(null);
  const [batchJobId, setBatchJobId] = useState<string | null>(null);
  const [batchProgress, setBatchProgress] = useState<number>(0);
  const [batchProcessed, setBatchProcessed] = useState<number>(0);
  const [batchTotal, setBatchTotal] = useState<number>(0);
  const [inputPreview, setInputPreview] = useState<string[][]>([]);
  const [outputPreview, setOutputPreview] = useState<string[][]>([]);
  const [outputPreviewJson, setOutputPreviewJson] = useState<string | null>(
    null
  );
  const [batchMetrics, setBatchMetrics] = useState<BatchMetrics | null>(null);
  const [inputPreviewNote, setInputPreviewNote] = useState<string | null>(
    "Upload a CSV file to see a preview."
  );
  const [outputPreviewNote, setOutputPreviewNote] = useState<string | null>(
    "Run a batch request to see the output preview."
  );

  const isDisabled = status === "loading" || text.trim().length === 0;
  const confidenceLabel = useMemo(() => {
    if (!result) return "--";
    return `${(result.confidence * 100).toFixed(2)}%`;
  }, [result]);

  useEffect(() => {
    return () => {
      if (batchDownloadUrl) {
        URL.revokeObjectURL(batchDownloadUrl);
      }
    };
  }, [batchDownloadUrl]);

  const resetAppState = () => {
      const ok = window.confirm(
          "Reset the app?\n\nThis will clear:\n- input text\n- batch file\n- results\n- metrics\n- progress"
        );

        if (!ok) return;
      
      sessionStorage.setItem("viewMode", "single");

      setText(INITIAL_TEXT);
      setStatus("idle");
      setErrorMessage(null);
      setResult(null);

      setBatchFile(null);
      setBatchStatus("idle");
      setBatchError(null);
      setBatchDownloadUrl(null);
      setBatchOutputFormat(null);
      setBatchFilename(null);
      setBatchMetricsFilename(null);
      setBatchJobId(null);
      setBatchProgress(0);
      setBatchProcessed(0);
      setBatchTotal(0);

      setInputPreview([]);
      setOutputPreview([]);
      setOutputPreviewJson(null);
      setBatchMetrics(null);

      setInputPreviewNote("Upload a CSV file to see a preview.");
      setOutputPreviewNote("Run a batch request to see the output preview.");
    };

  const handleBatchFileChange = async (file: File | null) => {
    setBatchFile(file);
    setInputPreview([]);
    setInputPreviewNote("Upload a CSV file to see a preview.");

    if (!file) return;

    const isCsv = file.name.toLowerCase().endsWith(".csv");
    if (!isCsv) {
      setInputPreviewNote("Input preview is available for CSV uploads only.");
      return;
    }

    try {
      const textContent = await file.text();
      const preview = parseCsvPreview(textContent);
      if (preview.length === 0) {
        setInputPreviewNote("No rows detected in the CSV file.");
        return;
      }
      setInputPreview(preview);
      setInputPreviewNote(null);
    } catch {
      setInputPreviewNote("Could not read CSV preview.");
    }
  };

  const handleSubmit = async (event: React.FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    const trimmed = text.trim();
    if (!trimmed) return;

    setStatus("loading");
    setErrorMessage(null);
    setResult(null);

    try {
      const response = await fetch("http://localhost:8000/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text: trimmed }),
      });

      if (!response.ok) {
        const message = await response.text();
        throw new Error(message || "Request failed");
      }

      const data = (await response.json()) as PredictionResponse;
      setResult(data);
      setStatus("idle");
    } catch (error) {
      setStatus("error");
      setErrorMessage(
        error instanceof Error ? error.message : "Unexpected error"
      );
    }
  };

  const handleBatchSubmit = async (
    event: React.FormEvent<HTMLFormElement>
  ) => {
    event.preventDefault();
    if (!batchFile || !batchOutputFormat) return;

    setBatchStatus("uploading");
    setBatchError(null);
    setOutputPreviewJson(null);
    setBatchMetrics(null);
    setBatchProgress(0);
    setBatchProcessed(0);
    setBatchTotal(0);
    if (batchDownloadUrl) {
      URL.revokeObjectURL(batchDownloadUrl);
      setBatchDownloadUrl(null);
    }
    setBatchMetricsFilename(null);

    try {
      const formData = new FormData();
      formData.append("file", batchFile);
      formData.append("output_format", batchOutputFormat);

      const response = await fetch("http://localhost:8000/batch/start", {
        method: "POST",
        body: formData,
      });

      if (!response.ok) {
        const message = await response.text();
        throw new Error(message || "Batch request failed");
      }

      const startPayload = (await response.json()) as { job_id: string };
      setBatchJobId(startPayload.job_id);
      setBatchStatus("processing");

      const pollStatus = async () => {
        const statusResponse = await fetch(
          `http://localhost:8000/batch/status/${startPayload.job_id}`
        );
        if (!statusResponse.ok) {
          const message = await statusResponse.text();
          throw new Error(message || "Status request failed");
        }

        const statusPayload = (await statusResponse.json()) as BatchStatusResponse;
        setBatchProgress(statusPayload.progress);
        setBatchProcessed(statusPayload.processed);
        setBatchTotal(statusPayload.total);
        if (statusPayload.metrics) {
          setBatchMetrics(statusPayload.metrics);
        }

        if (statusPayload.status === "completed") {
          const resultResponse = await fetch(
            `http://localhost:8000/batch/result/${startPayload.job_id}`
          );
          if (!resultResponse.ok) {
            const message = await resultResponse.text();
            throw new Error(message || "Result request failed");
          }

          const baseName =
            batchFile.name.replace(/\.(csv|json)$/i, "") || "predictions";
          const extension = batchOutputFormat === "json" ? "json" : "csv";
          if (statusPayload.metrics) {
            const metricsExtension =
              batchOutputFormat === "json" ? "json" : "csv";
            setBatchMetricsFilename(`${baseName}-metrics.${metricsExtension}`);
          }

          if (batchOutputFormat === "json") {
            const textContent = await resultResponse.text();
            const prettyJson = JSON.stringify(
              JSON.parse(textContent),
              null,
              2
            );
            setOutputPreview([]);
            setOutputPreviewJson(prettyJson);
            setOutputPreviewNote(null);

            const blob = new Blob([textContent], {
              type: "application/json",
            });
            const downloadUrl = URL.createObjectURL(blob);
            setBatchFilename(`${baseName}-results.${extension}`);
            setBatchDownloadUrl(downloadUrl);
            setBatchStatus("done");
            return true;
          }

          const csvText = await resultResponse.text();
          const preview = parseCsvAll(csvText);
          setOutputPreview(preview);
          setOutputPreviewNote(
            preview.length ? null : "No rows detected in the CSV output."
          );
          setOutputPreviewJson(null);

          const blob = new Blob([csvText], { type: "text/csv" });
          const downloadUrl = URL.createObjectURL(blob);
          setBatchFilename(`${baseName}-results.${extension}`);
          setBatchDownloadUrl(downloadUrl);
          setBatchStatus("done");
          return true;
        }

        if (statusPayload.status === "failed") {
          throw new Error(statusPayload.error || "Batch failed");
        }

        return false;
      };

      const pollLoop = async () => {
        let completed = false;
        while (!completed) {
          completed = await pollStatus();
          if (!completed) {
            await new Promise((resolve) => setTimeout(resolve, 600));
          }
        }
      };

      await pollLoop();
    } catch (error) {
      setBatchStatus("error");
      setBatchError(
        error instanceof Error ? error.message : "Unexpected error"
      );
    }
  };

  const handleResultsDownload = () => {
    if (!batchDownloadUrl || !batchFilename) return;
    const link = document.createElement("a");
    link.href = batchDownloadUrl;
    link.download = batchFilename;
    document.body.appendChild(link);
    link.click();
    link.remove();
  };

  const handleMetricsDownload = () => {
    if (!batchMetrics || !batchOutputFormat) return;
    const header = "accuracy,precision,recall,f1,tp,fp,tn,fn";
    const row =
      `${batchMetrics.accuracy},${batchMetrics.precision},` +
      `${batchMetrics.recall},${batchMetrics.f1},${batchMetrics.tp},` +
      `${batchMetrics.fp},${batchMetrics.tn},${batchMetrics.fn}`;
    const metricsText =
      batchOutputFormat === "json"
        ? JSON.stringify(batchMetrics, null, 2)
        : `${header}\n${row}`;
    const metricsBlob = new Blob([metricsText], {
      type: batchOutputFormat === "json" ? "application/json" : "text/csv",
    });
    const metricsUrl = URL.createObjectURL(metricsBlob);
    const link = document.createElement("a");
    link.href = metricsUrl;
    link.download =
      batchMetricsFilename ??
      `metrics.${batchOutputFormat === "json" ? "json" : "csv"}`;
    document.body.appendChild(link);
    link.click();
    link.remove();
    URL.revokeObjectURL(metricsUrl);
  };

  return (
    <div className={styles.page}>
      <main className={styles.main}>
        <section className={styles.hero}>
          <p className={styles.eyebrow}>Moralization Detection</p>
          <h1>Analyze moral framing in German texts.</h1>
          <p className={styles.subtitle}>
            CPU-only RoBERTa inference with single text and batch processing.
          </p>
        </section>

        <section className={styles.modeSwitch}>
          <div className={styles.modeTabs}>
            <button
              className={`${styles.modeTab} ${
                viewMode === "single" ? styles.modeTabActive : ""
              }`}
              type="button"
              onClick={() => {
                setViewMode("single");
                sessionStorage.setItem("viewMode", "single");
              }}
            >
              Single text
            </button>

            <button
              className={`${styles.modeTab} ${
                viewMode === "batch" ? styles.modeTabActive : ""
              }`}
              type="button"
              onClick={() => {
                setViewMode("batch");
                sessionStorage.setItem("viewMode", "batch");
              }}
            >
              Batch processing
            </button>
          </div>

          <p className={styles.modeHint}>
            {viewMode === "single"
              ? "Analyze one sentence at a time."
              : "Upload CSV or JSON and download results."}
          </p>
        </section>

        {viewMode === "single" ? (
          <>
            <section className={styles.panel}>
              <form className={styles.form} onSubmit={handleSubmit}>
                <label className={styles.label} htmlFor="textInput">
                  Text input
                </label>
                <textarea
                  id="textInput"
                  className={styles.textarea}
                  value={text}
                  onChange={(event) => setText(event.target.value)}
                  rows={7}
                  placeholder="Paste text to analyze for moralization..."
                />

                <div className={styles.actions}>
                  <button
                    className={styles.primaryButton}
                    type="submit"
                    disabled={isDisabled}
                  >
                    {status === "loading" ? "Analyzing..." : "Analyze"}
                  </button>
                  <span className={styles.hint}>Local model: XLM-RoBERTa</span>
                </div>
              </form>
            </section>

            {result && (
              <section className={styles.resultCard}>
                <div>
                  <p className={styles.resultLabel}>Prediction</p>
                  <p className={styles.resultValue}>
                    {result.label.replace("_", " ")}
                  </p>
                </div>
                <div>
                  <p className={styles.resultLabel}>Confidence</p>
                  <p className={styles.resultValue}>{confidenceLabel}</p>
                </div>
              </section>
            )}

            {status === "error" && (
              <p className={styles.errorMessage}>{errorMessage}</p>
            )}
          </>
        ) : (
          <section className={styles.batchPanel}>

            <form className={styles.batchForm} onSubmit={handleBatchSubmit}>
              <div className={styles.formatInfo}>
                <p className={styles.formatTitle}>Formatting</p>
                <div className={styles.formatList}>
                  <p><b>Optional columns:</b> id and label (moralization/no_moralization, true/false, 0/1).</p>
                  <p>If <b>no ids</b> are provided, ids are auto-generated.</p>
                  <p>If <b>no labels</b> are provided, metrics are not calculated.</p>
                </div>
                <div className={styles.formatSamples}>
                  <div>
                    <p className={styles.sampleLabel}>CSV</p>
                    <pre className={styles.formatSample}>{`id,text,label
1,Das ist absolut richtig.,moralization
2,So etwas darf niemand tolerieren.,1`}</pre>
                  </div>
                  <div>
                    <p className={styles.sampleLabel}>JSON</p>
                    <pre className={styles.formatSample}>{`[
  {"id": 1, "text": "Das ist absolut richtig.", "label": "moralization"},
  {"id": 2, "text": "So etwas darf niemand tolerieren.", "label": 1}
]`}</pre>
                  </div>
                </div>
              </div>

              <div className={styles.sectionBox}>
                <p className={styles.previewTitle}>Upload input file ...</p>
                <input
                  className={styles.fileInput}
                  type="file"
                  accept=".csv,.json,application/json,text/csv"
                  onChange={(event) =>
                    handleBatchFileChange(event.target.files?.[0] ?? null)
                  }
                />
              </div>

              {batchFile && (
                <div className={`${styles.previewCard} ${styles.fadeInSection}`}>
                  <p className={styles.previewTitle}>Input preview</p>
                  {inputPreview.length > 0 ? (
                    <div className={styles.previewScroll}>
                      <table className={styles.previewTable}>
                        <tbody>
                          {inputPreview.map((row, rowIndex) => (
                            <tr key={`input-row-${rowIndex}`}>
                              {row.map((cell, cellIndex) => (
                                <td key={`input-cell-${rowIndex}-${cellIndex}`}>
                                  {cell || "--"}
                                </td>
                              ))}
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </div>
                  ) : (
                    <p className={styles.previewEmpty}>{inputPreviewNote}</p>
                  )}
                </div>
              )}

              {batchFile && (
                <div className={`${styles.sectionBox} ${styles.fadeInSection}`}>
                  <p className={styles.previewTitle}>Select output file format ...</p>
                  <select
                    id="outputFormat"
                    className={`${styles.select} ${styles.primarySelect}`}
                    value={batchOutputFormat ?? ""}
                    onChange={(event) =>
                      setBatchOutputFormat(
                        event.target.value === "json"
                          ? "json"
                          : event.target.value === "csv"
                            ? "csv"
                            : null
                      )
                    }
                    required
                  >
                    <option value="" disabled>
                      Select format
                    </option>
                    <option value="csv">CSV</option>
                    <option value="json">JSON</option>
                  </select>
                </div>
              )}

              {batchFile && batchOutputFormat && (
                <div className={`${styles.batchActions} ${styles.sectionBox} ${styles.fadeInSection}`}>
                  <div className={styles.progressWrap}>
                    <button
                      className={styles.primaryButton}
                      type="submit"
                      disabled={
                        !batchFile || !batchOutputFormat || batchStatus === "uploading"
                      }
                    >
                      {batchStatus === "uploading" || batchStatus === "processing"
                        ? "Processing..."
                        : "Run batch"}
                    </button>
                    {(batchStatus === "uploading" || batchStatus === "processing") && (
                      <div className={styles.progressBar}>
                        <span className={styles.progressFill} />
                      </div>
                    )}
                    {batchStatus === "processing" && (
                      <p className={styles.progressText}>
                        {batchProgress}% ({batchProcessed}/{batchTotal})
                      </p>
                    )}
                  </div>
                </div>
              )}
            </form>

            {batchStatus === "error" && (
              <p className={styles.errorMessage}>{batchError}</p>
            )}

            <div className={styles.previewGrid}>
              {(batchStatus === "done" || outputPreviewJson || outputPreview.length > 0) && (
                <div className={`${styles.previewCard} ${styles.previewDark} ${styles.outputCard}`}>
                  <p className={styles.previewTitle}>
                    Output preview ({batchOutputFormat.toUpperCase()})
                  </p>
                  {batchOutputFormat === "csv" ? (
                    outputPreview.length > 0 ? (
                      <div className={styles.previewScroll}>
                        <table className={styles.previewTable}>
                          <tbody>
                            {outputPreview.map((row, rowIndex) => (
                              <tr key={`output-row-${rowIndex}`}>
                                {row.map((cell, cellIndex) => (
                                  <td key={`output-cell-${rowIndex}-${cellIndex}`}>
                                    {cell || "--"}
                                  </td>
                                ))}
                              </tr>
                            ))}
                          </tbody>
                        </table>
                      </div>
                    ) : (
                      <p className={styles.previewEmpty}>{outputPreviewNote}</p>
                    )
                  ) : outputPreviewJson ? (
                    <pre className={styles.previewJson}>{outputPreviewJson}</pre>
                  ) : (
                    <p className={styles.previewEmpty}>{outputPreviewNote}</p>
                  )}
                </div>
              )}
              {batchMetrics && (
                <div className={`${styles.resultCard} ${styles.metricsCard}`}>
                  <span className={styles.metricsTitle}>Metrics</span>
                  <div>
                    <p className={styles.resultLabel}>Accuracy</p>
                    <p className={styles.resultValue}>
                      {batchMetrics.accuracy}
                    </p>
                  </div>
                  <div>
                    <p className={styles.resultLabel}>Precision</p>
                    <p className={styles.resultValue}>
                      {batchMetrics.precision}
                    </p>
                  </div>
                  <div>
                    <p className={styles.resultLabel}>Recall</p>
                    <p className={styles.resultValue}>{batchMetrics.recall}</p>
                  </div>
                  <div>
                    <p className={styles.resultLabel}>F1</p>
                    <p className={styles.resultValue}>{batchMetrics.f1}</p>
                  </div>
                  <div>
                    <p className={styles.resultLabel}>TP</p>
                    <p className={styles.resultValue}>{batchMetrics.tp}</p>
                  </div>
                  <div>
                    <p className={styles.resultLabel}>FP</p>
                    <p className={styles.resultValue}>{batchMetrics.fp}</p>
                  </div>
                  <div>
                    <p className={styles.resultLabel}>TN</p>
                    <p className={styles.resultValue}>{batchMetrics.tn}</p>
                  </div>
                  <div>
                    <p className={styles.resultLabel}>FN</p>
                    <p className={styles.resultValue}>{batchMetrics.fn}</p>
                  </div>
                </div>
              )}
            </div>

            {(batchDownloadUrl || batchMetrics) && (
              <div className={`${styles.downloadSection} ${styles.fadeInSection}`}>
                <div className={styles.downloadLinks}>
                  {batchDownloadUrl && batchFilename && (
                    <button
                      className={`${styles.primaryButton} ${styles.downloadButton}`}
                      type="button"
                      onClick={handleResultsDownload}
                    >
                      Download results
                    </button>
                  )}
                  {batchMetrics && (
                    <button
                      className={`${styles.primaryButton} ${styles.downloadButton}`}
                      type="button"
                      onClick={handleMetricsDownload}
                    >
                      Download metrics
                    </button>
                  )}
                </div>
              </div>
            )}
            {(batchDownloadUrl || batchMetrics) && (
              <div className={`${styles.downloadSection} ${styles.fadeInSection}`}>
                <div className={styles.downloadLinks}>
                  {batchDownloadUrl && batchFilename && (
                    <button
                      className={`${styles.dangerButton} ${styles.downloadButton}`}
                      type="button"
                      onClick={resetAppState}
                    >
                      Restart
                    </button>
                  )}
                </div>
              </div>
            )}
          </section>
        )}
      <footer className={styles.footer}>
        <div className={styles.footerRow}>
          <span>
            © {currentYear} CHAI Lab - Department of German Language and Literature, University Heidelberg. All rights reserved.
          </span>
        </div>

        <div className={styles.footerRow}>
          <a className={styles.footerLink} href="/impressum">
            Legal Notice / Impressum
          </a>
          <span className={styles.footerDivider}> | </span>
          <a className={styles.footerLink} href="/datenschutz">
            Privacy Policy / Datenschutz
          </a>
        </div>
      </footer>
      </main>
    </div>
  );
}
