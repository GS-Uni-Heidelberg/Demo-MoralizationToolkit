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
const MAX_TEXT_LENGTH = 5000;
const MAX_BATCH_INSTANCES = 500;

const DEFAULT_TEXT =
  "Wer so handelt, liegt moralisch daneben, und die Gesellschaft darf das nicht akzeptieren.";

const INITIAL_TEXT = DEFAULT_TEXT;

const parseCsvText = (text: string): string[][] => {
  const rows: string[][] = [];
  let curCell = "";
  let curRow: string[] = [];
  let inQuotes = false;

  for (let i = 0; i < text.length; i += 1) {
    const ch = text[i];

    if (ch === '"') {
      if (inQuotes && text[i + 1] === '"') {
        curCell += '"';
        i += 1;
        continue;
      }
      inQuotes = !inQuotes;
      continue;
    }

    if (ch === ',' && !inQuotes) {
      curRow.push(curCell.trim());
      curCell = "";
      continue;
    }

    if ((ch === '\n' || ch === '\r') && !inQuotes) {
      if (ch === '\r' && text[i + 1] === '\n') {
        i += 1;
      }
      curRow.push(curCell.trim());
      rows.push(curRow);
      curRow = [];
      curCell = "";
      continue;
    }

    curCell += ch;
  }

  // Push any remaining data as the last row
  if (inQuotes) {
    // If quotes were not closed, still push what we have to avoid breaking preview
  }
  if (curCell.length > 0 || curRow.length > 0) {
    curRow.push(curCell.trim());
    rows.push(curRow);
  }

  // Filter out completely empty rows
  return rows.filter((r) => r.some((c) => c.length > 0));
};

const parseCsvAll = (text: string): string[][] => {
  return parseCsvText(text);
};

const formatJsonPreview = (text: string): string => {
  const parsed = JSON.parse(text) as unknown;
  return JSON.stringify(parsed, null, 2);
};

export default function Home() {
  const currentYear = new Date().getFullYear();
  const [text, setText] = useState(DEFAULT_TEXT);
  const [viewMode, setViewMode] = useState<ViewMode>("single");
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
  const [inputPreviewJson, setInputPreviewJson] = useState<string | null>(null);
  const [outputPreview, setOutputPreview] = useState<string[][]>([]);
  const [outputPreviewJson, setOutputPreviewJson] = useState<string | null>(
    null
  );
  const [batchMetrics, setBatchMetrics] = useState<BatchMetrics | null>(null);
  const [inputPreviewNote, setInputPreviewNote] = useState<string | null>(
    "Upload a CSV or JSON file to see a preview."
  );
  const [outputPreviewNote, setOutputPreviewNote] = useState<string | null>(
    "Run a batch request to see the output preview."
  );
  const [batchInputLimitError, setBatchInputLimitError] = useState<string | null>(
    null
  );

  const clearBatchRunState = () => {
    setBatchStatus("idle");
    setBatchError(null);
    setBatchJobId(null);
    setBatchProgress(0);
    setBatchProcessed(0);
    setBatchTotal(0);
    if (batchDownloadUrl) {
      URL.revokeObjectURL(batchDownloadUrl);
      setBatchDownloadUrl(null);
    }
    setBatchFilename(null);
    setBatchMetricsFilename(null);
    setOutputPreview([]);
    setOutputPreviewJson(null);
    setBatchMetrics(null);
    setOutputPreviewNote("Run a batch request to see the output preview.");
  };

  const confirmBatchReset = (message: string) => {
    if (typeof window === "undefined") return true;
    return window.confirm(message);
  };

  const isDisabled = status === "loading" || text.trim().length === 0;
  const isTextTooLong = text.length > MAX_TEXT_LENGTH;
  const isSingleAnalyzeDisabled = isDisabled || isTextTooLong;
  const confidenceLabel = useMemo(() => {
    if (!result) return "--";
    return `${(result.confidence * 100).toFixed(2)}%`;
  }, [result]);
  const hasUnsavedData =
    batchFile !== null;

  useEffect(() => {
    if (typeof window === "undefined") return;
    const stored = sessionStorage.getItem("viewMode") as ViewMode | null;
    if (stored === "single" || stored === "batch") {
      setViewMode(stored);
    }
  }, []);

  useEffect(() => {
    return () => {
      if (batchDownloadUrl) {
        URL.revokeObjectURL(batchDownloadUrl);
      }
    };
  }, [batchDownloadUrl]);

  const resetAppState = () => {
      const ok = window.confirm(
          "Reset the app?\nThis will clear your current progress and cannot be undone!"
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

      setInputPreviewNote("Upload a CSV or JSON file to see a preview.");
      setOutputPreviewNote("Run a batch request to see the output preview.");
    };
  
  
  useEffect(() => {
  const handler = (event: BeforeUnloadEvent) => {
      if (!hasUnsavedData) return;

      event.preventDefault();
      event.returnValue = "";
    };

    window.addEventListener("beforeunload", handler);
    return () => window.removeEventListener("beforeunload", handler);
  }, [hasUnsavedData]);
    
  const handleBatchFileChange = async (file: File | null) => {
    const hasDownstreamState =
      batchOutputFormat !== null ||
      batchStatus !== "idle" ||
      batchDownloadUrl !== null ||
      outputPreview.length > 0 ||
      outputPreviewJson !== null ||
      batchMetrics !== null;

    const isFileActuallyChanging =
      (batchFile === null && file !== null) ||
      (batchFile !== null && file === null) ||
      (batchFile !== null &&
        file !== null &&
        (batchFile.name !== file.name ||
          batchFile.size !== file.size ||
          batchFile.lastModified !== file.lastModified));

    if (isFileActuallyChanging && hasDownstreamState) {
      const ok = confirmBatchReset(
        "Changing the uploaded file will reset all later batch steps. Continue?"
      );
      if (!ok) return;
    }

    if (isFileActuallyChanging) {
      clearBatchRunState();
      setBatchOutputFormat(null);
    }

    setBatchFile(file);
    setBatchInputLimitError(null);
    setInputPreview([]);
    setInputPreviewJson(null);
    setInputPreviewNote("Upload a CSV or JSON file to see a preview.");

    if (!file) return;

    const isCsv = file.name.toLowerCase().endsWith(".csv");
    const isJson = file.name.toLowerCase().endsWith(".json");
    if (!isCsv && !isJson) {
      setInputPreviewNote("Input preview is available for CSV or JSON uploads only.");
      return;
    }

    try {
      const textContent = await file.text();
      if (isJson) {
        const parsedJson = JSON.parse(textContent) as unknown;
        const jsonInstanceCount = Array.isArray(parsedJson)
          ? parsedJson.length
          : parsedJson && typeof parsedJson === "object"
            ? 1
            : 0;

        if (jsonInstanceCount > MAX_BATCH_INSTANCES) {
          setBatchFile(null);
          setInputPreview([]);
          setInputPreviewJson(null);
          setBatchOutputFormat(null);
          setBatchInputLimitError(
            `Too many instances: ${jsonInstanceCount}. Maximum allowed is ${MAX_BATCH_INSTANCES}.`
          );
          setInputPreviewNote("Upload a CSV or JSON file to see a preview.");
          return;
        }

        const prettyJson = formatJsonPreview(textContent);
        setInputPreview([]);
        setInputPreviewJson(prettyJson);
        setInputPreviewNote(`Instances detected: ${jsonInstanceCount}`);
        return;
      }
      const allRows = parseCsvAll(textContent);
      if (allRows.length === 0) {
        setInputPreviewNote("No rows detected in the CSV file.");
        return;
      }
      const firstRow = allRows[0].map((cell) => cell.toLowerCase().trim());
      const hasHeader = firstRow.includes("text");
      const csvInstanceCount = hasHeader ? Math.max(allRows.length - 1, 0) : allRows.length;

      if (csvInstanceCount > MAX_BATCH_INSTANCES) {
        setBatchFile(null);
        setInputPreview([]);
        setInputPreviewJson(null);
        setBatchOutputFormat(null);
        setBatchInputLimitError(
          `Too many instances: ${csvInstanceCount}. Maximum allowed is ${MAX_BATCH_INSTANCES}.`
        );
        setInputPreviewNote("Upload a CSV or JSON file to see a preview.");
        return;
      }

      const preview = allRows
        .slice(0, MAX_PREVIEW_ROWS)
        .map((row) => row.slice(0, MAX_PREVIEW_COLS));
      setInputPreview(preview);
      setInputPreviewNote(`Instances detected: ${csvInstanceCount}`);
    } catch {
      setBatchInputLimitError(null);
      setInputPreviewNote("Could not read CSV preview.");
    }
  };

  const handleBatchOutputFormatChange = (value: string) => {
    const nextFormat =
      value === "json" ? "json" : value === "csv" ? "csv" : null;

    if (nextFormat === batchOutputFormat) return;

    const hasDownstreamState =
      batchStatus !== "idle" ||
      batchDownloadUrl !== null ||
      outputPreview.length > 0 ||
      outputPreviewJson !== null ||
      batchMetrics !== null;

    if (hasDownstreamState) {
      const ok = confirmBatchReset(
        "Changing the output format will reset the current batch results. Continue?"
      );
      if (!ok) return;
    }

    clearBatchRunState();
    setBatchOutputFormat(nextFormat);
  };

  const handleSubmit = async (event: React.FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    const trimmed = text.trim();
    if (!trimmed) return;
    if (trimmed.length > MAX_TEXT_LENGTH) {
      setStatus("error");
      setErrorMessage(
        `Input is too long (${trimmed.length} chars). Maximum allowed is ${MAX_TEXT_LENGTH}.`
      );
      return;
    }

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
            const prettyJson = formatJsonPreview(textContent);
            setOutputPreview([]);
            setOutputPreviewJson(prettyJson);
            setOutputPreviewNote(null);

            const blob = new Blob([prettyJson], {
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

  const truncateCell = (value: string | undefined | null, limit = 120) => {
    if (!value) return "";
    // collapse whitespace and newlines for preview
    const collapsed = value.replace(/\s+/g, " ").trim();
    if (collapsed.length <= limit) return collapsed;
    return collapsed.slice(0, limit) + "…";
  };

  return (
    <div className={styles.page}>
      <main className={styles.main}>
        <section className={styles.hero}>
          <p className={styles.eyebrow}>Moralization Detection Toolkit</p>
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
        </section>
        <section>
          <p className={styles.modeHint}>
            {viewMode === "single"
              ? "Analyze one sentence at a time."
              : "Upload CSV or JSON and download results."}
          </p>
        </section>

        {viewMode === "single" ? (
          <>
            <section className={styles.batchPanel}>
              <p className={styles.boxTitle}>Moralization Detection with Dictionary Approach (DiMi)</p>
              <p>TODO</p>

            </section>

            <section className={styles.batchPanel}>
              <p className={styles.boxTitle}>Moralization Detection with Language Models</p>
              <section className={styles.panel}>
                <form className={styles.form} onSubmit={handleSubmit}>
                  <label className={styles.label} htmlFor="textInputSecondary">
                    Text input
                  </label>
                  <textarea
                    id="textInputSecondary"
                    className={styles.textarea}
                    value={text}
                    onChange={(event) => setText(event.target.value)}
                    rows={7}
                    maxLength={MAX_TEXT_LENGTH}
                    placeholder="Paste text to analyze for moralization..."
                  />

                  <div className={styles.actions}>
                    <button
                      className={styles.primaryButton}
                      type="submit"
                      disabled={isSingleAnalyzeDisabled}
                    >
                      {status === "loading" ? "Analyzing..." : "Analyze"}
                    </button>
                    <span className={styles.hint}>
                      Local model: XLM-RoBERTa · {text.length}/{MAX_TEXT_LENGTH}
                    </span>
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
            </section>
          </>
        ) : (
          <section className={styles.batchPanel}>

            <form className={styles.batchForm} onSubmit={handleBatchSubmit}>
              <p className={styles.boxTitle}>Pipeline Moralization Detection (DiMi + Language Models)</p>
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
                {batchInputLimitError && (
                  <p className={styles.errorMessage}>{batchInputLimitError}</p>
                )}
              </div>

              {batchFile && (
                <div className={`${styles.previewCard} ${styles.fadeInSection}`}>
                  <p className={styles.previewTitle}>Input preview</p>
                  {inputPreviewJson ? (
                    <pre className={`${styles.previewJson} ${styles.previewJsonLight}`}>
                      {inputPreviewJson}
                    </pre>
                  ) : inputPreview.length > 0 ? (
                    <div className={styles.previewScroll}>
                      <table className={`${styles.previewTable} ${styles.previewJsonLight}`}>
                        <tbody>
                            {inputPreview.map((row, i) => (
                            <tr key={i}>
                              {row.map((cell, j) => (
                                <td key={j}>
                                  <div className={styles.cellTruncate} title={String(cell)}>
                                    {truncateCell(String(cell))}
                                  </div>
                                </td>
                              ))}
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </div>
                  ) : (
                    <p className={styles.previewEmpty}>
                      {inputPreviewNote || "No rows detected in the CSV file."}
                    </p>
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
                    onChange={(event) => handleBatchOutputFormatChange(event.target.value)}
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
                        !batchFile ||
                        !batchOutputFormat ||
                        batchStatus === "uploading" ||
                        !!batchInputLimitError
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
                    Output preview ({(batchOutputFormat ?? "csv").toUpperCase()})
                  </p>
                  {batchOutputFormat === "csv" ? (
                    outputPreview.length > 0 ? (
                      <div className={styles.previewScroll}>
                        <table className={styles.previewTable}>
                          <tbody>
                            {outputPreview.map((row, i) => (
                              <tr key={i}>
                                {row.map((cell, j) => (
                                  <td key={j}>
                                    <div className={styles.cellTruncate} title={String(cell)}>
                                      {truncateCell(String(cell))}
                                    </div>
                                  </td>
                                ))}
                              </tr>
                            ))}
                          </tbody>
                        </table>
                      </div>
                    ) : (
                      <p className={styles.previewEmpty}>
                        {outputPreviewNote || "No rows detected in the CSV file."}
                      </p>
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
          <a className={styles.footerLink} href="https://www.uni-heidelberg.de/en/imprint">
            Imprint / Impressum
          </a>
          <span className={styles.footerDivider}> | </span>
          <a className={styles.footerLink} href="https://www.uni-heidelberg.de/en/privacy-statement">
            Privacy Statement / Datenschutzerklärung
          </a>
        </div>
      </footer>
      </main>
    </div>
  );
}
