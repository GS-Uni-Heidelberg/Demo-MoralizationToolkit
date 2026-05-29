"use client";

import { useEffect, useMemo, useState } from "react";

import styles from "./page.module.css";

type PredictionResponse = {
  label: string;
  confidence: number;
};

type BatchStatus = "idle" | "uploading" | "error" | "done";
type ViewMode = "single" | "batch";

const MAX_PREVIEW_ROWS = 5;
const MAX_PREVIEW_COLS = 4;

const DEFAULT_TEXT =
  "Wer so handelt, liegt moralisch daneben, und die Gesellschaft darf das nicht akzeptieren.";

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

export default function Home() {
  const [text, setText] = useState(DEFAULT_TEXT);
  const [viewMode, setViewMode] = useState<ViewMode>("single");
  const [status, setStatus] = useState<"idle" | "loading" | "error">("idle");
  const [errorMessage, setErrorMessage] = useState<string | null>(null);
  const [result, setResult] = useState<PredictionResponse | null>(null);
  const [batchFile, setBatchFile] = useState<File | null>(null);
  const [batchStatus, setBatchStatus] = useState<BatchStatus>("idle");
  const [batchError, setBatchError] = useState<string | null>(null);
  const [batchDownloadUrl, setBatchDownloadUrl] = useState<string | null>(null);
  const [batchOutputFormat, setBatchOutputFormat] = useState<"csv" | "json">(
    "csv"
  );
  const [batchFilename, setBatchFilename] = useState<string | null>(null);
  const [inputPreview, setInputPreview] = useState<string[][]>([]);
  const [outputPreview, setOutputPreview] = useState<string[][]>([]);
  const [outputPreviewJson, setOutputPreviewJson] = useState<string | null>(
    null
  );
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
    if (!batchFile) return;

    setBatchStatus("uploading");
    setBatchError(null);
    setOutputPreviewJson(null);
    if (batchDownloadUrl) {
      URL.revokeObjectURL(batchDownloadUrl);
      setBatchDownloadUrl(null);
    }

    try {
      const formData = new FormData();
      formData.append("file", batchFile);
      formData.append("output_format", batchOutputFormat);

      const response = await fetch("http://localhost:8000/batch", {
        method: "POST",
        body: formData,
      });

      if (!response.ok) {
        const message = await response.text();
        throw new Error(message || "Batch request failed");
      }

      const blob = await response.blob();
      const downloadUrl = URL.createObjectURL(blob);
      const baseName = batchFile.name.replace(/\.(csv|json)$/i, "") || "predictions";
      const extension = batchOutputFormat === "json" ? "json" : "csv";

      if (batchOutputFormat === "csv") {
        const textContent = await blob.text();
        const preview = parseCsvPreview(textContent);
        setOutputPreview(preview);
        setOutputPreviewNote(
          preview.length ? null : "No rows detected in the CSV output."
        );
        setOutputPreviewJson(null);
      } else {
        const textContent = await blob.text();
        const prettyJson = JSON.stringify(JSON.parse(textContent), null, 2);
        setOutputPreview([]);
        setOutputPreviewJson(prettyJson);
        setOutputPreviewNote(null);
      }

      setBatchFilename(`${baseName}.${extension}`);
      setBatchDownloadUrl(downloadUrl);
      setBatchStatus("done");
    } catch (error) {
      setBatchStatus("error");
      setBatchError(
        error instanceof Error ? error.message : "Unexpected error"
      );
    }
  };

  return (
    <div className={styles.page}>
      <main className={styles.main}>
        <section className={styles.hero}>
          <p className={styles.eyebrow}>Moralization Detection Demo</p>
          <h1>Check moral framing in a single sentence.</h1>
          <p className={styles.subtitle}>
            CPU-only RoBERTa inference. No accounts, no storage, instant feedback.
          </p>
        </section>

        <section className={styles.modeSwitch}>
          <div className={styles.modeTabs}>
            <button
              className={`${styles.modeTab} ${
                viewMode === "single" ? styles.modeTabActive : ""
              }`}
              type="button"
              onClick={() => setViewMode("single")}
            >
              Single text
            </button>
            <button
              className={`${styles.modeTab} ${
                viewMode === "batch" ? styles.modeTabActive : ""
              }`}
              type="button"
              onClick={() => setViewMode("batch")}
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

            <section className={styles.resultCard}>
              <div>
                <p className={styles.resultLabel}>Prediction</p>
                <p className={styles.resultValue}>
                  {result ? result.label.replace("_", " ") : "No result yet"}
                </p>
              </div>
              <div>
                <p className={styles.resultLabel}>Confidence</p>
                <p className={styles.resultValue}>{confidenceLabel}</p>
              </div>
            </section>

            {status === "error" && (
              <p className={styles.errorMessage}>{errorMessage}</p>
            )}
          </>
        ) : (
          <section className={styles.batchPanel}>
            <div className={styles.batchHeader}
            >
              <div>
                <p className={styles.resultLabel}>Batch processing</p>
                <p className={styles.batchTitle}>Upload CSV or JSON</p>
              </div>
              <span className={styles.hint}>Column or field name: text</span>
            </div>

            <form className={styles.batchForm} onSubmit={handleBatchSubmit}>
              <input
                className={styles.fileInput}
                type="file"
                accept=".csv,.json,application/json,text/csv"
                onChange={(event) =>
                  handleBatchFileChange(event.target.files?.[0] ?? null)
                }
              />

              <div className={styles.formatInfo}>
                <p className={styles.formatTitle}>Formatting</p>
                <div className={styles.formatList}>
                  <p>CSV: header named text or first column is text.</p>
                  <p>JSON: array of objects with only the text field.</p>
                  <p>UTF-8 recommended, max 5000 chars per row.</p>
                </div>
                <div className={styles.formatSamples}>
                  <div>
                    <p className={styles.sampleLabel}>CSV</p>
                    <pre className={styles.formatSample}>{`text
Das ist absolut richtig.
So etwas darf niemand tolerieren.`}</pre>
                  </div>
                  <div>
                    <p className={styles.sampleLabel}>JSON</p>
                    <pre className={styles.formatSample}>{`[
  {"text": "Das ist absolut richtig."},
  {"text": "So etwas darf niemand tolerieren."}
]`}</pre>
                  </div>
                </div>
              </div>

              <div className={styles.batchActions}>
                <label className={styles.selectLabel} htmlFor="outputFormat">
                  Output format
                </label>
                <select
                  id="outputFormat"
                  className={styles.select}
                  value={batchOutputFormat}
                  onChange={(event) =>
                    setBatchOutputFormat(
                      event.target.value === "json" ? "json" : "csv"
                    )
                  }
                >
                  <option value="csv">CSV</option>
                  <option value="json">JSON</option>
                </select>

                <button
                  className={styles.secondaryButton}
                  type="submit"
                  disabled={!batchFile || batchStatus === "uploading"}
                >
                  {batchStatus === "uploading" ? "Processing..." : "Run batch"}
                </button>

                {batchDownloadUrl && batchFilename && (
                  <a
                    className={styles.downloadLink}
                    href={batchDownloadUrl}
                    download={batchFilename}
                  >
                    Download results
                  </a>
                )}
              </div>
            </form>

            {batchStatus === "error" && (
              <p className={styles.errorMessage}>{batchError}</p>
            )}

            <div className={styles.previewGrid}>
              <div className={styles.previewCard}>
                <p className={styles.previewTitle}>Input preview (CSV)</p>
                {inputPreview.length > 0 ? (
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
                ) : (
                  <p className={styles.previewEmpty}>{inputPreviewNote}</p>
                )}
              </div>
              <div className={styles.previewCard}>
                <p className={styles.previewTitle}>
                  Output preview ({batchOutputFormat.toUpperCase()})
                </p>
                {batchOutputFormat === "csv" ? (
                  outputPreview.length > 0 ? (
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
                  ) : (
                    <p className={styles.previewEmpty}>{outputPreviewNote}</p>
                  )
                ) : outputPreviewJson ? (
                  <pre className={styles.previewJson}>{outputPreviewJson}</pre>
                ) : (
                  <p className={styles.previewEmpty}>{outputPreviewNote}</p>
                )}
              </div>
            </div>
          </section>
        )}
      </main>
    </div>
  );
}
