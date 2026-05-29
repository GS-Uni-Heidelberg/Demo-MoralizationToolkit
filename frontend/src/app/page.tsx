"use client";

import { useMemo, useState } from "react";

import styles from "./page.module.css";

type PredictionResponse = {
  label: string;
  confidence: number;
};

const DEFAULT_TEXT =
  "People who do this are simply wrong, and society should not tolerate it.";

export default function Home() {
  const [text, setText] = useState(DEFAULT_TEXT);
  const [status, setStatus] = useState<"idle" | "loading" | "error">("idle");
  const [errorMessage, setErrorMessage] = useState<string | null>(null);
  const [result, setResult] = useState<PredictionResponse | null>(null);

  const isDisabled = status === "loading" || text.trim().length === 0;
  const confidenceLabel = useMemo(() => {
    if (!result) return "--";
    return `${(result.confidence * 100).toFixed(1)}%`;
  }, [result]);

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
      </main>
    </div>
  );
}
