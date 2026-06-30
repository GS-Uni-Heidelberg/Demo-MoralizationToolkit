"use client";

import { useEffect, useMemo, useRef, useState } from "react";

import styles from "./page.module.css";
import FloatingKeyButton from "../../components/FloatingKeyButton";

type PredictionResponse = {
  label: string;
  confidence: number;
  explanation?: string | null;
};

type BatchStatus = "idle" | "uploading" | "processing" | "error" | "done";
type ViewMode = "single" | "batch";

type LanguageCode = "de" | "en" | "fr" | "it";
type ModelCode = "xlm-roberta" | "claude" | "openai";

type LemmaMatchResult = {
  sentence_index: number;
  matched_lemmas: string[];
  context_sentences: string[];
  context_html: string[];
  center_sentence: string;
};

type LemmatizerResponse = {
  matches: LemmaMatchResult[];
  total_sentences: number;
};

type DimiPreviewRow = {
  id: string;
  full_text: string;
  text: string;
  dimi_matches: number;
  dimi_matched_lemmas: string[];
};

type BatchDimiStatus = "idle" | "processing" | "error" | "done";


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
  metrics?: Record<string, BatchMetrics>;
  error?: string;
};

const MAX_PREVIEW_ROWS = 5;
const MAX_PREVIEW_COLS = 5;
const MAX_TEXT_LENGTH = 5000;
const MAX_BATCH_INSTANCES = 200000;

const DEFAULT_TEXT =
  "Aber den weiteren Ausgleich, den es dort gibt, den Ausgleich zwischen Arm und Reich, halten wir in der Gesundheitsversicherung für wenig treffsicher und deswegen für sozial ungerecht.";

const INITIAL_TEXT = DEFAULT_TEXT;

const LANGUAGE_OPTIONS: Array<{
  code: LanguageCode;
  label: string;
  name: string;
}> = [
  { code: "de", label: "DE", name: "Deutsch" },
  { code: "en", label: "EN", name: "English" },
  { code: "fr", label: "FR", name: "Français" },
  { code: "it", label: "IT", name: "Italiano" },
];

const MODEL_OPTIONS: Array<{
  code: ModelCode;
  label: string;
  name: string;
}> = [
  { code: "xlm-roberta", label: "XLM-RoBERTa", name: "Fine-Tuned XLM-RoBERTa Model" },
  { code: "claude", label: "Claude Haiku 4.5", name: "Claude Haiku 4.5" },
  { code: "openai", label: "OpenAI GPT-5 mini", name: "OpenAI GPT-5 mini" },
];

const BATCH_MODEL_SUFFIXES: Record<ModelCode, string> = {
  "xlm-roberta": "roberta-finetuned",
  claude: "claude",
  openai: "openai",
};

const BATCH_MODEL_COLORS: Record<ModelCode, string> = {
  "xlm-roberta": "#f2c479",
  claude: "#6caac9",
  openai: "#c86a5a",
};


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

    if (ch === "," && !inQuotes) {
      curRow.push(curCell.trim());
      curCell = "";
      continue;
    }

    if ((ch === "\n" || ch === "\r") && !inQuotes) {
      if (ch === "\r" && text[i + 1] === "\n") {
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

  if (curCell.length > 0 || curRow.length > 0) {
    curRow.push(curCell.trim());
    rows.push(curRow);
  }

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
  const [dimiText, setDimiText] = useState(DEFAULT_TEXT);
  const [dimiLanguage, setDimiLanguage] = useState<LanguageCode>("de");
  const [dimiResult, setDimiResult] = useState<LemmatizerResponse | null>(null);
  const [dimiStatus, setDimiStatus] = useState<"idle" | "loading" | "error">("idle");
  const [dimiError, setDimiError] = useState<string | null>(null);
  const [dimiCopied, setDimiCopied] = useState<Record<number, boolean>>({});
  const [lemmaCount, setLemmaCount] = useState<number | null>(null);

  // Fetch lemma count whenever the selected language changes
  useEffect(() => {
    setLemmaCount(null);
    fetch(`http://localhost:8000/lemmas/${dimiLanguage}`)
      .then((res) => {
        if (!res.ok) throw new Error("Not found");
        return res.json() as Promise<{ language: string; lemmas: string[] }>;
      })
      .then((data) => setLemmaCount(data.lemmas.length))
      .catch(() => setLemmaCount(null));
  }, [dimiLanguage]);

  // Single-text moralization state
  const [text, setText] = useState(DEFAULT_TEXT);
  const [viewMode, setViewMode] = useState<ViewMode>("single");
  const [status, setStatus] = useState<"idle" | "loading" | "error">("idle");
  const [errorMessage, setErrorMessage] = useState<string | null>(null);
  const [result, setResult] = useState<PredictionResponse | null>(null);
  const [resultModel, setResultModel] = useState<ModelCode | null>(null);
  const [resultLanguage, setResultLanguage] = useState<LanguageCode | null>(null);
  const [selectedModel, setSelectedModel] = useState<ModelCode>("xlm-roberta");
  const [lmLanguage, setLmLanguage] = useState<LanguageCode>("de");

  // Batch state
  const [batchFile, setBatchFile] = useState<File | null>(null);
  const [batchStatus, setBatchStatus] = useState<BatchStatus>("idle");
  const [batchError, setBatchError] = useState<string | null>(null);
  const [batchDownloadUrl, setBatchDownloadUrl] = useState<string | null>(null);
  const [batchOutputFormat, setBatchOutputFormat] = useState<"csv" | "json" | null>(null);
  const [batchLanguage, setBatchLanguage] = useState<LanguageCode | null>(null);
  const [batchModels, setBatchModels] = useState<ModelCode[]>(["xlm-roberta"]);
  const [batchDimiStatus, setBatchDimiStatus] = useState<BatchDimiStatus>("idle");
  const [batchDimiError, setBatchDimiError] = useState<string | null>(null);
  const [batchDimiProgress, setBatchDimiProgress] = useState<number>(0);
  const [batchDimiProcessed, setBatchDimiProcessed] = useState<number>(0);
  const [batchDimiTotal, setBatchDimiTotal] = useState<number>(0);
  const [batchDimiStartedAt, setBatchDimiStartedAt] = useState<number | null>(null);
  const [batchDimiPreparedFile, setBatchDimiPreparedFile] = useState<File | null>(null);
  const [batchDimiSkipped, setBatchDimiSkipped] = useState<boolean>(false);
  const [dimiPreviewRows, setDimiPreviewRows] = useState<DimiPreviewRow[]>([]);
  const [dimiPreviewJson, setDimiPreviewJson] = useState<string | null>(null);
  const [dimiPreviewNote, setDimiPreviewNote] = useState<string | null>(
    "Run DiMi preprocessing to see the output preview."
  );
  const [batchFilename, setBatchFilename] = useState<string | null>(null);
  const [batchMetricsFilename, setBatchMetricsFilename] = useState<string | null>(null);
  const [batchProgress, setBatchProgress] = useState<number>(0);
  const [batchProcessed, setBatchProcessed] = useState<number>(0);
  const [batchTotal, setBatchTotal] = useState<number>(0);
  const [batchStartedAt, setBatchStartedAt] = useState<number | null>(null);
  const [skipNoDimiMatches, setSkipNoDimiMatches] = useState<boolean | null>(null);
  const [inputPreview, setInputPreview] = useState<string[][]>([]);
  const [inputPreviewJson, setInputPreviewJson] = useState<string | null>(null);
  const [outputPreview, setOutputPreview] = useState<string[][]>([]);
  const [outputPreviewJson, setOutputPreviewJson] = useState<string | null>(null);
  const [batchMetrics, setBatchMetrics] = useState<Record<string, BatchMetrics> | null>(null);
  const [inputPreviewNote, setInputPreviewNote] = useState<string | null>(
    "Upload a CSV or JSON file to see a preview."
  );
  const [outputPreviewNote, setOutputPreviewNote] = useState<string | null>(
    "Run a batch request to see the output preview."
  );
  const [batchInputLimitError, setBatchInputLimitError] = useState<string | null>(null);
  const [clockTick, setClockTick] = useState<number>(0);
  const [metricsHoveredModel, setMetricsHoveredModel] = useState<ModelCode | null>(null);
  const batchDimiRunIdRef = useRef(0);

  const clearBatchRunState = () => {
    setBatchStatus("idle");
    setBatchError(null);
    setBatchProgress(0);
    setBatchProcessed(0);
    setBatchTotal(0);
    setBatchStartedAt(null);
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

  const clearBatchDimiState = () => {
    batchDimiRunIdRef.current += 1;
    setBatchDimiStatus("idle");
    setBatchDimiError(null);
    setBatchDimiProgress(0);
    setBatchDimiProcessed(0);
    setBatchDimiTotal(0);
    setBatchDimiStartedAt(null);
    setBatchDimiPreparedFile(null);
    setBatchDimiSkipped(false);
    setSkipNoDimiMatches(null);
    setDimiPreviewRows([]);
    setDimiPreviewJson(null);
    setDimiPreviewNote("Run DiMi preprocessing to see the output preview.");
  };

  const confirmBatchReset = (message: string) => {
    if (typeof window === "undefined") return true;
    return window.confirm(message);
  };

  const formatDuration = (seconds: number) => {
    const safeSeconds = Math.max(0, Math.round(seconds));
    const hours = Math.floor(safeSeconds / 3600);
    const minutes = Math.floor((safeSeconds % 3600) / 60);
    const remainingSeconds = safeSeconds % 60;

    if (hours > 0) {
      return `${hours}h ${minutes}m ${remainingSeconds}s`;
    }

    if (minutes === 0) {
      return `${remainingSeconds}s`;
    }

    return `${minutes}m ${String(remainingSeconds).padStart(2, "0")}s`;
  };

  const estimateRemainingTime = (startedAt: number | null, progress: number) => {
    if (!startedAt || progress <= 0 || progress >= 100) return null;

    const elapsedSeconds = (Date.now() - startedAt) / 1000;
    const estimatedTotalSeconds = (elapsedSeconds * 100) / progress;
    return formatDuration(estimatedTotalSeconds - elapsedSeconds);
  };

  const hasPreparedDimiState = () => {
    return batchDimiPreparedFile !== null || batchDimiStatus === "done";
  };

  const handleSkipNoDimiMatchesChange = (nextChecked: boolean) => {
    if (nextChecked === skipNoDimiMatches) return;

    if (hasBatchResultState()) {
      const ok = confirmBatchReset(
        "Changing the skip-instances setting will reset the current batch results. Continue?"
      );
      if (!ok) return;
      clearBatchRunState();
    }

    setSkipNoDimiMatches(nextChecked);
  };

  const hasBatchResultState = () => {
    return (
      batchStatus !== "idle" ||
      batchDownloadUrl !== null ||
      outputPreview.length > 0 ||
      outputPreviewJson !== null ||
      batchMetrics !== null
    );
};

  const hasDimiResultState = () => {
    return (
      batchDimiStatus === "done" ||
      dimiStatus !== "idle" ||
      dimiResult !== null
    );
  };

  const handleBatchModelToggle = (modelCode: ModelCode) => {
    const nextModels = batchModels.includes(modelCode)
      ? batchModels.filter((code) => code !== modelCode)
      : [...batchModels, modelCode];

    if (nextModels.length === 0) {
      return;
    }

    const hasChanged =
      nextModels.length !== batchModels.length ||
      nextModels.some((code, index) => code !== batchModels[index]);

    if (!hasChanged) {
      return;
    }

    if (hasBatchResultState()) {
      const ok = confirmBatchReset(
        "Changing the selected models will reset the current batch results. Continue?"
      );
      if (!ok) return;
      clearBatchRunState();
    }

    setBatchModels(nextModels);
  };

  const isDisabled = status === "loading" || text.trim().length === 0;
  const isTextTooLong = text.length > MAX_TEXT_LENGTH;
  const isSingleAnalyzeDisabled = isDisabled || isTextTooLong;
  const selectedModelLabel = useMemo(() => {
    return MODEL_OPTIONS.find((option) => option.code === selectedModel)?.label ?? "XLM-RoBERTa";
  }, [selectedModel]);
  const xlmWarning = "Model only fine-tuned on German texts. Not tested on other languages!";
  const confidenceLabel = useMemo(() => {
    if (!result) return "--";
    return `${(result.confidence * 100).toFixed(2)}%`;
  }, [result]);
  const explanationText = useMemo(() => {
    if (!result) return "--";
    return result.explanation?.trim() || "--";
  }, [result]);

  const formatInstanceLabel = (count: number) => `${count} instance${count === 1 ? "" : "s"}`;
  const dimiProcessedInstanceCount = useMemo(() => dimiPreviewRows.length, [dimiPreviewRows]);
  const dimiSkippedInstanceCount = useMemo(
    () => dimiPreviewRows.filter((row) => row.dimi_matches === 0).length,
    [dimiPreviewRows]
  );
  const lmDetectionInstanceCount = useMemo(() => {
    if (batchDimiSkipped) {
      return batchDimiTotal;
    }
    if (skipNoDimiMatches === true) {
      return Math.max(dimiProcessedInstanceCount - dimiSkippedInstanceCount, 0);
    }
    return dimiProcessedInstanceCount;
  }, [
    batchDimiSkipped,
    batchDimiTotal,
    skipNoDimiMatches,
    dimiProcessedInstanceCount,
    dimiSkippedInstanceCount,
  ]);
  const dimiEta = useMemo(
    () => estimateRemainingTime(batchDimiStartedAt, batchDimiProgress),
    [batchDimiStartedAt, batchDimiProgress, clockTick]
  );
  const batchEta = useMemo(
    () => estimateRemainingTime(batchStartedAt, batchProgress),
    [batchStartedAt, batchProgress, clockTick]
  );
  const batchMetricComparison = useMemo(() => {
    if (!batchMetrics) return [];

    return batchModels.flatMap((modelCode) => {
      const suffix = BATCH_MODEL_SUFFIXES[modelCode];
      const metrics = batchMetrics[suffix];
      if (!metrics) return [];

      const modelLabel =
        MODEL_OPTIONS.find((option) => option.code === modelCode)?.label ?? suffix;
      const f1Value = Number.parseFloat(metrics.f1);

      return [
        {
          modelCode,
          modelLabel,
          suffix,
          f1Value: Number.isFinite(f1Value) ? f1Value : 0,
          metrics,
        },
      ];
    });
  }, [batchMetrics, batchModels]);
  const activeMetricComparison =
    batchMetricComparison.find((item) => item.modelCode === metricsHoveredModel) ??
    batchMetricComparison[0] ??
    null;

  const shouldShowResult =
    result !== null && resultModel === selectedModel && resultLanguage === lmLanguage;

  const hasUnsavedData = batchFile !== null;

  useEffect(() => {
    return () => {
      if (batchDownloadUrl) {
        URL.revokeObjectURL(batchDownloadUrl);
      }
    };
  }, [batchDownloadUrl]);

  useEffect(() => {
    const stored = sessionStorage.getItem("viewMode") as ViewMode | null;
    if (stored === "single" || stored === "batch") {
      setViewMode(stored);
    }
  }, []);

  useEffect(() => {
    const handler = (event: BeforeUnloadEvent) => {
      if (!hasUnsavedData) return;
      event.preventDefault();
      event.returnValue = "";
    };
    window.addEventListener("beforeunload", handler);
    return () => window.removeEventListener("beforeunload", handler);
  }, [hasUnsavedData]);

  useEffect(() => {
    if (batchStatus !== "uploading" && batchStatus !== "processing" && batchDimiStatus !== "processing") {
      return;
    }

    const intervalId = window.setInterval(() => {
      setClockTick((value) => value + 1);
    }, 1000);

    return () => window.clearInterval(intervalId);
  }, [batchStatus, batchDimiStatus]);

  const resetAppState = () => {
    const ok = window.confirm(
      "Reset the app?\nThis will clear your current progress and cannot be undone!"
    );
    if (!ok) return;

    sessionStorage.setItem("viewMode", "batch");
    setViewMode("batch");
    setText(INITIAL_TEXT);
    setStatus("idle");
    setErrorMessage(null);
    setResult(null);
    setSelectedModel("xlm-roberta");
    setLmLanguage("de");
    setResultModel(null);
    setResultLanguage(null);
    setDimiStatus("idle");
    setDimiError(null);
    setDimiResult(null);
    setDimiCopied({});
    clearBatchDimiState();
    setBatchModels(["xlm-roberta"]);

    setBatchFile(null);
    setBatchStatus("idle");
    setBatchError(null);
    setBatchDownloadUrl(null);
    setBatchOutputFormat(null);
    setBatchLanguage(null);
    setBatchDimiStatus("idle");
    setBatchDimiError(null);
    setBatchDimiProgress(0);
    setBatchDimiProcessed(0);
    setBatchDimiTotal(0);
    setBatchDimiPreparedFile(null);
    setBatchDimiSkipped(false);
    setSkipNoDimiMatches(null);
    setBatchFilename(null);
    setBatchMetricsFilename(null);
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


  const handleDimiSubmit = async (event: React.FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    const trimmed = dimiText.trim();
    if (!trimmed) {
      setDimiError("Please enter some text to analyze.");
      return;
    }

    setDimiError(null);
    setDimiStatus("loading");
    setDimiResult(null);
    setDimiCopied({});

    try {
      const response = await fetch("http://localhost:8000/lemmatize", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          text: trimmed,
          language: dimiLanguage,
        }),
      });

      if (!response.ok) {
        const message = await response.text();
        throw new Error(message || "Lemmatizer request failed");
      }

      const data = (await response.json()) as LemmatizerResponse;
      setDimiResult(data);
      setDimiStatus("idle");
    } catch (error) {
      setDimiStatus("error");
      setDimiError(error instanceof Error ? error.message : "Unexpected error");
    }
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
    setResultModel(null);
    setResultLanguage(null);

    const modelForRequest = selectedModel;
    const languageForRequest = lmLanguage;

    try {
      const response = await fetch("http://localhost:8000/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          text: trimmed,
          model: modelForRequest,
          language: languageForRequest,
        }),
      });

      if (!response.ok) {
        const message = await response.text();
        throw new Error(message || "Request failed");
      }

      const data = (await response.json()) as PredictionResponse;
      setResult(data);
      setResultModel(modelForRequest);
      setResultLanguage(languageForRequest);
      setStatus("idle");
    } catch (error) {
      setStatus("error");
      setErrorMessage(error instanceof Error ? error.message : "Unexpected error");
    }
  };


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
      setBatchLanguage(null);
      clearBatchDimiState();
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
          setBatchLanguage(null);
          clearBatchDimiState();
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
        setBatchLanguage(null);
        clearBatchDimiState();
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
    const nextFormat = value === "json" ? "json" : value === "csv" ? "csv" : null;
    if (nextFormat === batchOutputFormat) return;

    const hasDownstreamState =
      hasPreparedDimiState() ||
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
    setBatchLanguage(null);
    clearBatchDimiState();
  };

  const handleBatchLanguageChange = (value: string) => {
    const nextLanguage = LANGUAGE_OPTIONS.some((option) => option.code === value)
      ? (value as LanguageCode)
      : null;
    if (!nextLanguage || nextLanguage === batchLanguage) return;

    const hasDownstreamState =
      hasPreparedDimiState() ||
      batchStatus !== "idle" ||
      batchDownloadUrl !== null ||
      outputPreview.length > 0 ||
      outputPreviewJson !== null ||
      batchMetrics !== null;

    if (hasDownstreamState) {
      const ok = confirmBatchReset(
        "Changing the language will reset the current batch results. Continue?"
      );
      if (!ok) return;
    }

    clearBatchRunState();
    setBatchLanguage(nextLanguage);
    clearBatchDimiState();
  };

  const handleBatchDimiSubmit = async () => {
    if (!batchFile || !batchOutputFormat || !batchLanguage) return;

    if (hasBatchResultState() || hasDimiResultState()) {
      const ok = confirmBatchReset(
        "Running DiMi preprocessing will reset the current batch results. Continue?"
      );
      if (!ok) return;
      clearBatchRunState();
    }
    clearBatchDimiState();
    setBatchDimiStatus("processing");
    setBatchDimiError(null);
    setBatchDimiStartedAt(Date.now());
    setBatchDimiProgress(0);
    setBatchDimiProcessed(0);
    setBatchDimiTotal(0);
    setBatchDimiPreparedFile(null);
    setBatchDimiSkipped(false);
    setDimiPreviewRows([]);
    setDimiPreviewJson(null);
    setDimiPreviewNote(null);

    const runId = batchDimiRunIdRef.current;

    try {
      const textContent = await batchFile.text();
      if (runId !== batchDimiRunIdRef.current) return;
      const inputRows = parseBatchInput(textContent, batchFile.name);

      if (inputRows.length === 0) {
        throw new Error("No text rows were found in the uploaded file.");
      }

      if (inputRows.length > MAX_BATCH_INSTANCES) {
        throw new Error(
          `Too many instances: ${inputRows.length}. Maximum allowed is ${MAX_BATCH_INSTANCES}.`
        );
      }

      setBatchDimiTotal(inputRows.length);
      if (runId !== batchDimiRunIdRef.current) return;

      const startResponse = await fetch("http://localhost:8000/lemmatize/batch/start", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          texts: inputRows.map((row) => row.text),
          language: batchLanguage,
        }),
      });

      if (!startResponse.ok) {
        const message = await startResponse.text();
        throw new Error(message || "DiMi preprocessing failed");
      }

      const startPayload = (await startResponse.json()) as { job_id: string };
      if (runId !== batchDimiRunIdRef.current) return;

      const pollStatus = async () => {
        if (runId !== batchDimiRunIdRef.current) return true;

        const statusResponse = await fetch(
          `http://localhost:8000/lemmatize/batch/status/${startPayload.job_id}`
        );
        if (runId !== batchDimiRunIdRef.current) return true;
        if (!statusResponse.ok) {
          const message = await statusResponse.text();
          throw new Error(message || "DiMi status request failed");
        }

        const statusPayload = (await statusResponse.json()) as BatchStatusResponse;
        if (runId !== batchDimiRunIdRef.current) return true;
        setBatchDimiProcessed(statusPayload.processed);
        setBatchDimiProgress(statusPayload.progress);

        if (statusPayload.status === "completed") {
          const resultResponse = await fetch(
            `http://localhost:8000/lemmatize/batch/result/${startPayload.job_id}`
          );
          if (runId !== batchDimiRunIdRef.current) return true;
          if (!resultResponse.ok) {
            const message = await resultResponse.text();
            throw new Error(message || "DiMi result request failed");
          }

          const results = (await resultResponse.json()) as LemmatizerResponse[];
          if (runId !== batchDimiRunIdRef.current) return true;
          if (results.length !== inputRows.length) {
            throw new Error("DiMi preprocessing returned unexpected batch size.");
          }

          const previewRows: DimiPreviewRow[] = [];
          const augmentedRecords: Array<Record<string, unknown>> = [];

          results.forEach((result, index) => {
            const inputRow = inputRows[index];
            const matchRows = result.matches.length > 0 ? result.matches : [];
            let dimiCounter = 0;

            if (matchRows.length === 0) {
              const combinedId = `${inputRow.id}_${dimiCounter}`;
              const previewRow: DimiPreviewRow = {
                id: combinedId,
                full_text: inputRow.text,
                text: inputRow.text,
                dimi_matches: 0,
                dimi_matched_lemmas: [],
              };
              previewRows.push(previewRow);
              augmentedRecords.push(
                buildPreparedRecord(inputRow.record, {
                  id: combinedId,
                  text: inputRow.text,
                  fullText: inputRow.text,
                  dimiMatches: 0,
                  dimiMatchedLemmas: "",
                })
              );
            } else {
              matchRows.forEach((match) => {
                const contextText = match.context_sentences.join(" ").trim();
                const combinedId = `${inputRow.id}_${dimiCounter}`;
                const previewRow: DimiPreviewRow = {
                  id: combinedId,
                  full_text: inputRow.text,
                  text: contextText,
                  dimi_matches: result.matches.length,
                  dimi_matched_lemmas: [...new Set(match.matched_lemmas)],
                };

                previewRows.push(previewRow);
                augmentedRecords.push(
                  buildPreparedRecord(inputRow.record, {
                    id: combinedId,
                    text: contextText,
                    fullText: inputRow.text,
                    dimiMatches: result.matches.length,
                    dimiMatchedLemmas: previewRow.dimi_matched_lemmas.join("; "),
                  })
                );
                dimiCounter += 1;
              });
            }
          });

          if (previewRows.length > MAX_BATCH_INSTANCES) {
            throw new Error(
              `Too many DiMi instances: ${previewRows.length}. Maximum allowed is ${MAX_BATCH_INSTANCES}.`
            );
          }

          if (runId !== batchDimiRunIdRef.current) return true;

          const preparedName = batchFile.name.replace(/\.(csv|json)$/i, "") || "dimi-input";
          if (batchOutputFormat === "json") {
            const jsonText = JSON.stringify(augmentedRecords, null, 2);
            setDimiPreviewRows(previewRows);
            setDimiPreviewJson(jsonText);
            setDimiPreviewNote(previewRows.length > 0 ? null : "No DiMi matches found.");
            setBatchDimiPreparedFile(
              new File([jsonText], `${preparedName}-dimi.json`, { type: "application/json" })
            );
          } else {
            const csvText = buildCsvText(augmentedRecords);
            setDimiPreviewRows(previewRows);
            setDimiPreviewJson(csvText);
            setDimiPreviewNote(previewRows.length > 0 ? null : "No DiMi matches found.");
            setBatchDimiPreparedFile(
              new File([csvText], `${preparedName}-dimi.csv`, { type: "text/csv" })
            );
          }

          setBatchDimiStatus("done");
          return true;
        }

        if (statusPayload.status === "failed") {
          throw new Error(statusPayload.error || "DiMi preprocessing failed");
        }

        return false;
      };

      const pollLoop = async () => {
        let completed = false;
        while (!completed) {
          completed = await pollStatus();
          if (runId !== batchDimiRunIdRef.current) return;
          if (!completed) await new Promise((resolve) => setTimeout(resolve, 600));
        }
      };

      await pollLoop();

    } catch (error) {
      if (runId !== batchDimiRunIdRef.current) return;
      setBatchDimiStatus("error");
      setBatchDimiError(error instanceof Error ? error.message : "Unexpected error");
      setBatchDimiPreparedFile(null);
      setDimiPreviewRows([]);
      setDimiPreviewJson(null);
      setDimiPreviewNote("Run DiMi preprocessing to see the output preview.");
    }
  };

  const handleSkipBatchDimiSubmit = async () => {
    if (!batchFile || !batchOutputFormat || !batchLanguage) return;

    if (hasBatchResultState() || hasDimiResultState()) {
      const ok = confirmBatchReset(
        "Skipping DiMi preprocessing will reset the current batch results. Continue?"
      );
      if (!ok) return;
      clearBatchRunState();
    }

    setSkipNoDimiMatches(false);
    setBatchDimiError(null);
    setBatchDimiStartedAt(null);
    setBatchDimiProgress(0);
    setBatchDimiProcessed(0);
    setBatchDimiTotal(0);
    setBatchDimiPreparedFile(null);
    setBatchDimiSkipped(true);
    setDimiPreviewRows([]);
    setDimiPreviewJson(null);
    setDimiPreviewNote("DiMi preprocessing skipped.");

    try {
      const textContent = await batchFile.text();
      const inputRows = parseBatchInput(textContent, batchFile.name);

      if (inputRows.length === 0) {
        throw new Error("No text rows were found in the uploaded file.");
      }

      if (inputRows.length > MAX_BATCH_INSTANCES) {
        throw new Error(
          `Too many instances: ${inputRows.length}. Maximum allowed is ${MAX_BATCH_INSTANCES}.`
        );
      }

      setBatchDimiTotal(inputRows.length);

      const augmentedRecords: Array<Record<string, unknown>> = inputRows.map((inputRow) =>
        buildPreparedRecord(inputRow.record, {
          id: inputRow.id,
          text: inputRow.text,
          fullText: inputRow.text,
          dimiMatches: 0,
          dimiMatchedLemmas: "",
        })
      );

      setBatchDimiProcessed(inputRows.length);
      setBatchDimiProgress(100);

      const preparedName = batchFile.name.replace(/\.(csv|json)$/i, "") || "dimi-input";
      if (batchOutputFormat === "json") {
        const jsonText = JSON.stringify(augmentedRecords, null, 2);
        setBatchDimiPreparedFile(
          new File([jsonText], `${preparedName}-dimi.json`, { type: "application/json" })
        );
      } else {
        const csvText = buildCsvText(augmentedRecords);
        setBatchDimiPreparedFile(
          new File([csvText], `${preparedName}-dimi.csv`, { type: "text/csv" })
        );
      }

      setBatchDimiStatus("done");
    } catch (error) {
      setBatchDimiStatus("error");
      setBatchDimiError(error instanceof Error ? error.message : "Unexpected error");
      setBatchDimiPreparedFile(null);
      setDimiPreviewRows([]);
      setDimiPreviewJson(null);
      setDimiPreviewNote("Run DiMi preprocessing to see the output preview.");
    }
  };

  const handleBatchSubmit = async (event: React.FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    if (!batchFile || !batchOutputFormat || !batchLanguage || !batchDimiPreparedFile) return;
    if (batchModels.length === 0) return;

    if (hasBatchResultState()) {
      const ok = confirmBatchReset(
        "Running LM detection will reset the current batch results. Continue?"
      );
      if (!ok) return;
      clearBatchRunState();
    }

    setBatchStatus("uploading");
    setBatchError(null);
    setBatchStartedAt(Date.now());
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
      formData.append("file", batchDimiPreparedFile);
      formData.append("output_format", batchOutputFormat);
      formData.append("language", batchLanguage);
      formData.append("skip_no_dimi_matches", String(skipNoDimiMatches));
      batchModels.forEach((modelCode) => formData.append("models", modelCode));

      const response = await fetch("http://localhost:8000/batch/start", {
        method: "POST",
        body: formData,
      });

      if (!response.ok) {
        const message = await response.text();
        throw new Error(message || "Batch request failed");
      }

      const startPayload = (await response.json()) as { job_id: string };
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
        if (statusPayload.metrics) setBatchMetrics(statusPayload.metrics);

        if (statusPayload.status === "completed") {
          const resultResponse = await fetch(
            `http://localhost:8000/batch/result/${startPayload.job_id}`
          );
          if (!resultResponse.ok) {
            const message = await resultResponse.text();
            throw new Error(message || "Result request failed");
          }

          const baseName = batchDimiPreparedFile.name.replace(/\.(csv|json)$/i, "") || "predictions";
          const extension = batchOutputFormat === "json" ? "json" : "csv";
          if (statusPayload.metrics) {
            setBatchMetricsFilename(
              `${baseName}-metrics.${batchOutputFormat === "json" ? "json" : "csv"}`
            );
          }

          if (batchOutputFormat === "json") {
            const textContent = await resultResponse.text();
            const prettyJson = formatJsonPreview(textContent);
            setOutputPreview([]);
            setOutputPreviewJson(prettyJson);
            setOutputPreviewNote(null);
            const blob = new Blob([prettyJson], { type: "application/json" });
            setBatchFilename(`${baseName}-results.${extension}`);
            setBatchDownloadUrl(URL.createObjectURL(blob));
            setBatchStatus("done");
            return true;
          }

          const csvText = await resultResponse.text();
          const preview = parseCsvAll(csvText);
          setOutputPreview(preview);
          setOutputPreviewNote(preview.length ? null : "No rows detected in the CSV output.");
          setOutputPreviewJson(null);
          const blob = new Blob([csvText], { type: "text/csv" });
          setBatchFilename(`${baseName}-results.${extension}`);
          setBatchDownloadUrl(URL.createObjectURL(blob));
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
          if (!completed) await new Promise((resolve) => setTimeout(resolve, 600));
        }
      };

      await pollLoop();
    } catch (error) {
      setBatchStatus("error");
      setBatchError(error instanceof Error ? error.message : "Unexpected error");
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
    const entries = Object.entries(batchMetrics);
    const header = "model,accuracy,precision,recall,f1,tp,fp,tn,fn";
    const metricsText =
      batchOutputFormat === "json"
        ? JSON.stringify(batchMetrics, null, 2)
        : [
            header,
            ...entries.map(
              ([modelName, metrics]) =>
                [
                  modelName,
                  metrics.accuracy,
                  metrics.precision,
                  metrics.recall,
                  metrics.f1,
                  metrics.tp,
                  metrics.fp,
                  metrics.tn,
                  metrics.fn,
                ].join(",")
            ),
          ].join("\n");
    const metricsBlob = new Blob([metricsText], {
      type: batchOutputFormat === "json" ? "application/json" : "text/csv",
    });
    const metricsUrl = URL.createObjectURL(metricsBlob);
    const link = document.createElement("a");
    link.href = metricsUrl;
    link.download =
      batchMetricsFilename ?? `metrics.${batchOutputFormat === "json" ? "json" : "csv"}`;
    document.body.appendChild(link);
    link.click();
    link.remove();
    URL.revokeObjectURL(metricsUrl);
  };

  const truncateCell = (value: string | undefined | null, limit = 120) => {
    if (!value) return "";
    const collapsed = value.replace(/\s+/g, " ").trim();
    if (collapsed.length <= limit) return collapsed;
    return collapsed.slice(0, limit) + "…";
  };

  const stringifyCell = (value: unknown) => {
    if (value === null || value === undefined) return "";
    if (typeof value === "string") return value;
    if (typeof value === "number" || typeof value === "boolean") return String(value);
    return JSON.stringify(value);
  };

  const translateLabelValue = (value: unknown) => {
    const normalized = stringifyCell(value).trim().toLowerCase();
    if (normalized === "1" || normalized === "true" || normalized === "moralization") {
      return "moralization";
    }
    if (
      normalized === "0" ||
      normalized === "false" ||
      normalized === "no_moralization"
    ) {
      return "no_moralization";
    }
    return stringifyCell(value).trim();
  };

  const buildPreparedRecord = (
    inputRecord: Record<string, unknown>,
    values: {
      id: string;
      text: string;
      fullText: string;
      dimiMatches: number;
      dimiMatchedLemmas: string;
    }
  ) => {
    const reservedKeys = new Set([
      "id",
      "text",
      "full_text",
      "label",
      "dimi_matches",
      "dimi_matched_lemmas",
    ]);

    const preparedRecord: Record<string, unknown> = {
      id: values.id,
      text: values.text,
      full_text: values.fullText,
      dimi_matches: values.dimiMatches,
      dimi_matched_lemmas: values.dimiMatchedLemmas,
    };

    if (Object.prototype.hasOwnProperty.call(inputRecord, "label")) {
      preparedRecord.label = translateLabelValue(inputRecord.label);
    }

    Object.keys(inputRecord).forEach((key) => {
      if (reservedKeys.has(key)) return;
      preparedRecord[key] = inputRecord[key];
    });

    return preparedRecord;
  };

  const escapeCsvCell = (value: string) => {
    if (/[",\n\r]/.test(value)) {
      return `"${value.replace(/"/g, '""')}"`;
    }
    return value;
  };

  const buildCsvText = (rows: Array<Record<string, unknown>>) => {
    const columns = new Set<string>();
    rows.forEach((row) => {
      Object.keys(row).forEach((key) => columns.add(key));
    });
    const fieldnames = Array.from(columns);
    if (fieldnames.length === 0) return "";

    const header = fieldnames.join(",");
    const lines = rows.map((row) =>
      fieldnames.map((fieldname) => escapeCsvCell(stringifyCell(row[fieldname]))).join(",")
    );
    return [header, ...lines].join("\n");
  };

  const parseBatchInput = (textContent: string, fileName: string) => {
    const isJson = fileName.toLowerCase().endsWith(".json");
    if (isJson) {
      const parsedJson = JSON.parse(textContent) as unknown;
      const records = Array.isArray(parsedJson)
        ? parsedJson
        : parsedJson && typeof parsedJson === "object"
        ? [parsedJson]
        : [];

      return records
        .map((record, index) => {
          if (!record || typeof record !== "object") return null;
          const row = record as Record<string, unknown>;
          const text = stringifyCell(row.text ?? row.content ?? row.body ?? "").trim();
          if (!text) return null;
          return {
            index,
            record: row,
            text,
            id: stringifyCell(row.id ?? index + 1) || String(index + 1),
          };
        })
        .filter((item): item is { index: number; record: Record<string, unknown>; text: string; id: string } => item !== null);
    }

    const allRows = parseCsvAll(textContent);
    if (allRows.length === 0) return [];

    const headerRow = allRows[0].map((cell) => cell.trim());
    const hasHeader = headerRow.some((cell) => cell.toLowerCase() === "text");
    const startIndex = hasHeader ? 1 : 0;
    const textColumnIndex = hasHeader
      ? headerRow.findIndex((cell) => cell.toLowerCase() === "text")
      : 0;
    const idColumnIndex = hasHeader
      ? headerRow.findIndex((cell) => cell.toLowerCase() === "id")
      : -1;

    return allRows.slice(startIndex).map((row, index) => {
      const text = stringifyCell(row[textColumnIndex] ?? "").trim();
      const record: Record<string, unknown> = {};
      if (hasHeader) {
        headerRow.forEach((header, headerIndex) => {
          if (!header) return;
          record[header] = row[headerIndex] ?? "";
        });
      } else {
        row.forEach((cell, cellIndex) => {
          record[`column_${cellIndex + 1}`] = cell;
        });
      }

      const idValue =
        hasHeader && idColumnIndex >= 0 ? stringifyCell(row[idColumnIndex]) : String(index + 1);

      return {
        index,
        record,
        text,
        id: idValue || String(index + 1),
      };
    }).filter((item) => item.text.length > 0);
  };

  const renderDimiResults = () => {
    if (!dimiResult) return null;

    if (dimiResult.matches.length === 0) {
      return (
        <div className={styles.dimiResults}>
          <p className={`${styles.hint} ${styles.resultLabel}`}>
            No dictionary lemmas found in the text.
          </p>
        </div>
      );
    }

    return (
      <div className={styles.dimiResults}>
        <p className={`${styles.hint} ${styles.resultLabel}`}>
          {dimiResult.matches.length} Match{dimiResult.matches.length !== 1 ? "es" : ""} across{" "}
          {dimiResult.total_sentences} Sentence{dimiResult.total_sentences !== 1 ? "s" : ""}
        </p>

        {dimiResult.matches.map((match, i) => (
          <div
            key={i}
            className={styles.dimiMatchCard}
            role="button"
            tabIndex={0}
            onClick={async () => {
              try {
                const copyText = match.context_sentences.join(" ");
                await navigator.clipboard.writeText(copyText);
                setDimiCopied((s) => ({ ...(s || {}), [i]: true }));
                setTimeout(() =>
                  setDimiCopied((s) => {
                    const next = { ...(s || {}) };
                    delete next[i];
                    return next;
                  }),
                1500);
              } catch {
                // ignore clipboard errors
              }
            }}
            onKeyDown={(e) => {
              if (e.key === "Enter" || e.key === " ") {
                e.preventDefault();
                (e.currentTarget as HTMLElement).click();
              }
            }}
          >
            <span
              className={styles.copiedBadge}
              aria-hidden="false"
              aria-label={dimiCopied?.[i] ? "Copied" : "Click to copy"}
            >
              <span className={styles.copiedText}>
                {dimiCopied?.[i] ? "Copied" : "Click to copy"}
              </span>
              <svg
                className={styles.copiedSvg}
                width="14"
                height="14"
                viewBox="0 0 24 24"
                fill="none"
                xmlns="http://www.w3.org/2000/svg"
                aria-hidden="true"
              >
                <path
                  d="M16 1H4a2 2 0 0 0-2 2v14"
                  stroke="#0a1823"
                  strokeWidth="1.5"
                  strokeLinecap="round"
                  strokeLinejoin="round"
                />
                <rect
                  x="8"
                  y="4"
                  width="13"
                  height="13"
                  rx="2"
                  stroke="#0a1823"
                  strokeWidth="1.5"
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  fill="#fff8cc"
                />
              </svg>
            </span>

            <p className={styles.resultLabel}>Sentence {match.sentence_index + 1}</p>
            <p className={styles.dimiMatchMeta}>
              <span className={styles.resultLabelNormal}>
                MATCHED LEMMAS: <span className={styles.dimiLemmaList}>{match.matched_lemmas.join(", ")}</span>
              </span>
            </p>

            <div className={styles.dimiContext}>
              {match.context_html.map((sentence, j) => {
                const isCenterSentence = match.context_sentences[j] === match.center_sentence;

                return (
                  <span key={j}>
                    {isCenterSentence ? (
                      <span
                        className={styles.dimiCenterSentence}
                        dangerouslySetInnerHTML={{ __html: sentence }}
                      />
                    ) : (
                      <span className={styles.dimiContextSentence}>{sentence}</span>
                    )}
                    {" "}
                  </span>
                );
              })}
            </div>
          </div>
        ))}
      </div>
    );
  };


  return (
    <div className={styles.page}>
      <main className={styles.main}>
        <section className={styles.hero}>
          <p className={styles.eyebrow}>Moralization Detection Toolkit</p>
          <h1>Analyze moral framing in texts.</h1>
          <p className={styles.subtitle}>
            Moralization detection with dictionaries and language models.
          </p>
        </section>

        <section className={styles.modeSwitch}>
          <div className={styles.modeTabs}>
            <button
              className={`${styles.modeTab} ${viewMode === "single" ? styles.modeTabActive : ""}`}
              type="button"
              onClick={() => {
                setViewMode("single");
                sessionStorage.setItem("viewMode", "single");
              }}
            >
              Single text
            </button>
            <button
              className={`${styles.modeTab} ${viewMode === "batch" ? styles.modeTabActive : ""}`}
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
            {/* ── DiMi panel ── */}
            <section className={styles.batchPanel}>
              <p className={styles.boxTitle}>
                Moralization Detection with Dictionary Approach (DiMi)
              </p>

              <section className={styles.panel}>
                <form className={styles.form} onSubmit={handleDimiSubmit}>
                  <div className={styles.languageSwitch}>
                    <span className={styles.label}>Language</span>
                    <div className={styles.modeTabs}>
                      {LANGUAGE_OPTIONS.map((option) => (
                        <button
                          key={option.code}
                          className={`${styles.modeTab} ${
                            dimiLanguage === option.code ? styles.modeTabActive : ""
                          }`}
                          type="button"
                          onClick={() => setDimiLanguage(option.code)}
                          aria-pressed={dimiLanguage === option.code}
                          title={option.name}
                        >
                          {option.label}
                        </button>
                      ))}
                    </div>
                  </div>
                  <a className={styles.hint}>
                    {lemmaCount !== null ? `${lemmaCount} lemmas in ${LANGUAGE_OPTIONS.find((o) => o.code === dimiLanguage)?.name || dimiLanguage}-DiMi` : "loading lemmas…"}
                  </a>

                  <label className={styles.label} htmlFor="dimiTextInput">
                    Text input
                  </label>
                  <textarea
                    id="dimiTextInput"
                    className={styles.textarea}
                    value={dimiText}
                    onChange={(event) => setDimiText(event.target.value)}
                    rows={7}
                    maxLength={MAX_TEXT_LENGTH}
                    placeholder="Paste text to analyze using the DiMi lexicon..."
                  />

                  <div className={styles.actions}>
                    <div className={styles.analyzeControlStack}>
                      <button
                        className={styles.primaryButton}
                        type="submit"
                        disabled={dimiStatus === "loading"}
                      >
                        {dimiStatus === "loading" ? "Analyzing..." : "Analyze"}
                      </button>
                      {dimiStatus === "loading" && (
                        <div className={styles.progressBar} aria-hidden="true">
                          <span className={styles.progressFill} />
                        </div>
                      )}
                    </div>
                    {dimiStatus !== "loading" && (
                      <span className={styles.hint}>
                        {dimiText.length}/{MAX_TEXT_LENGTH}
                      </span>
                    )}
                  </div>
                </form>
              </section>

              {dimiStatus === "error" && dimiError && (
                <p className={styles.errorMessage}>{dimiError}</p>
              )}

              {renderDimiResults()}
            </section>

            {/* ── LM panel ── */}
            <section className={styles.batchPanel}>
              <p className={styles.boxTitle}>
                Moralization Detection with Language Models
              </p>
              <section className={styles.panel}>
                <form className={styles.form} onSubmit={handleSubmit}>
                  <div className={styles.languageSwitch}>
                    <span className={styles.label}>Language</span>
                    <div className={styles.modeTabs}>
                      {LANGUAGE_OPTIONS.map((option) => (
                        <button
                          key={option.code}
                          className={`${styles.modeTab} ${
                            lmLanguage === option.code ? styles.modeTabActive : ""
                          }`}
                          type="button"
                          onClick={() => {
                            setLmLanguage(option.code);
                            if (option.code !== "de" && selectedModel === "xlm-roberta") {
                              setSelectedModel("claude");
                            }
                          }}
                          aria-pressed={lmLanguage === option.code}
                          title={option.name}
                        >
                          {option.label}
                        </button>
                      ))}
                    </div>
                  </div>
                  <div className={styles.languageSwitch}>
                    <span className={styles.label}>Model</span>
                    <div className={styles.modeTabs}>
                      {MODEL_OPTIONS.map((option) => {
                        const isWarned = option.code === "xlm-roberta" && lmLanguage !== "de";

                        const button = (
                          <button
                            className={`${styles.modeTab} ${
                              selectedModel === option.code ? styles.modeTabActive : ""
                            }`}
                            type="button"
                            onClick={() => setSelectedModel(option.code)}
                            aria-pressed={selectedModel === option.code}
                            title={isWarned ? xlmWarning : option.name}
                          >
                            {option.label}
                          </button>
                        );

                        return (
                          <span key={option.code} className={styles.modeTabWrap}>
                            {button}
                          </span>
                        );
                      })}
                    </div>
                  </div>
                  {selectedModel === "xlm-roberta" && lmLanguage !== "de" && (
                    <p className={styles.modelWarning}>{xlmWarning}</p>
                  )}
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
                    <div className={styles.analyzeControlStack}>
                      <button
                        className={styles.primaryButton}
                        type="submit"
                        disabled={isSingleAnalyzeDisabled}
                      >
                        {status === "loading" ? "Analyzing..." : "Analyze"}
                      </button>
                      {status === "loading" && (
                        <div className={styles.progressBar} aria-hidden="true">
                          <span className={styles.progressFill} />
                        </div>
                      )}
                    </div>
                    {status !== "loading" && (
                      <span className={styles.hint}>
                        {text.length}/{MAX_TEXT_LENGTH}
                      </span>
                    )}
                  </div>
                </form>
              </section>

              {shouldShowResult && result && (
                <section className={styles.resultCard}>
                  <div>
                    <p className={styles.resultLabel}>Prediction</p>
                    <p className={styles.resultValue}>{result.label.replace("_", " ")}</p>
                  </div>
                  <div>
                    <p className={styles.resultLabel}>
                      {selectedModel === "xlm-roberta" ? "Confidence" : "Explanation"}
                    </p>
                    <p
                      className={`${styles.resultValue} ${
                        selectedModel === "xlm-roberta" ? "" : styles.resultValueSmall
                      }`}
                    >
                      {selectedModel === "xlm-roberta" ? confidenceLabel : explanationText}
                    </p>
                  </div>
                </section>
              )}

              {status === "error" && (
                <p className={styles.errorMessage}>{errorMessage}</p>
              )}
            </section>
          </>
        ) : (
          /* ── Batch panel ── */
          <section className={styles.batchPanel}>
            <form className={styles.batchForm} onSubmit={handleBatchSubmit}>
              <p className={styles.boxTitle}>
                Pipeline Moralization Detection (DiMi + Language Models)
              </p>
              <div className={styles.formatInfo}>
                <p className={styles.formatTitle}>Formatting</p>
                <div className={styles.formatList}>
                  <p>
                    <b>Optional columns:</b> id and label (moralization/no_moralization,
                    true/false, 0/1).
                  </p>
                  <p>If <b>no ids</b> are provided, ids are auto-generated.</p>
                  <p>If <b>no labels</b> are provided, metrics are not calculated.</p>
                  <p>Additional columns in jsons and csvs are kept as is.</p>
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
                  <div className={styles.languageSwitch}>
                    <span className={styles.label}>Select output file format...</span>
                    <div className={styles.modeTabs}>
                      {[
                        { code: "csv", label: "CSV", name: "CSV" },
                        { code: "json", label: "JSON", name: "JSON" },
                      ].map((option) => (
                        <button
                          key={option.code}
                          className={`${styles.modeTab} ${
                            batchOutputFormat === option.code ? styles.modeTabActive : ""
                          }`}
                          type="button"
                          onClick={() => handleBatchOutputFormatChange(option.code)}
                          aria-pressed={batchOutputFormat === option.code}
                          title={option.name}
                        >
                          {option.label}
                        </button>
                      ))}
                    </div>
                  </div>
                </div>
              )}

              {batchFile && batchOutputFormat && (
                <div className={`${styles.sectionBox} ${styles.fadeInSection}`}>
                  <div className={styles.languageSwitch}>
                    <span className={styles.label}>Select the input language...</span>
                    <div className={styles.modeTabs}>
                      {LANGUAGE_OPTIONS.map((option) => (
                        <button
                          key={option.code}
                          className={`${styles.modeTab} ${
                            batchLanguage === option.code ? styles.modeTabActive : ""
                          }`}
                          type="button"
                          onClick={() => handleBatchLanguageChange(option.code)}
                          aria-pressed={batchLanguage === option.code}
                          title={option.name}
                        >
                          {option.label}
                        </button>
                      ))}
                    </div>
                  </div>
                </div>
              )}

                          {batchFile && batchOutputFormat && batchLanguage && (
                            <div className={`${styles.sectionBox} ${styles.fadeInSection}`}>
                              <div className={styles.languageSwitch}>
                                <span className={styles.label}>Select models...</span>
                                <div className={styles.modeTabs}>
                                  {MODEL_OPTIONS.map((option) => (
                                    <button
                                      key={option.code}
                                      className={`${styles.modeTab} ${
                                        batchModels.includes(option.code) ? styles.modeTabActive : ""
                                      }`}
                                      type="button"
                                      onClick={() => handleBatchModelToggle(option.code)}
                                      aria-pressed={batchModels.includes(option.code)}
                                      title={option.name}
                                    >
                                      {option.label}
                                    </button>
                                  ))}
                                </div>
                              </div>
                              {batchModels.includes("xlm-roberta") && batchLanguage !== "de" && (
                                <p className={styles.modelWarning}>{xlmWarning}</p>
                              )}
                            </div>
                          )}

              {batchFile && batchOutputFormat && batchLanguage && (
                <div className={`${styles.sectionBox} ${styles.fadeInSection}`}>
                  <div className={styles.boxHeader}>
                    <p className={styles.previewTitle}>Run DiMi preprocessing ...</p>
                  </div>
                  <div className={styles.progressWrap}>
                    <div className={styles.dimiActionRow}>
                      <button
                        className={`${styles.primaryButton} ${styles.compactButton}`}
                        type="button"
                        onClick={() => handleBatchDimiSubmit()}
                        disabled={
                          batchDimiStatus === "processing" ||
                          batchStatus === "uploading" ||
                          !!batchInputLimitError
                        }
                      >
                        {batchDimiStatus === "processing" ? "Processing..." : "Run DiMi"}
                      </button>
                      <button
                        className={styles.inlineTextButton}
                        type="button"
                        onClick={() => handleSkipBatchDimiSubmit()}
                        disabled={
                          batchDimiStatus === "processing" ||
                          batchStatus === "uploading" ||
                          !!batchInputLimitError
                        }
                      >
                        Skip DiMi preprocessing
                      </button>
                    </div>
                    {batchDimiStatus === "processing" && (
                      <>
                        <div className={styles.progressBar}>
                          <span className={styles.progressFill} />
                        </div>
                        <p className={styles.progressText}>
                          {batchDimiProgress}% ({batchDimiProcessed}/{batchDimiTotal})
                          {dimiEta ? ` · about ${dimiEta} remaining` : ""}
                        </p>
                      </>
                    )}
                  </div>
                  {batchDimiStatus === "error" && batchDimiError && (
                    <p className={styles.errorMessage}>{batchDimiError}</p>
                  )}
                </div>
              )}

              {batchDimiStatus === "done" && batchDimiPreparedFile && !batchDimiSkipped && (
                <div className={`${styles.previewCard} ${styles.previewDark} ${styles.outputCard}`}>
                  <p className={styles.previewTitle}>
                    DiMi preview ({(batchOutputFormat ?? "csv").toUpperCase()})
                  </p>
                  <p className={styles.hint}>
                    {formatInstanceLabel(dimiProcessedInstanceCount)} processed after DiMi.
                  </p>
                  {batchOutputFormat === "json" ? (
                    dimiPreviewJson ? (
                      <pre className={styles.previewJson}>
                        {dimiPreviewJson}
                      </pre>
                    ) : (
                      <p className={styles.previewEmpty}>
                        {dimiPreviewNote || "No DiMi matches found."}
                      </p>
                    )
                  ) : dimiPreviewRows.length > 0 ? (
                    <div className={styles.previewScroll}>
                      <table className={styles.previewTable}>
                        <thead>
                          <tr>
                            <th>id</th>
                            <th>text</th>
                            <th>full_text</th>
                            <th>dimi_matches</th>
                            <th>dimi_matched_lemmas</th>
                          </tr>
                        </thead>
                        <tbody>
                          {dimiPreviewRows.map((row) => (
                            <tr key={row.id}>
                              <td>{row.id}</td>
                              <td>
                                <div className={styles.cellTruncate} title={row.full_text}>
                                  {truncateCell(row.full_text)}
                                </div>
                              </td>
                              <td>
                                <div className={styles.cellTruncate} title={row.text}>
                                  {truncateCell(row.text)}
                                </div>
                              </td>
                              <td>{row.dimi_matches}</td>
                              <td>
                                <div
                                  className={styles.cellTruncate}
                                  title={row.dimi_matched_lemmas.join(", ")}
                                >
                                  {truncateCell(row.dimi_matched_lemmas.join(", "))}
                                </div>
                              </td>
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </div>
                  ) : (
                    <p className={styles.previewEmpty}>
                      {dimiPreviewNote || "No DiMi matches found."}
                    </p>
                  )}
                </div>
              )}

              {batchDimiStatus === "done" && batchDimiPreparedFile && batchDimiSkipped && (
                <div className={`${styles.previewCard} ${styles.previewDark} ${styles.outputCard}`}>
                  <p className={styles.previewTitle}>DiMi preprocessing skipped...</p>
                </div>
              )}

              {batchDimiStatus === "done" && batchDimiPreparedFile && !batchDimiSkipped && (
                <div className={`${styles.downloadSection} ${styles.fadeInSection}`}>
                  <div className={styles.downloadLinks}>
                    <button
                      className={`${styles.primaryButton} ${styles.downloadButton}`}
                      type="button"
                      onClick={() => {
                        const link = document.createElement("a");
                        link.href = URL.createObjectURL(batchDimiPreparedFile);
                        link.download = batchDimiPreparedFile.name;
                        document.body.appendChild(link);
                        link.click();
                        link.remove();
                        URL.revokeObjectURL(link.href);
                      }}
                    >
                      Download DiMi output
                    </button>
                  </div>
                </div>
              )}

              {batchDimiStatus === "done" && batchDimiPreparedFile && !batchDimiSkipped && (
                <div className={`${styles.sectionBox} ${styles.fadeInSection}`}>
                  <div className={styles.boxHeader}>
                    <p className={styles.previewTitle}>
                      Skip instances with no DiMi Matches; these are also ignored for the metrics
                      calculation
                    </p>
                  </div>
                  <p className={styles.hint}>
                    {formatInstanceLabel(dimiSkippedInstanceCount)} would be skipped out of{" "}
                    {formatInstanceLabel(dimiProcessedInstanceCount)} (total DiMi matches).
                  </p>
                  <br/>
                  <div className={styles.languageSwitch}>
                    <div className={styles.modeTabs}>
                      <button
                        type="button"
                        className={`${styles.modeTab} ${skipNoDimiMatches === false ? styles.modeTabActive : ""}`}
                        onClick={() => handleSkipNoDimiMatchesChange(false)}
                        aria-pressed={skipNoDimiMatches === false}
                      >
                        Keep all
                      </button>
                      <button
                        type="button"
                        className={`${styles.modeTab} ${skipNoDimiMatches === true ? styles.modeTabActive : ""}`}
                        onClick={() => handleSkipNoDimiMatchesChange(true)}
                        aria-pressed={skipNoDimiMatches === true}
                      >
                        Skip Intances with no DiMi matches
                      </button>
                    </div>
                  </div>
                </div>
              )}

              {batchDimiStatus === "done" && batchDimiPreparedFile && skipNoDimiMatches !== null && (
                <div className={`${styles.sectionBox} ${styles.fadeInSection}`}>
                  <div className={styles.boxHeader}>
                    <p className={styles.previewTitle}>
                      Run LM Detection ({formatInstanceLabel(lmDetectionInstanceCount)})...
                    </p>
                  </div>

                  <div className={styles.progressWrap}>
                    <button
                      className={`${styles.primaryButton} ${styles.compactButton}`}
                      type="submit"
                      disabled={
                        !batchFile ||
                        !batchOutputFormat ||
                        !batchLanguage ||
                        batchStatus === "uploading" ||
                        batchStatus === "processing" ||
                        !!batchInputLimitError
                      }
                    >
                      {batchStatus === "uploading" || batchStatus === "processing"
                        ? "Processing..."
                        : "Run LM Detection"}
                    </button>

                    {(batchStatus === "uploading" || batchStatus === "processing") && (
                      <>
                        <div className={styles.progressBar}>
                          <span className={styles.progressFill} />
                        </div>

                        <p className={styles.progressText}>
                          {batchProgress}% ({batchProcessed}/{batchTotal})
                          {batchEta ? ` · about ${batchEta} remaining` : ""}
                        </p>
                      </>
                    )}
                  </div>
                </div>
              )}
            </form>

            {batchStatus === "error" && (
              <p className={styles.errorMessage}>{batchError}</p>
            )}

            <div className={styles.previewGrid}>
              {(batchStatus === "done" ||
                outputPreviewJson ||
                outputPreview.length > 0) && (
                <div
                  className={`${styles.previewCard} ${styles.previewDark} ${styles.outputCard}`}
                >
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
                                    <div
                                      className={styles.cellTruncate}
                                      title={String(cell)}
                                    >
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

              {batchMetricComparison.length > 0 && (
                <div className={`${styles.previewCard} ${styles.previewDark} ${styles.metricsChartCard}`}>
                  <div className={styles.metricsChartHeader}>
                    <span className={styles.metricsTitle}>F1 comparison</span>
                    <span className={styles.metricsChartHint}>
                      F1 Score comparison between the models.
                    </span>
                  </div>

                  <div className={styles.metricsChart} role="list" aria-label="Model F1 comparison">
                    {batchMetricComparison.map((item) => {
                      const isActive = activeMetricComparison?.modelCode === item.modelCode;
                      const barHeight = Math.max(item.f1Value * 100, 6);

                      return (
                        <button
                          key={item.suffix}
                          type="button"
                          className={`${styles.metricsBarGroup} ${
                            isActive ? styles.metricsBarGroupActive : ""
                          }`}
                          role="listitem"
                          onMouseEnter={() => setMetricsHoveredModel(item.modelCode)}
                          onFocus={() => setMetricsHoveredModel(item.modelCode)}
                          onMouseLeave={() => setMetricsHoveredModel(null)}
                          onBlur={() => setMetricsHoveredModel(null)}
                          aria-label={`${item.modelLabel} F1 ${(item.f1Value * 100).toFixed(2)}%`}
                        >
                          <div className={styles.metricsBarTrack}>
                            <div
                              className={styles.metricsBarFill}
                              style={{
                                height: `${barHeight}%`,
                                backgroundColor: BATCH_MODEL_COLORS[item.modelCode],
                              }}
                            />
                          </div>
                          <div className={styles.metricsBarLabel}>{item.modelLabel}</div>
                          <div className={styles.metricsBarValue}>{(item.f1Value * 100).toFixed(2)}%</div>
                        </button>
                      );
                    })}
                  </div>

                </div>
              )}

              {batchMetrics &&
                batchModels.map((modelCode) => {
                  const suffix = BATCH_MODEL_SUFFIXES[modelCode];
                  const metrics = batchMetrics[suffix];
                  if (!metrics) return null;

                  const modelLabel =
                    MODEL_OPTIONS.find((option) => option.code === modelCode)?.label ?? suffix;

                  return (
                    <div key={suffix} className={`${styles.resultCard} ${styles.metricsCard}`}>
                      <span className={styles.metricsTitle}>Metrics - {modelLabel}</span>
                      <div>
                        <p className={styles.resultLabel}>Accuracy</p>
                        <p className={styles.resultValue}>{metrics.accuracy}</p>
                      </div>
                      <div>
                        <p className={styles.resultLabel}>Precision</p>
                        <p className={styles.resultValue}>{metrics.precision}</p>
                      </div>
                      <div>
                        <p className={styles.resultLabel}>Recall</p>
                        <p className={styles.resultValue}>{metrics.recall}</p>
                      </div>
                      <div>
                        <p className={styles.resultLabel}>F1</p>
                        <p className={styles.resultValue}>{metrics.f1}</p>
                      </div>
                      <div>
                        <p className={styles.resultLabel}>TP</p>
                        <p className={styles.resultValue}>{metrics.tp}</p>
                      </div>
                      <div>
                        <p className={styles.resultLabel}>FP</p>
                        <p className={styles.resultValue}>{metrics.fp}</p>
                      </div>
                      <div>
                        <p className={styles.resultLabel}>TN</p>
                        <p className={styles.resultValue}>{metrics.tn}</p>
                      </div>
                      <div>
                        <p className={styles.resultLabel}>FN</p>
                        <p className={styles.resultValue}>{metrics.fn}</p>
                      </div>
                    </div>
                  );
                })}
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
              © {currentYear} CHAI Lab - Department of German Language and Literature,
              University Heidelberg. All rights reserved.
            </span>
          </div>
          <div className={styles.footerRow}>
            <a
              className={styles.footerLink}
              href="https://www.uni-heidelberg.de/en/imprint"
            >
              Imprint / Impressum
            </a>
            <span className={styles.footerDivider}> | </span>
            <a
              className={styles.footerLink}
              href="https://www.uni-heidelberg.de/en/privacy-statement"
            >
              Privacy Statement / Datenschutzerklärung
            </a>
          </div>
        </footer>
      </main>
      <FloatingKeyButton />
    </div>
  );
}