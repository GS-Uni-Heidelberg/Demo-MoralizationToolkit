"use client";

import { useEffect, useRef, useState } from "react";
import styles from "./FloatingKeyPanel.module.css";

type BillingSettingsResponse = {
  free_tier_daily_credits: number;
};

type BillingStatusResponse = {
  token_type: "free_tier" | "api_token";
  credits_remaining: number;
  free_tier_daily_credits?: number | null;
};

const API_BASE_URL = "http://localhost:8000";
const TOKEN_STORAGE_KEY = "apiToken";
const LEGACY_TOKEN_STORAGE_KEY = "apiKey";
const DEFAULT_FREE_TIER_CREDITS = 5;

export default function FloatingKeyPanel() {
  const [open, setOpen] = useState(false);
  const [apiToken, setApiToken] = useState("");
  const [savedToken, setSavedToken] = useState("");
  const [freeTierCredits, setFreeTierCredits] = useState(DEFAULT_FREE_TIER_CREDITS);
  const [remainingCredits, setRemainingCredits] = useState<number | null>(null);
  const [statusMessage, setStatusMessage] = useState<string | null>(null);

  const wrapperRef = useRef<HTMLDivElement | null>(null);

  const refreshBillingStatus = async (tokenOverride?: string) => {
    const token = tokenOverride ?? savedToken.trim();
    try {
      const response = await fetch(`${API_BASE_URL}/billing/status`, {
        headers: token ? { "X-API-Token": token } : undefined,
      });

      if (!response.ok) return;

      const data = (await response.json()) as BillingStatusResponse;
      setRemainingCredits(data.credits_remaining);
      if (data.token_type === "free_tier" && typeof data.free_tier_daily_credits === "number") {
        setFreeTierCredits(data.free_tier_daily_credits);
      }
    } catch {
      // Keep the last known value if the backend is unreachable.
    }
  };

  useEffect(() => {
    const storedToken =
      localStorage.getItem(TOKEN_STORAGE_KEY) ?? localStorage.getItem(LEGACY_TOKEN_STORAGE_KEY) ?? "";
    const trimmedToken = storedToken.trim();

    if (trimmedToken) {
      setApiToken(trimmedToken);
      setSavedToken(trimmedToken);
    }

    const loadSettings = async () => {
      try {
        const response = await fetch(`${API_BASE_URL}/settings`);
        if (!response.ok) return;

        const data = (await response.json()) as BillingSettingsResponse;
        if (typeof data.free_tier_daily_credits === "number") {
          setFreeTierCredits(data.free_tier_daily_credits);
        }
      } catch {
        // Keep the default if the backend is unreachable.
      }
    };

    void loadSettings();
  }, []);

  useEffect(() => {
    void refreshBillingStatus();

    const interval = window.setInterval(() => {
      void refreshBillingStatus();
    }, 15000);

    const handleFocus = () => {
      void refreshBillingStatus();
    };

    window.addEventListener("focus", handleFocus);
    return () => {
      window.clearInterval(interval);
      window.removeEventListener("focus", handleFocus);
    };
  }, [savedToken]);

  useEffect(() => {
    const handleOutsideClick = (event: MouseEvent) => {
      if (!open) return;
      if (wrapperRef.current && !wrapperRef.current.contains(event.target as Node)) {
        setOpen(false);
      }
    };

    document.addEventListener("mousedown", handleOutsideClick);
    return () => document.removeEventListener("mousedown", handleOutsideClick);
  }, [open]);

  const saveKey = async () => {
    const trimmedToken = apiToken.trim();

    if (trimmedToken) {
      try {
        const response = await fetch(`${API_BASE_URL}/billing/status`, {
          headers: { "X-API-Token": trimmedToken },
        });

        if (!response.ok) {
          const message = await response.text();
          throw new Error(message || "Invalid API token.");
        }

        const data = (await response.json()) as BillingStatusResponse;
        localStorage.setItem(TOKEN_STORAGE_KEY, trimmedToken);
        localStorage.setItem(LEGACY_TOKEN_STORAGE_KEY, trimmedToken);
        setSavedToken(trimmedToken);
        setRemainingCredits(data.credits_remaining);
        window.dispatchEvent(new Event("billingstatuschange"));
        setStatusMessage("API token saved. Requests will use it automatically.");
        setOpen(false);
      } catch (error: unknown) {
        const message = error instanceof Error ? error.message : "Invalid API token.";
        setStatusMessage(message);
        return;
      }
    } else {
      localStorage.removeItem(TOKEN_STORAGE_KEY);
      localStorage.removeItem(LEGACY_TOKEN_STORAGE_KEY);
      setSavedToken("");
      setApiToken("");
      void refreshBillingStatus("");
      window.dispatchEvent(new Event("billingstatuschange"));
      setStatusMessage(`Free tier active: ${freeTierCredits} credit(s) per day.`);
      setOpen(false);
    }
  };

  const clearKey = () => {
    localStorage.removeItem(TOKEN_STORAGE_KEY);
    localStorage.removeItem(LEGACY_TOKEN_STORAGE_KEY);
    setApiToken("");
    setSavedToken("");
    void refreshBillingStatus("");
    window.dispatchEvent(new Event("billingstatuschange"));
    setStatusMessage(`Free tier active: ${freeTierCredits} credit(s) per day.`);
  };

  const hasToken = savedToken.trim().length > 0;
  const coins = remainingCredits ?? freeTierCredits;

  return (
    <div className={styles.wrapper} ref={wrapperRef}>
      <button
        className={`${styles.fab} ${coins === 0 ? styles.fabEmpty : ""}`}
        onClick={() => setOpen((v) => !v)}
      >
        {coins === 0 ? "💰 Add More Coins" : `💰 ${coins}`}
      </button>

      <div className={`${styles.panel} ${open ? styles.open : ""}`}>
        <div className={styles.header}>
          <span>API Token</span>
          <button className={styles.close} onClick={() => setOpen(false)}>
            ✕
          </button>
        </div>

        <span className={styles.info}>
        <b>Don't have a key and your coins are empty?</b>
        <br /> Contact us via {" "} <a href="mailto:support@moralizer.ai" className={styles.link}> email{" "}
        </a>
        to get one or wait 24h for your coins to refill automatically. 
        The free tier includes {freeTierCredits} credit(s) per day for all users! 
        Credits are only consumed for external prompting models. 
        Local DiMi and XLM-RoBERTa runs stay free.
        </span>

        <div className={styles.body}>
          <input
            className={styles.input}
            value={apiToken}
            onChange={(event) => setApiToken(event.target.value)}
            placeholder="mk_..."
            spellCheck={false}
            autoCapitalize="off"
            autoComplete="off"
            autoCorrect="off"
          />

          <div className={styles.actions}>
            <button className={styles.save} onClick={saveKey}>
              Save token
            </button>
            {hasToken ? (
              <button className={styles.clear} onClick={clearKey}>
                Clear
              </button>
            ) : null}
          </div>
        </div>

        <div className={styles.footer}>
          {statusMessage ? <div className={styles.status}>{statusMessage}</div> : null}
        </div>
      </div>
    </div>
  );
}
