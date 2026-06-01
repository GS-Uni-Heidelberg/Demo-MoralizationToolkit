"use client";

import { useEffect, useRef, useState } from "react";
import styles from "./FloatingKeyPanel.module.css";

const DEFAULT_COINS = 5;
const REFILL_TIME = 24 * 60 * 60 * 1000; // 24h

export default function FloatingKeyPanel() {
  const [open, setOpen] = useState(false);
  const [apiKey, setApiKey] = useState("");
  const [coins, setCoins] = useState<number>(0);

  const wrapperRef = useRef<HTMLDivElement | null>(null);

  // Load API key + coins + refill logic
  useEffect(() => {
    const savedKey = localStorage.getItem("apiKey");
    const savedCoins = localStorage.getItem("coins");
    const lastRefill = localStorage.getItem("lastRefill");

    if (savedKey) {
      setApiKey(savedKey);
      setCoins(999); // “unlimited” mode for API users (adjust later if backend exists)
      return;
    }

    const now = Date.now();

    if (!savedCoins || !lastRefill) {
      // first visit (no key)
      setCoins(DEFAULT_COINS);
      localStorage.setItem("coins", String(DEFAULT_COINS));
      localStorage.setItem("lastRefill", String(now));
      return;
    }

    const elapsed = now - Number(lastRefill);

    if (elapsed >= REFILL_TIME) {
      setCoins(DEFAULT_COINS);
      localStorage.setItem("coins", String(DEFAULT_COINS));
      localStorage.setItem("lastRefill", String(now));
    } else {
      setCoins(Number(savedCoins));
    }
  }, []);

  // Persist coins (only in demo mode)
  useEffect(() => {
    if (!apiKey) {
      localStorage.setItem("coins", String(coins));
    }
  }, [coins, apiKey]);

  // Auto refill when coins hit 0
  useEffect(() => {
    if (apiKey) return;
    if (coins !== 0) return;

    const timer = setTimeout(() => {
      const now = Date.now();
      setCoins(DEFAULT_COINS);
      localStorage.setItem("coins", String(DEFAULT_COINS));
      localStorage.setItem("lastRefill", String(now));
    }, REFILL_TIME);

    return () => clearTimeout(timer);
  }, [coins, apiKey]);

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

  const saveKey = () => {
    localStorage.setItem("apiKey", apiKey);

    if (apiKey.trim().length > 0) {
      setCoins(999); // switch to “pro mode”
    } else {
      setCoins(DEFAULT_COINS);
    }

    setOpen(false);
  };

  return (
    <div className={styles.wrapper} ref={wrapperRef}>
      {/* Floating button */}
      <button
      className={`${styles.fab} ${coins === 0 ? styles.fabEmpty : ""}`}
      onClick={() => setOpen((v) => !v)}>
        {coins === 0 ? (
           "💰 Add More Coins"
        ) : (
          `💰 ${coins}`
        )}
      </button>

      {/* Panel */}
      <div className={`${styles.panel} ${open ? styles.open : ""}`}>
        <div className={styles.header}>
          <span>Enter API Key</span>
          <button className={styles.close} onClick={() => setOpen(false)}>
            ✕
          </button>
        </div>
        <span className={styles.info}>
          Don't have a key and your coins are empty? <br /> 
          Contact us via {" "}
          <a href="mailto:support@moralizer.ai" className={styles.link}>
            email{" "}
          </a>
          to get one or wait 24h for your coins to refill automatically.
        </span>

        <div className={styles.body}>

          <input
            className={styles.input}
            value={apiKey}
            onChange={(e) => setApiKey(e.target.value)}
            placeholder="md-..."
          />

          <button className={styles.save} onClick={saveKey}>
            Save Key
          </button>
        </div>

        <div className={styles.footer}>
          💰 Coins remaining: <b>{coins}</b>
        </div>
      </div>
    </div>
  );
}