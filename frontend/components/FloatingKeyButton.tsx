"use client";

import { useEffect, useRef, useState } from "react";
import styles from "./FloatingKeyPanel.module.css";

export default function FloatingKeyPanel() {
  const [open, setOpen] = useState(false);
  const [apiKey, setApiKey] = useState("");
  const [coins, setCoins] = useState(0); // demo value
  const wrapperRef = useRef<HTMLDivElement | null>(null);

  // load saved key
  useEffect(() => {
    const saved = localStorage.getItem("apiKey");
    if (saved) setApiKey(saved);
  }, []);

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