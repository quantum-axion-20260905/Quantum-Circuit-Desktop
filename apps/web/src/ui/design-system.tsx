"use client";

import React from "react";

export type ButtonVariant = "primary" | "secondary" | "accent" | "success" | "danger" | "ghost";
export type ButtonSize = "sm" | "md";
export type CardTone = "light" | "dark";

export function Button({
  variant = "secondary",
  size = "sm",
  className = "",
  ...props
}: React.ButtonHTMLAttributes<HTMLButtonElement> & { variant?: ButtonVariant; size?: ButtonSize }) {
  return <button {...props} className={`qc-button qc-button-${variant} qc-button-${size} ${className}`.trim()} />;
}

export function Card({ tone = "light", className = "", ...props }: React.HTMLAttributes<HTMLElement> & { tone?: CardTone }) {
  return <section {...props} className={`qc-card qc-card-${tone} ${className}`.trim()} />;
}

export function ProgressBar({ value, tone = "accent", label }: { value: number; tone?: "accent" | "success"; label?: string }) {
  const percent = Math.round(Math.max(0, Math.min(1, value)) * 100);
  return (
    <div className="qc-progress" role="progressbar" aria-valuemin={0} aria-valuemax={100} aria-valuenow={percent} aria-label={label}>
      <div className={`qc-progress-fill qc-progress-${tone}`} style={{ width: `${percent}%` }} />
    </div>
  );
}

export function MetricGrid({ children, className = "" }: { children: React.ReactNode; className?: string }) {
  return <div className={`qc-metric-grid ${className}`.trim()}>{children}</div>;
}

export function Metric({ label, value, tone }: { label: string; value: React.ReactNode; tone?: "default" | "info" | "success" | "warning" | "danger" }) {
  return (
    <div className="qc-metric">
      <div className="qc-label">{label}</div>
      <div className={`qc-metric-value qc-metric-${tone ?? "default"}`}>{value}</div>
    </div>
  );
}

export function Field({ label, children, className = "" }: { label: string; children: React.ReactNode; className?: string }) {
  return <label className={`qc-field ${className}`.trim()}><span>{label}</span>{children}</label>;
}
