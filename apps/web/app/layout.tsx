import type { ReactNode } from "react";
import "./globals.css";

export const metadata = {
  title: "Quantum Circuit",
  description: "Research-grade quantum circuit editor + compute"
};

export default function RootLayout({ children }: { children: ReactNode }) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}

