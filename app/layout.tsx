import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "AIFred — Multi-Agent Trading Intelligence",
  description:
    "7-agent AI-powered trading system with deep learning, NLP sentiment analysis, and adaptive risk management.",
  icons: { icon: "/favicon.ico" },
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en" className="dark">
      <body className="bg-[#06060a] text-white antialiased">{children}</body>
    </html>
  );
}
