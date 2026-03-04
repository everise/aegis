/**
 * Icons for planning model providers (OpenRouter, Mock, etc.).
 */

import { type FC, type SVGProps } from "react";

type IconProps = SVGProps<SVGSVGElement>;

/** OpenRouter icon – layered routing paths */
export const OpenRouterIcon: FC<IconProps> = (props) => (
  <svg viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg" {...props}>
    <defs>
      <linearGradient id="or-grad" x1="2" y1="2" x2="22" y2="22" gradientUnits="userSpaceOnUse">
        <stop offset="0%" stopColor="#6366F1" />
        <stop offset="100%" stopColor="#8B5CF6" />
      </linearGradient>
    </defs>
    <rect x="2" y="2" width="20" height="20" rx="4" fill="url(#or-grad)" />
    <path
      d="M7 8h4l2 2-2 2H7M17 16h-4l-2-2 2-2h4"
      stroke="white"
      strokeWidth="1.5"
      strokeLinecap="round"
      strokeLinejoin="round"
    />
    <circle cx="12" cy="12" r="1.5" fill="white" />
  </svg>
);

/** Mock provider icon – test flask / beaker */
export const MockIcon: FC<IconProps> = (props) => (
  <svg viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg" {...props}>
    <defs>
      <linearGradient id="mock-grad" x1="4" y1="2" x2="20" y2="22" gradientUnits="userSpaceOnUse">
        <stop offset="0%" stopColor="#10B981" />
        <stop offset="100%" stopColor="#059669" />
      </linearGradient>
    </defs>
    <rect x="2" y="2" width="20" height="20" rx="4" fill="url(#mock-grad)" />
    <path
      d="M10 5v5l-3 6h10l-3-6V5M9 5h6"
      stroke="white"
      strokeWidth="1.5"
      strokeLinecap="round"
      strokeLinejoin="round"
    />
    <circle cx="11" cy="14" r="1" fill="white" opacity="0.7" />
    <circle cx="14" cy="13" r="0.7" fill="white" opacity="0.5" />
  </svg>
);

/** Vertex AI icon – Google Cloud triangle motif */
export const VertexIcon: FC<IconProps> = (props) => (
  <svg viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg" {...props}>
    <defs>
      <linearGradient id="vertex-grad" x1="4" y1="2" x2="20" y2="22" gradientUnits="userSpaceOnUse">
        <stop offset="0%" stopColor="#4285F4" />
        <stop offset="100%" stopColor="#34A853" />
      </linearGradient>
    </defs>
    <rect x="2" y="2" width="20" height="20" rx="4" fill="url(#vertex-grad)" />
    <path
      d="M12 6L6 18h12L12 6z"
      stroke="white"
      strokeWidth="1.5"
      strokeLinejoin="round"
    />
    <circle cx="12" cy="14" r="1.5" fill="white" />
  </svg>
);

/** Map provider ID → icon component */
export const MODEL_ICONS: Record<string, FC<IconProps>> = {
  openrouter: OpenRouterIcon,
  mock: MockIcon,
  vertex: VertexIcon,
};

/** Get icon for a provider ID, returns undefined if not found */
export function getModelIcon(modelId: string): FC<IconProps> | undefined {
  return MODEL_ICONS[modelId];
}
