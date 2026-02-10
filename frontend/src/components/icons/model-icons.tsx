/**
 * Official-style SVG icons for planning model providers.
 */

import { type FC, type SVGProps } from "react";

type IconProps = SVGProps<SVGSVGElement>;

/** Google Gemini icon – the four-pointed star gradient */
export const GeminiIcon: FC<IconProps> = (props) => (
  <svg viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg" {...props}>
    <defs>
      <linearGradient id="gemini-grad" x1="0" y1="0" x2="24" y2="24" gradientUnits="userSpaceOnUse">
        <stop offset="0%" stopColor="#4285F4" />
        <stop offset="25%" stopColor="#9B72CB" />
        <stop offset="50%" stopColor="#D96570" />
        <stop offset="75%" stopColor="#D96570" />
        <stop offset="100%" stopColor="#9B72CB" />
      </linearGradient>
    </defs>
    <path
      d="M12 2C12 2 14.5 8.5 17 11C19.5 13.5 22 12 22 12C22 12 19.5 14.5 17 17C14.5 19.5 12 22 12 22C12 22 9.5 19.5 7 17C4.5 14.5 2 12 2 12C2 12 4.5 13.5 7 11C9.5 8.5 12 2 12 2Z"
      fill="url(#gemini-grad)"
    />
  </svg>
);

/** Moonshot Kimi icon – crescent moon stylization */
export const KimiIcon: FC<IconProps> = (props) => (
  <svg viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg" {...props}>
    <defs>
      <linearGradient id="kimi-grad" x1="4" y1="2" x2="20" y2="22" gradientUnits="userSpaceOnUse">
        <stop offset="0%" stopColor="#1A1A2E" />
        <stop offset="100%" stopColor="#16213E" />
      </linearGradient>
    </defs>
    <circle cx="12" cy="12" r="10" fill="url(#kimi-grad)" />
    <path
      d="M14.5 4.5C11.5 5.5 9.5 8.5 9.5 12C9.5 15.5 11.5 18.5 14.5 19.5C10.5 20.5 6.5 17.5 5.5 13.5C4.5 9.5 6.5 5.5 10.5 3.5C11.8 3.0 13.2 3.5 14.5 4.5Z"
      fill="#E0E0FF"
    />
    <circle cx="16" cy="7" r="1" fill="#FFD700" />
    <circle cx="18" cy="10" r="0.6" fill="#FFD700" opacity="0.7" />
  </svg>
);

/** Alibaba Qwen VL icon – eye/vision motif with gradient */
export const QwenVLIcon: FC<IconProps> = (props) => (
  <svg viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg" {...props}>
    <defs>
      <linearGradient id="qwen-grad" x1="2" y1="6" x2="22" y2="18" gradientUnits="userSpaceOnUse">
        <stop offset="0%" stopColor="#6F3AFA" />
        <stop offset="100%" stopColor="#3B82F6" />
      </linearGradient>
    </defs>
    <path
      d="M12 5C7 5 2.73 8.11 1 12C2.73 15.89 7 19 12 19C17 19 21.27 15.89 23 12C21.27 8.11 17 5 12 5Z"
      fill="url(#qwen-grad)"
      opacity="0.15"
    />
    <path
      d="M12 5C7 5 2.73 8.11 1 12C2.73 15.89 7 19 12 19C17 19 21.27 15.89 23 12C21.27 8.11 17 5 12 5Z"
      stroke="url(#qwen-grad)"
      strokeWidth="1.5"
      strokeLinecap="round"
      strokeLinejoin="round"
    />
    <circle cx="12" cy="12" r="3.5" fill="url(#qwen-grad)" />
    <circle cx="12" cy="12" r="1.5" fill="white" />
  </svg>
);

/** Map model ID → icon component */
export const MODEL_ICONS: Record<string, FC<IconProps>> = {
  gemini: GeminiIcon,
  kimi: KimiIcon,
  "qwen-vl": QwenVLIcon,
};

/** Get icon for a model ID, returns undefined if not found */
export function getModelIcon(modelId: string): FC<IconProps> | undefined {
  return MODEL_ICONS[modelId];
}
