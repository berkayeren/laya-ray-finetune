// The 7 inline SVG icons of 11-UI-SPEC.md (Design System, icon library row): send, chevron-down, chevron-right, info,
// check, alert-triangle and x. Each is 16px, drawn in currentColor with stroke width 1.75 on a 24-unit grid, and
// aria-hidden: an icon always sits next to a text label or inside a control that carries its own accessible name.
// No icon is fetched: the paths are written here, and the only URL is the SVG namespace.

import type { ReactNode } from "react";

const SVG_NS = "http://www.w3.org/2000/svg";

function Icon({ children }: { children: ReactNode }) {
  return (
    <svg
      xmlns={SVG_NS}
      width={16}
      height={16}
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth={1.75}
      strokeLinecap="round"
      strokeLinejoin="round"
      aria-hidden="true"
      focusable="false"
    >
      {children}
    </svg>
  );
}

export function SendIcon() {
  return (
    <Icon>
      <path d="M22 2 11 13" />
      <path d="M22 2 15 22l-4-9-9-4z" />
    </Icon>
  );
}

export function ChevronDownIcon() {
  return (
    <Icon>
      <path d="m6 9 6 6 6-6" />
    </Icon>
  );
}

export function ChevronRightIcon() {
  return (
    <Icon>
      <path d="m9 18 6-6-6-6" />
    </Icon>
  );
}

export function InfoIcon() {
  return (
    <Icon>
      <circle cx="12" cy="12" r="10" />
      <path d="M12 16v-4" />
      <path d="M12 8h.01" />
    </Icon>
  );
}

export function CheckIcon() {
  return (
    <Icon>
      <path d="M20 6 9 17l-5-5" />
    </Icon>
  );
}

export function AlertTriangleIcon() {
  return (
    <Icon>
      <path d="M10.29 3.86 1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z" />
      <path d="M12 9v4" />
      <path d="M12 17h.01" />
    </Icon>
  );
}

export function XIcon() {
  return (
    <Icon>
      <path d="M18 6 6 18" />
      <path d="m6 6 12 12" />
    </Icon>
  );
}
