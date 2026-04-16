/// <reference types="vite/client" />

interface ImportMetaEnv {
  readonly VITE_MATOMO_ENABLED?: string;
  readonly VITE_MATOMO_URL?: string;
  readonly VITE_MATOMO_SITE_ID?: string;
}

interface ImportMeta {
  readonly env: ImportMetaEnv;
}

// Declare static asset imports so TypeScript doesn't complain
declare module '*.png' {
  const src: string
  export default src
}
declare module '*.svg' {
  const src: string
  export default src
}
declare module '*.jpg' {
  const src: string
  export default src
}

