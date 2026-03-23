/// <reference types="vite/client" />

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

