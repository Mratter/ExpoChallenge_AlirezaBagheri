/** Vite's `?raw` suffix returns a module's file contents as a string. */
declare module '*?raw' {
  const content: string
  export default content
}
