import react from '@vitejs/plugin-react'
import { defineConfig } from 'vitest/config'

export default defineConfig({
  plugins: [react()],
  server: {
    host: '127.0.0.1',
    port: 4173,
    strictPort: true,
    proxy: { '/api': 'http://127.0.0.1:4117', '/health': 'http://127.0.0.1:4117' },
  },
  test: {
    environment: 'jsdom',
    setupFiles: './tests/setup.ts',
  },
})
