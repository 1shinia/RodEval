import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import tailwindcss from '@tailwindcss/vite'
import path from 'path'

export default defineConfig({
  plugins: [react(), tailwindcss()],
  resolve: {
    alias: {
      '@': path.resolve(__dirname, './src'),
    },
  },
  build: {
    rolldownOptions: {
      output: {
        manualChunks(id) {
          if (id.includes('/node_modules/katex/')) return 'katex'
          if (id.includes('/node_modules/react-syntax-highlighter/') || id.includes('/node_modules/refractor/') || id.includes('/node_modules/prismjs/')) {
            return 'syntax-highlighter'
          }
          return undefined
        },
      },
    },
  },
  server: {
    port: 5173,
    host: '0.0.0.0',
    proxy: {
      '/api/v1': {
        target: 'http://127.0.0.1:9002',
        changeOrigin: true,
        xfwd: true,
      },
      '/health': {
        target: 'http://127.0.0.1:9002',
        changeOrigin: true,
        xfwd: true,
      },
    },
  },
})
